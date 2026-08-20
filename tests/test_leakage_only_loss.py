# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ultralytics.data.augment import BaseMixTransform, CopyPaste, Mosaic
from ultralytics.utils.loss import v8DetectionLoss


class DummyDetectHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.nc = 3
        self.reg_max = 2


class DummyDetectionModel(nn.Module):
    def __init__(self, records):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.model = nn.ModuleList([nn.Identity(), DummyDetectHead()])
        self.args = SimpleNamespace(box=1.0, cls=1.0, dfl=1.0)
        self._leakage_only_dataset_records = records


class DummyAugmentDataset:
    def __init__(self):
        self.im_files = ["normal_a.jpg", "leak.jpg", "normal_b.jpg"]
        self.leakage_only_files = frozenset({"leak.jpg"})
        self.mixing_indices = [0, 2]
        self.mixing_indices_set = set(self.mixing_indices)
        self.buffer = [1]
        self.cache = None
        self.mix_calls = []

    def is_leakage_only(self, im_file):
        return im_file in self.leakage_only_files

    def get_mixing_indices(self):
        return self.mixing_indices

    def get_image_and_label(self, index):
        self.mix_calls.append(index)
        return {"im_file": self.im_files[index]}


class RecordingMixTransform(BaseMixTransform):
    def apply_image(self, labels, params=None):
        labels["mixed_files"] = [item["im_file"] for item in labels["mix_labels"]]
        return labels

    def apply_instances(self, labels, params=None):
        return labels

    def apply_semantic(self, labels, params=None):
        return labels


class DummyInstances:
    segments = [object()]


def test_leakage_only_images_are_skipped_and_excluded_from_mixing_sources():
    dataset = DummyAugmentDataset()
    transform = RecordingMixTransform(dataset, p=1.0)

    normal = transform({"im_file": "normal_a.jpg"})
    assert normal["mixed_files"] in (["normal_a.jpg"], ["normal_b.jpg"])
    assert all(not dataset.is_leakage_only(path) for path in normal["mixed_files"])

    calls_before = len(dataset.mix_calls)
    leakage = {"im_file": "leak.jpg"}
    assert transform(leakage) is leakage
    assert "mixed_files" not in leakage
    assert len(dataset.mix_calls) == calls_before

    mosaic = Mosaic(dataset, p=1.0, n=4)
    assert all(index in dataset.mixing_indices for index in mosaic.get_indexes())

    copy_paste = CopyPaste(dataset, p=1.0, mode="flip")
    copy_paste_labels = {"im_file": "leak.jpg", "instances": DummyInstances()}
    assert copy_paste(copy_paste_labels) is copy_paste_labels


def test_leakage_only_does_not_override_augmentation_probabilities(monkeypatch):
    import demo_base

    monkeypatch.setenv("LEAKAGE_ONLY_LIST", "configured.txt")
    captured = {}

    class DummyModel:
        def __init__(self, *args, **kwargs):
            pass

        def load(self, *args, **kwargs):
            pass

        def add_callback(self, *args, **kwargs):
            pass

        def train(self, **kwargs):
            captured.update(kwargs)

    demo_base.model_train(
        "yolov8x.yaml",
        "weights.pt",
        network=DummyModel,
        mosaic=0.7,
        mixup=0.2,
        cutmix=0.1,
        copy_paste=0.3,
    )

    assert {key: captured[key] for key in ("mosaic", "mixup", "cutmix", "copy_paste")} == {
        "mosaic": 0.7,
        "mixup": 0.2,
        "cutmix": 0.1,
        "copy_paste": 0.3,
    }


def make_criterion(monkeypatch, tmp_path, records, contents="leak.jpg\n", tal_topk=10):
    list_path = tmp_path / "leakage_only.txt"
    list_path.write_text(contents, encoding="utf-8")
    monkeypatch.setenv("LEAKAGE_ONLY_LIST", str(list_path))
    return v8DetectionLoss(DummyDetectionModel(records), tal_topk=tal_topk)


def test_leakage_only_masked_classification_and_box_dfl_gradients(monkeypatch, tmp_path):
    criterion = make_criterion(
        monkeypatch,
        tmp_path,
        [("/dataset/normal/normal.jpg", [0, 1, 2]), ("C:/dataset/leak/leak.jpg", [2])],
    )
    pred_scores = torch.zeros((2, 1, 3), requires_grad=True)
    target_scores = torch.zeros_like(pred_scores)
    bce = criterion.bce(pred_scores, target_scores)
    masked_bce = bce * criterion._get_leakage_only_loss_mask(
        ["/dataset/normal/normal.jpg", "C:/dataset/leak/leak.jpg"], 2, pred_scores.device, pred_scores.dtype
    )
    masked_bce.sum().backward()
    assert pred_scores.grad[0].ne(0).all()
    assert pred_scores.grad[1, :, :2].eq(0).all()
    assert pred_scores.grad[1, :, 2].ne(0).all()

    pred_dist = torch.randn((1, 1, 8), requires_grad=True)
    pred_bboxes = torch.tensor([[[0.2, 0.2, 0.7, 0.7]]], requires_grad=True)
    anchor_points = torch.tensor([[0.5, 0.5]])
    target_bboxes = torch.tensor([[[0.25, 0.25, 0.75, 0.75]]])
    target_scores = torch.tensor([[[0.0, 0.0, 1.0]]])
    fg_mask = torch.tensor([[True]])
    box_loss, dfl_loss = criterion.bbox_loss(
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        torch.tensor(1.0),
        fg_mask,
        torch.tensor([640.0, 640.0]),
        torch.tensor([8.0]),
    )
    (box_loss + dfl_loss).backward()
    assert pred_bboxes.grad is not None
    assert pred_dist.grad is not None
    assert pred_bboxes.grad.abs().sum() > 0
    assert pred_dist.grad.abs().sum() > 0


def test_leakage_only_mask_applies_inside_detection_loss(monkeypatch, tmp_path):
    criterion = make_criterion(
        monkeypatch,
        tmp_path,
        [("/dataset/normal/normal.jpg", [0]), ("C:/dataset/leak/leak.jpg", [2])],
        tal_topk=1,
    )
    preds = {
        "boxes": torch.zeros((2, 8, 3), requires_grad=True),
        "scores": torch.zeros((2, 3, 3), requires_grad=True),
        "feats": [torch.zeros((2, 1, 1, 1)) for _ in range(3)],
    }
    batch = {
        "batch_idx": torch.tensor([0, 1]),
        "cls": torch.tensor([[0], [2]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]),
        "im_file": ["/dataset/normal/normal.jpg", "C:/dataset/leak/leak.jpg"],
    }
    _, loss, _ = criterion.get_assigned_targets_and_loss(preds, batch)
    loss.sum().backward()
    assert preds["scores"].grad[0].ne(0).all()
    assert preds["scores"].grad[1, :2, :].eq(0).all()
    assert preds["scores"].grad[1, 2, :].ne(0).all()
    assert preds["boxes"].grad is not None
    assert preds["boxes"].grad.abs().sum() > 0


@pytest.mark.parametrize(
    ("records", "contents", "error"),
    [
        ([("/dataset/a/leak.jpg", [2])], "missing.jpg\n", "not present"),
        ([("/dataset/a/leak.jpg", [0, 2])], "leak.jpg\n", "outside class 2"),
        (
            [("/dataset/a/leak.jpg", [2]), ("/dataset/b/leak.jpg", [2])],
            "leak.jpg\n",
            "Duplicate training image filenames",
        ),
    ],
)
def test_leakage_only_dataset_validation(monkeypatch, tmp_path, records, contents, error):
    with pytest.raises(ValueError, match=error):
        make_criterion(monkeypatch, tmp_path, records, contents)


@pytest.mark.parametrize("image_path", ["/dataset/leak/leak.jpg", "C:/dataset/leak/leak.jpg"])
def test_leakage_only_list_accepts_full_training_paths(monkeypatch, tmp_path, image_path):
    criterion = make_criterion(
        monkeypatch,
        tmp_path,
        [(image_path, [2])],
        contents=f"{image_path}\n",
    )
    mask = criterion._get_leakage_only_loss_mask([image_path], 1, torch.device("cpu"), torch.float32)
    assert torch.equal(mask[0, 0, :2], torch.zeros(2))

def test_leakage_only_list_must_exist_and_contain_a_name(monkeypatch, tmp_path):
    missing = tmp_path / "missing.txt"
    monkeypatch.setenv("LEAKAGE_ONLY_LIST", str(missing))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        v8DetectionLoss(DummyDetectionModel([]))

    empty = tmp_path / "empty.txt"
    empty.write_text("\n# comment\n", encoding="utf-8")
    monkeypatch.setenv("LEAKAGE_ONLY_LIST", str(empty))
    with pytest.raises(ValueError, match="empty"):
        v8DetectionLoss(DummyDetectionModel([]))


def test_leakage_only_validation_criterion_does_not_require_training_metadata(monkeypatch, tmp_path):
    list_path = tmp_path / "leakage_only.txt"
    list_path.write_text("/dataset/leak/leak.jpg\n", encoding="utf-8")
    monkeypatch.setenv("LEAKAGE_ONLY_LIST", str(list_path))
    model = DummyDetectionModel([])
    model.eval()

    criterion = v8DetectionLoss(model)
    assert criterion.leakage_only_files is None


def test_leakage_only_list_is_loaded_once(monkeypatch, tmp_path):
    criterion = make_criterion(
        monkeypatch,
        tmp_path,
        [("C:/dataset/leak/leak.jpg", [2])],
    )
    (tmp_path / "leakage_only.txt").write_text("different.jpg\n", encoding="utf-8")
    mask = criterion._get_leakage_only_loss_mask(["C:/dataset/leak/leak.jpg"], 1, torch.device("cpu"), torch.float32)
    assert torch.equal(mask[0, 0, :2], torch.zeros(2))
