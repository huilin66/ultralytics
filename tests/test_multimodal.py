# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the pixel-aligned multi-modal extension."""

import os
from copy import deepcopy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from ultralytics.data.augment import Mosaic, RandomPerspective
from ultralytics.data.multimodal import Modalities, MultiModalDataset, MultiModalFormat
from ultralytics.models.multimodal import MultiModalYOLO
from ultralytics.models.multimodal.fusion import parse_fusion_spec
from ultralytics.models.multimodal.modules import ModalFold, ModalSplit, ModalUnfold, MultiModalFusion
from ultralytics.models.multimodal.predict import MultiModalDetectionPredictor
from ultralytics.models.multimodal.pretrained import load_coco_pretrained
from ultralytics.models.multimodal.tasks import MultiModalDetectionModel
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils import DEFAULT_CFG, YAML
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.torch_utils import init_seeds


def _write_dataset(root, depth_shape=(12, 20)):
    """Create a two-image RGB-thermal-depth detection dataset with extension-mapped companion files."""
    for split in ("train", "val"):
        for name in ("rgb", "thermal", "depth"):
            (root / "images" / name / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / "rgb" / split).mkdir(parents=True, exist_ok=True)
        for index in range(2):
            stem = f"sample_{index}"
            assert cv2.imwrite(
                str(root / "images" / "rgb" / split / f"{stem}.jpg"), np.full((12, 20, 3), 10, dtype=np.uint8)
            )
            assert cv2.imwrite(
                str(root / "images" / "thermal" / split / f"{stem}.png"), np.full((12, 20), 20, dtype=np.uint8)
            )
            assert cv2.imwrite(
                str(root / "images" / "depth" / split / f"{stem}.png"), np.full(depth_shape, 30, dtype=np.uint8)
            )
            (root / "labels" / "rgb" / split / f"{stem}.txt").write_text("0 0.5 0.5 0.5 0.5\n")


def _data(root):
    """Return the multi-modal data dictionary used by the test dataset."""
    return {
        "path": root,
        "channels": 5,
        "names": {0: "object"},
        "modalities": [
            {"name": "rgb", "path": "images/rgb", "channels": 3, "color": "bgr"},
            {"name": "thermal", "path": "images/thermal", "channels": 1, "suffix": ".png"},
            {"name": "depth", "path": "images/depth", "channels": 1, "suffix": ".png"},
        ],
    }


def test_multimodal_dataset_stacks_paired_modalities(tmp_path):
    """A sample should contain RGB, thermal and depth channels in its declared order."""
    _write_dataset(tmp_path)
    data = _data(tmp_path)
    dataset = MultiModalDataset(
        img_path=tmp_path / "images" / "rgb" / "train",
        imgsz=32,
        augment=False,
        batch_size=2,
        data=data,
        hyp=DEFAULT_CFG,
    )

    raw = dataset.get_image_and_label(0)["img"]
    sample = dataset[0]["img"]
    assert raw.shape == (20, 32, 5)
    assert sample.shape == (5, 32, 32)
    assert sample[0, 6, 0].item() == pytest.approx(10, abs=4)  # JPEG compression can change RGB slightly.
    assert sample[3, 6, 0].item() == 20
    assert sample[4, 6, 0].item() == 30


def test_modalities_reject_unaligned_images(tmp_path):
    """All modality images must have an identical HxW before augmentation."""
    _write_dataset(tmp_path, depth_shape=(13, 20))
    modalities = Modalities(_data(tmp_path))
    with pytest.raises(ValueError, match="not pixel-aligned"):
        modalities.load(tmp_path / "images" / "rgb" / "train" / "sample_0.jpg")


def test_modalities_keep_the_staged_path_when_primary_is_a_symbolic_link(tmp_path):
    """Companion lookup must use the data-YAML layout rather than resolve a linked primary image to its source."""
    source = tmp_path / "source" / "visible"
    primary_root = tmp_path / "staged" / "images" / "primary"
    auxiliary_root = tmp_path / "staged" / "images" / "auxiliary"
    source.mkdir(parents=True)
    (primary_root / "train").mkdir(parents=True)
    (auxiliary_root / "train").mkdir(parents=True)
    source_image = source / "sample.png"
    primary_image = primary_root / "train" / "sample.png"
    auxiliary_image = auxiliary_root / "train" / "sample.png"
    assert cv2.imwrite(str(source_image), np.full((12, 20, 3), 10, dtype=np.uint8))
    assert cv2.imwrite(str(auxiliary_image), np.full((12, 20, 3), 20, dtype=np.uint8))
    try:
        os.symlink(source_image, primary_image)
    except OSError as error:
        pytest.skip(f"Symbolic links are unavailable in this test environment: {error}")

    modalities = Modalities(
        {
            "path": tmp_path / "staged",
            "channels": 6,
            "modalities": [
                {"name": "primary", "path": "images/primary", "channels": 3, "suffix": ".png"},
                {"name": "auxiliary", "path": "images/auxiliary", "channels": 3, "suffix": ".png"},
            ],
        }
    )

    assert modalities.paths(primary_image) == (primary_image, auxiliary_image)
    assert modalities.load(primary_image).shape == (12, 20, 6)


def test_multimodal_augmentations_keep_the_channel_stack(tmp_path):
    """Standard geometry transforms operate on the full stack without dropping or splitting a modality."""
    _write_dataset(tmp_path)
    hyp = deepcopy(DEFAULT_CFG)
    hyp.mosaic = hyp.mixup = hyp.copy_paste = hyp.cutmix = 0.0
    dataset = MultiModalDataset(
        img_path=tmp_path / "images" / "rgb" / "train",
        imgsz=32,
        augment=True,
        batch_size=2,
        data=_data(tmp_path),
        hyp=hyp,
    )
    assert dataset[0]["img"].shape == (5, 32, 32)


@pytest.mark.parametrize("channels", [3, 4, 6])
@pytest.mark.parametrize("perspective", [0.0, 0.001])
def test_multichannel_warp_fills_every_channel(channels, perspective):
    """Affine and perspective padding must use 114 for every sensor channel."""
    transform = RandomPerspective(perspective=perspective)
    matrix = np.array([[1, 0, 2], [0, 1, 2], [0, 0, 1]], dtype=np.float32)
    labels = {"img": np.full((8, 8, channels), 20, dtype=np.uint8)}
    result = transform.apply_image(labels, {"M": matrix, "size": (8, 8)})["img"]
    assert result.shape == (8, 8, channels)
    assert np.all(result[0, 0] == 114)
    assert np.all(result[4, 4] == 20)


@pytest.mark.parametrize("channels", [4, 6])
@pytest.mark.parametrize("empty", [False, True])
def test_multimodal_leakage_mosaic_backward(tmp_path, monkeypatch, channels, empty):
    """Paired Mosaic sources, including empty leakage labels, retain class-specific loss gradients."""
    init_seeds(0)
    _write_dataset(tmp_path)
    data = _data(tmp_path)
    data["modalities"].pop()
    data["channels"] = channels
    data["names"] = {0: "Hollow", 1: "Temperature", 2: "Leakage"}
    if channels == 6:
        data["modalities"][1]["channels"] = 3
        for path in (tmp_path / "images" / "thermal" / "train").glob("*.png"):
            assert cv2.imwrite(str(path), np.full((12, 20, 3), 20, dtype=np.uint8))
    (tmp_path / "labels/rgb/train/sample_1.txt").write_text("" if empty else "2 0.5 0.5 0.5 0.5\n")
    list_path = tmp_path / "leakage.txt"
    list_path.write_text("/server/images/rgb/train/sample_1.jpg\n")
    monkeypatch.setenv("LEAKAGE_ONLY_LIST", str(list_path))
    monkeypatch.setattr(Mosaic, "get_indexes", lambda self: [0, 1, 1])
    hyp = deepcopy(DEFAULT_CFG)
    hyp.mosaic = 1.0
    hyp.mixup = hyp.cutmix = hyp.copy_paste = hyp.degrees = hyp.translate = hyp.scale = hyp.shear = 0.0
    dataset = MultiModalDataset(
        img_path=tmp_path / "images/rgb/train", imgsz=32, augment=True, batch_size=2, data=data, hyp=hyp
    )
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    assert batch["img"].shape == (2, channels, 32, 32)
    if empty:
        assert not batch["cls"].eq(2).any()
    assert set(batch["cls_loss_spatial_mask"].unique().tolist()) == {0, 1}
    model = MultiModalDetectionModel(
        {
            "nc": 3,
            "channels": channels,
            "backbone": [[-1, 1, "Conv", [16, 3, 2]]],
            "head": [[[0], 1, "Detect", ["nc"]]],
        },
        verbose=False,
    ).train()
    model.args = hyp
    model._leakage_only_dataset_records = DetectionTrainer._leakage_only_dataset_records(dataset)
    criterion = v8DetectionLoss(model)
    preds = model(batch["img"].float() / 255)
    preds["scores"].retain_grad()
    loss, _ = criterion(preds, batch)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    anchors, strides = make_anchors(preds["feats"], model.stride, 0.5)
    mask = criterion._get_leakage_only_spatial_loss_mask(
        batch["cls_loss_spatial_mask"], anchors, strides, (32, 32), torch.float32
    ).permute(0, 2, 1)
    gradients = preds["scores"].grad
    assert (mask == 0).any()
    assert gradients[mask == 0].eq(0).all()
    assert gradients[:, 2].abs().sum() > 0
    assert model.model[0].conv.weight.grad.abs().sum() > 0


def test_modal_split_supports_any_branch_count():
    """Channel sections define branch count without a two-branch special case."""
    split = ModalSplit([3, 1, 2, 4])
    outputs = split(torch.zeros(1, 10, 8, 8))
    assert [output.shape[1] for output in outputs] == [3, 1, 2, 4]
    with pytest.raises(ValueError, match="expected 10 channels"):
        split(torch.zeros(1, 9, 8, 8))


def test_multimodal_fusion_operators_validate_feature_geometry():
    """Concatenation and addition should enforce their documented channel and geometry contracts."""
    features = [torch.ones(1, 2, 8, 8), torch.full((1, 2, 8, 8), 2.0)]
    assert MultiModalFusion([2, 2], "concat")(features).shape == (1, 4, 8, 8)
    assert torch.equal(MultiModalFusion([2, 2], "add")(features), torch.full((1, 2, 8, 8), 3.0))
    with pytest.raises(ValueError, match="equal input channel"):
        MultiModalFusion([2, 3], "add")
    with pytest.raises(ValueError, match="matching batch and spatial"):
        MultiModalFusion([2, 2], "concat")([features[0], torch.ones(1, 2, 4, 8)])


def test_modal_fold_unfold_uses_one_shared_batch():
    """Fold and unfold should preserve modality order while giving a stage one larger batch."""
    features = [torch.ones(2, 2, 8, 8), torch.full((2, 2, 8, 8), 2.0), torch.full((2, 2, 8, 8), 3.0)]
    folded = ModalFold([2, 2, 2])(features)
    assert folded.shape == (6, 2, 8, 8)
    unfolded = ModalUnfold(3)(folded)
    assert all(torch.equal(actual, expected) for actual, expected in zip(unfolded, features))


def test_fusion_spec_validates_modes_channels_and_weight_sharing():
    """Fusion metadata should reject inconsistent channel layouts and impossible sharing requests."""
    config = {
        "channels": 5,
        "multimodal": {
            "input_sections": [3, 1, 1],
            "fusion": "BF",
            "operator": "concat",
            "fusion_points": ["P3", "P4", "P5"],
            "share_weight": True,
        },
    }
    spec = parse_fusion_spec(config)
    assert spec.input_sections == (3, 1, 1)
    assert spec.shared_stages == ("encoder", "nape")

    config["multimodal"]["fusion"] = "IF"
    config["multimodal"]["fusion_points"] = []
    with pytest.raises(ValueError, match="share_weight"):
        parse_fusion_spec(config)


def test_multimodal_fusion_yaml_builds_a_multi_branch_model():
    """A registered multi-input YAML module should preserve the standard parser and model lifecycle."""
    cfg = {
        "nc": 2,
        "channels": 3,
        "multimodal": {
            "input_sections": [1, 1, 1],
            "fusion": "BF",
            "operator": "add",
            "fusion_points": ["P3"],
            "share_weight": False,
        },
        "backbone": [
            [-1, 1, "ModalSplit", [[1, 1, 1]]],
            [0, 1, "Index", [1, 0]],
            [0, 1, "Index", [1, 1]],
            [0, 1, "Index", [1, 2]],
            [[1, 2, 3], 1, "MultiModalFusion", ["add"]],
            [-1, 1, "Conv", [16, 3, 2]],
        ],
        "head": [[[5], 1, "Detect", ["nc"]]],
    }
    model = MultiModalDetectionModel(cfg, verbose=False)
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.zeros(1, 3, 64, 64))
    assert prediction[0].shape[0] == 1
    assert MultiModalYOLO("ultralytics/cfg/models/multimodal/yolo11-mm3-bf-seg.yaml", verbose=False).task == "segment"
    cfg["multimodal"]["share_weight"] = True
    with pytest.raises(ValueError, match="ModalFold and ModalUnfold"):
        MultiModalDetectionModel(cfg, verbose=False)


def test_shared_weight_yaml_runs_one_stage_on_folded_modalities():
    """A shared stage should run once on the folded batch rather than duplicate branch weights."""
    cfg = {
        "nc": 2,
        "channels": 2,
        "multimodal": {
            "input_sections": [1, 1],
            "fusion": "BF",
            "operator": "add",
            "fusion_points": ["P3"],
            "share_weight": True,
        },
        "backbone": [
            [-1, 1, "ModalSplit", [[1, 1]]],
            [0, 1, "Index", [1, 0]],
            [-1, 1, "Conv", [8, 3, 1]],
            [0, 1, "Index", [1, 1]],
            [-1, 1, "Conv", [8, 3, 1]],
            [[2, 4], 1, "ModalFold", []],
            [-1, 1, "Conv", [16, 3, 2]],
            [-1, 1, "ModalUnfold", [2]],
            [7, 1, "Index", [16, 0]],
            [7, 1, "Index", [16, 1]],
            [[8, 9], 1, "MultiModalFusion", ["add"]],
        ],
        "head": [[[10], 1, "Detect", ["nc"]]],
    }
    model = MultiModalDetectionModel(cfg, verbose=False).eval()
    with torch.inference_mode():
        prediction = model(torch.zeros(1, 2, 64, 64))
    assert model.model[6].conv.weight.shape[1] == 8
    assert prediction[0].shape[0] == 1


@pytest.mark.parametrize("facades", [False, True])
def test_if_coco_transfer_expands_the_first_convolution(tmp_path, facades):
    """IF should retain YOLOv8 weights while copying RGB channels into the expanded first convolution."""
    source = DetectionModel("ultralytics/cfg/models/v8/yolov8.yaml", verbose=False)
    cfg = yaml_model_load("ultralytics/cfg/mmodels/yolov8x-mm3-if.yaml")
    cfg["scale"] = "n"
    target = MultiModalDetectionModel(cfg, verbose=False)
    if facades:
        cfg_path = tmp_path / "yolov8n-mm3-if.yaml"
        YAML.save(cfg_path, cfg)
        target_facade = MultiModalYOLO(cfg_path, verbose=False)
        source_facade = YOLO("ultralytics/cfg/models/v8/yolov8.yaml", verbose=False)
        target, source = target_facade.model, source_facade.model
        report = load_coco_pretrained(target_facade, source_facade, verbose=False)
    else:
        report = load_coco_pretrained(target, source, verbose=False)

    assert torch.equal(target.model[0].conv.weight[:, :3], source.model[0].conv.weight)
    assert torch.count_nonzero(target.model[0].conv.weight[:, 3:]) == 0
    assert report.transformed_tensors == 1
    assert report.skipped_tensors == 0
    raw = np.full((32, 32, 5), [10, 20, 90, 30, 40], dtype=np.uint8)
    modalities = Modalities(_data(tmp_path))
    formatted = MultiModalFormat(modalities)._format_img(raw).float().unsqueeze(0)
    assert formatted[0, :, 0, 0].tolist() == [90, 20, 10, 30, 40]
    rgb = torch.tensor([90, 20, 10]).float().view(1, 3, 1, 1).expand(1, 3, 32, 32)
    torch.testing.assert_close(target.model[0].eval()(formatted), source.model[0].eval()(rgb))


def test_multimodal_disk_cache_separates_modality_order(tmp_path):
    """A changed sensor layout must not reuse fused images with the old channel order."""
    _write_dataset(tmp_path)
    data = _data(tmp_path)
    first = MultiModalDataset(
        img_path=tmp_path / "images/rgb/train", imgsz=32, augment=False, data=data, hyp=DEFAULT_CFG
    )
    first.cache_images_to_disk(0)
    data["modalities"][1:] = reversed(data["modalities"][1:])
    second = MultiModalDataset(
        img_path=tmp_path / "images/rgb/train", imgsz=32, augment=False, data=data, hyp=DEFAULT_CFG
    )
    assert first._cache_file(0) != second._cache_file(0)
    image = second.load_image(0)[0]
    assert np.all(image[..., 3] == 30)
    assert np.all(image[..., 4] == 20)

    def fail_save(*args, **kwargs):
        raise OSError("disk write failed")

    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as patch:
        patch.setattr(np, "save", fail_save)
        with pytest.raises(OSError, match="disk write failed"):
            second.cache_images_to_disk(0)


def test_multimodal_rectangular_sort_preserves_cache_pairing(tmp_path):
    """Rectangular validation must load the cache belonging to the sorted image path."""
    _write_dataset(tmp_path)
    for name, shape, suffix in [
        ("rgb", (20, 12, 3), ".jpg"),
        ("thermal", (20, 12), ".png"),
        ("depth", (20, 12), ".png"),
    ]:
        assert cv2.imwrite(str(tmp_path / f"images/{name}/train/sample_0{suffix}"), np.full(shape, 71, np.uint8))
    kwargs = dict(
        img_path=tmp_path / "images/rgb/train",
        imgsz=32,
        batch_size=2,
        augment=False,
        data=_data(tmp_path),
        hyp=DEFAULT_CFG,
    )
    original = MultiModalDataset(**kwargs)
    for index in range(len(original)):
        original.cache_images_to_disk(index)
    sorted_dataset = MultiModalDataset(**kwargs, rect=True)
    assert sorted_dataset.im_files[0].endswith("sample_1.jpg")
    assert np.all(sorted_dataset.load_image(0)[0][..., 0] == 10)


@pytest.mark.parametrize("from_img", [False, True])
def test_three_channel_prediction_distinguishes_files_and_arrays(tmp_path, from_img):
    """File inputs load paired sensors even when the primary loader already supplies three channels."""
    predictor = MultiModalDetectionPredictor(overrides={"save": False})
    predictor.model = SimpleNamespace(channels=3, fp16=False)
    predictor.device = torch.device("cpu")
    predictor.source_type = SimpleNamespace(from_img=from_img)
    predictor.batch = ([str(tmp_path / "primary.png")], None, None)
    paired = np.full((8, 8, 3), [10, 20, 30], dtype=np.uint8)
    predictor.modalities = SimpleNamespace(load=lambda path: paired, to_rgb=lambda image: image)
    predictor.pre_transform = lambda images: images
    image = np.full((8, 8, 3), 99, dtype=np.uint8)
    result = predictor.preprocess([image])
    expected = torch.tensor([99, 99, 99] if from_img else [10, 20, 30]) / 255
    torch.testing.assert_close(result[0, :, 0, 0], expected)


def test_multimodal_prediction_rejects_video_sources(monkeypatch):
    """Image pairing must not try to read a video path as a companion image."""
    from ultralytics.models.yolo.detect.predict import DetectionPredictor

    monkeypatch.setattr(
        DetectionPredictor, "setup_source", lambda self, *args: setattr(self, "dataset", SimpleNamespace(mode="video"))
    )
    predictor = MultiModalDetectionPredictor(overrides={"save": False})
    with pytest.raises(ValueError, match="paired images"):
        predictor.setup_source("video.mp4")


@pytest.mark.parametrize("method", ["val", "predict", "track"])
def test_multimodal_demo_inference_accepts_overrides(monkeypatch, method):
    """Caller confidence and tracker options override demo defaults without duplicate keywords."""
    import demo_multimodal_base as demo

    captured = {}

    def infer(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(demo, "MultiModalYOLO", lambda *args: SimpleNamespace(**{method: infer}))
    kwargs = {"conf": 0.5}
    if method == "track":
        kwargs["tracker"] = "bytetrack.yaml"
    if method != "val":
        kwargs["source"] = "images"
    getattr(demo, f"model_{method}")("weights.pt", data="data.yaml", weight_name=False, **kwargs)
    assert captured["conf"] == 0.5
    if method == "track":
        assert captured["tracker"] == "bytetrack.yaml"


def test_multimodal_trainer_initializes_actual_training_weights():
    """Semantic transfer belongs to the trainer model rather than the temporary YAML facade."""
    from ultralytics.models.multimodal.train import MultiModalDetectionTrainer

    source = DetectionModel("ultralytics/cfg/models/v8/yolov8.yaml", verbose=False)
    cfg = yaml_model_load("ultralytics/cfg/mmodels/yolov8x-mm3-if.yaml")
    cfg["scale"] = "n"
    trainer = object.__new__(MultiModalDetectionTrainer)
    trainer.args = SimpleNamespace(pretrained="source.pt", cls_remap=False)
    trainer.data = {"nc": 3, "channels": 5, "modalities": [{"channels": 3}, {"channels": 1}, {"channels": 1}]}
    trainer.resume = False
    model = trainer.get_model(cfg, weights=source, verbose=False)
    torch.testing.assert_close(model.model[0].conv.weight[:, :3], source.model[0].conv.weight)
    assert torch.count_nonzero(model.model[0].conv.weight[:, 3:]) == 0
    with torch.no_grad():
        model.model[0].conv.weight.fill_(0.75)
    finetuned = trainer.get_model(cfg, weights=model, verbose=False)
    torch.testing.assert_close(finetuned.model[0].conv.weight, model.model[0].conv.weight)
    trainer.data["modalities"] = [{"channels": 1}, {"channels": 3}, {"channels": 1}]
    with pytest.raises(ValueError, match="modality sections"):
        trainer.get_model(cfg, verbose=False)


def test_multimodal_prediction_rejects_wrong_sensor_sections(tmp_path, monkeypatch):
    """Matching channel totals cannot substitute for matching ordered sensor sections."""
    import ultralytics.models.multimodal.predict as multimodal_predict
    from ultralytics.models.yolo.detect.predict import DetectionPredictor

    data = _data(tmp_path)
    data["modalities"][:2] = reversed(data["modalities"][:2])
    monkeypatch.setattr(multimodal_predict, "check_det_dataset", lambda path: data)
    monkeypatch.setattr(
        DetectionPredictor, "setup_source", lambda self, *args: setattr(self, "dataset", SimpleNamespace(mode="image"))
    )
    predictor = MultiModalDetectionPredictor(overrides={"data": "data.yaml", "save": False})
    predictor.model = SimpleNamespace(
        channels=5, model=SimpleNamespace(fusion_spec=SimpleNamespace(input_sections=(3, 1, 1)))
    )
    with pytest.raises(ValueError, match="modality sections"):
        predictor.setup_source("image.jpg")


def test_mm2_transfer_averages_equal_width_auxiliary_input():
    """A three-channel thermal branch must honor mean initialization despite matching RGB width."""
    source = DetectionModel("ultralytics/cfg/models/v8/yolov8.yaml", verbose=False)
    cfg = yaml_model_load("ultralytics/cfg/mmodels/yolov8x-mm2-ef.yaml")
    cfg["scale"] = "n"
    target = MultiModalDetectionModel(cfg, verbose=False)
    load_coco_pretrained(target, source, verbose=False)
    auxiliary = cfg["multimodal"]["pretrained"]["copies"][1]["target"][0]
    expected = source.model[0].conv.weight.mean(1, keepdim=True).expand(-1, 3, -1, -1)
    torch.testing.assert_close(target.model[auxiliary].conv.weight, expected)


@pytest.mark.parametrize("method", ["write", "flush"])
def test_multimodal_console_propagates_io_failures(method):
    """Console capture must report disk errors instead of silently losing logs."""
    from demo_multimodal_base import Tee

    def fail(*args):
        raise OSError("disk full")

    tee = Tee(SimpleNamespace(write=fail, flush=fail))
    with pytest.raises(OSError, match="disk full"):
        getattr(tee, method)(*(["text"] if method == "write" else []))


@pytest.mark.parametrize(
    "filename",
    [
        "yolov8x-mm3-ef.yaml",
        "yolov8x-mm3-nif.yaml",
        "yolov8x-mm3-bf.yaml",
        "yolov8x-mm3-nf.yaml",
        "yolov8x-mm3-hf.yaml",
    ],
)
def test_multibranch_coco_transfer_uses_explicit_yaml_mappings(filename):
    """Every non-IF template should copy only declared layers and initialize its new fusion projections."""
    source = DetectionModel("ultralytics/cfg/models/v8/yolov8.yaml", verbose=False)
    cfg = yaml_model_load(f"ultralytics/cfg/mmodels/{filename}")
    cfg["scale"] = "n"
    target = MultiModalDetectionModel(cfg, verbose=False)
    report = load_coco_pretrained(target, source, verbose=False)

    assert report.copied_tensors > 300
    assert report.transformed_tensors == 2
    assert report.initialized_layers
    assert report.skipped_tensors == 0
