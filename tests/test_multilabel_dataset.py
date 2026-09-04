"""Tests for physical-box and n-hot-label dataset semantics."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

from multilabel_yolo.dataset import MultiLabelLabelError, MultiLabelYOLODataset, parse_multilabel_label_file


def _write_image(root: Path, stem: str, label_text: str | None):
    """Create one small YOLO image/label pair for a test."""
    image_dir, label_dir = root / "images", root / "labels"
    image_dir.mkdir(exist_ok=True)
    label_dir.mkdir(exist_ok=True)
    Image.new("RGB", (32, 32), (80, 100, 120)).save(image_dir / f"{stem}.jpg")
    if label_text is not None:
        (label_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8")


def _dataset(root: Path, augment=False, mosaic=None, mixup=0.0):
    cfg = get_cfg(
        DEFAULT_CFG,
        overrides={
            "imgsz": 32,
            "rect": False,
            "cache": False,
            "task": "detect",
            "single_cls": False,
            "classes": None,
            "fraction": 1.0,
            "mosaic": (1.0 if augment else 0.0) if mosaic is None else mosaic,
            "mixup": mixup,
            "copy_paste": 0.0,
            "fliplr": 1.0 if augment else 0.0,
            "flipud": 1.0 if augment else 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
        },
    )
    return MultiLabelYOLODataset(
        img_path=root / "images",
        imgsz=32,
        batch_size=2,
        augment=augment,
        hyp=cfg,
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.5,
        prefix="test: ",
        task="detect",
        classes=None,
        data={"nc": 5, "names": {i: str(i) for i in range(5)}},
    )


def test_two_label_object_remains_one_box(tmp_path):
    """A two-label line must remain one physical object after formatting."""
    _write_image(tmp_path, "a", "1,3 0.5 0.5 0.2 0.3\n")
    dataset = _dataset(tmp_path)
    sample = dataset[0]
    assert sample["bboxes"].shape == (1, 4)
    assert sample["cls_nhot"].shape == (1, 5)
    assert torch.equal(sample["cls_nhot"][0], torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32))
    assert sample["cls_nhot"][0].sum() == 2


def test_single_label_and_duplicate_normalization(tmp_path):
    """Single labels remain compatible and duplicate IDs are removed."""
    label = tmp_path / "a.txt"
    label.write_text("1,1,3 0.5 0.5 0.2 0.3\n", encoding="utf-8")
    combinations, bboxes, duplicates = parse_multilabel_label_file(label, 5)
    assert combinations == [(1, 3)]
    assert bboxes.shape == (1, 4)
    assert duplicates == 0

    _write_image(tmp_path, "a", "1,1,3 0.5 0.5 0.2 0.3\n")
    _write_image(tmp_path, "b", "2 0.5 0.5 0.2 0.3\n")
    dataset = _dataset(tmp_path)
    sample = dataset[1]  # sorted image names: a, b
    assert sample["bboxes"].shape[0] == 1
    assert sample["cls_nhot"][0, 2] == 1
    assert sample["cls_nhot"][0].sum() == 1


@pytest.mark.parametrize("class_field", ["-1,2", "5,2"])
def test_invalid_class_ids_fail_fast(tmp_path, class_field):
    """Out-of-range IDs must not be silently converted into background samples."""
    label = tmp_path / "bad.txt"
    label.write_text(f"{class_field} 0.5 0.5 0.2 0.3\n", encoding="utf-8")
    with pytest.raises(MultiLabelLabelError):
        parse_multilabel_label_file(label, 5)


def test_collate_keeps_physical_alignment(tmp_path):
    """Collation concatenates n-hot rows exactly once per physical bbox."""
    _write_image(tmp_path, "a", "1,3 0.5 0.5 0.2 0.3\n")
    _write_image(tmp_path, "b", "2 0.4 0.4 0.1 0.2\n")
    dataset = _dataset(tmp_path)
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    assert batch["bboxes"].shape == (2, 4)
    assert batch["cls_nhot"].shape == (2, 5)
    assert batch["batch_idx"].shape == (2,)
    assert torch.equal(batch["cls_nhot"].sum(1), torch.tensor([2.0, 1.0]))


def test_mosaic_and_flip_keep_transport_alignment(tmp_path):
    """Native geometric transforms may remove/reorder rows, but not their labels."""
    for index in range(4):
        combo = "1,3" if index % 2 == 0 else "2"
        _write_image(tmp_path, str(index), f"{combo} 0.5 0.5 0.3 0.3\n")
    dataset = _dataset(tmp_path, augment=True)
    sample = dataset[0]
    decoded = dataset.codec.to_nhot(sample["cls"], dtype=torch.float32)
    assert sample["bboxes"].shape[0] == sample["cls_nhot"].shape[0]
    assert torch.equal(decoded, sample["cls_nhot"])


def test_mixup_keeps_transport_alignment(tmp_path):
    """MixUp concatenation preserves one n-hot row per physical box."""
    for index in range(2):
        combo = "1,3" if index == 0 else "2"
        _write_image(tmp_path, str(index), f"{combo} 0.5 0.5 0.3 0.3\n")
    dataset = _dataset(tmp_path, augment=True, mosaic=0.0, mixup=1.0)
    sample = dataset[0]
    decoded = dataset.codec.to_nhot(sample["cls"], dtype=torch.float32)
    assert sample["bboxes"].shape[0] == sample["cls_nhot"].shape[0]
    assert torch.equal(decoded, sample["cls_nhot"])
