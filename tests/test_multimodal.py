# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the pixel-aligned multi-modal extension."""

from copy import deepcopy

import cv2
import numpy as np
import pytest
import torch

from ultralytics.data.multimodal import Modalities, MultiModalDataset
from ultralytics.models.multimodal import MultiModalYOLO
from ultralytics.models.multimodal.tasks import MultiModalDetectionModel
from ultralytics.nn.modules import ModalSplit
from ultralytics.utils import DEFAULT_CFG


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


def test_modal_split_supports_any_branch_count():
    """Channel sections define branch count without a two-branch special case."""
    split = ModalSplit([3, 1, 2, 4])
    outputs = split(torch.zeros(1, 10, 8, 8))
    assert [output.shape[1] for output in outputs] == [3, 1, 2, 4]
    with pytest.raises(ValueError, match="expected 10 channels"):
        split(torch.zeros(1, 9, 8, 8))


def test_modal_split_yaml_builds_a_multi_branch_model():
    """ModalSplit and stock Index/Concat modules build a model without parser special cases."""
    cfg = {
        "nc": 2,
        "channels": 5,
        "backbone": [
            [-1, 1, "ModalSplit", [[3, 1, 1]]],
            [0, 1, "Index", [3, 0]],
            [0, 1, "Index", [1, 1]],
            [0, 1, "Index", [1, 2]],
            [[1, 2, 3], 1, "Concat", [1]],
            [-1, 1, "Conv", [16, 3, 2]],
        ],
        "head": [[[5], 1, "Detect", ["nc"]]],
    }
    model = MultiModalDetectionModel(cfg, verbose=False)
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.zeros(1, 5, 64, 64))
    assert prediction[0].shape[0] == 1
    assert MultiModalYOLO("ultralytics/cfg/models/11/yolo11-mm3-seg.yaml", verbose=False).task == "segment"
