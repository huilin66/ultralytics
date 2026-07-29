# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the pixel-aligned multi-modal extension."""

from copy import deepcopy

import cv2
import numpy as np
import pytest
import torch

from ultralytics.data.multimodal import Modalities, MultiModalDataset
from ultralytics.models.multimodal import MultiModalYOLO
from ultralytics.models.multimodal.fusion import parse_fusion_spec
from ultralytics.models.multimodal.modules import ModalFold, ModalSplit, ModalUnfold, MultiModalFusion
from ultralytics.models.multimodal.pretrained import load_coco_pretrained
from ultralytics.models.multimodal.tasks import MultiModalDetectionModel
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
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


def test_if_coco_transfer_expands_the_first_convolution():
    """IF should retain YOLOv8 weights while copying RGB channels into the expanded first convolution."""
    source = DetectionModel("ultralytics/cfg/models/v8/yolov8.yaml", verbose=False)
    cfg = yaml_model_load("ultralytics/cfg/mmodels/yolov8x-mm3-if.yaml")
    cfg["scale"] = "n"
    target = MultiModalDetectionModel(cfg, verbose=False)
    report = load_coco_pretrained(target, source, verbose=False)

    assert torch.equal(target.model[0].conv.weight[:, :3], source.model[0].conv.weight)
    assert torch.count_nonzero(target.model[0].conv.weight[:, 3:]) == 0
    assert report.transformed_tensors == 1
    assert report.skipped_tensors == 0


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
