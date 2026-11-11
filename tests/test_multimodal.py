# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the pixel-aligned multi-modal extension."""

from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from ultralytics.data.loaders import LoadImagesAndVideos
from ultralytics.data.multimodal import Modalities, MultiModalDataset
from ultralytics.models.multimodal import MultiModalYOLO
from ultralytics.models.multimodal.config import load_multimodal_config
from ultralytics.models.multimodal.fusion import parse_fusion_spec
from ultralytics.models.multimodal.modules import ModalFold, ModalSplit, ModalUnfold, MultiModalFusion
from ultralytics.models.multimodal.predict import MultiModalPredictorMixin
from ultralytics.models.multimodal.pretrained import load_coco_pretrained
from ultralytics.models.multimodal.tasks import MultiModalDetectionModel, MultiModalSegmentationModel
from ultralytics.nn.modules import Conv
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


def test_coco_transfer_accepts_the_multimodal_facade():
    """The public facade must be accepted directly, since it is an nn.Module without a nested ``.model``."""
    source = DetectionModel("ultralytics/cfg/models/v8/yolov8.yaml", verbose=False)
    facade = MultiModalYOLO("ultralytics/cfg/mmodels/yolov8n-mm3-if.yaml", task="detect")
    report = load_coco_pretrained(facade, source, verbose=False)

    assert report.copied_tensors > 300
    assert report.transformed_tensors == 1


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
def test_multibranch_coco_transfer_derives_the_yaml_mappings(filename):
    """Every non-IF template should copy the donor graph derived from its own layer declarations."""
    source = DetectionModel("ultralytics/cfg/models/v8/yolov8.yaml", verbose=False)
    cfg = yaml_model_load(f"ultralytics/cfg/mmodels/{filename}")
    cfg["scale"] = "n"
    target = MultiModalDetectionModel(cfg, verbose=False)
    report = load_coco_pretrained(target, source, verbose=False)

    assert report.copied_tensors > 300
    assert report.transformed_tensors == 2
    assert report.initialized_layers
    assert report.skipped_tensors == 0


def test_every_template_inherits_the_shared_contract():
    """Every template must resolve base.yaml and declare only its own transfer policy."""
    for path in sorted(Path("ultralytics/cfg/mmodels").glob("*.yaml")):
        if path.name == "base.yaml":
            continue
        config = load_multimodal_config(path)
        multimodal = config["multimodal"]
        assert config["channels"] == sum(multimodal["input_sections"]), path.name
        assert multimodal["operator"] == "concat", path.name
        assert multimodal["share_weight"] is False, path.name
        assert multimodal["pretrained"] == {"source": "auto"}, path.name
        assert load_multimodal_config(config) == config, path.name


DONOR_CONFIGS = {
    "yolov8": "ultralytics/cfg/models/v8/yolov8.yaml",
    "yolov9": "ultralytics/cfg/models/v9/yolov9{scale}.yaml",
    "yolov10": "ultralytics/cfg/models/v10/yolov10{scale}.yaml",
    "yolo11": "ultralytics/cfg/models/11/yolo11.yaml",
    "yolo12": "ultralytics/cfg/models/12/yolo12.yaml",
    "yolo26": "ultralytics/cfg/models/26/yolo26.yaml",
}


def _template_family_and_scale(stem):
    """Return the donor family and scale encoded in a template file stem."""
    family = next((name for name in sorted(DONOR_CONFIGS, key=len, reverse=True) if stem.startswith(name)), None)
    if family is None:
        raise AssertionError(f"{stem} does not name a supported architecture family.")
    scale = next(char for char in stem[len(family) :] if char in "nsmlxtceb")
    return family, scale


def _build_donor(family, scale):
    """Build the COCO donor graph of one family and scale from the upstream YAML."""
    cfg = yaml_model_load(DONOR_CONFIGS[family].format(scale=scale))
    if "scales" in cfg:
        cfg["scale"] = scale
    return DetectionModel(cfg, verbose=False)


def test_every_template_loads_its_family_coco_checkpoint():
    """Every template must accept a COCO checkpoint of its own architecture family and scale."""
    groups = {}
    for path in sorted(Path("ultralytics/cfg/mmodels").glob("*.yaml")):
        if path.name != "base.yaml":
            groups.setdefault(_template_family_and_scale(path.stem), []).append(path)
    assert len(groups) == 27, "expected one donor per family and scale"
    for (family, scale), paths in groups.items():
        donor = _build_donor(family, scale)
        for path in paths:
            cfg = yaml_model_load(path)
            if "scales" in cfg:
                cfg["scale"] = scale
            target = MultiModalDetectionModel(cfg, verbose=False)
            report = load_coco_pretrained(target, donor, verbose=False)
            assert report.copied_tensors > 0, path.name
            assert report.skipped_tensors == 0, path.name
        del donor


def test_yolo26x_multimodal_model_keeps_the_native_end2end_head():
    """A YOLO26x template must build, infer end to end and backpropagate into every modality branch."""
    cfg = yaml_model_load("ultralytics/cfg/mmodels/yolo26x-mm3-bf.yaml")
    model = MultiModalDetectionModel(cfg, nc=2, verbose=False)
    model.args = DEFAULT_CFG
    assert model.end2end is True
    assert model.yaml["channels"] == 5
    assert len(model.model[37].channels) == 3  # one fused P3 tensor per modality

    model.eval()
    with torch.inference_mode():
        prediction, _ = model(torch.zeros(1, 5, 64, 64))
    assert prediction.shape == (1, 84, 6)  # 84 anchors at 64 px, 4 box + 2 class columns

    model.train()
    batch = {
        "img": torch.rand(2, 5, 64, 64),
        "cls": torch.zeros(2, 1, 1),
        "bboxes": torch.tensor([[32.0, 32.0, 24.0, 24.0], [40.0, 20.0, 16.0, 12.0]]),
        "batch_idx": torch.tensor([0, 1]),
    }
    loss, _ = model(batch)
    loss.sum().backward()
    for layer in (2, 14, 26):  # first convolution of the RGB, thermal and depth backbones
        gradient = model.model[layer].conv.weight.grad
        assert gradient is not None and gradient.abs().sum() > 0


@pytest.mark.parametrize(
    "mode, branch_layers",
    [
        ("if", (0,)),
        ("ef", (2, 12, 22)),
        ("nif", (2, 12, 22)),
        ("bf", (2, 14, 26)),
        ("nf", (2, 14, 26)),
        ("hf", (2, 14, 26)),
    ],
)
def test_yolo26x_fusion_modes_build_and_backpropagate(mode, branch_layers):
    """Every YOLO26x fusion template must build, infer end to end and train all input paths."""
    cfg = yaml_model_load(f"ultralytics/cfg/mmodels/yolo26x-mm3-{mode}.yaml")
    cfg["scale"] = "n"
    model = MultiModalDetectionModel(cfg, nc=2, verbose=False)
    model.args = DEFAULT_CFG
    assert model.end2end is True

    model.eval()
    with torch.inference_mode():
        prediction, _ = model(torch.zeros(1, 5, 64, 64))
    assert prediction.shape == (1, 84, 6)

    model.train()
    batch = {
        "img": torch.rand(2, 5, 64, 64),
        "cls": torch.zeros(2, 1, 1),
        "bboxes": torch.tensor([[32.0, 32.0, 24.0, 24.0], [40.0, 20.0, 16.0, 12.0]]),
        "batch_idx": torch.tensor([0, 1]),
    }
    loss, _ = model(batch)
    loss.sum().backward()
    for layer in branch_layers:
        gradient = model.model[layer].conv.weight.grad
        assert gradient is not None and gradient.abs().sum() > 0

    fusion_layers = {
        "if": (),
        "ef": (31, 33, 35),
        "nif": (31, 33, 38),
        "bf": (37, 39, 41),
        "nf": (73, 75, 77),
        "hf": (82, 84, 86),
    }[mode]
    for layer in fusion_layers:
        assert len(model.model[layer].channels) == 3


@pytest.mark.parametrize(
    "tag, channels",
    [("31", 4), ("11", 2)],
)
@pytest.mark.parametrize(
    "mode, branch_layers, fusion_count",
    [
        ("if", (0,), 0),
        ("ef", (2, 12), 3),
        ("nif", (2, 12), 3),
        ("bf", (2, 14), 3),
        ("nf", (2, 14), 3),
        ("hf", (2, 14), 3),
    ],
)
def test_yolo26x_mm2_templates_match_the_dataset_modality_split(tag, channels, mode, branch_layers, fusion_count):
    """Each two-modality template must build, infer and train both branches for its declared channel split."""
    cfg = yaml_model_load(f"ultralytics/cfg/mmodels/yolo26x-mm2-{tag}-{mode}.yaml")
    cfg["scale"] = "n"
    model = MultiModalDetectionModel(cfg, nc=2, verbose=False)
    model.args = DEFAULT_CFG
    assert model.end2end is True
    assert model.yaml["channels"] == channels
    sections = cfg["multimodal"]["input_sections"]
    assert sections == ([3, 1] if tag == "31" else [1, 1])
    if mode == "if":
        assert model.model[0].conv.weight.shape[1] == channels  # one stem over the stacked input
    else:
        assert [model.model[layer].conv.weight.shape[1] for layer in branch_layers] == sections

    fusions = [(index, layer) for index, layer in enumerate(model.model) if type(layer).__name__ == "MultiModalFusion"]
    assert len(fusions) == fusion_count
    assert all(len(layer.channels) == 2 for _, layer in fusions)  # two modalities per fusion node

    model.eval()
    with torch.inference_mode():
        prediction, _ = model(torch.zeros(1, channels, 64, 64))
    assert prediction.shape == (1, 84, 6)

    model.train()
    batch = {
        "img": torch.rand(2, channels, 64, 64),
        "cls": torch.zeros(2, 1, 1),
        "bboxes": torch.tensor([[32.0, 32.0, 24.0, 24.0], [40.0, 20.0, 16.0, 12.0]]),
        "batch_idx": torch.tensor([0, 1]),
    }
    loss, _ = model(batch)
    loss.sum().backward()
    for layer in branch_layers:
        gradient = model.model[layer].conv.weight.grad
        assert gradient is not None and gradient.abs().sum() > 0


@pytest.mark.parametrize(
    "path, sections, fusion_inputs",
    [
        ("ultralytics/cfg/mmodels/yolov8x-mm3-if.yaml", [3, 1, 1], 0),
        ("ultralytics/cfg/mmodels/yolov8x-mm3-ef.yaml", [3, 1, 1], 3),
        ("ultralytics/cfg/mmodels/yolov8x-mm3-nif.yaml", [3, 1, 1], 3),
        ("ultralytics/cfg/mmodels/yolov8x-mm3-bf.yaml", [3, 1, 1], 3),
        ("ultralytics/cfg/mmodels/yolov8x-mm3-nf.yaml", [3, 1, 1], 3),
        ("ultralytics/cfg/mmodels/yolov8x-mm3-hf.yaml", [3, 1, 1], 3),
        ("ultralytics/cfg/mmodels/yolov8x-mm2-if.yaml", [3, 3], 0),
        ("ultralytics/cfg/mmodels/yolov8x-mm2-ef.yaml", [3, 3], 3),
        ("ultralytics/cfg/mmodels/yolov8x-mm2-nif.yaml", [3, 3], 3),
        ("ultralytics/cfg/mmodels/yolov8x-mm2-bf.yaml", [3, 3], 3),
        ("ultralytics/cfg/mmodels/yolov8x-mm2-nf.yaml", [3, 3], 3),
        ("ultralytics/cfg/mmodels/yolov8x-mm2-hf.yaml", [3, 3], 3),
        ("ultralytics/cfg/models/multimodal/yolov8x-mm3-pre-sppf.yaml", [3, 1, 1], 3),
    ],
)
def test_yolov8x_templates_build_and_backpropagate(path, sections, fusion_inputs):
    """Every YOLOv8 template must build, infer and train each modality branch it declares."""
    cfg = yaml_model_load(path)
    cfg["scale"] = "n"
    model = MultiModalDetectionModel(cfg, nc=2, verbose=False)
    model.args = DEFAULT_CFG
    channels = sum(sections)
    assert model.yaml["channels"] == channels
    assert list(model.yaml["multimodal"]["input_sections"]) == sections

    fusions = [layer for layer in model.model if type(layer).__name__ == "MultiModalFusion"]
    assert len(fusions) == (3 if fusion_inputs else 0)
    assert all(len(layer.channels) == len(sections) for layer in fusions)

    model.eval()
    with torch.inference_mode():
        prediction, _ = model(torch.zeros(1, channels, 64, 64))
    assert prediction.shape == (1, 4 + 2, 84)  # decoded predictions: 4 box + 2 class channels, 84 anchors

    model.train()
    batch = {
        "img": torch.rand(2, channels, 64, 64),
        "cls": torch.zeros(2, 1, 1),
        "bboxes": torch.tensor([[32.0, 32.0, 24.0, 24.0], [40.0, 20.0, 16.0, 12.0]]),
        "batch_idx": torch.tensor([0, 1]),
    }
    loss, _ = model(batch)
    loss.sum().backward()
    if fusion_inputs == 0:
        stems = [model.model[0]]  # single stem over the stacked input
    else:
        stems = [layer for layer in model.model if isinstance(layer, Conv) and layer.conv.in_channels in sections]
        assert len(stems) == len(sections)
    for stem in stems:
        gradient = stem.conv.weight.grad
        assert gradient is not None and gradient.abs().sum() > 0


def test_yolo11_segmentation_template_builds_and_backpropagates():
    """The YOLO11 segmentation template must keep its native (predictions, protos) output on top of fusion."""
    cfg = yaml_model_load("ultralytics/cfg/models/multimodal/yolo11-mm3-bf-seg.yaml")
    cfg["scale"] = "n"
    model = MultiModalSegmentationModel(cfg, nc=2, verbose=False)
    model.args = DEFAULT_CFG

    model.eval()
    with torch.inference_mode():
        (prediction, protos), _ = model(torch.zeros(1, 5, 64, 64))
    assert prediction.shape == (1, 4 + 2 + 32, 84)  # box + class + 32 mask coefficients
    assert protos.shape == (1, 32, 16, 16)

    model.train()
    batch = {
        "img": torch.rand(2, 5, 64, 64),
        "cls": torch.zeros(2, 1, 1),
        "bboxes": torch.tensor([[32.0, 32.0, 24.0, 24.0], [40.0, 20.0, 16.0, 12.0]]),
        "batch_idx": torch.tensor([0, 1]),
    }
    loss, _ = model(batch)
    loss.sum().backward()
    stems = [layer for layer in model.model if isinstance(layer, Conv) and layer.conv.in_channels in (3, 1)]
    assert len(stems) == 3
    for stem in stems:
        gradient = stem.conv.weight.grad
        assert gradient is not None and gradient.abs().sum() > 0


@pytest.mark.parametrize("arch", ("yolov8s", "yolov9s", "yolov10s", "yolo11s", "yolo12s"))
@pytest.mark.parametrize("mode", ("if", "ef", "nif", "bf", "nf", "hf"))
def test_other_yolo_templates_build_and_backpropagate(arch, mode):
    """Every YOLOv8-v12 fusion template must keep its architecture's native head output and gradient path."""
    cfg = yaml_model_load(f"ultralytics/cfg/mmodels/{arch}-mm3-{mode}.yaml")
    config = load_multimodal_config(cfg)
    assert config["channels"] == 5
    assert config["multimodal"]["fusion"] == mode.upper()
    assert list(config["multimodal"]["input_sections"]) == [3, 1, 1]

    model = MultiModalDetectionModel(cfg, nc=2, verbose=False)
    model.args = DEFAULT_CFG
    fusions = [layer for layer in model.model if type(layer).__name__ == "MultiModalFusion"]
    assert len(fusions) == (0 if mode == "if" else 3)
    assert all(len(layer.channels) == 3 for layer in fusions)
    detect = next(layer for layer in model.model if type(layer).__name__ in ("Detect", "v10Detect"))
    assert len(detect.f) == 3

    model.eval()
    with torch.inference_mode():
        prediction, _ = model(torch.zeros(1, 5, 64, 64))
    expected = (1, 84, 6) if getattr(detect, "end2end", False) else (1, 6, 84)
    assert prediction.shape == expected

    model.train()
    batch = {
        "img": torch.rand(2, 5, 64, 64),
        "cls": torch.zeros(2, 1, 1),
        "bboxes": torch.tensor([[32.0, 32.0, 24.0, 24.0], [40.0, 20.0, 16.0, 12.0]]),
        "batch_idx": torch.tensor([0, 1]),
    }
    loss, _ = model(batch)
    loss.sum().backward()
    if mode == "if":
        stems = [model.model[0]]
    else:
        stems = [layer for layer in model.model if isinstance(layer, Conv) and layer.conv.in_channels in (3, 1)]
        assert len(stems) == 3
    for stem in stems:
        gradient = stem.conv.weight.grad
        assert gradient is not None and gradient.abs().sum() > 0


SCALE_SETS = {
    "yolov8": "nsmlx",
    "yolov9": "tsmce",
    "yolov10": "nsmblx",
    "yolo11": "nsmlx",
    "yolo12": "nsmlx",
}


@pytest.mark.parametrize("arch", tuple(SCALE_SETS))
@pytest.mark.parametrize("mode", ("if", "bf"))
def test_every_template_scale_builds_and_grows_with_the_scale(arch, mode):
    """Each architecture must ship every upstream scale, and a bigger scale must mean more parameters."""
    counts = {}
    for scale in SCALE_SETS[arch]:
        cfg = yaml_model_load(f"ultralytics/cfg/mmodels/{arch}{scale}-mm3-{mode}.yaml")
        config = load_multimodal_config(cfg)
        assert config["channels"] == 5
        assert config["multimodal"]["fusion"] == mode.upper()
        assert list(config["multimodal"]["input_sections"]) == [3, 1, 1]
        if cfg.get("scales"):
            expected_scale = scale if scale in "nsmlx" else next(iter(cfg["scales"]))
            assert expected_scale in cfg["scales"]
        model = MultiModalDetectionModel(cfg, nc=2, verbose=False)
        model.args = DEFAULT_CFG
        detect = next(layer for layer in model.model if type(layer).__name__ in ("Detect", "v10Detect"))
        assert len(detect.f) == 3
        model.eval()
        with torch.inference_mode():
            prediction, _ = model(torch.zeros(1, 5, 64, 64))
        expected = (1, 84, 6) if getattr(detect, "end2end", False) else (1, 6, 84)
        assert prediction.shape == expected
        counts[scale] = sum(parameter.numel() for parameter in model.parameters())

        model.train()
        batch = {
            "img": torch.rand(2, 5, 64, 64),
            "cls": torch.zeros(2, 1, 1),
            "bboxes": torch.tensor([[32.0, 32.0, 24.0, 24.0], [40.0, 20.0, 16.0, 12.0]]),
            "batch_idx": torch.tensor([0, 1]),
        }
        loss, _ = model(batch)
        loss.sum().backward()
        if mode == "if":
            stems = [layer for layer in model.model if isinstance(layer, Conv) and layer.conv.in_channels == 5]
            assert stems
        else:
            stems = [layer for layer in model.model if isinstance(layer, Conv) and layer.conv.in_channels in (3, 1)]
            assert stems and len(stems) % 3 == 0
        for stem in stems:
            gradient = stem.conv.weight.grad
            assert gradient is not None and gradient.abs().sum() > 0
    ordered = [counts[scale] for scale in SCALE_SETS[arch]]
    assert ordered == sorted(ordered), counts
    assert len(set(ordered)) == len(ordered), counts


def test_modalities_pair_resolved_split_paths_with_a_differently_spelled_root(tmp_path):
    """A split path resolved to another spelling of the same directory must still find its companions."""
    real = tmp_path / "real"
    (real / "primary" / "train").mkdir(parents=True)
    (real / "auxiliary" / "train").mkdir(parents=True)
    assert cv2.imwrite(str(real / "primary" / "train" / "s.png"), np.full((12, 20, 3), 10, dtype=np.uint8))
    assert cv2.imwrite(str(real / "auxiliary" / "train" / "s.png"), np.full((12, 20, 3), 20, dtype=np.uint8))
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Directory symbolic links are unavailable in this test environment: {error}")

    modalities = Modalities(
        {
            "path": alias,
            "channels": 6,
            "modalities": [
                {"name": "primary", "path": "primary", "channels": 3, "suffix": ".png"},
                {"name": "auxiliary", "path": "auxiliary", "channels": 3, "suffix": ".png"},
            ],
        }
    )

    stacked = modalities.load(real / "primary" / "train" / "s.png")
    assert stacked.shape == (12, 20, 6)
    assert stacked[0, 0, 0] == 10
    assert stacked[0, 0, 3] == 20


def test_multimodal_predictor_keeps_primary_paths_for_a_list_source(tmp_path):
    """A list source must keep real image paths so the predictor can find each sample's companion modalities."""
    _write_dataset(tmp_path)
    primary = tmp_path / "images" / "rgb" / "train" / "sample_0.jpg"
    predictor = MultiModalPredictorMixin()
    predictor.args = SimpleNamespace(batch=1, vid_stride=1)
    predictor.model = SimpleNamespace(channels=5)

    loader = predictor._as_file_loader([str(primary)])
    assert isinstance(loader, LoadImagesAndVideos)  # not LoadPilAndNumpy, which renames samples to image<i>.jpg
    paths, images, _ = next(iter(loader))
    assert paths == [str(primary.absolute())]

    predictor.batch = (paths, images, None)
    predictor.modalities = Modalities(_data(tmp_path))
    assert [str(path) for path in predictor._primary_paths()] == paths
    assert predictor.modalities.load(predictor._primary_paths()[0]).shape == (12, 20, 5)

    predictor.batch = (["image0.jpg"], images, None)
    with pytest.raises(ValueError, match="no file paths"):
        predictor._primary_paths()
