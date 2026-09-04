"""Tests for image-level multi-label classification."""

from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from ultralytics.engine.results import Results

from multilabel_yolo.classification_dataset import (
    MultiLabelClassificationDataset,
    MultiLabelClassificationLabelError,
    load_multilabel_classification_data,
    parse_image_multilabel_label_file,
)
from multilabel_yolo.classification_loss import MultiLabelClassificationLoss
from multilabel_yolo.classification_metrics import MultiLabelClassificationMetrics
from multilabel_yolo.classification_model import MultiLabelClassificationModel
from multilabel_yolo.classification_pipeline import classify_detection_crops


def _args(imgsz=32):
    """Return the classification augmentation options needed by the dataset."""
    return SimpleNamespace(
        imgsz=imgsz,
        crop_fraction=1.0,
        scale=0.5,
        fliplr=0.0,
        flipud=0.0,
        auto_augment=None,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        erasing=0.0,
    )


def _write_image_split(root: Path, split: str):
    """Create a small flat image split and matching sidecar labels."""
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for index, labels in enumerate(("0,2", "1")):
        Image.new("RGB", (48, 40), (80 + index * 20, 100, 120)).save(image_dir / f"{index}.jpg")
        (label_dir / f"{index}.txt").write_text(labels, encoding="utf-8")


def test_image_multilabel_dataset_returns_one_nhot_per_image(tmp_path):
    """Image labels are not expanded and do not depend on directory names."""
    _write_image_split(tmp_path, "train")
    dataset = MultiLabelClassificationDataset(
        tmp_path / "images" / "train",
        _args(),
        nc=3,
        label_root=tmp_path / "labels" / "train",
        augment=False,
    )
    sample = dataset[0]
    assert sample["img"].shape == (3, 32, 32)
    assert torch.equal(sample["cls_nhot"], torch.tensor([1.0, 0.0, 1.0]))
    assert torch.equal(sample["cls"], sample["cls_nhot"])
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    assert batch["cls_nhot"].shape == (2, 3)
    assert torch.equal(batch["cls_nhot"].sum(1), torch.tensor([2.0, 1.0]))


def test_image_multilabel_parser_rejects_missing_and_invalid_labels(tmp_path):
    """Annotation errors fail fast rather than silently becoming negatives."""
    missing = tmp_path / "missing.txt"
    with pytest.raises(MultiLabelClassificationLabelError):
        parse_image_multilabel_label_file(missing, 3)
    invalid = tmp_path / "invalid.txt"
    invalid.write_text("0 3", encoding="utf-8")
    with pytest.raises(MultiLabelClassificationLabelError):
        parse_image_multilabel_label_file(invalid, 3)


def test_image_multilabel_loss_and_metrics():
    """BCEWithLogits and thresholded metrics handle independent labels."""
    logits = torch.tensor([[4.0, -4.0, 3.0], [-3.0, 3.0, 2.0]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    loss, items = MultiLabelClassificationLoss(nc=3)(logits, {"cls_nhot": target})
    assert torch.isfinite(loss)
    assert items.shape == (1,)
    loss.backward()

    metrics = MultiLabelClassificationMetrics({0: "a", 1: "b", 2: "c"}, threshold=0.5)
    result = metrics.process([target], [logits.detach().sigmoid()])
    assert result["metrics/precision"] == pytest.approx(1.0)
    assert result["metrics/recall"] == pytest.approx(1.0)
    assert result["metrics/f1_micro"] == pytest.approx(1.0)
    assert result["metrics/mAP"] == pytest.approx(1.0)
    assert metrics.per_class[2]["support"] == 2


def test_image_multilabel_yaml_resolution(tmp_path):
    """The YAML resolves split image and label roots deterministically."""
    _write_image_split(tmp_path, "train")
    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images/train",
                "names": ["a", "b", "c"],
                "labels": "labels",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    data = load_multilabel_classification_data(yaml_file)
    assert data["nc"] == 3
    assert data["val"] == data["train"]
    assert data["labels"]["train"] == (tmp_path / "labels" / "train").resolve()


def test_yolo_classification_model_uses_raw_logits():
    """The native YOLO classification head is reused with the new criterion."""
    model = MultiLabelClassificationModel(
        str(Path(__file__).parents[1] / "ultralytics/cfg/models/11/yolo11n-cls.yaml"),
        nc=3,
        verbose=False,
    )
    model.train()
    logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 3)
    loss, _ = model.loss(
        {"img": torch.randn(2, 3, 32, 32), "cls_nhot": torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)},
        logits,
    )
    assert torch.isfinite(loss)
    loss.backward()
    model.eval()
    with torch.no_grad():
        probabilities, raw_logits = model(torch.randn(2, 3, 32, 32))
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    assert torch.allclose(probabilities, raw_logits.sigmoid())


def test_detection_crops_keep_box_order_for_multilabel_classifier():
    """The two-stage helper maps each crop prediction back to its box."""
    detection = Results(
        np.zeros((20, 30, 3), dtype=np.uint8),
        path="image.jpg",
        names={0: "object"},
        boxes=torch.tensor([[1, 2, 10, 12, 0.9, 0], [12, 4, 28, 18, 0.8, 0]]),
    )

    class FakeClassifier:
        def predict(self, source, **kwargs):
            self.source = source
            self.kwargs = kwargs
            return [
                SimpleNamespace(
                    multilabel_scores=torch.tensor([0.8, 0.2]),
                    multilabel=torch.tensor([True, False]),
                    multilabel_indices=torch.tensor([0]),
                    multilabel_names=["clean"],
                )
                for _ in source
            ]

    classifier = FakeClassifier()
    output = classify_detection_crops(detection, classifier, threshold=0.5)
    assert output is detection
    assert len(output.detection_multilabel_names) == 2
    assert output.detection_multilabel_names == [["clean"], ["clean"]]
    assert [crop.shape[:2] for crop in classifier.source] == [(10, 9), (14, 16)]
    assert classifier.kwargs["conf"] == 0.5


def test_yolo_public_multilabel_classification_one_epoch(tmp_path, monkeypatch):
    """The custom trainer/validator/predictor work through the public API."""
    # Loading a checkpoint for final evaluation imports this repository's
    # optional deployment extension. It is unrelated to image classification.
    exporter_stub = types.ModuleType("deform_conv2d_onnx_exporter")
    exporter_stub.register_deform_conv2d_onnx_op = lambda: None
    monkeypatch.setitem(sys.modules, "deform_conv2d_onnx_exporter", exporter_stub)

    from ultralytics import YOLO

    _write_image_split(tmp_path, "train")
    _write_image_split(tmp_path, "val")
    data_file = tmp_path / "multilabel_cls.yaml"
    data_file.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images/train",
                "val": "images/val",
                "labels": "labels",
                "names": ["a", "b", "c"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    from multilabel_yolo.classification_predictor import MultiLabelClassificationPredictor
    from multilabel_yolo.classification_trainer import MultiLabelClassificationTrainer
    from multilabel_yolo.classification_validator import MultiLabelClassificationValidator

    model = YOLO(
        str(Path(__file__).parents[1] / "ultralytics/cfg/models/11/yolo11n-cls.yaml"), task="classify"
    )
    model.train(
        trainer=MultiLabelClassificationTrainer,
        data=str(data_file),
        epochs=1,
        imgsz=32,
        batch=2,
        workers=0,
        device="cpu",
        amp=False,
        pretrained=False,
        plots=False,
        verbose=False,
        patience=1,
        project=str(tmp_path / "runs"),
        name="classifier",
        exist_ok=True,
    )
    weights = tmp_path / "runs" / "classifier" / "weights" / "last.pt"
    assert weights.exists()

    reloaded = YOLO(str(weights), task="classify")
    metrics = reloaded.val(
        validator=MultiLabelClassificationValidator,
        data=str(data_file),
        imgsz=32,
        batch=2,
        workers=0,
        device="cpu",
        plots=False,
    )
    assert "metrics/mAP" in metrics.results_dict
    predictions = reloaded.predict(
        predictor=MultiLabelClassificationPredictor,
        source=str(tmp_path / "images" / "val" / "0.jpg"),
        imgsz=32,
        conf=0.5,
        device="cpu",
        verbose=False,
    )
    assert predictions and predictions[0].probs.data.shape == (3,)
    assert predictions[0].multilabel_indices.ndim == 1

    detections = Results(
        np.zeros((40, 40, 3), dtype=np.uint8),
        path="image.jpg",
        names={0: "object"},
        boxes=torch.tensor([[4, 5, 30, 35, 0.9, 0]]),
    )
    enriched = classify_detection_crops(detections, reloaded, threshold=0.5, imgsz=32, device="cpu")
    assert len(enriched.detection_multilabel_names) == 1
