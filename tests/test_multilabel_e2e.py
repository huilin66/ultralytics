"""YOLOv10 end-to-end integration checks for the true multi-label adapter."""

from pathlib import Path
import sys
from types import SimpleNamespace
import types

import torch
import yaml
from PIL import Image
from ultralytics import YOLO

from multilabel_yolo.inference import prepare_prediction_for_multilabel_nms
from multilabel_yolo.model import MultiLabelDetectionModel
from multilabel_yolo.predictor import MultiLabelDetectionPredictor
from multilabel_yolo.trainer import MultiLabelDetectionTrainer


def _write_toy_split(root: Path, split: str, label: str):
    """Write one tiny image and one physical multi-label object."""
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), (100, 120, 140)).save(image_dir / "0.jpg")
    (label_dir / "0.txt").write_text(label, encoding="utf-8")


def test_yolov10_end_to_end_multilabel_forward_and_decode():
    """The YOLOv10 branches train with n-hot loss and decode without dropping labels."""
    model = MultiLabelDetectionModel(
        str(Path(__file__).parents[1] / "ultralytics/cfg/models/v10/yolov10n.yaml"),
        nc=5,
        verbose=False,
    )
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    assert model.end2end
    assert model.model[-1].end2end

    image = torch.randn(1, 3, 64, 64)
    model.train()
    train_preds = model(image)
    assert set(train_preds) == {"one2many", "one2one"}
    assert len(train_preds["one2many"]) == len(train_preds["one2one"]) == 3

    batch = {
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[99.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
        "cls_nhot": torch.tensor([[1, 0, 1, 0, 1]], dtype=torch.float32),
    }
    total, items = model.loss(batch, train_preds)
    assert torch.isfinite(total)
    assert torch.isfinite(items).all()
    total.backward()

    model.eval()
    with torch.no_grad():
        eval_preds = model(image)
    assert eval_preds[0].shape[-1] == 6  # native YOLOv10 output is single-label postprocessed
    prepared = prepare_prediction_for_multilabel_nms(eval_preds, model)
    assert prepared.shape[:2] == (1, 9)  # xywh + five independent class scores
    assert prepared.shape[-1] == 84  # 8x8 + 4x4 + 2x2 anchors at 64px


def test_yolov10_public_trainer_one_epoch(tmp_path, monkeypatch):
    """The public trainer can execute a YOLOv10 E2E multi-label epoch."""
    # The repository imports this optional deployment extension while loading
    # the final checkpoint for validation. It is unrelated to the E2E loss.
    exporter_stub = types.ModuleType("deform_conv2d_onnx_exporter")
    exporter_stub.register_deform_conv2d_onnx_op = lambda: None
    monkeypatch.setitem(sys.modules, "deform_conv2d_onnx_exporter", exporter_stub)
    _write_toy_split(tmp_path, "train", "1,3 0.5 0.5 0.3 0.3\n")
    _write_toy_split(tmp_path, "val", "1,3 0.5 0.5 0.3 0.3\n")
    data_file = tmp_path / "toy.yaml"
    data_file.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "c0", 1: "c1", 2: "c2", 3: "c3", 4: "c4"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    model = YOLO(str(Path(__file__).parents[1] / "ultralytics/cfg/models/v10/yolov10n.yaml"))
    model.train(
        trainer=MultiLabelDetectionTrainer,
        data=str(data_file),
        epochs=1,
        imgsz=64,
        batch=1,
        workers=0,
        device="cpu",
        amp=False,
        pretrained=False,
        plots=False,
        verbose=False,
        patience=1,
        project=str(tmp_path / "runs"),
        name="yolov10",
        exist_ok=True,
    )
    weights = tmp_path / "runs" / "yolov10" / "weights" / "last.pt"
    assert weights.exists()
    reloaded = YOLO(str(weights))
    predictions = reloaded.predict(
        predictor=MultiLabelDetectionPredictor,
        source=str(tmp_path / "images" / "val" / "0.jpg"),
        imgsz=64,
        conf=0.01,
        device="cpu",
        verbose=False,
    )
    assert predictions and predictions[0].boxes.data.shape[1] == 6
