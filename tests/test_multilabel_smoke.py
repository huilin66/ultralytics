"""One-epoch end-to-end smoke test for the public multi-label entry points."""

from pathlib import Path
import sys
import types

import yaml
from PIL import Image

from ultralytics import YOLO

from multilabel_yolo.predictor import MultiLabelDetectionPredictor
from multilabel_yolo.trainer import MultiLabelDetectionTrainer
from multilabel_yolo.validator import MultiLabelDetectionValidator


def _make_split(root: Path, split: str, rows):
    """Create a tiny image split with one physical row per object."""
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for index, label in enumerate(rows):
        name = f"{index}.jpg"
        Image.new("RGB", (64, 64), (90, 110, 130)).save(image_dir / name)
        if label is not None:
            (label_dir / name.replace(".jpg", ".txt")).write_text(label, encoding="utf-8")


def test_one_epoch_train_reload_val_and_predict(tmp_path, monkeypatch):
    """The custom trainer, checkpoint loader, validator, and predictor all run."""
    # The repository's exporter imports this optional local extension even for
    # PyTorch validation.  Keep the smoke test independent of that extra op.
    exporter_stub = types.ModuleType("deform_conv2d_onnx_exporter")
    exporter_stub.register_deform_conv2d_onnx_op = lambda: None
    monkeypatch.setitem(sys.modules, "deform_conv2d_onnx_exporter", exporter_stub)
    _make_split(
        tmp_path,
        "train",
        [
            "1,3 0.5 0.5 0.3 0.3\n",
            "2 0.4 0.4 0.2 0.2\n",
            "0,2,4 0.6 0.6 0.2 0.2\n",
            None,
        ],
    )
    _make_split(tmp_path, "val", ["1,3 0.5 0.5 0.3 0.3\n", None])
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
    model_yaml = Path(__file__).parents[1] / "ultralytics/cfg/models/11/yolo11.yaml"
    project = tmp_path / "runs"

    model = YOLO(str(model_yaml))
    model.train(
        trainer=MultiLabelDetectionTrainer,
        data=str(data_file),
        epochs=1,
        imgsz=64,
        batch=2,
        workers=0,
        device="cpu",
        amp=False,
        pretrained=False,
        plots=False,
        verbose=False,
        patience=1,
        project=str(project),
        name="train",
        exist_ok=True,
    )
    weights = project / "train" / "weights" / "last.pt"
    assert weights.exists()

    reloaded = YOLO(str(weights))
    metrics = reloaded.val(
        validator=MultiLabelDetectionValidator,
        data=str(data_file),
        imgsz=64,
        batch=2,
        workers=0,
        device="cpu",
        plots=False,
        project=str(project),
        name="val",
        exist_ok=True,
    )
    assert metrics is not None

    predictions = reloaded.predict(
        predictor=MultiLabelDetectionPredictor,
        source=str(tmp_path / "images/val/0.jpg"),
        imgsz=64,
        conf=0.01,
        device="cpu",
        verbose=False,
    )
    assert predictions and predictions[0].boxes.data.shape[1] == 6
