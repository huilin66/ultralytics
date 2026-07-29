"""Reusable train, validation, prediction, tracking and export helpers for MultiModalYOLO."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from ultralytics.models.multimodal import MultiModalYOLO

DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 640
DEFAULT_TASK = "detect"
VAL_CONFIDENCE = 0.001
PREDICT_CONFIDENCE = 0.25


class Tee:
    """Write console output to each supplied stream."""

    def __init__(self, *streams):
        """Initialize a console tee for the supplied writable streams."""
        self.streams = streams

    def write(self, data: str) -> int:
        """Write data to every healthy stream."""
        for stream in self.streams:
            try:
                stream.write(data)
            except OSError:
                pass
        return len(data)

    def flush(self) -> None:
        """Flush every healthy stream."""
        for stream in self.streams:
            try:
                stream.flush()
            except OSError:
                pass


def tee_log_to_run_dir(trainer) -> None:
    """Mirror trainer console output to ``console.log`` in the run directory."""
    log_path = Path(trainer.save_dir) / "console.log"
    log_file = log_path.open("w", buffering=1, encoding="utf-8")
    trainer._multimodal_console_streams = (sys.stdout, sys.stderr)
    trainer._multimodal_console_log = log_file
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)


def close_tee_log(trainer) -> None:
    """Restore console streams and close the trainer console log when training ends."""
    streams = getattr(trainer, "_multimodal_console_streams", None)
    if streams:
        sys.stdout, sys.stderr = streams
        del trainer._multimodal_console_streams
    log_file = getattr(trainer, "_multimodal_console_log", None)
    if log_file:
        log_file.close()
        del trainer._multimodal_console_log


def resolve_weights(weight_path: str | Path, task: str, weight_name: bool) -> Path:
    """Resolve a run name to its best checkpoint, or return an explicit checkpoint path."""
    path = Path(weight_path)
    return Path("runs") / task / path / "weights" / "best.pt" if weight_name else path


def build_model(model_path: str | Path, task: str = DEFAULT_TASK) -> MultiModalYOLO:
    """Create the multimodal facade used by every lifecycle helper."""
    return MultiModalYOLO(model_path, task=task)


def model_train(
    cfg_path: str | Path,
    data: str | Path,
    pretrain_path: str | Path | None = None,
    *,
    task: str = DEFAULT_TASK,
    batch: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = DEFAULT_IMGSZ,
    device: str | int | torch.device | None = None,
    auto_optim: bool = True,
    console_log: bool = True,
    **kwargs: Any,
):
    """Train a multimodal model using a data YAML that declares every aligned modality."""
    model = build_model(cfg_path, task)
    if pretrain_path:
        model.load(pretrain_path)
    if console_log:
        model.add_callback("on_train_start", tee_log_to_run_dir)
        model.add_callback("on_train_end", close_tee_log)

    train_params = {
        "data": str(data),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "val": True,
        "patience": epochs,
    }
    if not auto_optim:
        train_params.update({"optimizer": "AdamW", "lr0": 1e-4})
    train_params.update(kwargs)
    train_params.setdefault("name", f"{Path(data).stem}-[{Path(cfg_path).stem}]")
    try:
        return model.train(**train_params)
    finally:
        if console_log:
            close_tee_log(getattr(model, "trainer", None)) if getattr(model, "trainer", None) else None


def model_val(
    weight_path: str | Path,
    data: str | Path,
    *,
    task: str = DEFAULT_TASK,
    weight_name: bool = True,
    batch: int = DEFAULT_BATCH_SIZE,
    device: str | int | torch.device | None = None,
    save_txt: bool = False,
    **kwargs: Any,
):
    """Validate a checkpoint with the data YAML needed to recreate modality pairing."""
    weights = resolve_weights(weight_path, task, weight_name)
    model = build_model(weights, task)
    return model.val(
        data=str(data),
        batch=batch,
        conf=VAL_CONFIDENCE,
        device=device,
        save_txt=save_txt,
        **kwargs,
    )


def model_predict(
    weight_path: str | Path,
    source: str | Path,
    data: str | Path,
    *,
    task: str = DEFAULT_TASK,
    weight_name: bool = True,
    batch: int = DEFAULT_BATCH_SIZE,
    device: str | int | torch.device | None = None,
    save: bool = True,
    save_txt: bool = True,
    stream: bool = True,
    **kwargs: Any,
) -> None:
    """Predict from primary-modality files, using the data YAML to map companion modalities."""
    weights = resolve_weights(weight_path, task, weight_name)
    model = build_model(weights, task)
    results = model.predict(
        source=str(source),
        data=str(data),
        batch=batch,
        conf=PREDICT_CONFIDENCE,
        device=device,
        save=save,
        save_txt=save_txt,
        stream=stream,
        **kwargs,
    )
    for _ in results:
        pass


def model_track(
    weight_path: str | Path,
    source: str | Path,
    data: str | Path,
    *,
    task: str = DEFAULT_TASK,
    weight_name: bool = True,
    batch: int = DEFAULT_BATCH_SIZE,
    device: str | int | torch.device | None = None,
    **kwargs: Any,
) -> None:
    """Track primary-modality files while using the data YAML for aligned companion images."""
    weights = resolve_weights(weight_path, task, weight_name)
    model = build_model(weights, task)
    results = model.track(
        source=str(source),
        data=str(data),
        batch=batch,
        conf=PREDICT_CONFIDENCE,
        device=device,
        tracker="botsort.yaml",
        persist=True,
        stream=True,
        **kwargs,
    )
    for _ in results:
        pass


def model_export(
    weight_path: str | Path,
    *,
    task: str = DEFAULT_TASK,
    weight_name: bool = True,
    format: str = "onnx",
    device: str | int | torch.device | None = None,
    **kwargs: Any,
):
    """Export a trained multimodal checkpoint through Ultralytics' standard exporter."""
    weights = resolve_weights(weight_path, task, weight_name)
    return build_model(weights, task).export(format=format, device=device, **kwargs)


def yolov8(cfg_path: str | Path, data: str | Path, pretrain_path: str | Path | None = None, **kwargs: Any):
    """Train a YOLOv8 multimodal YAML and reject an accidental non-YOLOv8 configuration."""
    if "yolov8" not in Path(cfg_path).stem:
        raise ValueError(f"Expected a YOLOv8 configuration, got {cfg_path!s}.")
    return model_train(cfg_path, data, pretrain_path, **kwargs)
