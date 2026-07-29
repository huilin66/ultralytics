"""Command-line detection demo for the YOLOv8 multi-modal configuration templates."""

from __future__ import annotations

import argparse
from pathlib import Path

import demo_multimodal_base as base

DEFAULT_MODEL = Path("ultralytics/cfg/mmodels/yolov8x-mm3-bf.yaml")


def parse_args() -> argparse.Namespace:
    """Parse a multimodal detection command without hard-coding dataset or checkpoint paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "val", "predict", "track", "export"), default="train")
    parser.add_argument("--data", required=True, help="Multi-modal data YAML with modality paths and channels.")
    parser.add_argument("--model", default=DEFAULT_MODEL, type=Path, help="Model YAML used for training.")
    parser.add_argument("--weights", type=Path, help="Explicit trained checkpoint for val, predict, track or export.")
    parser.add_argument("--pretrained", type=Path, help="Optional single-modal checkpoint used to initialize training.")
    parser.add_argument("--source", type=Path, help="Primary-modality image, directory or video for predict or track.")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", help="Optional Ultralytics run name.")
    parser.add_argument("--format", default="onnx", help="Export format when --mode export.")
    parser.add_argument(
        "--adamw", action="store_true", help="Use AdamW with lr0=1e-4 instead of auto optimizer selection."
    )
    return parser.parse_args()


def main() -> None:
    """Run one multimodal detect lifecycle operation."""
    args = parse_args()
    if args.mode == "train":
        kwargs = {"epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch, "device": args.device}
        if args.name:
            kwargs["name"] = args.name
        base.yolov8(args.model, args.data, args.pretrained, auto_optim=not args.adamw, **kwargs)
        return

    if not args.weights:
        raise SystemExit("--weights is required for val, predict, track and export.")
    if args.mode in {"predict", "track"} and not args.source:
        raise SystemExit("--source is required for predict and track.")

    common = {"data": args.data, "weight_name": False, "batch": args.batch, "device": args.device}
    if args.mode == "val":
        base.model_val(args.weights, **common)
    elif args.mode == "predict":
        base.model_predict(args.weights, args.source, **common)
    elif args.mode == "track":
        base.model_track(args.weights, args.source, **common)
    else:
        base.model_export(args.weights, weight_name=False, format=args.format, device=args.device)


if __name__ == "__main__":
    main()
