"""Validate a true multi-label YOLO detector.

Example:
    conda run -n mdet python scripts/val_multilabel.py \
        --model runs/multilabel/weights/best.pt --data path/to/data_multilabel.yaml
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from multilabel_yolo.validator import MultiLabelDetectionValidator


def main():
    """Run validation with explicit multi-label NMS and GT expansion at metric time."""
    parser = argparse.ArgumentParser(description="Validate a true multi-label YOLO detector")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    kwargs = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "conf": args.conf,
        "iou": args.iou,
        "plots": args.plots,
        "task": "detect",
    }
    if args.device is not None:
        kwargs["device"] = args.device
    YOLO(args.model).val(validator=MultiLabelDetectionValidator, **kwargs)


if __name__ == "__main__":
    main()
