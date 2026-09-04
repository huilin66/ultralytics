"""Validate a YOLO image-level multi-label classifier."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from multilabel_yolo.classification_validator import MultiLabelClassificationValidator


def parse_args():
    """Parse common validation arguments."""
    parser = argparse.ArgumentParser(description="Validate YOLO image-level multi-label predictions")
    parser.add_argument("--model", required=True, help="Multi-label classifier checkpoint")
    parser.add_argument("--data", required=True, help="Image-level multi-label dataset YAML")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Override the YAML label threshold")
    return parser.parse_args()


def main():
    """Run validation through the public YOLO API."""
    args = parse_args()
    kwargs = {
        "data": args.data,
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "task": "classify",
    }
    if args.device is not None:
        kwargs["device"] = args.device
    # ``conf`` is used as the explicit threshold by the custom validator when
    # supplied. Native validation's default 0.001 is ignored.
    if args.threshold is not None:
        kwargs["conf"] = args.threshold
    YOLO(args.model, task="classify").val(validator=MultiLabelClassificationValidator, **kwargs)


if __name__ == "__main__":
    main()
