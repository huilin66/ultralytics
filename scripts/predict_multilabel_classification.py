"""Predict all labels for images or detection-result crops."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from multilabel_yolo.classification_predictor import MultiLabelClassificationPredictor


def parse_args():
    """Parse prediction arguments."""
    parser = argparse.ArgumentParser(description="Predict independent labels for each image or crop")
    parser.add_argument("--model", required=True, help="Multi-label classifier checkpoint")
    parser.add_argument("--source", required=True, help="Image, directory, or list-compatible source")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    """Run prediction and print all selected class names."""
    args = parse_args()
    kwargs = {
        "source": args.source,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.threshold,
        "verbose": False,
    }
    if args.device is not None:
        kwargs["device"] = args.device
    results = YOLO(args.model, task="classify").predict(
        predictor=MultiLabelClassificationPredictor, **kwargs
    )
    for result in results:
        print(f"{result.path}: {result.multilabel_names}")


if __name__ == "__main__":
    main()
