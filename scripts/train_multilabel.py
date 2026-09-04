"""Train a true multi-label YOLO detector.

Example:
    conda run -n mdet python scripts/train_multilabel.py \
        --model yolo11n.pt --data path/to/data_multilabel.yaml --epochs 100
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_compat import add_bool_argument
from ultralytics import YOLO

from multilabel_yolo.trainer import MultiLabelDetectionTrainer


def parse_args():
    """Parse the small set of commonly used training arguments."""
    parser = argparse.ArgumentParser(description="Train YOLO with one physical box and n-hot labels")
    parser.add_argument("--model", required=True, help="YOLO .pt checkpoint or model YAML")
    parser.add_argument("--data", required=True, help="Dataset YAML with comma-separated multi-label txt files")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    add_bool_argument(parser, "--amp", default=True)
    return parser.parse_args()


def main():
    """Run multi-label training through the public YOLO API."""
    args = parse_args()
    kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "seed": args.seed,
        "amp": args.amp,
        "task": "detect",
    }
    for key in ("device", "project", "name"):
        value = getattr(args, key)
        if value is not None:
            kwargs[key] = value
    YOLO(args.model).train(trainer=MultiLabelDetectionTrainer, **kwargs)


if __name__ == "__main__":
    main()
