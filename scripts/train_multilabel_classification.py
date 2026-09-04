"""Train a YOLO image-level multi-label classifier.

Example:
    conda run -n mdet python scripts/train_multilabel_classification.py \
        --model ultralytics/cfg/models/11/yolo11n-cls.yaml \
        --data path/to/multilabel_cls.yaml --epochs 100
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_compat import add_bool_argument
from ultralytics import YOLO

from multilabel_yolo.classification_trainer import MultiLabelClassificationTrainer


def parse_args():
    """Parse common training arguments."""
    parser = argparse.ArgumentParser(description="Train YOLO with one n-hot label vector per image")
    parser.add_argument("--model", required=True, help="YOLO classification checkpoint or model YAML")
    parser.add_argument("--data", required=True, help="Image-level multi-label dataset YAML")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    add_bool_argument(parser, "--pretrained", default=True)
    add_bool_argument(parser, "--amp", default=True)
    return parser.parse_args()


def main():
    """Run training through the public YOLO API."""
    args = parse_args()
    kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "seed": args.seed,
        "pretrained": args.pretrained,
        "amp": args.amp,
        "task": "classify",
    }
    for key in ("device", "project", "name"):
        value = getattr(args, key)
        if value is not None:
            kwargs[key] = value
    YOLO(args.model, task="classify").train(trainer=MultiLabelClassificationTrainer, **kwargs)


if __name__ == "__main__":
    main()
