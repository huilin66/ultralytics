"""Train the second stage of the detector + multi-label classifier baseline.

The detector checkpoint is intentionally supplied as an input rather than
silently retrained here.  Use ``train_mdet_experiments.py`` for the detector
stage, then use this script on crops produced from that detector.  The crop
dataset is an image-level multi-label dataset: each crop has one sidecar text
file containing comma- or whitespace-separated class IDs.

Example:

    python scripts/train_two_stage.py `
        --detector-checkpoint runs/experiments/E3_yolov10x_stage2/weights/best.pt `
        --model ultralytics/cfg/models/11/yolo11n-cls.yaml `
        --data path/to/detection_crops_multilabel.yaml `
        --project runs/experiments --name E6_two_stage_yolov10x
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the image-level multi-label classifier for the two-stage baseline"
    )
    parser.add_argument(
        "--detector-checkpoint",
        required=True,
        help="checkpoint produced by the first-stage detector experiment; recorded for provenance",
    )
    parser.add_argument("--model", required=True, help="YOLO classification checkpoint or model YAML")
    parser.add_argument("--data", required=True, help="image-level multi-label crop dataset YAML")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/experiments")
    parser.add_argument("--name", default="E6_two_stage_classifier")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true", help="print the training configuration only")
    return parser


def _record_manifest(args: argparse.Namespace) -> None:
    """Write the detector/classifier provenance next to the classifier run."""
    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)
    record = {
        "experiment": "E6_two_stage_detector_plus_multilabel_classifier",
        "detector_checkpoint": str(Path(args.detector_checkpoint).expanduser()),
        "classifier_model": args.model,
        "classifier_data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "seed": args.seed,
        "pretrained": args.pretrained,
        "amp": args.amp,
        "project": args.project,
        "name": args.name,
        "protocol": "classifier is trained on detector-aligned crops; evaluate with predicted crops for the final comparison",
    }
    with (project / "two_stage_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
    print(json.dumps(record, ensure_ascii=False, indent=2))


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the multi-label classifier training stage."""
    args = _build_parser().parse_args(argv)
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if not Path(args.detector_checkpoint).expanduser().exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {args.detector_checkpoint}")

    _record_manifest(args)
    if args.dry_run:
        return

    from ultralytics import YOLO
    from multilabel_yolo.classification_trainer import MultiLabelClassificationTrainer

    kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "seed": args.seed,
        "pretrained": args.pretrained,
        "amp": args.amp,
        "exist_ok": args.exist_ok,
        "task": "classify",
    }
    YOLO(args.model, task="classify").train(trainer=MultiLabelClassificationTrainer, **kwargs)


if __name__ == "__main__":
    main()
