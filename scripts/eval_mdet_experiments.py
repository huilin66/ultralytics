"""Evaluation helpers for mdet experiments that do not require retraining.

Currently this script provides the HO comparison.  ``native`` keeps the
checkpoint's normal head selection; ``one2many`` explicitly switches the
mdet head to the one-to-many branch.  ``both`` reloads the checkpoint for each
mode so that the two measurements are independent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_compat import add_bool_argument


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate non-training mdet experiment comparisons")
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    ho = subparsers.add_parser("ho", help="E2.4: compare native HO inference with one-to-many inference")
    ho.add_argument("--weights", required=True, help="trained mdet/HO checkpoint")
    ho.add_argument("--data", required=True, help="mdet dataset YAML")
    ho.add_argument("--mode", choices=("native", "one2many", "both"), default="both")
    ho.add_argument("--device", default="0")
    ho.add_argument("--imgsz", type=int, default=640)
    ho.add_argument("--batch", type=int, default=32)
    ho.add_argument("--workers", type=int, default=8)
    ho.add_argument("--conf", type=float, default=None, help="optional validation confidence threshold")
    ho.add_argument("--project", default="runs/experiments")
    ho.add_argument("--name", default="E2_4_HO")
    add_bool_argument(ho, "--plots", default=True)
    return parser


def _load_model(weights: str):
    """Load an mdet checkpoint through the task-specific YOLO wrapper."""
    from ultralytics import YOLO

    return YOLO(weights, task="mdetect")


def _set_one2many(model) -> None:
    """Switch the final mdet head to its one-to-many inference branch."""
    detector = getattr(model, "model", None)
    layers = getattr(detector, "model", None)
    if layers is None or not layers:
        raise RuntimeError("Could not locate the mdet model head")
    head = layers[-1]
    switch = getattr(head, "use_one2many_head", None)
    if switch is None:
        raise RuntimeError("The loaded checkpoint does not expose use_one2many_head(); it is not an HO mdet model")
    switch()


def _evaluate_one(args: argparse.Namespace, mode: str) -> None:
    """Evaluate a checkpoint in one selected head mode."""
    model = _load_model(args.weights)
    if mode == "one2many":
        _set_one2many(model)
    kwargs = {
        "data": args.data,
        "device": args.device,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": f"{args.name}_{mode}",
        "plots": args.plots,
    }
    if args.conf is not None:
        kwargs["conf"] = args.conf
    print(f"[eval] mode={mode}, weights={args.weights}")
    model.val(**kwargs)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the selected evaluation."""
    args = _build_parser().parse_args(argv)
    if args.experiment != "ho":
        raise ValueError(f"Unsupported evaluation: {args.experiment}")
    modes = ("native", "one2many") if args.mode == "both" else (args.mode,)
    for mode in modes:
        _evaluate_one(args, mode)


if __name__ == "__main__":
    main()
