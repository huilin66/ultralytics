"""Evaluate attribute ordinal error and probability calibration on the Test split.

The evaluator reuses the mdetect validator's matching protocol: an attribute
sample is included only when the predicted object has the correct object class
and its box is matched to a ground-truth box at IoU >= 0.5.  The validator
already retains the complete per-attribute softmax distributions for those
matches, so this script does not retrain a model or use hard-label text files.

Example::

    python scripts/eval_attribute_calibration.py \
        --data ultralytics/cfg/mayolo_r1/mayolo_v3.yaml \
        --model YOLOv10x=/path/to/yolov10x.pt \
        --model MAYOLOx=/path/to/mayolox.pt \
        --device 0 \
        --ordinal-attributes all \
        --project runs/experiments/E6_attribute_quality \
        --name mayolox_vs_yolov10x

The current DSD attributes are binary.  Consequently, when ``--ordinal-
attributes all`` is used, ordinal MAE is exactly the binary attribute error
rate and is reported with an explicit ``ordinal_binary_degenerate`` flag.
Calibration metrics remain meaningful because they use the full softmax
probability vectors.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


QUALITY_KEYS = (
    "ECE_macro_test",
    "Brier_macro_test",
    "NLL_macro_test",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate ordinal error and calibration on matched Test attribute predictions"
    )
    parser.add_argument("--data", required=True, help="mdet dataset YAML")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="LABEL=WEIGHTS[::native|one2many]; repeat for each model",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=15,
        help="Number of equal-width bins used by top-label ECE",
    )
    parser.add_argument(
        "--ordinal-attributes",
        default="all",
        help="Comma-separated attribute names/indices, 'all', or 'none'",
    )
    parser.add_argument("--project", default="runs/experiments/E3_attribute_calibration")
    parser.add_argument("--name", default="mayolox_vs_yolov10x")
    return parser


def _parse_model_spec(spec: str) -> tuple[str, str, str]:
    model_spec, separator, mode = spec.partition("::")
    label, equals, weights = model_spec.partition("=")
    if not equals or not label or not weights:
        raise ValueError(f"Invalid --model specification: {spec!r}; expected LABEL=WEIGHTS[::MODE]")
    mode = mode or "native"
    if mode not in {"native", "one2many"}:
        raise ValueError(f"Unsupported head mode {mode!r} in {spec!r}")
    return label, weights, mode


def _set_one2many(model: Any) -> None:
    """Switch an mdet model to the one-to-many inference branch."""
    detector = getattr(model, "model", None)
    layers = getattr(detector, "model", None)
    if layers is None or not layers:
        raise RuntimeError("Could not locate the mdet model head")
    switch = getattr(layers[-1], "use_one2many_head", None)
    if switch is None:
        raise RuntimeError("The loaded checkpoint does not expose use_one2many_head()")
    switch()


def _as_float(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _select_ordinal_attributes(selector: str, per_attribute: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the attributes for which ordinal MAE should be reported."""
    normalized = selector.strip().lower()
    if normalized in {"", "none", "off"}:
        return []
    if normalized in {"all", "*"}:
        return list(per_attribute)

    by_name = {str(row["attribute"]): row for row in per_attribute}
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for token in (item.strip() for item in selector.split(",")):
        if not token:
            continue
        if token.isdigit():
            index = int(token)
            matches = [row for row in per_attribute if int(row["attribute_index"]) == index]
            if matches:
                selected.extend(matches)
            else:
                missing.append(token)
        elif token in by_name:
            selected.append(by_name[token])
        else:
            missing.append(token)
    if missing:
        available = ", ".join(by_name)
        raise ValueError(f"Unknown ordinal attribute(s): {', '.join(missing)}. Available: {available}")

    # Preserve the model's attribute order and avoid duplicate selections.
    selected_indices = {int(row["attribute_index"]) for row in selected}
    return [row for row in per_attribute if int(row["attribute_index"]) in selected_indices]


def _evaluate_one(args: argparse.Namespace, label: str, weights: str, mode: str) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(weights, task="mdetect")
    if mode == "one2many":
        _set_one2many(model)

    eval_name = f"{args.name}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', label)}_{mode}"
    kwargs: dict[str, Any] = {
        "data": args.data,
        "split": "test",
        "device": args.device,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "iou": args.iou,
        "project": args.project,
        "name": eval_name,
        "plots": False,
        "verbose": False,
        "eval_att_by_class": True,
        "calibration_bins": args.calibration_bins,
    }
    if args.conf is not None:
        kwargs["conf"] = args.conf

    print(f"[eval] model={label}, mode={mode}, weights={weights}")
    metrics = model.val(**kwargs)
    detailed = getattr(getattr(metrics, "attributes", None), "detailed", None)
    if detailed is None:
        raise RuntimeError(
            "Detailed test attributes were not collected. Ensure the updated mdetect validator "
            "and attribute_metrics.py are loaded."
        )

    overall = detailed["overall"]
    per_attribute = [dict(row) for row in detailed["per_attribute"]]
    ordinal_rows = _select_ordinal_attributes(args.ordinal_attributes, per_attribute)
    level_count = len({int(row["level"]) for row in detailed["per_level"]})
    ordinal_is_binary = bool(ordinal_rows) and level_count == 2

    summary: dict[str, Any] = {
        "model": label,
        "mode": mode,
        "weight": weights,
        "matched_support": int(overall["matched_support"]),
        **{key: _as_float(overall[key]) for key in QUALITY_KEYS},
        "ordinal_attributes": ",".join(str(row["attribute"]) for row in ordinal_rows) or "none",
        "ordinal_attribute_count": len(ordinal_rows),
        "Ordinal_MAE_test": (
            float(np.mean([row["Ordinal_MAE_test"] for row in ordinal_rows])) if ordinal_rows else None
        ),
        "Ordinal_MAE_normalized_test": (
            float(np.mean([row["Ordinal_MAE_normalized_test"] for row in ordinal_rows]))
            if ordinal_rows
            else None
        ),
        "ordinal_level_count": level_count if ordinal_rows else None,
        "ordinal_binary_degenerate": ordinal_is_binary if ordinal_rows else None,
    }

    attribute_rows: list[dict[str, Any]] = []
    ordinal_indices = {int(row["attribute_index"]) for row in ordinal_rows}
    for row in per_attribute:
        row.update(
            {
                "model": label,
                "mode": mode,
                "weight": weights,
                "ordinal_evaluation": int(row["attribute_index"]) in ordinal_indices,
            }
        )
        attribute_rows.append(row)

    return {"summary": summary, "per_attribute": attribute_rows}


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    if args.calibration_bins < 1:
        raise ValueError("--calibration-bins must be positive")

    results = [_evaluate_one(args, *_parse_model_spec(spec)) for spec in args.model]
    output_dir = Path(args.project) / args.name
    _write_rows(output_dir / "summary.csv", [result["summary"] for result in results])
    _write_rows(
        output_dir / "per_attribute.csv",
        [row for result in results for row in result["per_attribute"]],
    )
    print(f"[summary] {output_dir / 'summary.csv'}")
    print(f"[details] {output_dir / 'per_attribute.csv'}")


if __name__ == "__main__":
    main()
