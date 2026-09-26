"""Recompute detailed test attribute/level metrics and compare them to mdet metrics.

The evaluator deliberately reuses the task validator's IoU=0.5 and class-correct
matches.  The validator now retains the matched ground-truth levels and the
corresponding softmax probabilities, so this script can compare a fresh
calculation against the values already exposed by ``MDetMetrics``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


INTEGRATED_KEYS = {
    "OA_test": "metrics/OA(A)",
    "F1_macro_test": "metrics/f1_macro(A)",
    "F1_macro_global_test": "metrics/f1_macro_global(A)",
    "P_macro_test": "metrics/P_macro(A)",
    "R_macro_test": "metrics/R_macro(A)",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate detailed test attribute/level metrics")
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
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--project", default="runs/experiments/E3_risk_level_test")
    parser.add_argument("--name", default="comparison")
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
    """Switch an mdet model to its one-to-many inference head."""
    detector = getattr(model, "model", None)
    layers = getattr(detector, "model", None)
    if layers is None or not layers:
        raise RuntimeError("Could not locate the mdet model head")
    switch = getattr(layers[-1], "use_one2many_head", None)
    if switch is None:
        raise RuntimeError("The loaded checkpoint does not expose use_one2many_head()")
    switch()


def _as_float(value: Any) -> float:
    """Convert numpy/torch scalar values to a CSV-friendly float."""
    return float(value.item() if hasattr(value, "item") else value)


def _dataset_class_names(data: str) -> dict[int, str]:
    """Read object-class names from the dataset definition, not checkpoint metadata."""
    from ultralytics.utils import yaml_load

    names = yaml_load(data).get("names", {})
    if isinstance(names, (list, tuple)):
        return {index: str(name) for index, name in enumerate(names)}
    return {int(index): str(name) for index, name in names.items()}


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write heterogeneous detail rows while preserving first-seen column order."""
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


def _evaluate_one(args: argparse.Namespace, label: str, weights: str, mode: str) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(weights, task="mdetect")
    if mode == "one2many":
        _set_one2many(model)

    eval_name = f"{args.name}_{label}_{mode}".replace(" ", "_")
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
    }
    if args.conf is not None:
        kwargs["conf"] = args.conf

    print(f"[eval] model={label}, mode={mode}, weights={weights}")
    metrics = model.val(**kwargs)
    values = metrics.results_dict
    detailed = getattr(getattr(metrics, "attributes", None), "detailed", None)
    if detailed is None:
        raise RuntimeError(
            "Detailed level metrics were not collected. Ensure split=test and the updated mdetect validator is loaded."
        )

    # ``mAP`` is the mean over object classes.  Keep the class-wise AP values
    # as well so section 6.1 can report exactly which object class contributes
    # to the aggregate detection scores.
    box_metrics = getattr(metrics, "box", None)
    if box_metrics is None:
        raise RuntimeError("The test validator did not expose box metrics for class-wise AP export.")
    class_ap50 = np.asarray(getattr(box_metrics, "ap50", []), dtype=np.float64)
    class_ap5095 = np.asarray(getattr(box_metrics, "ap", []), dtype=np.float64)
    class_precision = np.asarray(getattr(box_metrics, "p", []), dtype=np.float64)
    class_recall = np.asarray(getattr(box_metrics, "r", []), dtype=np.float64)
    class_indices = np.asarray(getattr(box_metrics, "ap_class_index", []), dtype=np.int64)
    if not (
        len(class_ap50)
        == len(class_ap5095)
        == len(class_precision)
        == len(class_recall)
        == len(class_indices)
    ):
        raise RuntimeError(
            "Class-wise AP arrays have inconsistent lengths: "
            f"AP50={len(class_ap50)}, AP50-95={len(class_ap5095)}, "
            f"Precision={len(class_precision)}, Recall={len(class_recall)}, "
            f"class_index={len(class_indices)}."
        )
    class_names = _dataset_class_names(args.data)
    if not class_names:
        class_names = getattr(metrics, "names", None) or getattr(model, "names", None) or {}
    if isinstance(class_names, (list, tuple)):
        class_names = {index: name for index, name in enumerate(class_names)}
    per_class = [
        {
            "model": label,
            "mode": mode,
            "weight": weights,
            "class_index": int(class_index),
            "class_name": str(class_names.get(int(class_index), int(class_index))),
            "AP50_test": float(ap50),
            "AP50-95_test": float(ap5095),
        }
        for class_index, precision, recall, ap50, ap5095 in zip(
            class_indices, class_precision, class_recall, class_ap50, class_ap5095
        )
    ]
    for row, precision, recall in zip(per_class, class_precision, class_recall):
        row["Precision_test"] = float(precision)
        row["Recall_test"] = float(recall)

    # MDetectionValidator exports a box confusion matrix built from the same
    # IoU>=0.5/class-correct matching basis as the attribute metrics.  The
    # standard plotting confusion matrix uses an independent confidence
    # threshold and is therefore not suitable for this table.
    box_confusion = np.asarray(getattr(metrics, "box_confusion_matrix", []), dtype=np.float64)
    box_confusion_rows = []
    if box_confusion.ndim == 2 and box_confusion.shape[0] == box_confusion.shape[1]:
        background_index = box_confusion.shape[0] - 1
        for predicted_index in range(box_confusion.shape[0]):
            for true_index in range(box_confusion.shape[1]):
                predicted_name = "background" if predicted_index == background_index else str(
                    class_names.get(predicted_index, predicted_index)
                )
                true_name = "background" if true_index == background_index else str(
                    class_names.get(true_index, true_index)
                )
                box_confusion_rows.append(
                    {
                        "model": label,
                        "mode": mode,
                        "weight": weights,
                        "predicted_index": predicted_index,
                        "predicted_name": predicted_name,
                        "true_index": true_index,
                        "true_name": true_name,
                        "count": int(round(box_confusion[predicted_index, true_index])),
                    }
                )
    else:
        raise RuntimeError(
            "The Test validator did not expose the exact square box confusion matrix. "
            "Ensure the updated mdetect validator is loaded."
        )

    detailed_by_class = getattr(getattr(metrics, "attributes", None), "detailed_by_class", None) or {}
    per_class_attribute = []
    per_class_attribute_detail = []
    for class_index, class_detail in sorted(detailed_by_class.items()):
        class_index = int(class_index)
        class_name = str(class_names.get(class_index, class_index))
        overall_by_class = class_detail["overall"]
        per_class_attribute.append(
            {
                "model": label,
                "mode": mode,
                "weight": weights,
                "class_index": class_index,
                "class_name": class_name,
                "matched_instances": int(overall_by_class["matched_instances"]),
                "matched_attribute_decisions": int(overall_by_class["matched_attribute_decisions"]),
                "OA_test": _as_float(overall_by_class["OA_test"]),
                "F1_macro_test": _as_float(overall_by_class["F1_macro_test"]),
                "F1_macro_global_test": _as_float(overall_by_class["F1_macro_global_test"]),
                "F1_micro_test": _as_float(overall_by_class["F1_micro_test"]),
                "P_macro_test": _as_float(overall_by_class["P_macro_test"]),
                "R_macro_test": _as_float(overall_by_class["R_macro_test"]),
                "PR_AUC_macro_test": _as_float(overall_by_class["PR_AUC_macro_test"]),
            }
        )
        for detail in class_detail["per_attribute"]:
            detail = dict(detail)
            detail.update(
                {
                    "model": label,
                    "mode": mode,
                    "weight": weights,
                    "class_index": class_index,
                    "class_name": class_name,
                }
            )
            per_class_attribute_detail.append(detail)

    overall = detailed["overall"]
    row: dict[str, Any] = {
        "model": label,
        "mode": mode,
        "weight": weights,
        "mAP50_test": _as_float(values["metrics/mAP50(B)"]),
        "mAP50-95_test": _as_float(values["metrics/mAP50-95(B)"]),
        "matched_instances": int(overall["matched_instances"]),
        "matched_attribute_decisions": int(overall["matched_attribute_decisions"]),
        "F1_micro_test": _as_float(overall["F1_micro_test"]),
        "PR_AUC_macro_test": _as_float(overall["PR_AUC_macro_test"]),
        "ECE_macro_test": _as_float(overall["ECE_macro_test"]),
        "Brier_macro_test": _as_float(overall["Brier_macro_test"]),
        "NLL_macro_test": _as_float(overall["NLL_macro_test"]),
        "Ordinal_MAE_macro_test": _as_float(overall["Ordinal_MAE_macro_test"]),
        "Ordinal_MAE_normalized_macro_test": _as_float(
            overall["Ordinal_MAE_normalized_macro_test"]
        ),
    }
    for metric_name, integrated_key in INTEGRATED_KEYS.items():
        integrated = _as_float(values[integrated_key])
        recomputed = _as_float(overall[metric_name])
        row[f"{metric_name}_integrated"] = integrated
        row[f"{metric_name}_recomputed"] = recomputed
        row[f"{metric_name}_delta"] = recomputed - integrated

    attributes = getattr(metrics, "attributes", None)
    integrated_f1 = np.asarray(getattr(attributes, "all_f1_macro", []), dtype=np.float64)
    integrated_precision = np.asarray(getattr(attributes, "all_precision", []), dtype=np.float64)
    integrated_recall = np.asarray(getattr(attributes, "all_recall", []), dtype=np.float64)
    for detail in detailed["per_attribute"]:
        index = int(detail["attribute_index"])
        detail.update(
            {
                "model": label,
                "mode": mode,
                "weight": weights,
                "integrated_F1_macro_test": float(integrated_f1[index]) if index < len(integrated_f1) else None,
                "integrated_P_macro_test": float(integrated_precision[index]) if index < len(integrated_precision) else None,
                "integrated_R_macro_test": float(integrated_recall[index]) if index < len(integrated_recall) else None,
            }
        )
        if index < len(integrated_f1):
            detail["delta_F1_macro_test"] = detail["F1_macro_test"] - float(integrated_f1[index])
        if index < len(integrated_precision):
            detail["delta_P_macro_test"] = detail["P_macro_test"] - float(integrated_precision[index])
        if index < len(integrated_recall):
            detail["delta_R_macro_test"] = detail["R_macro_test"] - float(integrated_recall[index])

    per_level = []
    for detail in detailed["per_level"] + detailed["level_macro"]:
        detail = dict(detail)
        detail.update({"model": label, "mode": mode, "weight": weights})
        per_level.append(detail)

    confusion_rows = []
    confusion = np.asarray(detailed["confusion"])
    for attribute_index, matrix in enumerate(confusion):
        attribute_name = detailed["per_attribute"][attribute_index]["attribute"]
        for true_level in range(matrix.shape[0]):
            for predicted_level in range(matrix.shape[1]):
                confusion_rows.append(
                    {
                        "model": label,
                        "mode": mode,
                        "weight": weights,
                        "attribute_index": attribute_index,
                        "attribute": attribute_name,
                        "true_level": true_level,
                        "predicted_level": predicted_level,
                        "count": int(matrix[true_level, predicted_level]),
                    }
                )

    return {
        "summary": row,
        "per_class": per_class,
        "box_confusion": box_confusion_rows,
        "per_class_attribute": per_class_attribute,
        "per_class_attribute_detail": per_class_attribute_detail,
        "per_attribute": detailed["per_attribute"],
        "per_level": per_level,
        "confusion": confusion_rows,
    }


def main() -> None:
    args = _parser().parse_args()
    output_dir = Path(args.project) / args.name
    results = [_evaluate_one(args, *_parse_model_spec(spec)) for spec in args.model]
    _write_rows(output_dir / "summary.csv", [result["summary"] for result in results])
    _write_rows(output_dir / "per_class_test.csv", [row for result in results for row in result["per_class"]])
    _write_rows(
        output_dir / "box_confusion_test.csv",
        [row for result in results for row in result["box_confusion"]],
    )
    _write_rows(
        output_dir / "per_class_attribute_test.csv",
        [row for result in results for row in result["per_class_attribute"]],
    )
    _write_rows(
        output_dir / "per_class_attribute_detail_test.csv",
        [row for result in results for row in result["per_class_attribute_detail"]],
    )
    _write_rows(output_dir / "per_attribute_test.csv", [row for result in results for row in result["per_attribute"]])
    _write_rows(output_dir / "per_level_test.csv", [row for result in results for row in result["per_level"]])
    _write_rows(output_dir / "confusion_test.csv", [row for result in results for row in result["confusion"]])
    print(f"[summary] {output_dir / 'summary.csv'}")
    print(f"[details] {output_dir / 'per_class_test.csv'}")
    print(f"[details] {output_dir / 'box_confusion_test.csv'}")
    print(f"[details] {output_dir / 'per_class_attribute_test.csv'}")
    print(f"[details] {output_dir / 'per_class_attribute_detail_test.csv'}")
    print(f"[details] {output_dir / 'per_attribute_test.csv'}")
    print(f"[details] {output_dir / 'per_level_test.csv'}")
    print(f"[details] {output_dir / 'confusion_test.csv'}")


if __name__ == "__main__":
    main()
