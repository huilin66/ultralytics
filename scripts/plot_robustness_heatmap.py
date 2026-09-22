"""Plot compact test-set robustness sensitivity heatmaps from summary.csv.

The input format is the summary produced by ``scripts/eval_robustness.py``.
It must contain one ``clean`` row per model and rows identified by
``condition`` and ``severity`` for the generated robustness variants.

The default figure is a 2x3 panel:

    rows    : Test mAP50 and Test Macro-F1
    columns : mild, moderate, and severe
    cells   : change from the corresponding clean-test result

The script only reads the summary; it does not load checkpoints or rerun
inference.  It also writes a long-format delta CSV beside the figure so that
the plotted values remain auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


METRICS: Tuple[Tuple[str, str], ...] = (
    ("mAP50_test", "Test mAP50"),
    ("F1_macro_test", "Test Macro-F1"),
)
SEVERITIES: Tuple[str, ...] = ("mild", "moderate", "severe")
CONDITION_ORDER: Tuple[str, ...] = (
    "low_light",
    "overexposure",
    "local_shadow",
    "local_glare",
    "gaussian_blur",
    "motion_blur",
    "local_occlusion",
    "perspective",
    "affine",
    "scale",
    "translate",
    "gaussian_noise",
    "fog",
    "rain",
)
CONDITION_LABELS: Mapping[str, str] = {
    "low_light": "Low light",
    "overexposure": "Overexposure",
    "local_shadow": "Local shadow",
    "local_glare": "Local glare",
    "gaussian_blur": "Gaussian blur",
    "motion_blur": "Motion blur",
    "local_occlusion": "Occlusion",
    "perspective": "Perspective",
    "affine": "Affine",
    "scale": "Scale",
    "translate": "Translate",
    "gaussian_noise": "Gaussian noise",
    "fog": "Fog",
    "rain": "Rain",
}
DEFAULT_MODELS: Tuple[str, ...] = (
    "YOLOv8x",
    "YOLOv9e",
    "YOLOv10x",
    "YOLOv11x",
    "YOLOv12x",
    "YOLOv13x",
    "YOLOv26x",
    "RT-DETR-X",
    "MAYOLOx",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate mAP50/Macro-F1 robustness heatmaps from eval_robustness summary.csv"
    )
    parser.add_argument("--summary", required=True, help="summary.csv produced by eval_robustness.py")
    parser.add_argument(
        "--output",
        default="runs/experiments/E5_robustness/seed0_v1/robustness_heatmap.png",
        help="Output PNG path; PDF and derived delta CSV are written beside it",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model order; unspecified models are appended in first-seen order",
    )
    parser.add_argument(
        "--mode",
        choices=("delta", "relative_percent", "absolute"),
        default="delta",
        help="Cell value: variant-clean, percentage change, or raw test metric",
    )
    parser.add_argument(
        "--no-mean",
        action="store_true",
        help="Do not append the per-model mean over conditions",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Write numeric values into cells; disabled by default for readability",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def _as_float(value: object) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric_value(row: Mapping[str, str], metric: str) -> Optional[float]:
    """Read canonical or legacy metric names from a robustness summary."""

    aliases = {
        "mAP50_test": ("mAP50_test", "metrics/mAP50(B)", "mAP50"),
        "F1_macro_test": ("F1_macro_test", "metrics/f1_macro(A)", "F1_macro"),
    }
    for key in aliases[metric]:
        if key in row:
            value = _as_float(row[key])
            if value is not None:
                return value
    return None


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in summary: {path}")
    required = {"model", "variant"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Summary is missing required columns: {sorted(missing)}")
    return rows


def _condition_and_severity(row: Mapping[str, str]) -> Tuple[str, str]:
    condition = (row.get("condition") or "").strip().lower()
    severity = (row.get("severity") or "").strip().lower()
    if condition and severity:
        return condition, severity
    variant = (row.get("variant") or "").strip().lower()
    for level in SEVERITIES:
        suffix = f"_{level}"
        if variant.endswith(suffix):
            return variant[: -len(suffix)], level
    return condition or variant, severity


def _ordered_models(rows: Sequence[Mapping[str, str]], requested: Optional[Sequence[str]]) -> List[str]:
    available = list(dict.fromkeys(row.get("model", "") for row in rows if row.get("model", "")))
    if requested:
        selected = [model for model in requested if model in available]
        selected.extend(model for model in available if model not in selected)
        return selected
    selected = [model for model in DEFAULT_MODELS if model in available]
    selected.extend(model for model in available if model not in selected)
    return selected


def _ordered_conditions(rows: Sequence[Mapping[str, str]]) -> List[str]:
    available = {
        _condition_and_severity(row)[0]
        for row in rows
        if (row.get("variant") or "").strip().lower() != "clean"
    }
    selected = [condition for condition in CONDITION_ORDER if condition in available]
    selected.extend(sorted(available.difference(selected)))
    if not selected:
        raise ValueError("No non-clean robustness variants found in summary")
    return selected


def _build_lookup(rows: Sequence[Mapping[str, str]]) -> Tuple[Dict[str, Dict[str, str]], Dict[Tuple[str, str, str], Dict[str, str]]]:
    clean: Dict[str, Dict[str, str]] = {}
    variants: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row in rows:
        model = row.get("model", "")
        variant = (row.get("variant") or "").strip().lower()
        if not model:
            continue
        if variant == "clean":
            clean[model] = dict(row)
            continue
        condition, severity = _condition_and_severity(row)
        if condition and severity:
            variants[(model, condition, severity)] = dict(row)
    return clean, variants


def _cell_value(raw: Optional[float], clean: Optional[float], mode: str) -> Optional[float]:
    if raw is None:
        return None
    if mode == "absolute":
        return raw
    if clean is None:
        return None
    delta = raw - clean
    if mode == "relative_percent":
        return 100.0 * delta / clean if clean else None
    return delta


def _matrix(
    models: Sequence[str],
    conditions: Sequence[str],
    severity: str,
    metric: str,
    mode: str,
    clean: Mapping[str, Mapping[str, str]],
    variants: Mapping[Tuple[str, str, str], Mapping[str, str]],
    add_mean: bool,
) -> Tuple[np.ndarray, List[str], List[Dict[str, object]]]:
    values = np.full((len(models), len(conditions)), np.nan, dtype=np.float64)
    audit: List[Dict[str, object]] = []
    for row_index, model in enumerate(models):
        clean_value = _metric_value(clean.get(model, {}), metric)
        for column_index, condition in enumerate(conditions):
            row = variants.get((model, condition, severity), {})
            raw_value = _metric_value(row, metric)
            value = _cell_value(raw_value, clean_value, mode)
            if value is not None:
                values[row_index, column_index] = value
            audit.append(
                {
                    "model": model,
                    "metric": metric,
                    "severity": severity,
                    "condition": condition,
                    "clean_value": clean_value,
                    "variant_value": raw_value,
                    "plot_value": value,
                }
            )
    labels = [CONDITION_LABELS.get(condition, condition.replace("_", " ")) for condition in conditions]
    if add_mean:
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(values, axis=1, keepdims=True)
        values = np.concatenate((values, mean), axis=1)
        labels.append("Mean")
    return values, labels, audit


def _limits(matrices: Sequence[np.ndarray], mode: str) -> Tuple[float, float]:
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices if np.isfinite(matrix).any()])
    if mode in {"delta", "relative_percent"}:
        limit = max(float(np.max(np.abs(finite))), 1e-6)
        return -limit, limit
    return float(np.min(finite)), float(np.max(finite))


def _format_value(value: Optional[float], mode: str) -> str:
    if value is None:
        return ""
    return f"{value:.4f}" if mode != "relative_percent" else f"{value:.2f}"


def _write_audit(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    fields = ("model", "metric", "severity", "condition", "clean_value", "variant_value", "plot_value")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = _parser().parse_args()
    summary_path = Path(args.summary).expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    rows = _read_rows(summary_path)
    models = _ordered_models(rows, args.models)
    conditions = _ordered_conditions(rows)
    clean, variants = _build_lookup(rows)
    missing_clean = [model for model in models if model not in clean]
    if missing_clean:
        raise ValueError(f"Missing clean rows for model(s): {missing_clean}")

    matrices: Dict[Tuple[str, str], np.ndarray] = {}
    labels: Dict[str, List[str]] = {}
    audits: List[Dict[str, object]] = []
    for metric, _ in METRICS:
        for severity in SEVERITIES:
            matrix, condition_labels, audit = _matrix(
                models,
                conditions,
                severity,
                metric,
                args.mode,
                clean,
                variants,
                add_mean=not args.no_mean,
            )
            matrices[(metric, severity)] = matrix
            labels[metric] = condition_labels
            audits.extend(audit)
    vmin, vmax = _limits(list(matrices.values()), args.mode)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    if args.mode in {"delta", "relative_percent"}:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap = "RdBu"
    else:
        norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-6)
        cmap = "viridis"

    figure, axes = plt.subplots(2, 3, figsize=(20, 8.5), squeeze=False, constrained_layout=True)
    last_images = []
    for metric_index, (metric, metric_label) in enumerate(METRICS):
        for severity_index, severity in enumerate(SEVERITIES):
            axis = axes[metric_index, severity_index]
            matrix = np.ma.masked_invalid(matrices[(metric, severity)])
            image = axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
            last_images.append(image)
            axis.set_title(f"{metric_label} — {severity.title()}")
            axis.set_xticks(np.arange(len(labels[metric])))
            axis.set_xticklabels(labels[metric], rotation=55, ha="right", fontsize=8)
            axis.set_yticks(np.arange(len(models)))
            axis.set_yticklabels(models if severity_index == 0 else [], fontsize=9)
            axis.tick_params(length=0)
            axis.set_xticks(np.arange(-0.5, len(labels[metric]), 1), minor=True)
            axis.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=0.5)
            axis.tick_params(which="minor", bottom=False, left=False)
            if not args.no_mean:
                axis.axvline(len(conditions) - 0.5, color="black", linewidth=1.0)
            if args.annotate:
                for row_index in range(matrix.shape[0]):
                    for column_index in range(matrix.shape[1]):
                        value = matrix[row_index, column_index]
                        if not np.ma.is_masked(value):
                            axis.text(
                                column_index,
                                row_index,
                                _format_value(float(value), args.mode),
                                ha="center",
                                va="center",
                                fontsize=6,
                            )
        colorbar = figure.colorbar(last_images[-1], ax=axes[metric_index, :].tolist(), shrink=0.86, pad=0.02)
        colorbar.set_label(
            "Δ from clean test" if args.mode == "delta" else "% change from clean" if args.mode == "relative_percent" else "Test metric"
        )

    figure.suptitle("Test-set robustness sensitivity", fontsize=15)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    audit_path = output.with_name(f"{output.stem}_data.csv")
    _write_audit(audit_path, audits)
    metadata = {
        "summary": str(summary_path),
        "mode": args.mode,
        "models": models,
        "conditions": conditions,
        "severities": list(SEVERITIES),
        "metrics": [metric for metric, _ in METRICS],
        "include_mean": not args.no_mean,
    }
    output.with_name(f"{output.stem}_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[figure] {output}")
    print(f"[figure] {output.with_suffix('.pdf')}")
    print(f"[data] {audit_path}")


if __name__ == "__main__":
    main()
