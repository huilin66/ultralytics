"""Plot robustness sensitivity as compact variant-wise line charts.

The script consumes the ``summary.csv`` produced by
``scripts/eval_robustness.py``.  It creates a 2x3 figure:

    rows    : Test mAP50 and Test Macro-F1
    columns : Illumination, Image degradation, Viewpoint and scale
    x-axis  : the nine evaluated models
    color   : robustness Variant
    marker  : severity (mild circle, moderate square, severe triangle)

By default, plotted values are changes from each model's clean-test result.
No checkpoint loading or inference is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from plot_robustness_heatmap import (
    CONDITION_LABELS,
    DEFAULT_MODELS,
    METRICS,
    SEVERITIES,
    _as_float,
    _build_lookup,
    _cell_value,
    _metric_value,
    _ordered_conditions,
    _ordered_models,
    _read_rows,
)


CATEGORY_ORDER: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    (
        "illumination",
        "Illumination",
        ("low_light", "overexposure", "local_shadow", "local_glare"),
    ),
    (
        "image_degradation",
        "Image degradation",
        ("gaussian_blur", "motion_blur", "local_occlusion", "gaussian_noise", "fog", "rain"),
    ),
    (
        "viewpoint_scale",
        "Viewpoint and scale",
        ("perspective", "affine", "scale", "translate"),
    ),
)
SEVERITY_MARKERS: Mapping[str, str] = {
    "mild": "o",
    "moderate": "s",
    "severe": "^",
}
SEVERITY_LABELS: Mapping[str, str] = {
    "mild": "Mild",
    "moderate": "Moderate",
    "severe": "Severe",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot robustness line charts from summary.csv")
    parser.add_argument("--summary", required=True, help="summary.csv produced by eval_robustness.py")
    parser.add_argument(
        "--output",
        default="runs/experiments/E5_robustness/seed0_v1/robustness_sensitivity_lines.png",
        help="Output PNG path; PDF and audit data are written beside it",
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
        help="Line values: variant-clean, percentage change, or raw test metric",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def _limits(matrices: Sequence[np.ndarray], mode: str) -> Tuple[float, float]:
    finite_parts = [matrix[np.isfinite(matrix)] for matrix in matrices if np.isfinite(matrix).any()]
    if not finite_parts:
        raise ValueError("No finite metric values were found for plotting")
    finite = np.concatenate(finite_parts)
    if mode in {"delta", "relative_percent"}:
        limit = max(float(np.max(np.abs(finite))), 1e-6)
        return -limit, limit
    lower, upper = float(np.min(finite)), float(np.max(finite))
    return lower, upper if upper > lower else lower + 1e-6


def _write_audit(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    fields = (
        "model",
        "metric",
        "category",
        "variant",
        "severity",
        "clean_value",
        "variant_value",
        "plot_value",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _display_value(value: Optional[float], mode: str) -> str:
    if value is None:
        return ""
    return f"{value:.4f}" if mode != "relative_percent" else f"{value:.2f}"


def main() -> None:
    args = _parser().parse_args()
    summary_path = Path(args.summary).expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    rows = _read_rows(summary_path)
    models = _ordered_models(rows, args.models)
    available_conditions = set(_ordered_conditions(rows))
    clean, variants = _build_lookup(rows)
    missing_clean = [model for model in models if model not in clean]
    if missing_clean:
        raise ValueError(f"Missing clean rows for model(s): {missing_clean}")

    series: Dict[Tuple[str, str, str, str], np.ndarray] = {}
    audit: List[Dict[str, object]] = []
    all_matrices_by_metric: Dict[str, List[np.ndarray]] = {metric: [] for metric, _ in METRICS}
    category_conditions: Dict[str, List[str]] = {}

    for category_key, _, condition_names in CATEGORY_ORDER:
        selected = [condition for condition in condition_names if condition in available_conditions]
        if not selected:
            raise ValueError(f"No variants found for category {category_key!r}")
        category_conditions[category_key] = selected
        for metric, _ in METRICS:
            for condition in selected:
                for severity in SEVERITIES:
                    values = np.full(len(models), np.nan, dtype=np.float64)
                    for model_index, model in enumerate(models):
                        clean_value = _metric_value(clean[model], metric)
                        row = variants.get((model, condition, severity), {})
                        raw_value = _metric_value(row, metric)
                        value = _cell_value(raw_value, clean_value, args.mode)
                        if value is not None:
                            values[model_index] = value
                        audit.append(
                            {
                                "model": model,
                                "metric": metric,
                                "category": category_key,
                                "variant": condition,
                                "severity": severity,
                                "clean_value": clean_value,
                                "variant_value": raw_value,
                                "plot_value": value,
                            }
                        )
                    series[(metric, category_key, condition, severity)] = values
                    all_matrices_by_metric[metric].append(values)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(2, 3, figsize=(20, 11), squeeze=False)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.31, wspace=0.18, hspace=0.48)
    x = np.arange(len(models))
    color_maps = [plt.get_cmap("tab10"), plt.get_cmap("tab10"), plt.get_cmap("tab10")]
    all_legend_handles: List[List[Line2D]] = []

    for metric_index, (metric, metric_label) in enumerate(METRICS):
        ymin, ymax = _limits(all_matrices_by_metric[metric], args.mode)
        padding = max((ymax - ymin) * 0.06, 1e-4)
        for category_index, (category_key, category_label, _) in enumerate(CATEGORY_ORDER):
            axis = axes[metric_index, category_index]
            conditions = category_conditions[category_key]
            color_map = color_maps[category_index]
            variant_handles: List[Line2D] = []
            for variant_index, condition in enumerate(conditions):
                color = color_map(variant_index / max(1, len(conditions) - 1))
                variant_handles.append(
                    Line2D([0], [0], color=color, linewidth=2.0, label=CONDITION_LABELS.get(condition, condition))
                )
                for severity in SEVERITIES:
                    values = series[(metric, category_key, condition, severity)]
                    axis.plot(
                        x,
                        values,
                        color=color,
                        marker=SEVERITY_MARKERS[severity],
                        markersize=5.2,
                        linewidth=1.45,
                        alpha=0.9,
                    )
            axis.axhline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.65)
            axis.set_ylim(ymin - padding, ymax + padding)
            axis.set_title(f"{metric_label} — {category_label}")
            axis.set_xticks(x)
            axis.set_xticklabels(models, rotation=38, ha="right", fontsize=8)
            axis.grid(axis="y", color="0.82", linewidth=0.7)
            axis.tick_params(axis="both", labelsize=8)
            if category_index == 0:
                if args.mode == "delta":
                    ylabel = "Δ from clean test"
                elif args.mode == "relative_percent":
                    ylabel = "% change from clean"
                else:
                    ylabel = "Test metric"
                axis.set_ylabel(ylabel)
            else:
                axis.set_ylabel("")
            if metric_index == 1:
                axis.set_xlabel("Model")
            # Keep one Variant legend per category column.  Putting legends
            # on both metric rows makes the upper-row legend overlap the
            # lower-row titles when the figure is compressed.
            if metric_index == len(METRICS) - 1:
                axis.legend(
                    handles=variant_handles,
                    title="Variant",
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.34),
                    ncol=2 if len(conditions) > 4 else 2,
                    fontsize=7.5,
                    title_fontsize=8,
                    frameon=False,
                    handlelength=1.8,
                    columnspacing=0.8,
                )

    severity_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker=SEVERITY_MARKERS[severity],
            linestyle="-",
            linewidth=1.2,
            markersize=5.5,
            label=SEVERITY_LABELS[severity],
        )
        for severity in SEVERITIES
    ]
    figure.legend(
        handles=severity_handles,
        title="Severity",
        # Keep it inside the first column, beside the first column's Variant
        # legend at the same vertical level.
        loc="lower left",
        bbox_to_anchor=(0.04, 0.17),
        ncol=1,
        frameon=False,
        fontsize=9,
        title_fontsize=9,
    )
    figure.suptitle("Test-set robustness sensitivity by variant and severity", fontsize=15)

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    audit_path = output.with_name(f"{output.stem}_data.csv")
    _write_audit(audit_path, audit)
    metadata = {
        "summary": str(summary_path),
        "mode": args.mode,
        "models": models,
        "categories": {
            key: conditions for key, conditions in category_conditions.items()
        },
        "severities": list(SEVERITIES),
        "metrics": [metric for metric, _ in METRICS],
        "severity_markers": dict(SEVERITY_MARKERS),
    }
    output.with_name(f"{output.stem}_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[figure] {output}")
    print(f"[figure] {output.with_suffix('.pdf')}")
    print(f"[data] {audit_path}")


if __name__ == "__main__":
    main()
