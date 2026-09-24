"""Plot per-attribute confusion matrices for multi-attribute detection.

The input is ``confusion_test.csv`` produced by
``scripts/eval_attribute_levels.py``.  Only the attribute levels are shown;
unmatched detections and missed objects are intentionally not represented as a
``background`` class because the attribute metrics are conditional on a
correct-class, IoU>=0.5 detection--ground-truth match.

The script writes:

* one 2x5 overview figure per model;
* one two-panel figure per attribute, comparing the requested models; and
* raw and row-normalized CSV values used by the plots.

Example::

    python scripts/plot_attribute_confusion_matrices.py \
        --confusion runs/experiments/E3_risk_level_test/comparison/confusion_test.csv \
        --data ultralytics/cfg/mayolo_r1/mayolo_v3.yaml \
        --models YOLOv10x MAYOLOx \
        --output runs/experiments/E3_risk_level_test/comparison/confusion_figures
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_MODELS: Tuple[str, ...] = ("YOLOv10x", "MAYOLOx")
DEFAULT_LEVEL_NAMES: Tuple[str, ...] = ("No risk", "High risk")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate raw and row-normalized per-attribute confusion-matrix figures"
    )
    parser.add_argument(
        "--confusion",
        required=True,
        help="confusion_test.csv produced by eval_attribute_levels.py",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="optional dataset YAML used to recover attribute names",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(DEFAULT_MODELS),
        help="model labels and plotting order (default: YOLOv10x MAYOLOx)",
    )
    parser.add_argument(
        "--head-mode",
        choices=("native", "one2many", "all"),
        default="native",
        help="filter the head mode in the confusion CSV",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="output directory for PNG/PDF figures and plotted CSV values",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="write PNG files only",
    )
    return parser


def _as_int(value: object, field: str) -> int:
    try:
        number = int(float(str(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if number < 0:
        raise ValueError(f"{field} must be non-negative, got {number}")
    return number


def _as_count(value: object) -> int:
    try:
        number = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid confusion count: {value!r}") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Confusion count must be finite and non-negative, got {value!r}")
    return int(round(number))


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_") or "unnamed"


def _read_attribute_names(data_path: Optional[str]) -> List[str]:
    if not data_path:
        return []
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {path}")
    try:
        import yaml

        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ImportError as exc:
        raise RuntimeError("PyYAML is required when --data is used") from exc
    attributes = config.get("attributes", {})
    if isinstance(attributes, Mapping):
        return [str(name) for name in attributes.keys()]
    if isinstance(attributes, Sequence) and not isinstance(attributes, (str, bytes)):
        return [str(name) for name in attributes]
    return []


def _read_rows(path: Path, head_mode: str) -> List[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in confusion CSV: {path}")
    required = {"model", "attribute_index", "true_level", "predicted_level", "count"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Confusion CSV is missing columns: {sorted(missing)}")
    if head_mode != "all" and "mode" in rows[0]:
        rows = [row for row in rows if (row.get("mode") or "native").strip().lower() == head_mode]
    if not rows:
        raise ValueError(f"No rows remain after --head-mode {head_mode!r} filtering")
    return rows


def _build_matrices(
    rows: Sequence[Mapping[str, str]],
    models: Sequence[str],
    data_attribute_names: Sequence[str],
) -> tuple[dict[str, dict[int, np.ndarray]], list[str], int]:
    """Aggregate CSV rows into model -> attribute index -> confusion matrix."""
    selected = {model.lower(): model for model in models}
    filtered = [row for row in rows if (row.get("model") or "").strip().lower() in selected]
    if not filtered:
        available = sorted({row.get("model", "") for row in rows})
        raise ValueError(f"None of the requested models were found. Available: {available}")

    max_attribute = max(_as_int(row["attribute_index"], "attribute_index") for row in filtered)
    max_level = max(
        max(_as_int(row["true_level"], "true_level"), _as_int(row["predicted_level"], "predicted_level"))
        for row in filtered
    )
    attribute_count = max(max_attribute + 1, len(data_attribute_names))
    level_count = max_level + 1
    matrices: dict[str, dict[int, np.ndarray]] = {
        model: {index: np.zeros((level_count, level_count), dtype=np.int64) for index in range(attribute_count)}
        for model in models
    }

    for row in filtered:
        model_key = (row.get("model") or "").strip().lower()
        model = selected[model_key]
        attribute_index = _as_int(row["attribute_index"], "attribute_index")
        true_level = _as_int(row["true_level"], "true_level")
        predicted_level = _as_int(row["predicted_level"], "predicted_level")
        matrices[model][attribute_index][true_level, predicted_level] += _as_count(row["count"])

    names = list(data_attribute_names)
    for row in filtered:
        index = _as_int(row["attribute_index"], "attribute_index")
        csv_name = (row.get("attribute") or "").strip()
        if csv_name and index >= len(names):
            names.extend([f"attribute_{i}" for i in range(len(names), index + 1)])
            names[index] = csv_name
    if len(names) < attribute_count:
        names.extend(f"attribute_{i}" for i in range(len(names), attribute_count))
    return matrices, names[:attribute_count], level_count


def _level_labels(level_count: int) -> list[str]:
    if level_count == 2:
        return list(DEFAULT_LEVEL_NAMES)
    return [f"Level {index}" for index in range(level_count)]


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    denominator = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, denominator, out=np.zeros_like(matrix, dtype=np.float64), where=denominator != 0)


def _write_value_rows(
    output: Path,
    matrices: Mapping[str, Mapping[int, np.ndarray]],
    names: Sequence[str],
    models: Sequence[str],
) -> None:
    rows: list[dict[str, object]] = []
    for model in models:
        for attribute_index, matrix in matrices[model].items():
            normalized = _row_normalize(matrix)
            for true_level in range(matrix.shape[0]):
                for predicted_level in range(matrix.shape[1]):
                    rows.append(
                        {
                            "model": model,
                            "attribute_index": attribute_index,
                            "attribute": names[attribute_index],
                            "true_level": true_level,
                            "predicted_level": predicted_level,
                            "count": int(matrix[true_level, predicted_level]),
                            "row_normalized": float(normalized[true_level, predicted_level]),
                        }
                    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(figure, output: Path, dpi: int, no_pdf: bool) -> None:
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    if not no_pdf:
        figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")


def _draw_matrix(axis, matrix: np.ndarray, labels: Sequence[str], normalized: bool, vmax: float) -> None:
    import matplotlib.pyplot as plt

    values = _row_normalize(matrix) if normalized else matrix
    image = axis.imshow(values, cmap="Blues", vmin=0.0, vmax=vmax, aspect="equal")
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    for true_level in range(values.shape[0]):
        for predicted_level in range(values.shape[1]):
            value = values[true_level, predicted_level]
            text = f"{value:.1%}" if normalized else f"{int(matrix[true_level, predicted_level])}"
            color = "white" if value > vmax * 0.55 else "black"
            axis.text(predicted_level, true_level, text, ha="center", va="center", color=color, fontsize=8)
    return image


def _plot_model_overview(
    model: str,
    matrices: Mapping[int, np.ndarray],
    names: Sequence[str],
    level_labels: Sequence[str],
    output_dir: Path,
    dpi: int,
    no_pdf: bool,
    normalized: bool,
    vmax: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 5, figsize=(13.0, 5.8), squeeze=False)
    axes_flat = axes.ravel()
    for attribute_index, axis in enumerate(axes_flat):
        if attribute_index >= len(names):
            axis.axis("off")
            continue
        _draw_matrix(axis, matrices[attribute_index], level_labels, normalized, vmax)
        axis.set_title(names[attribute_index], fontsize=9)
    figure.suptitle(f"{model}: attribute confusion matrices", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    suffix = "row_normalized" if normalized else "counts"
    _save_figure(figure, output_dir / f"{_safe_slug(model)}_confusion_{suffix}.png", dpi, no_pdf)
    plt.close(figure)


def _plot_attribute_comparison(
    attribute_index: int,
    name: str,
    matrices: Mapping[str, Mapping[int, np.ndarray]],
    models: Sequence[str],
    level_labels: Sequence[str],
    output_dir: Path,
    dpi: int,
    no_pdf: bool,
    normalized: bool,
    vmax: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(models), figsize=(5.8 * len(models), 3.9), squeeze=False)
    axes_flat = axes.ravel()
    for axis, model in zip(axes_flat, models):
        _draw_matrix(axis, matrices[model][attribute_index], level_labels, normalized, vmax)
        axis.set_title(model)
    figure.suptitle(f"{name}: attribute confusion matrix", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    suffix = "row_normalized" if normalized else "counts"
    filename = f"attribute_{attribute_index:02d}_{_safe_slug(name)}_{suffix}.png"
    _save_figure(figure, output_dir / filename, dpi, no_pdf)
    plt.close(figure)


def main() -> None:
    args = _parser().parse_args()
    if len(args.models) != 2:
        raise ValueError("This comparison figure expects exactly two models: YOLOv10x and MAYOLOx")

    confusion_path = Path(args.confusion)
    rows = _read_rows(confusion_path, args.head_mode)
    data_names = _read_attribute_names(args.data)
    matrices, names, level_count = _build_matrices(rows, args.models, data_names)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    level_labels = _level_labels(level_count)

    _write_value_rows(output_dir / "confusion_matrix_values.csv", matrices, names, args.models)

    raw_vmax = max(
        int(matrix.max())
        for model in args.models
        for matrix in matrices[model].values()
    )
    raw_vmax = max(raw_vmax, 1)
    for model in args.models:
        _plot_model_overview(
            model,
            matrices[model],
            names,
            level_labels,
            output_dir,
            args.dpi,
            args.no_pdf,
            normalized=False,
            vmax=raw_vmax,
        )
        _plot_model_overview(
            model,
            matrices[model],
            names,
            level_labels,
            output_dir,
            args.dpi,
            args.no_pdf,
            normalized=True,
            vmax=1.0,
        )

    for attribute_index, name in enumerate(names):
        _plot_attribute_comparison(
            attribute_index,
            name,
            matrices,
            args.models,
            level_labels,
            output_dir,
            args.dpi,
            args.no_pdf,
            normalized=False,
            vmax=raw_vmax,
        )
        _plot_attribute_comparison(
            attribute_index,
            name,
            matrices,
            args.models,
            level_labels,
            output_dir,
            args.dpi,
            args.no_pdf,
            normalized=True,
            vmax=1.0,
        )

    print(f"[figures] {output_dir}")
    print(f"[values] {output_dir / 'confusion_matrix_values.csv'}")


if __name__ == "__main__":
    main()
