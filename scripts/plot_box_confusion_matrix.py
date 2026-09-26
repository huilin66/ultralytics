"""Plot object-detection box confusion matrices.

The input CSV is produced by ``scripts/eval_attribute_levels.py``.  Rows are
predicted classes, columns are true classes, and the final background class
keeps false positives and missed detections visible.  The normalized plot is
column-normalized by the true class, matching the convention used by the
Ultralytics confusion-matrix implementation.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


DEFAULT_MODELS = ("YOLOv10x", "MAYOLOx")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot object-detection box confusion matrices")
    parser.add_argument("--confusion", required=True, help="box_confusion_test.csv from eval_attribute_levels.py")
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--head-mode", choices=("native", "one2many", "all"), default="all")
    parser.add_argument("--output", required=True, help="Output directory for figures and CSV values")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-pdf", action="store_true", help="Write PNG files only")
    return parser


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_") or "unnamed"


def _read_matrices(
    path: Path, models: Sequence[str], head_mode: str
) -> tuple[dict[str, np.ndarray], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in box confusion CSV: {path}")
    required = {"model", "predicted_index", "predicted_name", "true_index", "true_name", "count"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Box confusion CSV is missing columns: {sorted(missing)}")
    if head_mode != "all" and "mode" in rows[0]:
        rows = [row for row in rows if (row.get("mode") or "native").strip().lower() == head_mode]
    selected = {model.lower(): model for model in models}
    rows = [row for row in rows if (row.get("model") or "").strip().lower() in selected]
    if not rows:
        raise ValueError(f"None of the requested models were found in {path}")

    size = 1 + max(
        max(int(float(row["predicted_index"])), int(float(row["true_index"]))) for row in rows
    )
    names = [f"class_{index}" for index in range(size)]
    matrices = {model: np.zeros((size, size), dtype=np.int64) for model in models}
    for row in rows:
        model = selected[(row.get("model") or "").strip().lower()]
        predicted_index = int(float(row["predicted_index"]))
        true_index = int(float(row["true_index"]))
        names[predicted_index] = row["predicted_name"]
        names[true_index] = row["true_name"]
        matrices[model][predicted_index, true_index] += int(float(row["count"]))
    names[-1] = "background"
    return matrices, names


def _column_normalize(matrix: np.ndarray) -> np.ndarray:
    denominator = matrix.sum(axis=0, keepdims=True).astype(float)
    return np.divide(matrix, denominator, out=np.zeros_like(matrix, dtype=float), where=denominator > 0)


def _write_values(path: Path, matrices: Mapping[str, np.ndarray], names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "predicted_index", "predicted_name", "true_index", "true_name", "count", "true_normalized"],
        )
        writer.writeheader()
        for model, matrix in matrices.items():
            normalized = _column_normalize(matrix)
            for predicted_index, true_index in np.ndindex(matrix.shape):
                writer.writerow(
                    {
                        "model": model,
                        "predicted_index": predicted_index,
                        "predicted_name": names[predicted_index],
                        "true_index": true_index,
                        "true_name": names[true_index],
                        "count": int(matrix[predicted_index, true_index]),
                        "true_normalized": f"{normalized[predicted_index, true_index]:.8f}",
                    }
                )


def _plot(
    matrices: Mapping[str, np.ndarray],
    names: Sequence[str],
    output: Path,
    dpi: int,
    no_pdf: bool,
    normalized: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(matrices)
    values = {model: _column_normalize(matrix) if normalized else matrix for model, matrix in matrices.items()}
    vmax = 1.0 if normalized else max(1, max(float(matrix.max()) for matrix in values.values()))
    figure, axes = plt.subplots(1, len(models), figsize=(5.1 * len(models), 4.7), squeeze=False)
    axes_flat = axes.ravel()
    image = None
    for axis, model in zip(axes_flat, models):
        image = axis.imshow(values[model], cmap="Blues", vmin=0.0, vmax=vmax)
        axis.set_title(model)
        axis.set_xlabel("True class")
        axis.set_ylabel("Predicted class")
        axis.set_xticks(range(len(names)), names, rotation=35, ha="right")
        axis.set_yticks(range(len(names)), names)
        for predicted_index, true_index in np.ndindex(values[model].shape):
            value = values[model][predicted_index, true_index]
            text = f"{value:.2f}" if normalized else f"{int(value)}"
            axis.text(true_index, predicted_index, text, ha="center", va="center", fontsize=9)
    figure.colorbar(image, ax=axes_flat.tolist(), fraction=0.035, pad=0.04)
    title = "Box confusion matrix (true-normalized)" if normalized else "Box confusion matrix (counts)"
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    suffix = "true_normalized" if normalized else "counts"
    output.mkdir(parents=True, exist_ok=True)
    png = output / f"box_confusion_{suffix}.png"
    figure.savefig(png, dpi=dpi, bbox_inches="tight")
    if not no_pdf:
        figure.savefig(output / f"box_confusion_{suffix}.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = _parser().parse_args()
    matrices, names = _read_matrices(Path(args.confusion), args.models, args.head_mode)
    output = Path(args.output)
    _write_values(output / "box_confusion_matrix_values.csv", matrices, names)
    _plot(matrices, names, output, args.dpi, args.no_pdf, normalized=False)
    _plot(matrices, names, output, args.dpi, args.no_pdf, normalized=True)
    print(f"[figures] {output}")
    print(f"[values] {output / 'box_confusion_matrix_values.csv'}")


if __name__ == "__main__":
    main()
