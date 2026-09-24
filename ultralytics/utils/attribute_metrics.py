"""Detailed metrics for matched multi-attribute, multi-level predictions."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


_EPS = 1e-12


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return a finite ratio, using zero for an undefined class statistic."""
    return float(numerator / denominator) if denominator > 0 else 0.0


def _mean_or_zero(values: Sequence[float]) -> float:
    """Average finite values without allowing empty/undefined level rows to poison a summary."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else 0.0


def _binary_stats(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    """Compute one-vs-rest metrics from a binary confusion tuple."""
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    specificity = _safe_divide(tn, tn + fp)
    balanced_accuracy = (recall + specificity) / 2.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def _binary_pr_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    """Compute trapezoidal area under a binary precision-recall curve.

    ``scores`` are continuous probabilities, not thresholded predictions.  The
    curve is constructed by sorting descending scores and treating the target
    class as positive.  A class with no positive target is undefined and
    returns NaN so macro aggregation can ignore it explicitly.
    """
    targets = np.asarray(targets, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if targets.shape != scores.shape:
        raise ValueError(f"PR-AUC target/score shape mismatch: {targets.shape} vs {scores.shape}")
    positives = int(targets.sum())
    if not targets.size or positives == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_targets = targets[order].astype(np.float64)
    tp = np.cumsum(sorted_targets)
    fp = np.cumsum(1.0 - sorted_targets)
    precision = tp / np.maximum(tp + fp, _EPS)
    recall = tp / positives

    # Include the conventional (recall=0, precision=1) endpoint.
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    trapezoid = getattr(np, "trapezoid", np.trapz if hasattr(np, "trapz") else None)
    if trapezoid is None:  # pragma: no cover - compatibility fallback for unusual NumPy builds
        raise RuntimeError("NumPy does not provide trapezoidal integration")
    return float(trapezoid(precision, recall))


def _probability_quality_metrics(targets: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> dict[str, float]:
    """Compute top-label calibration and ordinal metrics for one attribute.

    The inputs are restricted to the same matched detection--ground-truth
    pairs used by the attribute metrics.  ``probabilities`` must contain one
    softmax distribution per sample.  ECE is the standard top-label ECE;
    Brier and NLL use the complete multiclass probability vector.

    ``Ordinal_MAE`` treats the level indices as ordered.  This is meaningful
    only when the attribute levels have a real order.  For two levels it is
    exactly the binary misclassification rate, which is why the caller may
    choose not to report it separately.
    """
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != targets.size:
        raise ValueError(
            "targets/probabilities shape mismatch for probability quality metrics: "
            f"targets={targets.shape}, probabilities={probabilities.shape}"
        )
    if not probabilities.shape[1] or not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite and contain at least one level")
    if targets.size and (targets.min() < 0 or targets.max() >= probabilities.shape[1]):
        raise ValueError("target level is outside the probability vector")
    if not targets.size:
        return {
            "ECE_test": 0.0,
            "Brier_test": 0.0,
            "NLL_test": 0.0,
            "Ordinal_MAE_test": 0.0,
            "Ordinal_MAE_normalized_test": 0.0,
        }

    predicted = probabilities.argmax(axis=1)
    confidence = probabilities[np.arange(targets.size), predicted]
    correctness = (predicted == targets).astype(np.float64)

    # Top-label expected calibration error.  The final bin includes a
    # confidence of exactly one.
    bins = max(int(bins), 1)
    bin_ids = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    ece = 0.0
    for bin_index in range(bins):
        selected = bin_ids == bin_index
        if selected.any():
            ece += float(selected.mean()) * abs(float(confidence[selected].mean()) - float(correctness[selected].mean()))

    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(targets.size), targets] = 1.0
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    true_probability = np.clip(probabilities[np.arange(targets.size), targets], _EPS, 1.0)
    nll = float(np.mean(-np.log(true_probability)))
    ordinal_mae = float(np.mean(np.abs(predicted.astype(np.float64) - targets.astype(np.float64))))
    level_count = probabilities.shape[1]
    ordinal_mae_normalized = ordinal_mae / float(level_count - 1) if level_count > 1 else 0.0
    return {
        "ECE_test": float(ece),
        "Brier_test": brier,
        "NLL_test": nll,
        "Ordinal_MAE_test": ordinal_mae,
        "Ordinal_MAE_normalized_test": float(ordinal_mae_normalized),
    }


def _attribute_names(attribute_names: Mapping[str, Any] | Sequence[str] | None, count: int) -> list[str]:
    """Normalize the project's attribute-name containers to a list."""
    if isinstance(attribute_names, Mapping):
        names = list(attribute_names.keys())
    elif attribute_names is None:
        names = []
    else:
        names = list(attribute_names)
    return [str(names[i]) if i < len(names) else f"attribute_{i}" for i in range(count)]


def compute_attribute_level_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    attribute_names: Mapping[str, Any] | Sequence[str] | None = None,
    calibration_bins: int = 15,
) -> dict[str, Any]:
    """Compute overall, per-attribute, and per-level test metrics.

    Args:
        targets: Integer ground-truth levels with shape ``(N, attributes)``.
        probabilities: Softmax probabilities with shape ``(N, attributes, levels)``.
        attribute_names: Optional mapping/list used only for readable row names.

    Returns:
        A dictionary containing ``overall``, ``per_attribute``, ``per_level``,
        and ``confusion`` rows.  The rows use the ``*_test`` names consumed by
        the experiment summary tables.  The overall result explicitly
        distinguishes ``matched_instances`` (matched objects) from
        ``matched_attribute_decisions`` (matched objects multiplied by the
        number of attributes).
    """
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.ndim != 2:
        raise ValueError(f"targets must have shape (N, attributes), got {targets.shape}")
    if probabilities.ndim != 3:
        raise ValueError(f"probabilities must have shape (N, attributes, levels), got {probabilities.shape}")
    if probabilities.shape[:2] != targets.shape:
        raise ValueError(f"target/probability shape mismatch: {targets.shape} vs {probabilities.shape}")
    if probabilities.shape[2] < 2:
        raise ValueError("Detailed level metrics require at least two mutually exclusive levels")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")

    levels = probabilities.shape[2]
    target_levels = targets.astype(np.int64)
    if not np.array_equal(targets, target_levels):
        raise ValueError("targets must contain integer level indices")
    if target_levels.size and (target_levels.min() < 0 or target_levels.max() >= levels):
        raise ValueError(f"target level is outside [0, {levels - 1}]")
    targets = target_levels
    matched_instances = int(targets.shape[0])
    attribute_count = targets.shape[1]
    names = _attribute_names(attribute_names, attribute_count)
    predicted = probabilities.argmax(axis=2)
    confusion = np.zeros((attribute_count, levels, levels), dtype=np.int64)
    for attribute in range(attribute_count):
        np.add.at(confusion[attribute], (targets[:, attribute].astype(int), predicted[:, attribute]), 1)

    per_level: list[dict[str, Any]] = []
    per_attribute: list[dict[str, Any]] = []
    for attribute in range(attribute_count):
        matrix = confusion[attribute]
        total = int(matrix.sum())
        attribute_level_rows = []
        quality = _probability_quality_metrics(
            targets[:, attribute], probabilities[:, attribute, :], bins=calibration_bins
        )
        for level in range(levels):
            tp = int(matrix[level, level])
            fp = int(matrix[:, level].sum() - tp)
            fn = int(matrix[level, :].sum() - tp)
            tn = int(total - tp - fp - fn)
            stats = _binary_stats(tp, fp, fn, tn)
            row = {
                "attribute": names[attribute],
                "attribute_index": attribute,
                "level": level,
                "support": int(tp + fn),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "Precision_test": stats["precision"],
                "Recall_test": stats["recall"],
                "F1_test": stats["f1"],
                "Balanced_Accuracy_test": stats["balanced_accuracy"],
                "PR_AUC_test": _binary_pr_auc(targets[:, attribute] == level, probabilities[:, attribute, level]),
            }
            per_level.append(row)
            attribute_level_rows.append(row)

        attr_f1 = _mean_or_zero([row["F1_test"] for row in attribute_level_rows])
        attr_precision = _mean_or_zero([row["Precision_test"] for row in attribute_level_rows])
        attr_recall = _mean_or_zero([row["Recall_test"] for row in attribute_level_rows])
        attr_pr_auc = _mean_or_zero([row["PR_AUC_test"] for row in attribute_level_rows])
        per_attribute.append(
            {
                "attribute": names[attribute],
                "attribute_index": attribute,
                "support": total,
                "OA_test": _safe_divide(np.trace(matrix), total),
                "F1_macro_test": attr_f1,
                "F1_macro_global_test": attr_f1,
                "F1_micro_test": _safe_divide(np.trace(matrix), total),
                "P_macro_test": attr_precision,
                "R_macro_test": attr_recall,
                "PR_AUC_macro_test": attr_pr_auc,
                **quality,
            }
        )

    pooled = confusion.sum(axis=0)
    pooled_total = int(pooled.sum())
    pooled_tp = np.diag(pooled).astype(np.float64)
    pooled_fp = pooled.sum(axis=0).astype(np.float64) - pooled_tp
    pooled_fn = pooled.sum(axis=1).astype(np.float64) - pooled_tp
    pooled_tn = pooled_total - pooled_tp - pooled_fp - pooled_fn
    pooled_stats = [_binary_stats(tp, fp, fn, tn) for tp, fp, fn, tn in zip(pooled_tp, pooled_fp, pooled_fn, pooled_tn)]
    pooled_f1 = _mean_or_zero([stats["f1"] for stats in pooled_stats])
    pooled_precision = _mean_or_zero([stats["precision"] for stats in pooled_stats])
    pooled_recall = _mean_or_zero([stats["recall"] for stats in pooled_stats])
    micro_tp = float(pooled_tp.sum())
    micro_fp = float(pooled_fp.sum())
    micro_fn = float(pooled_fn.sum())
    micro_precision = _safe_divide(micro_tp, micro_tp + micro_fp)
    micro_recall = _safe_divide(micro_tp, micro_tp + micro_fn)
    micro_f1 = _safe_divide(2.0 * micro_precision * micro_recall, micro_precision + micro_recall)
    overall_pr_auc = _mean_or_zero([row["PR_AUC_test"] for row in per_level])

    overall = {
        "OA_test": _safe_divide(np.trace(pooled), pooled_total),
        "F1_macro_test": _mean_or_zero([row["F1_macro_test"] for row in per_attribute]),
        "F1_macro_global_test": pooled_f1,
        "F1_micro_test": micro_f1,
        "P_macro_test": _mean_or_zero([row["P_macro_test"] for row in per_attribute]),
        "R_macro_test": _mean_or_zero([row["R_macro_test"] for row in per_attribute]),
        "PR_AUC_macro_test": overall_pr_auc,
        "ECE_macro_test": _mean_or_zero([row["ECE_test"] for row in per_attribute]),
        "Brier_macro_test": _mean_or_zero([row["Brier_test"] for row in per_attribute]),
        "NLL_macro_test": _mean_or_zero([row["NLL_test"] for row in per_attribute]),
        "Ordinal_MAE_macro_test": _mean_or_zero([row["Ordinal_MAE_test"] for row in per_attribute]),
        "Ordinal_MAE_normalized_macro_test": _mean_or_zero(
            [row["Ordinal_MAE_normalized_test"] for row in per_attribute]
        ),
        # One row in ``targets`` represents one correctly matched object.
        "matched_instances": matched_instances,
        # The pooled attribute confusion matrix contains one decision for
        # every attribute of every matched object.
        "matched_attribute_decisions": pooled_total,
    }

    # Level macro rows average the same level over attributes, preserving the
    # level-specific interpretation instead of collapsing level 0 and level 1.
    level_macro: list[dict[str, Any]] = []
    for level in range(levels):
        rows = [row for row in per_level if row["level"] == level]
        level_macro.append(
            {
                "attribute": "all",
                "attribute_index": -1,
                "level": level,
                "support": int(sum(row["support"] for row in rows)),
                "tp": None,
                "fp": None,
                "fn": None,
                "tn": None,
                "Precision_test": _mean_or_zero([row["Precision_test"] for row in rows]),
                "Recall_test": _mean_or_zero([row["Recall_test"] for row in rows]),
                "F1_test": _mean_or_zero([row["F1_test"] for row in rows]),
                "Balanced_Accuracy_test": _mean_or_zero([row["Balanced_Accuracy_test"] for row in rows]),
                "PR_AUC_test": _mean_or_zero([row["PR_AUC_test"] for row in rows]),
            }
        )

    return {
        "overall": overall,
        "per_attribute": per_attribute,
        "per_level": per_level,
        "level_macro": level_macro,
        "confusion": confusion,
    }
