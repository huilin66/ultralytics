"""Metrics for image-level multi-label classification."""

from __future__ import annotations

import torch


class MultiLabelClassificationMetrics:
    """Compute thresholded multi-label metrics and class-wise average precision."""

    keys = [
        "metrics/precision",
        "metrics/recall",
        "metrics/f1_macro",
        "metrics/f1_micro",
        "metrics/mAP",
        "metrics/subset_accuracy",
    ]

    def __init__(self, names=None, threshold=0.5):
        self.names = names or {}
        self.threshold = float(threshold)
        self.speed = {}
        self.save_dir = None
        self.per_class = {}
        self.results_dict = {key: 0.0 for key in self.keys}
        self.results_dict["fitness"] = 0.0

    @staticmethod
    def _average_precision(scores, targets):
        """Compute the area under the precision-recall staircase for one class."""
        positive_count = int(targets.sum())
        if positive_count == 0:
            return float("nan")
        order = torch.argsort(scores, descending=True)
        truth = targets[order].float()
        precision = truth.cumsum(0) / torch.arange(1, truth.numel() + 1, dtype=torch.float32)
        return float((precision * truth).sum() / positive_count)

    @staticmethod
    def _safe_div(numerator, denominator):
        """Divide without producing NaN for an absent class or empty dataset."""
        return numerator / denominator.clamp_min(1e-9)

    def process(self, targets, scores):
        """Aggregate predictions and targets into scalar and per-class metrics."""
        if not targets or not scores:
            self.results_dict = {key: 0.0 for key in self.keys}
            self.results_dict["fitness"] = 0.0
            return self.results_dict
        targets = torch.cat([torch.as_tensor(value).cpu() for value in targets], dim=0).float()
        scores = torch.cat([torch.as_tensor(value).cpu() for value in scores], dim=0).float()
        if targets.ndim != 2 or scores.shape != targets.shape:
            raise ValueError(f"Targets and scores must both have shape [N, nc], got {targets.shape} and {scores.shape}")
        if targets.numel() and not torch.all((targets == 0) | (targets == 1)):
            raise ValueError("Multi-label classification targets must be binary")

        predicted = scores >= self.threshold
        target_bool = targets.bool()
        tp = (predicted & target_bool).sum(0).float()
        fp = (predicted & ~target_bool).sum(0).float()
        fn = (~predicted & target_bool).sum(0).float()
        precision = self._safe_div(tp, tp + fp)
        recall = self._safe_div(tp, tp + fn)
        f1 = self._safe_div(2 * precision * recall, precision + recall)
        support = target_bool.sum(0)
        valid = support > 0
        macro_slice = valid if valid.any() else torch.ones_like(valid, dtype=torch.bool)

        micro_tp, micro_fp, micro_fn = tp.sum(), fp.sum(), fn.sum()
        micro_precision = self._safe_div(micro_tp, micro_tp + micro_fp)
        micro_recall = self._safe_div(micro_tp, micro_tp + micro_fn)
        micro_f1 = self._safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)
        ap = torch.tensor(
            [self._average_precision(scores[:, col], target_bool[:, col]) for col in range(targets.shape[1])],
            dtype=torch.float32,
        )
        map_value = float(torch.nanmean(ap[valid]) if valid.any() else torch.nan_to_num(ap).mean())
        macro_precision = float(precision[macro_slice].mean())
        macro_recall = float(recall[macro_slice].mean())
        macro_f1 = float(f1[macro_slice].mean())
        subset_accuracy = float((predicted == target_bool).all(1).float().mean())

        self.per_class = {
            int(index): {
                "name": self.names.get(int(index), str(index)),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "ap": float(ap[index]) if torch.isfinite(ap[index]) else 0.0,
                "support": int(support[index]),
            }
            for index in range(targets.shape[1])
        }
        self.results_dict = {
            "metrics/precision": macro_precision,
            "metrics/recall": macro_recall,
            "metrics/f1_macro": macro_f1,
            "metrics/f1_micro": float(micro_f1),
            "metrics/mAP": map_value,
            "metrics/subset_accuracy": subset_accuracy,
        }
        # Keep the training selection scalar bounded and meaningful when a
        # dataset contains classes with very different frequencies.
        self.results_dict["fitness"] = 0.5 * map_value + 0.5 * float(micro_f1)
        return self.results_dict


__all__ = ("MultiLabelClassificationMetrics",)
