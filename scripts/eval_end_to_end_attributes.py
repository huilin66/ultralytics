"""Evaluate end-to-end two-level attribute metrics for multi-attribute detection.

The existing attribute metrics are conditional: they are computed only on
class-correct detections matched to ground truth at IoU >= 0.5.  This script
keeps those metrics untouched and additionally computes the same two-level
Precision, Recall, Macro-F1, and Micro-F1 formulation end to end.  Missed
ground-truth objects contribute FN for their ground-truth level, and unmatched
predictions contribute FP for their predicted level, including the Absent
level.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end mdet attribute metrics")
    parser.add_argument("--data", required=True, help="mdet dataset YAML")
    parser.add_argument(
        "--data-root", default=None, help="Optional runtime override for the dataset YAML path field"
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="LABEL=WEIGHTS[::native|one2many]; repeat for each model",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Detection confidence threshold for end-to-end evaluation (default: 0.001)",
    )
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--iou-match", type=float, default=0.5, help="GT/prediction matching IoU")
    parser.add_argument("--project", default="runs/experiments/E2E_attribute_metrics")
    parser.add_argument("--name", default="comparison")
    return parser


def _greedy_class_correct_matches(
    gt_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    pred_classes: torch.Tensor,
    iou_match: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-to-one GT/prediction indices using the validator's greedy rule."""

    if not len(gt_boxes) or not len(pred_boxes):
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    from ultralytics.utils.metrics import box_iou

    iou = box_iou(gt_boxes, pred_boxes).detach().cpu().numpy()
    valid = (iou >= iou_match) & (
        gt_classes[:, None].detach().cpu().numpy() == pred_classes[None, :].detach().cpu().numpy()
    )
    pairs = np.argwhere(valid)
    if not len(pairs):
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    # Greedy one-to-one assignment: consume the globally highest-IoU valid
    # pair first, then skip any pair sharing an already assigned GT or
    # prediction.  This is intentionally explicit rather than relying on
    # ``np.unique`` ordering, which can change the greedy decision.
    pairs = pairs[np.argsort(iou[pairs[:, 0], pairs[:, 1]])[::-1]]
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    selected: list[tuple[int, int]] = []
    for gi, pi in pairs:
        gi = int(gi)
        pi = int(pi)
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        selected.append((gi, pi))
    if not selected:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    selected_array = np.asarray(selected, dtype=np.int64)
    return selected_array[:, 0], selected_array[:, 1]


class EndToEndMDetectionValidator:
    """MDetectionValidator plus two-level end-to-end accounting."""

    use_rtdetr_validator = False

    def __new__(cls, *args: Any, **kwargs: Any):
        # Import lazily so argument parsing and --help do not initialize the
        # full model stack.
        if cls.use_rtdetr_validator:
            from ultralytics.models.rtdetr.val import RTDETRMDetectionValidator as BaseMDetectionValidator
        else:
            from ultralytics.models.yolo.mdetect.val import MDetectionValidator as BaseMDetectionValidator

        raw_args = kwargs.get("args", {})
        match_iou = float(raw_args.get("iou_match", 0.5)) if isinstance(raw_args, dict) else 0.5

        class _Validator(BaseMDetectionValidator):
            def init_metrics(self, model):
                super().init_metrics(model)
                self.e2e_tp = np.zeros((self.na, self.nal), dtype=np.int64)
                self.e2e_fp = np.zeros((self.na, self.nal), dtype=np.int64)
                self.e2e_fn = np.zeros((self.na, self.nal), dtype=np.int64)
                self.e2e_gt_instances = 0
                self.e2e_pred_instances = 0
                self.e2e_matched_instances = 0
                self.e2e_gt_level_support = np.zeros((self.na, self.nal), dtype=np.int64)

            def _decode_attribute_levels(self, pred_attributes: torch.Tensor) -> torch.Tensor:
                if self.multiclass_attributes:
                    return pred_attributes.reshape(-1, self.na, self.nal).argmax(dim=-1).long()
                risk_enlarge = torch.tensor(self.args.risk_enlarge, device=pred_attributes.device).view(1, -1)
                return torch.floor(pred_attributes * risk_enlarge * self.nal).long().clamp(0, self.nal - 1)

            def _update_e2e_image(self, si, pred, batch):
                pbatch = self._prepare_batch(si, batch)
                gt_classes = pbatch["cls"].clone()
                gt_boxes = pbatch["bbox"].clone()
                gt_attributes = pbatch["mdet_attributes"].clone().long()
                predn = self._prepare_pred(pred, pbatch)

                # YOLO-style postprocessing applies the confidence threshold
                # inside NMS, whereas RT-DETR's query decoder returns its
                # top-k queries directly.  Apply the same operating-point
                # threshold here so both validator families use the declared
                # end-to-end confidence threshold.
                confidence_threshold = float(self.args.conf or 0.0)
                if len(predn):
                    predn = predn[predn[:, 4] >= confidence_threshold]

                if self.args.single_cls:
                    predn[:, 5] = 0

                if self.args.filter_small != 0:
                    img_height, img_width = pbatch["ori_shape"]
                    min_w = img_width * self.args.filter_small
                    min_h = img_height * self.args.filter_small
                    gt_wh = gt_boxes[:, 2:4] - gt_boxes[:, 0:2]
                    pred_wh = predn[:, 2:4] - predn[:, 0:2]
                    keep_gt = (gt_wh[:, 0] >= min_w) & (gt_wh[:, 1] >= min_h)
                    keep_pred = (pred_wh[:, 0] >= min_w) & (pred_wh[:, 1] >= min_h)
                    gt_classes = gt_classes[keep_gt]
                    gt_boxes = gt_boxes[keep_gt]
                    gt_attributes = gt_attributes[keep_gt]
                    predn = predn[keep_pred]

                gt_attributes_np = gt_attributes.detach().cpu().numpy()
                self.e2e_gt_instances += len(gt_classes)
                self.e2e_pred_instances += len(predn)
                if len(gt_attributes_np):
                    for level in range(self.nal):
                        self.e2e_gt_level_support[:, level] += (gt_attributes_np == level).sum(axis=0)

                if len(predn):
                    pred_levels = self._decode_attribute_levels(
                        predn[:, 6 : 6 + self.attribute_channels]
                    ).detach().cpu().numpy()
                else:
                    pred_levels = np.empty((0, self.na), dtype=np.int64)
                gt_idx, pred_idx = _greedy_class_correct_matches(
                    gt_boxes,
                    predn[:, :4],
                    gt_classes,
                    predn[:, 5].long(),
                    match_iou,
                )
                self.e2e_matched_instances += len(gt_idx)

                matched_gt = np.zeros(len(gt_classes), dtype=bool)
                matched_pred = np.zeros(len(predn), dtype=bool)
                for gi, pi in zip(gt_idx, pred_idx):
                    matched_gt[gi] = True
                    matched_pred[pi] = True
                    gt_levels = gt_attributes_np[gi]
                    predicted_levels = pred_levels[pi]
                    for attribute in range(self.na):
                        gt_level = int(gt_levels[attribute])
                        predicted_level = int(predicted_levels[attribute])
                        if gt_level == predicted_level:
                            self.e2e_tp[attribute, gt_level] += 1
                        else:
                            self.e2e_fp[attribute, predicted_level] += 1
                            self.e2e_fn[attribute, gt_level] += 1

                if len(gt_attributes_np):
                    for gt_levels in gt_attributes_np[~matched_gt]:
                        for attribute, gt_level in enumerate(gt_levels):
                            self.e2e_fn[attribute, int(gt_level)] += 1
                if len(pred_levels):
                    for predicted_levels in pred_levels[~matched_pred]:
                        for attribute, predicted_level in enumerate(predicted_levels):
                            self.e2e_fp[attribute, int(predicted_level)] += 1

            def _e2e_summary(self) -> dict[str, Any]:
                precision = np.divide(
                    self.e2e_tp,
                    self.e2e_tp + self.e2e_fp,
                    out=np.zeros((self.na, self.nal), dtype=np.float64),
                    where=(self.e2e_tp + self.e2e_fp) > 0,
                )
                recall = np.divide(
                    self.e2e_tp,
                    self.e2e_tp + self.e2e_fn,
                    out=np.zeros((self.na, self.nal), dtype=np.float64),
                    where=(self.e2e_tp + self.e2e_fn) > 0,
                )
                f1 = np.divide(
                    2 * precision * recall,
                    precision + recall,
                    out=np.zeros((self.na, self.nal), dtype=np.float64),
                    where=(precision + recall) > 0,
                )
                names = list(self.attribute_names.keys())
                per_attribute = []
                for i in range(self.na):
                    attribute_precision = float(np.mean(precision[i]))
                    attribute_recall = float(np.mean(recall[i]))
                    attribute_f1 = float(np.mean(f1[i]))
                    per_attribute.append(
                        {
                            "attribute_index": i,
                            "attribute": names[i] if i < len(names) else str(i),
                            "gt_absent_support": int(self.e2e_gt_level_support[i, 0]),
                            "gt_present_support": int(self.e2e_gt_level_support[i, 1])
                            if self.nal > 1
                            else 0,
                            "TP": int(self.e2e_tp[i].sum()),
                            "FP": int(self.e2e_fp[i].sum()),
                            "FN": int(self.e2e_fn[i].sum()),
                            "Precision_macro_e2e": attribute_precision,
                            "Recall_macro_e2e": attribute_recall,
                            "F1_macro_e2e": attribute_f1,
                        }
                    )
                macro_precision = float(np.mean(precision))
                macro_recall = float(np.mean(recall))
                macro_f1 = float(np.mean(f1))
                micro_tp = float(self.e2e_tp.sum())
                micro_fp = float(self.e2e_fp.sum())
                micro_fn = float(self.e2e_fn.sum())
                micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
                micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
                micro_f1 = (
                    2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
                    if micro_precision + micro_recall
                    else 0.0
                )
                return {
                    "detection_recall_e2e": self.e2e_matched_instances / self.e2e_gt_instances
                    if self.e2e_gt_instances
                    else 0.0,
                    "gt_instances": self.e2e_gt_instances,
                    "pred_instances": self.e2e_pred_instances,
                    "matched_instances": self.e2e_matched_instances,
                    "F1_macro_e2e": macro_f1,
                    "Precision_macro_e2e": macro_precision,
                    "Recall_macro_e2e": macro_recall,
                    "Precision_micro_e2e": micro_precision,
                    "Recall_micro_e2e": micro_recall,
                    "F1_micro_e2e": micro_f1,
                    "per_attribute": per_attribute,
                }

            def update_metrics(self, preds, batch):
                # Keep the original validator metrics exactly as implemented;
                # the extra counter is independent of its statistics.
                super().update_metrics(preds, batch)
                for si, pred in enumerate(preds):
                    self._update_e2e_image(si, pred, batch)

            def finalize_metrics(self, *args, **kwargs):
                super().finalize_metrics(*args, **kwargs)
                self.metrics.end_to_end = self._e2e_summary()

        return _Validator(*args, **kwargs)


class EndToEndRTDETRValidator(EndToEndMDetectionValidator):
    """End-to-end accounting using RT-DETR's attribute-aware postprocessor."""

    use_rtdetr_validator = True


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_one(args: argparse.Namespace, data_yaml: str, label: str, weights: str, mode: str) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(weights, task="mdetect")
    if mode == "one2many":
        _set_one2many(model)
    eval_name = f"{args.name}_{label}_{mode}".replace(" ", "_")
    validator_cls = (
        EndToEndRTDETRValidator
        if type(getattr(model, "model", None)).__name__ == "RTDETRDetectionModel"
        else EndToEndMDetectionValidator
    )
    metrics = model.val(
        validator=validator_cls,
        data=data_yaml,
        split=args.split,
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=eval_name,
        plots=False,
        verbose=False,
    )
    end_to_end = metrics.end_to_end
    results = metrics.results_dict
    suffix = args.split
    summary = {
        "model": label,
        "mode": mode,
        "weight": weights,
        "split": args.split,
        "conf": args.conf,
        "nms_iou": args.iou,
        "matching_iou": args.iou_match,
        f"mAP50_{suffix}": float(results["metrics/mAP50(B)"]),
        f"mAP50-95_{suffix}": float(results["metrics/mAP50-95(B)"]),
        f"F1_macro_conditional_{suffix}": float(results["metrics/f1_macro(A)"]),
        f"F1_micro_conditional_{suffix}": float(results["metrics/f1_macro_global(A)"]),
        **{key: value for key, value in end_to_end.items() if key != "per_attribute"},
    }
    per_attribute = [
        {"model": label, "mode": mode, "weight": weights, "conf": args.conf, **row}
        for row in end_to_end["per_attribute"]
    ]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"summary": summary, "per_attribute": per_attribute}


def main() -> None:
    args = _parser().parse_args()
    from mdet_eval_utils import resolved_data_yaml

    output_dir = Path(args.project) / args.name
    results: list[dict[str, Any]] = []
    with resolved_data_yaml(args.data, args.data_root) as data_yaml:
        for spec in args.model:
            label, weights, mode = _parse_model_spec(spec)
            print(f"[eval-e2e] model={label}, mode={mode}, conf={args.conf}, weights={weights}")
            results.append(_evaluate_one(args, data_yaml, label, weights, mode))
    _write_csv(output_dir / "summary.csv", [result["summary"] for result in results])
    _write_csv(output_dir / "per_attribute.csv", [row for result in results for row in result["per_attribute"]])
    print(f"[summary] {output_dir / 'summary.csv'}")
    print(f"[details] {output_dir / 'per_attribute.csv'}")
    for result in results:
        print(result["summary"])


if __name__ == "__main__":
    main()
