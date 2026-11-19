"""Compute end-to-end positive-attribute metrics for the E5/E6 baselines.

E5 is a one-box multi-label detector.  E6 is a detector followed by a
multi-label crop classifier.  The original E5/E6 summaries are conditional on
matched detections; this script keeps those values at the legacy confidence
floor and adds fixed-operating-point end-to-end counts at ``--e2e-conf``.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from attribute_dataset_utils import label_path_for_image, load_dataset_spec, parse_mdet_label_file
from eval_e5_e6 import (
    E5ReportValidator,
    _attribute_metrics,
    _crop_box,
    _metric_row,
    _write_summary,
)
from eval_end_to_end_attributes import _greedy_class_correct_matches
from multilabel_yolo.classification_predictor import MultiLabelClassificationPredictor
from ultralytics import YOLO
from ultralytics.utils import yaml_load, yaml_save
from ultralytics.utils import ops
from ultralytics.models.yolo.mdetect.predict import MDetectionPredictor
from mdet_eval_utils import resolved_data_yaml


E2E_FIELDS = (
    "detection_recall_e2e",
    "gt_instances",
    "pred_instances",
    "matched_instances",
    "Precision_e2e",
    "Recall_e2e",
    "F1_macro_e2e",
)


def _e2e_summary(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    gt_instances: int,
    pred_instances: int,
    matched_instances: int,
) -> dict[str, float | int]:
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) > 0,
    )
    return {
        "detection_recall_e2e": matched_instances / gt_instances if gt_instances else 0.0,
        "gt_instances": int(gt_instances),
        "pred_instances": int(pred_instances),
        "matched_instances": int(matched_instances),
        "Precision_e2e": float(precision.mean()) if len(precision) else 0.0,
        "Recall_e2e": float(recall.mean()) if len(recall) else 0.0,
        "F1_macro_e2e": float(f1.mean()) if len(f1) else 0.0,
    }


def _update_positive_counts(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    gt_present: np.ndarray,
    pred_present: np.ndarray,
) -> None:
    """Accumulate binary Present-level decisions for one matched object."""

    tp += gt_present & pred_present
    fp += ~gt_present & pred_present
    fn += gt_present & ~pred_present


class EndToEndE5Validator(E5ReportValidator):
    """E5 validator with an independent fixed-confidence end-to-end counter."""

    e2e_conf = 0.25
    e2e_iou_match = 0.5

    def init_metrics(self, model):
        super().init_metrics(model)
        # ``E5ReportValidator`` changes the detection class count from the
        # detector head (12 signboard/attribute output channels) to the two
        # signboard classes.  Recreate the base confusion matrix after that
        # change so that the inherited detector bookkeeping remains shape
        # consistent.
        self.box_confusion_matrix = np.zeros((self.nc + 1, self.nc + 1), dtype=np.int64)
        self._last_box_confusion = None
        self.e2e_tp = np.zeros(self.attribute_count, dtype=np.int64)
        self.e2e_fp = np.zeros(self.attribute_count, dtype=np.int64)
        self.e2e_fn = np.zeros(self.attribute_count, dtype=np.int64)
        self.e2e_gt_instances = 0
        self.e2e_pred_instances = 0
        self.e2e_matched_instances = 0

    def _update_e2e_image(self, si, pred, batch):
        pbatch = self._prepare_batch(si, batch)
        gt_classes = pbatch["cls"].clone().long().reshape(-1)
        gt_boxes = pbatch["bbox"].clone()
        gt_attributes = pbatch["mdet_attributes"].clone().detach().cpu().numpy() > 0.5
        predn = self._prepare_pred(pred, pbatch)
        confidence = float(self.e2e_conf)
        if len(predn):
            predn = predn[predn[:, 4] >= confidence]

        self.e2e_gt_instances += len(gt_classes)
        self.e2e_pred_instances += len(predn)
        pred_classes = predn[:, 5].long() if len(predn) else torch.empty(0, dtype=torch.long, device=gt_boxes.device)
        gt_idx, pred_idx = _greedy_class_correct_matches(
            gt_boxes,
            predn[:, :4],
            gt_classes,
            pred_classes,
            float(self.e2e_iou_match),
        )
        self.e2e_matched_instances += len(gt_idx)
        matched_gt = np.zeros(len(gt_classes), dtype=bool)
        matched_pred = np.zeros(len(predn), dtype=bool)
        pred_attributes = predn[:, 6 : 6 + self.attribute_count].detach().cpu().numpy() > 0.5
        for gi, pi in zip(gt_idx, pred_idx):
            matched_gt[gi] = True
            matched_pred[pi] = True
            _update_positive_counts(self.e2e_tp, self.e2e_fp, self.e2e_fn, gt_attributes[gi], pred_attributes[pi])
        if len(gt_attributes):
            self.e2e_fn += gt_attributes[~matched_gt].sum(axis=0).astype(np.int64)
        if len(pred_attributes):
            self.e2e_fp += pred_attributes[~matched_pred].sum(axis=0).astype(np.int64)

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        for si, pred in enumerate(preds):
            self._update_e2e_image(si, pred, batch)

    def finalize_metrics(self, *args, **kwargs):
        super().finalize_metrics(*args, **kwargs)
        self.metrics.end_to_end = _e2e_summary(
            self.e2e_tp,
            self.e2e_fp,
            self.e2e_fn,
            self.e2e_gt_instances,
            self.e2e_pred_instances,
            self.e2e_matched_instances,
        )


@contextmanager
def _resolved_multilabel_yaml(data_yaml: str, data_root: str | None):
    """Remap E5 split manifests without modifying the backup dataset."""

    if not data_root:
        yield data_yaml
        return
    source_yaml = Path(data_yaml)
    root = Path(data_root)
    config = yaml_load(source_yaml)
    with tempfile.TemporaryDirectory(prefix="e5_data_") as temporary:
        temporary_root = Path(temporary)
        config["path"] = str(root)
        for split in ("train", "val", "test"):
            split_value = config.get(split)
            if not isinstance(split_value, str) or not split_value.endswith(".txt"):
                continue
            source_manifest = Path(split_value)
            if not source_manifest.is_absolute():
                source_manifest = source_yaml.parent / split_value
            lines = source_manifest.read_text(encoding="utf-8").splitlines()
            remapped: list[str] = []
            for line in lines:
                normalized = line.strip().replace("\\", "/")
                marker = "mayolo_v3_multilabel/"
                relative = normalized.split(marker, 1)[1] if marker in normalized else f"images/{Path(normalized).name}"
                remapped.append(str(root / Path(relative)))
            manifest = temporary_root / f"{split}.txt"
            manifest.write_text("\n".join(remapped) + "\n", encoding="utf-8")
            config[split] = str(manifest)
        resolved = temporary_root / "data.yaml"
        yaml_save(resolved, config)
        yield str(resolved)


def _summary_row(label: str, weight: str, metrics, split: str = "test") -> dict[str, Any]:
    values = metrics.results_dict
    end_to_end = metrics.end_to_end
    suffix = split
    row: dict[str, Any] = {
        "model": label,
        "weight": weight,
        "mode": "test",
        f"mAP50_{suffix}": float(values["metrics/mAP50(B)"]),
        f"mAP50-95_{suffix}": float(values["metrics/mAP50-95(B)"]),
        f"F1_macro_conditional_{suffix}": float(values["metrics/f1_macro(A)"]),
        f"F1_micro_conditional_{suffix}": float(values["metrics/f1_macro_global(A)"]),
    }
    row.update({field: end_to_end[field] for field in E2E_FIELDS})
    return row


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_model_spec(spec: str) -> tuple[str, str]:
    label, separator, weight = spec.partition("=")
    if not separator or not label or not weight:
        raise ValueError(f"Invalid model specification {spec!r}; expected LABEL=WEIGHTS")
    return label, weight


def evaluate_e5(args: argparse.Namespace) -> None:
    specs = args.model or [f"E5={args.weights}"]
    rows: list[dict[str, Any]] = []
    EndToEndE5Validator.e2e_conf = float(args.e2e_conf)
    EndToEndE5Validator.e2e_iou_match = float(args.iou_match)
    output_dir = Path(args.project) / args.name
    with _resolved_multilabel_yaml(args.data, args.data_root) as data_yaml:
        for spec in specs:
            label, weight = _parse_model_spec(spec)
            print(f"[eval-e2e-e5] model={label}, weight={weight}")
            model = YOLO(weight, task="detect")
            metrics = model.val(
                data=data_yaml,
                split="test",
                validator=EndToEndE5Validator,
                device=args.device,
                imgsz=args.imgsz,
                batch=args.batch,
                workers=args.workers,
                conf=args.conf,
                iou=args.iou,
                project=args.project,
                name=f"{args.name}_{label}",
                plots=False,
                verbose=False,
            )
            rows.append(_summary_row(label, weight, metrics))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    _write_rows(output_dir / "summary.csv", rows)
    print(f"[summary] {output_dir / 'summary.csv'}")
    for row in rows:
        print(row)


def _xywh_to_xyxy(box: Iterable[float], width: int, height: int) -> np.ndarray:
    x, y, w, h = [float(value) for value in box]
    return np.asarray(
        [(x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height],
        dtype=np.float32,
    )


def _e6_end_to_end(results, spec, classifier, args, output_dir: Path) -> dict[str, float | int]:
    """Run the classifier on every retained detection, including unmatched ones."""

    output_dir.mkdir(parents=True, exist_ok=True)
    source_images = {path.resolve(): path for path in spec.splits["test"]}
    records: list[tuple[int, int]] = []
    crop_paths: list[Path] = []
    gt_by_image: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    gt_instances = 0
    pred_instances = 0
    matched_instances = 0
    tp = np.zeros(10, dtype=np.int64)
    fp = np.zeros(10, dtype=np.int64)
    fn = np.zeros(10, dtype=np.int64)

    for image_index, result in enumerate(results):
        image_path = Path(result.path).resolve()
        source_path = source_images.get(image_path, image_path)
        if not source_path.exists():
            gt_by_image.append((np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64), np.zeros((0, 10), dtype=bool), np.zeros(0, dtype=bool)))
            continue
        label_path = label_path_for_image(source_path, spec.image_root, spec.labels_root)
        gt_objects = parse_mdet_label_file(label_path, expected_attributes=len(spec.attribute_names))
        image = result.orig_img
        height, width = image.shape[:2]
        gt_boxes = np.asarray([_xywh_to_xyxy(obj.bbox, width, height) for obj in gt_objects], dtype=np.float32)
        gt_classes = np.asarray([obj.object_class for obj in gt_objects], dtype=np.int64)
        gt_attributes = np.asarray([np.asarray(obj.attributes, dtype=np.float32) > 0.5 for obj in gt_objects], dtype=bool).reshape(-1, 10)
        gt_by_image.append((gt_boxes, gt_classes, gt_attributes, np.ones(len(gt_objects), dtype=bool)))
        gt_instances += len(gt_objects)

        raw_predictions = result.boxes.data.detach().cpu().numpy() if result.boxes is not None else np.zeros((0, 6), dtype=np.float32)
        pred_instances += len(raw_predictions)
        pred_boxes = raw_predictions[:, :4] if len(raw_predictions) else np.zeros((0, 4), dtype=np.float32)
        pred_classes = raw_predictions[:, 5].astype(np.int64) if len(raw_predictions) else np.zeros(0, dtype=np.int64)
        gt_idx, pred_idx = _greedy_class_correct_matches(
            torch.from_numpy(gt_boxes),
            torch.from_numpy(pred_boxes),
            torch.from_numpy(gt_classes),
            torch.from_numpy(pred_classes),
            args.iou_match,
        )
        matched_instances += len(gt_idx)
        matched_by_pred = {int(pi): int(gi) for gi, pi in zip(gt_idx, pred_idx)}
        matched_gt = set(int(gi) for gi in gt_idx)
        for gt_index in range(len(gt_objects)):
            if gt_index not in matched_gt:
                fn += gt_attributes[gt_index].astype(np.int64)
        for prediction_index, prediction in enumerate(raw_predictions):
            crop_left, crop_top, crop_right, crop_bottom = _crop_box(prediction[:4], width, height, args.crop_pad)
            if crop_right <= crop_left or crop_bottom <= crop_top:
                continue
            crop_path = output_dir / f"{image_index:04d}_{prediction_index:04d}.png"
            import cv2
            from PIL import Image

            crop = image[crop_top:crop_bottom, crop_left:crop_right]
            Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(crop_path)
            crop_paths.append(crop_path)
            records.append((image_index, matched_by_pred.get(prediction_index, -1)))

    if crop_paths:
        classifier_results = classifier.predict(
            source=[str(path) for path in crop_paths],
            predictor=MultiLabelClassificationPredictor,
            device=args.device,
            imgsz=args.classifier_imgsz,
            batch=args.classifier_batch,
            conf=args.classifier_threshold,
            verbose=False,
        )
        predictions = np.asarray(
            [result.multilabel_scores.detach().cpu().numpy() >= args.classifier_threshold for result in classifier_results],
            dtype=bool,
        ).reshape(-1, 10)
    else:
        predictions = np.zeros((0, 10), dtype=bool)

    for (image_index, gt_index), prediction in zip(records, predictions):
        gt_attributes = gt_by_image[image_index][2]
        if gt_index >= 0:
            _update_positive_counts(tp, fp, fn, gt_attributes[gt_index], prediction)
        else:
            fp += prediction.astype(np.int64)
    return _e2e_summary(tp, fp, fn, gt_instances, pred_instances, matched_instances)


def evaluate_e6(args: argparse.Namespace) -> None:
    detector_specs = args.detector_model or [f"E6={args.detector_weights}"]
    classifier_specs = dict(_parse_model_spec(spec) for spec in (args.classifier_model or []))
    rows: list[dict[str, Any]] = []
    with resolved_data_yaml(args.source_data, args.source_data_root) as source_data:
        spec = load_dataset_spec(source_data)
        for spec_text in detector_specs:
            label, detector_weight = _parse_model_spec(spec_text)
            classifier_weight = classifier_specs.get(label, args.classifier_weights)
            print(f"[eval-e2e-e6] model={label}, detector={detector_weight}")
            print(f"[eval-e2e-e6] model={label}, classifier={classifier_weight}")
            classifier = YOLO(classifier_weight, task="classify")
            detector = YOLO(detector_weight, task="mdetect")
            detector_metrics = detector.val(
                data=source_data,
                split="test",
                device=args.device,
                imgsz=args.imgsz,
                batch=args.batch,
                workers=args.workers,
                conf=args.conf,
                iou=args.iou,
                project=args.project,
                name=f"{args.name}_{label}_detector",
                plots=False,
                verbose=False,
            )
            conditional_results = detector.predict(
                source=[str(path) for path in spec.splits["test"]],
                predictor=MDetectionPredictor,
                device=args.device,
                imgsz=args.imgsz,
                batch=args.batch,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                verbose=False,
            )
            from eval_e5_e6 import _matched_crops

            crop_dir = Path(args.project) / f"{args.name}_{label}_conditional_crops"
            crops, targets = _matched_crops(conditional_results, spec, crop_dir, args.iou_match, args.crop_pad)
            classifier_results = classifier.predict(
                source=[str(path) for path in crops],
                predictor=MultiLabelClassificationPredictor,
                device=args.device,
                imgsz=args.classifier_imgsz,
                batch=args.classifier_batch,
                conf=args.classifier_threshold,
                verbose=False,
            ) if crops else []
            scores = np.asarray(
                [result.multilabel_scores.detach().cpu().numpy() for result in classifier_results], dtype=np.float32
            )
            predictions = scores >= args.classifier_threshold
            conditional = _attribute_metrics(targets, predictions)
            e2e_results = detector.predict(
                source=[str(path) for path in spec.splits["test"]],
                predictor=MDetectionPredictor,
                device=args.device,
                imgsz=args.imgsz,
                batch=args.batch,
                conf=args.e2e_conf,
                iou=args.iou,
                max_det=args.max_det,
                verbose=False,
            )
            e2e = _e6_end_to_end(
                e2e_results,
                spec,
                classifier,
                args,
                Path(args.project) / f"{args.name}_{label}_e2e_crops",
            )
            values = {
                "model": label,
                "weight": detector_weight,
                "mode": "two_stage_test",
                "mAP50_test": float(detector_metrics.results_dict["metrics/mAP50(B)"]),
                "mAP50-95_test": float(detector_metrics.results_dict["metrics/mAP50-95(B)"]),
                **conditional,
                **e2e,
            }
            rows.append(values)
            del detector
            del classifier
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    output_dir = Path(args.project) / args.name
    _write_rows(output_dir / "summary.csv", rows)
    print(f"[summary] {output_dir / 'summary.csv'}")
    for row in rows:
        print(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("e5", "e6"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--classifier-batch", type=int, default=16)
    parser.add_argument("--classifier-imgsz", type=int, default=224)
    parser.add_argument("--classifier-threshold", type=float, default=0.5)
    parser.add_argument("--conf", type=float, default=0.001, help="Legacy conditional/mAP confidence floor")
    parser.add_argument("--e2e-conf", type=float, default=0.25, help="Fixed end-to-end detection threshold")
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--iou-match", type=float, default=0.5)
    parser.add_argument("--crop-pad", type=float, default=0.10)
    parser.add_argument("--project", default="runs/experiments/E2E_attribute_metrics")
    parser.add_argument("--name", default="e5_e6")
    parser.add_argument("--model", action="append", help="E5 LABEL=WEIGHTS; repeat for each seed")
    parser.add_argument("--data", default="ultralytics/cfg/mayolo_r1/mayolo_v3.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--weights", default="runs/experiments/E5_multilabel/yolov10x/weights/best.pt")
    parser.add_argument("--detector-model", action="append", help="E6 LABEL=DETECTOR_WEIGHTS; repeat for each seed")
    parser.add_argument(
        "--detector-weights",
        default="runs/experiments/E3_versions/E3_versions_yolov10x_w4_0p5_seed_0_stage2/weights/best.pt",
    )
    parser.add_argument(
        "--classifier-weights",
        default="runs/experiments/E6_two_stage_yolov10x/detector_yolov10x_classifier_yolov10x_cls/weights/best.pt",
        help="Fallback classifier weights for E6; use --classifier-model for per-seed weights",
    )
    parser.add_argument(
        "--classifier-model",
        action="append",
        help="E6 LABEL=CLASSIFIER_WEIGHTS; repeat to pair a classifier with each detector label",
    )
    parser.add_argument("--source-data", default="ultralytics/cfg/mayolo_r1/mayolo_v3.yaml")
    parser.add_argument("--source-data-root", default=None)
    parser.add_argument("--max-det", type=int, default=300)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.experiment == "e5":
        evaluate_e5(args)
    else:
        evaluate_e6(args)


if __name__ == "__main__":
    main()
