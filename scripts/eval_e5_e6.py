"""Generate Test summaries for the E5 and E6 comparison baselines.

E5 uses one physical detection box with two object classes and ten binary
attributes encoded in the twelve output channels.  E6 evaluates the E3
detector and applies the trained multi-label classifier to crops of its
predicted boxes.  Both paths use the project's matched-box attribute metric
definitions at IoU=0.5.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from attribute_dataset_utils import label_path_for_image, load_dataset_spec, parse_mdet_label_file
from multilabel_yolo.classification_predictor import MultiLabelClassificationPredictor
from multilabel_yolo.dataset import MultiLabelYOLODataset
from multilabel_yolo.inference import prepare_prediction_for_multilabel_nms
from ultralytics import YOLO
from ultralytics.data import build_dataloader
from ultralytics.models.yolo.mdetect.predict import MDetectionPredictor
from ultralytics.models.yolo.mdetect.val import MDetectionValidator
from ultralytics.utils import colorstr, ops
from ultralytics.utils.metrics import MConfusionMatrix


METRIC_FIELDS = (
    "mAP50_test",
    "mAP50-95_test",
    "OA_test",
    "F1_macro_test",
    "F1_macro_global_test",
    "P_macro_test",
    "R_macro_test",
)


def _attribute_names(count: int) -> dict[str, list[str]]:
    """Return binary attribute metadata compatible with the mdetect metrics."""

    return {f"attribute_{index}": ["no", "yes"] for index in range(count)}


def _metric_row(values: dict[str, Any]) -> dict[str, float]:
    """Extract the seven public Test metrics from a result dictionary."""

    return {field: float(values[field]) for field in METRIC_FIELDS}


def _write_summary(path: Path, weight: str, mode: str, values: dict[str, Any]) -> None:
    """Write one standard test summary row."""

    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"weight": weight, "mode": mode, **_metric_row(values)}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


class E5ReportValidator(MDetectionValidator):
    """Evaluate E5 as two object classes plus ten binary attributes."""

    object_count = 2
    attribute_count = 10

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build the one-box n-hot dataset for the requested split."""

        stride = int(self.stride.max()) if isinstance(self.stride, torch.Tensor) else int(self.stride)
        return MultiLabelYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            rect=True,
            cache=self.args.cache or None,
            single_cls=False,
            stride=stride,
            pad=0.5,
            prefix=colorstr(f"{mode}: "),
            task="detect",
            classes=None,
            data=self.data,
            fraction=1.0,
        )

    def get_dataloader(self, dataset_path, batch_size):
        """Build a dataloader whose labels retain the physical n-hot vector."""

        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="test")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)

    def preprocess(self, batch):
        """Convert the twelve n-hot channels to object and attribute targets."""

        cls_nhot = batch.pop("cls_nhot")
        batch["cls"] = cls_nhot[:, : self.object_count].argmax(dim=1, keepdim=True).float()
        batch["mdet_attributes"] = cls_nhot[:, self.object_count :].float()
        return super().preprocess(batch)

    def init_metrics(self, model):
        """Configure the inherited mdetect metric container for E5's layout."""

        super().init_metrics(model)
        self.model = model
        self.nc = self.object_count
        self.na = self.attribute_count
        self.nal = 2
        self.attribute_channels = self.attribute_count
        self.multiclass_attributes = False
        self.names = {index: model.names[index] for index in range(self.object_count)}
        self.attribute_names = _attribute_names(self.attribute_count)
        self.metrics.names = self.names
        self.metrics.attribute_names = self.attribute_names
        self.metrics.nc = self.nc
        self.metrics.na = self.na
        self.metrics.nal = self.nal
        self.metrics.reset_attribute_metrics()
        self.confusion_matrix = MConfusionMatrix(
            nc=self.nc,
            na=self.na,
            nal=self.nal,
            attribute_channels=self.attribute_channels,
            attribute_names=self.attribute_names,
            conf=self.args.conf,
            risk_enlarge=self.args.risk_enlarge,
            eval_att_by_class=self.args.eval_att_by_class,
        )

    def postprocess(self, preds):
        """Run NMS on object channels while carrying the ten attribute scores."""

        preds = prepare_prediction_for_multilabel_nms(preds, self.model)
        if not isinstance(preds, torch.Tensor) or preds.shape[1] < 4 + self.object_count + self.attribute_count:
            raise RuntimeError(f"Unexpected E5 prediction shape: {getattr(preds, 'shape', None)}")
        # E5 stores [object_0, object_1, attribute_0, ..., attribute_9].
        reordered = torch.cat(
            (preds[:, :4], preds[:, 4 : 4 + self.object_count], preds[:, 4 + self.object_count : 4 + 12]),
            dim=1,
        )
        return ops.non_max_suppression_with_attributes(
            reordered,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=self.object_count,
            na=self.attribute_count,
            rotated=self.args.task == "obb",
        )


def evaluate_e5(args: argparse.Namespace) -> None:
    """Evaluate E5 and write its one-row summary."""

    model = YOLO(args.weights, task="detect")
    metrics = model.val(
        data=args.data,
        split="test",
        validator=E5ReportValidator,
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=args.name,
        plots=False,
    )
    raw_values = metrics.results_dict
    values = {
        "mAP50_test": raw_values["metrics/mAP50(B)"],
        "mAP50-95_test": raw_values["metrics/mAP50-95(B)"],
        "OA_test": raw_values["metrics/OA(A)"],
        "F1_macro_test": raw_values["metrics/f1_macro(A)"],
        "F1_macro_global_test": raw_values["metrics/f1_macro_global(A)"],
        "P_macro_test": raw_values["metrics/P_macro(A)"],
        "R_macro_test": raw_values["metrics/R_macro(A)"],
    }
    _write_summary(Path(args.summary), args.weights, "test", values)
    print(json.dumps({"summary": args.summary, **_metric_row(values)}, ensure_ascii=False, indent=2))


def _xywh_to_xyxy(box: Iterable[float], width: int, height: int) -> np.ndarray:
    """Convert normalized xywh coordinates to pixel xyxy coordinates."""

    x, y, w, h = [float(value) for value in box]
    return np.asarray(
        [(x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height],
        dtype=np.float32,
    )


def _box_iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    """Compute IoU between one xyxy box and an array of xyxy boxes."""

    if len(many) == 0:
        return np.zeros(0, dtype=np.float32)
    left = np.maximum(one[0], many[:, 0])
    top = np.maximum(one[1], many[:, 1])
    right = np.minimum(one[2], many[:, 2])
    bottom = np.minimum(one[3], many[:, 3])
    intersection = np.maximum(right - left, 0) * np.maximum(bottom - top, 0)
    area_one = max(float(one[2] - one[0]), 0.0) * max(float(one[3] - one[1]), 0.0)
    area_many = np.maximum(many[:, 2] - many[:, 0], 0) * np.maximum(many[:, 3] - many[:, 1], 0)
    return intersection / np.maximum(area_one + area_many - intersection, 1e-9)


def _crop_box(box: np.ndarray, width: int, height: int, pad: float) -> tuple[int, int, int, int]:
    """Create the padded pixel crop used by the E6 classifier."""

    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * pad
    pad_y = (y2 - y1) * pad
    return (
        max(0, int(np.floor(x1 - pad_x))),
        max(0, int(np.floor(y1 - pad_y))),
        min(width, int(np.ceil(x2 + pad_x))),
        min(height, int(np.ceil(y2 + pad_y))),
    )


def _matched_crops(results, spec, output_dir: Path, iou_threshold: float, pad: float):
    """Match detector boxes to source objects and save classifier crops."""

    output_dir.mkdir(parents=True, exist_ok=True)
    crops: list[Path] = []
    targets: list[np.ndarray] = []
    image_count = 0
    matched_count = 0
    source_images = {path.resolve(): path for path in spec.splits["test"]}

    for result in results:
        image_path = Path(result.path).resolve()
        source_path = source_images.get(image_path, image_path)
        if not source_path.exists():
            continue
        label_path = label_path_for_image(source_path, spec.image_root, spec.labels_root)
        gt_objects = parse_mdet_label_file(label_path, expected_attributes=len(spec.attribute_names))
        image = result.orig_img
        height, width = image.shape[:2]
        gt_boxes = np.asarray([_xywh_to_xyxy(obj.bbox, width, height) for obj in gt_objects], dtype=np.float32)
        gt_classes = np.asarray([obj.object_class for obj in gt_objects], dtype=np.int64)
        predictions = result.boxes.data.detach().cpu().numpy() if result.boxes is not None else np.zeros((0, 6))
        order = np.argsort(-predictions[:, 4]) if len(predictions) else np.zeros(0, dtype=np.int64)
        used = set()
        image_count += 1
        for prediction_index in order:
            prediction = predictions[prediction_index]
            candidates = np.where(gt_classes == int(prediction[5]))[0]
            candidates = np.asarray([index for index in candidates if index not in used], dtype=np.int64)
            if len(candidates) == 0:
                continue
            overlaps = _box_iou(prediction[:4], gt_boxes[candidates])
            best = int(candidates[int(overlaps.argmax())])
            if float(overlaps.max()) < iou_threshold:
                continue
            used.add(best)
            crop_left, crop_top, crop_right, crop_bottom = _crop_box(prediction[:4], width, height, pad)
            if crop_right <= crop_left or crop_bottom <= crop_top:
                continue
            crop = image[crop_top:crop_bottom, crop_left:crop_right]
            crop_path = output_dir / f"{matched_count:06d}.png"
            Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(crop_path)
            crops.append(crop_path)
            targets.append(np.asarray(gt_objects[best].attributes, dtype=np.float32) > 0.5)
            matched_count += 1

    print(f"[E6] images={image_count}, matched_crops={matched_count}")
    return crops, np.asarray(targets, dtype=bool).reshape(-1, 10)


def _attribute_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Compute the project's OA/F1/P/R metrics from binary attributes."""

    confusion = np.zeros((targets.shape[1], 2, 2), dtype=np.float64)
    for attribute in range(targets.shape[1]):
        for target, prediction in zip(targets[:, attribute], predictions[:, attribute]):
            confusion[attribute, int(target), int(prediction)] += 1

    def stats(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        true_positive = np.diag(matrix)
        false_positive = matrix.sum(0) - true_positive
        false_negative = matrix.sum(1) - true_positive
        precision = true_positive / (true_positive + false_positive + 1e-8)
        recall = true_positive / (true_positive + false_negative + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return precision, recall, f1

    if not len(targets):
        return {key: 0.0 for key in ("OA_test", "F1_macro_test", "F1_macro_global_test", "P_macro_test", "R_macro_test")}
    per_attribute = [stats(matrix) for matrix in confusion]
    pooled = stats(confusion.sum(0))
    pooled_precision, pooled_recall, pooled_f1 = pooled
    true_positive = np.diag(confusion.sum(0)).sum()
    false_positive = (confusion.sum(0).sum(0) - np.diag(confusion.sum(0))).sum()
    false_negative = (confusion.sum(0).sum(1) - np.diag(confusion.sum(0))).sum()
    micro_precision = true_positive / (true_positive + false_positive + 1e-8)
    micro_recall = true_positive / (true_positive + false_negative + 1e-8)
    oa = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8)
    return {
        "OA_test": float(oa),
        "F1_macro_test": float(np.mean([value[2].mean() for value in per_attribute])),
        "F1_macro_global_test": float(pooled_f1.mean()),
        "P_macro_test": float(np.mean([value[0].mean() for value in per_attribute])),
        "R_macro_test": float(np.mean([value[1].mean() for value in per_attribute])),
    }


def evaluate_e6(args: argparse.Namespace) -> None:
    """Evaluate the complete E6 detector-plus-classifier pipeline."""

    spec = load_dataset_spec(args.source_data)
    detector = YOLO(args.detector_weights, task="mdetect")
    detector_metrics = detector.val(
        data=args.source_data,
        split="test",
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=f"{args.name}_detector",
        plots=False,
    )
    detector_values = detector_metrics.results_dict
    results = detector.predict(
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
    crop_dir = Path(args.project) / f"{args.name}_pred_crops"
    crops, targets = _matched_crops(results, spec, crop_dir, args.iou_match, args.crop_pad)
    classifier = YOLO(args.classifier_weights, task="classify")
    classifier_results = classifier.predict(
        source=[str(path) for path in crops],
        predictor=MultiLabelClassificationPredictor,
        device=args.device,
        imgsz=args.classifier_imgsz,
        batch=args.classifier_batch,
        conf=args.classifier_threshold,
        verbose=False,
    ) if crops else []
    scores = np.asarray([result.multilabel_scores.detach().cpu().numpy() for result in classifier_results], dtype=np.float32)
    predictions = scores >= args.classifier_threshold
    values = {
        "mAP50_test": detector_values["metrics/mAP50(B)"],
        "mAP50-95_test": detector_values["metrics/mAP50-95(B)"],
        **_attribute_metrics(targets, predictions),
    }
    _write_summary(Path(args.summary), args.classifier_weights, "two_stage_test", values)
    print(json.dumps({"summary": args.summary, **_metric_row(values)}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Build the E5/E6 evaluation parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("e5", "e6"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--classifier-batch", type=int, default=16)
    parser.add_argument("--classifier-imgsz", type=int, default=224)
    parser.add_argument("--classifier-threshold", type=float, default=0.5)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--iou-match", type=float, default=0.5)
    parser.add_argument("--crop-pad", type=float, default=0.10)
    parser.add_argument("--project", default="runs/experiments/eval_e5_e6")
    parser.add_argument("--name", default="test")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--weights", default="runs/experiments/E5_multilabel/yolov10x/weights/best.pt")
    parser.add_argument("--data", default="/localnvme/data/billboard/mayolo_v3_multilabel/data.yaml")
    parser.add_argument("--detector-weights", default="runs/experiments/E3_versions/E3_versions_yolov10x_w4_0p5_seed_0_stage2/weights/best.pt")
    parser.add_argument("--classifier-weights", default="runs/experiments/E6_two_stage_yolov10x/detector_yolov10x_classifier_yolov10x_cls/weights/best.pt")
    parser.add_argument("--source-data", default="ultralytics/cfg/mayolo_r1/mayolo_v3.yaml")
    parser.add_argument("--max-det", type=int, default=300)
    return parser


def main() -> None:
    """Run one selected Test evaluation."""

    args = build_parser().parse_args()
    if args.experiment == "e5":
        evaluate_e5(args)
    else:
        evaluate_e6(args)


if __name__ == "__main__":
    main()
