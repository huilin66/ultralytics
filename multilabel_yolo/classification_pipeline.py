"""Utilities for applying the image classifier to detector result crops."""

from __future__ import annotations

from ultralytics.engine.results import Results

from .classification_predictor import MultiLabelClassificationPredictor


def classify_detection_crops(detector_results, classifier, threshold=0.5, **kwargs):
    """Classify each detector box crop and attach its label set to the result.

    Args:
        detector_results (Results | list[Results]): Results from a detector.
        classifier (ultralytics.YOLO): A YOLO wrapper containing a trained
            image-level multi-label classifier.
        threshold (float): Per-class sigmoid threshold.
        **kwargs: Additional arguments forwarded to ``classifier.predict``.

    Returns:
        (Results | list[Results]): The original result object(s), augmented with
        ``detection_multilabel_*`` lists aligned to ``result.boxes``. The crop
        classification Results objects are also available in
        ``detection_multilabel_results``.

    Notes:
        Ultralytics detector ``orig_img`` arrays are BGR, which is the format
        expected by the custom predictor's normal OpenCV conversion path.
    """
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"Classification threshold must be in [0, 1], got {threshold}")
    single = isinstance(detector_results, Results)
    results = [detector_results] if single else list(detector_results)
    crops, references = [], []

    for result_index, result in enumerate(results):
        boxes = result.boxes
        result.detection_multilabel_results = []
        result.detection_multilabel_scores = []
        result.detection_multilabel = []
        result.detection_multilabel_indices = []
        result.detection_multilabel_names = []
        if boxes is None:
            continue
        # Keep every attached list aligned with the detector's box order.
        count = len(boxes)
        result.detection_multilabel_results = [None] * count
        result.detection_multilabel_scores = [None] * count
        result.detection_multilabel = [None] * count
        result.detection_multilabel_indices = [None] * count
        result.detection_multilabel_names = [None] * count
        height, width = result.orig_img.shape[:2]
        for box_index, box in enumerate(boxes.xyxy):
            x1, y1, x2, y2 = box.detach().cpu().round().long().tolist()
            x1, y1 = max(0, min(x1, width - 1)), max(0, min(y1, height - 1))
            x2, y2 = min(x2, width), min(y2, height)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(result.orig_img[y1:y2, x1:x2].copy())
            references.append((result_index, box_index))

    if crops:
        predict_kwargs = dict(kwargs)
        predict_kwargs.update(
            {
                "predictor": MultiLabelClassificationPredictor,
                "conf": float(threshold),
                "verbose": False,
            }
        )
        # A YOLO wrapper reuses an existing predictor. Replace a previously
        # created native single-label predictor so the required multi-label
        # result fields are guaranteed to be present.
        if getattr(classifier, "predictor", None) is not None and not isinstance(
            classifier.predictor, MultiLabelClassificationPredictor
        ):
            classifier.predictor = None
        crop_results = list(classifier.predict(source=crops, **predict_kwargs))
        if len(crop_results) != len(references):
            raise RuntimeError(
                f"Classifier returned {len(crop_results)} crop results for {len(references)} detector boxes"
            )
        for (result_index, box_index), crop_result in zip(references, crop_results):
            result = results[result_index]
            result.detection_multilabel_results[box_index] = crop_result
            result.detection_multilabel_scores[box_index] = crop_result.multilabel_scores.detach().cpu()
            result.detection_multilabel[box_index] = crop_result.multilabel.detach().cpu()
            result.detection_multilabel_indices[box_index] = crop_result.multilabel_indices.detach().cpu()
            result.detection_multilabel_names[box_index] = list(crop_result.multilabel_names)
    return results[0] if single else results


__all__ = ("classify_detection_crops",)
