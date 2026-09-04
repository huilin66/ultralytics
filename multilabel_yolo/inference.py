"""Prediction helpers for true multi-label YOLO heads."""

import torch

from ultralytics.utils import ops


def _detection_head(model):
    """Find the final YOLO detection head in a raw model or an AutoBackend wrapper."""
    candidates = [model, getattr(model, "module", None), getattr(model, "model", None)]
    for candidate in candidates:
        modules = getattr(candidate, "model", None)
        if modules is None:
            continue
        try:
            head = modules[-1]
        except (IndexError, KeyError, TypeError):
            continue
        if hasattr(head, "_inference") and hasattr(head, "end2end"):
            return head
    return None


def prepare_prediction_for_multilabel_nms(preds, model):
    """Return predictions in the regular ``[B, 4 + nc, anchors]`` NMS layout.

    A YOLOv10 head normally post-processes its one-to-one branch into
    ``[B, max_det, 6]`` and keeps only the best class for each candidate. That
    representation has already lost the other positive labels. During PyTorch
    inference the head also returns the raw one-to-one branch, so decode that
    branch and let the multi-label NMS retain every class above the threshold.
    Non-end-to-end predictions are returned unchanged.
    """
    payload = preds
    if isinstance(payload, (tuple, list)) and len(payload) > 1 and isinstance(payload[1], dict):
        payload = payload[1]
    if not isinstance(payload, dict) or not {"one2many", "one2one"}.issubset(payload):
        return preds

    head = _detection_head(model)
    branch = payload["one2one"]
    if head is None or not isinstance(branch, (tuple, list)) or not branch:
        return preds
    if not all(isinstance(feature, torch.Tensor) and feature.ndim == 4 for feature in branch):
        return preds

    # MDetect has an additional attribute payload and is not part of this
    # true object multi-label adapter. Only convert a plain nc-class head.
    nc = getattr(head, "nc", None)
    if nc is None:
        return preds

    decoded = head._inference(branch)  # E2E heads decode boxes as xyxy.
    if decoded.ndim != 3 or decoded.shape[1] != 4 + nc:
        return preds
    decoded = decoded.transpose(1, 2).contiguous()
    decoded[..., :4] = ops.xyxy2xywh(decoded[..., :4])
    return decoded.transpose(1, 2).contiguous()
