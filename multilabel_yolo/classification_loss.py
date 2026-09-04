"""Loss for image-level multi-label classification."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class MultiLabelClassificationLoss:
    """Independent sigmoid/BCE loss for an image-level n-hot target.

    A softmax/cross-entropy classifier is unsuitable here because labels are
    not mutually exclusive.  The model head remains the native YOLO
    classification head; only its loss and decoding semantics change.
    """

    def __init__(self, nc=None):
        self.nc = int(nc) if nc is not None else None

    @staticmethod
    def _logits(preds):
        """Extract logits from training output or the native eval tuple."""
        if isinstance(preds, (tuple, list)):
            if len(preds) < 2:
                raise ValueError("Multi-label classification needs raw logits, but the model output has one item")
            # Classify.forward returns (softmax probabilities, raw logits) in
            # eval mode and raw logits directly in train mode.
            preds = preds[1]
        if not isinstance(preds, torch.Tensor) or preds.ndim != 2:
            raise ValueError(f"Classification logits must have shape [B, nc], got {type(preds).__name__}")
        return preds

    def __call__(self, preds, batch):
        """Return total BCE loss and a detached one-element loss vector."""
        logits = self._logits(preds)
        target = batch.get("cls_nhot")
        if target is None:
            raise KeyError("Multi-label classification batches must contain 'cls_nhot'")
        if target.ndim != 2 or target.shape != logits.shape:
            raise ValueError(f"cls_nhot must have shape {tuple(logits.shape)}, got {tuple(target.shape)}")
        if target.numel() and not torch.all((target == 0) | (target == 1)):
            raise ValueError("Image-level cls_nhot targets must be binary")
        if self.nc is not None and logits.shape[1] != self.nc:
            raise ValueError(f"Expected {self.nc} classifier outputs, got {logits.shape[1]}")

        target = target.to(device=logits.device, dtype=logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        return loss, loss.detach().reshape(1)


__all__ = ("MultiLabelClassificationLoss",)
