"""Prediction for image-level multi-label classification."""

from __future__ import annotations

import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.classify.predict import ClassificationPredictor
from ultralytics.utils import DEFAULT_CFG, ops

from .classification_loss import MultiLabelClassificationLoss


class MultiLabelClassificationPredictor(ClassificationPredictor):
    """Return independent class probabilities and thresholded label sets."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.threshold = float(self.args.conf if self.args.conf is not None else 0.5)

    def _exported_multilabel_output(self):
        """Return whether a backend returns sigmoid scores rather than logits."""
        backend_model = getattr(self.model, "model", self.model)
        if not getattr(backend_model, "multilabel", False):
            return False
        head = getattr(backend_model, "model", None)
        return bool(head is not None and getattr(head[-1], "export", False))

    def postprocess(self, preds, img, orig_imgs):
        """Convert logits to sigmoid scores and attach all selected labels."""
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        if isinstance(preds, (tuple, list)) and len(preds) >= 2:
            logits = MultiLabelClassificationLoss._logits(preds)
            scores = logits.float().sigmoid()
        elif self._exported_multilabel_output():
            scores = torch.as_tensor(preds).float().clamp_(0.0, 1.0)
        else:
            logits = MultiLabelClassificationLoss._logits(preds)
            scores = logits.float().sigmoid()
        names = self.model.names
        results = []
        for score, orig_img, img_path in zip(scores, orig_imgs, self.batch[0]):
            selected = score >= self.threshold
            result = Results(orig_img, path=img_path, names=names, probs=score)
            # These attributes are intentionally explicit instead of changing
            # Results/Probs globally, so the native single-label API remains
            # backwards compatible.
            result.multilabel_scores = score
            result.multilabel = selected
            result.multilabel_indices = torch.where(selected)[0]
            result.multilabel_names = [names[int(index)] for index in result.multilabel_indices]
            results.append(result)
        return results


__all__ = ("MultiLabelClassificationPredictor",)
