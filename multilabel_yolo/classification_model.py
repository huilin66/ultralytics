"""YOLO classification model with image-level multi-label semantics."""

from ultralytics.nn.tasks import ClassificationModel

from .classification_loss import MultiLabelClassificationLoss


class MultiLabelClassificationModel(ClassificationModel):
    """Native YOLO classification architecture using independent class logits."""

    def init_criterion(self):
        """Use BCE-with-logits instead of mutually-exclusive cross entropy."""
        return MultiLabelClassificationLoss(nc=self.yaml.get("nc"))

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Build the native model and mark only its classification head."""
        super()._from_yaml(cfg, ch, nc, verbose)
        self.multilabel = True
        self.model[-1].multilabel = True


__all__ = ("MultiLabelClassificationModel",)
