"""Detection model wrapper selecting the true multi-label criterion."""

from ultralytics.nn.tasks import DetectionModel

from .loss import MultiLabelDetectionLoss


class MultiLabelDetectionModel(DetectionModel):
    """Native YOLO detection architecture with a multi-label criterion."""

    def init_criterion(self):
        """Return the n-hot target loss for standard (non-end-to-end) YOLO heads."""
        if getattr(self, "end2end", False):
            raise NotImplementedError("MultiLabelDetectionModel currently supports standard YOLO heads, not end2end heads")
        return MultiLabelDetectionLoss(self)
