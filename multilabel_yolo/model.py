"""Detection model wrapper selecting the true multi-label criterion."""

from ultralytics.nn.tasks import DetectionModel

from .loss import MultiLabelDetectionLoss


class MultiLabelDetectionModel(DetectionModel):
    """Native YOLO detection architecture with a multi-label criterion."""

    def init_criterion(self):
        """Return the n-hot target loss for standard and end-to-end YOLO heads."""
        return MultiLabelDetectionLoss(self)
