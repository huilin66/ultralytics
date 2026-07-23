# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Validation entry points for pixel-aligned multi-modal detection and segmentation."""

from __future__ import annotations

from ultralytics.data.multimodal import build_multimodal_dataset
from ultralytics.models import yolo


class MultiModalValidatorMixin:
    """Build the same fused dataset for standalone validation as for training."""

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build a validation dataset whose modality files remain paired with the primary image path."""
        return build_multimodal_dataset(self.args, img_path, batch, self.data, mode, False, self.stride)


class MultiModalDetectionValidator(MultiModalValidatorMixin, yolo.detect.DetectionValidator):
    """Detection validator backed by ``MultiModalDataset``."""


class MultiModalSegmentationValidator(MultiModalValidatorMixin, yolo.segment.SegmentationValidator):
    """Instance-segmentation validator backed by ``MultiModalDataset``."""
