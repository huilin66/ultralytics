# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Training entry points for pixel-aligned multi-modal detection and segmentation."""

from __future__ import annotations

from copy import copy
from pathlib import Path

from ultralytics.data.multimodal import build_multimodal_dataset
from ultralytics.models import yolo
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

from .tasks import MultiModalDetectionModel, MultiModalSegmentationModel
from .val import MultiModalDetectionValidator, MultiModalSegmentationValidator


class MultiModalTrainerMixin:
    """Share multi-modal dataset creation and channel validation between detection and segmentation trainers."""

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build pixel-aligned multi-modal train or validation data."""
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_multimodal_dataset(self.args, img_path, batch, self.data, mode, mode == "val", stride)

class MultiModalDetectionTrainer(MultiModalTrainerMixin, yolo.detect.DetectionTrainer):
    """Detection trainer backed by ``MultiModalDataset``."""

    def get_model(self, cfg: str | None = None, weights: str | Path | None = None, verbose: bool = True):
        """Build a detection model with the channel count declared by the paired dataset."""
        model = self.set_model_names_for_load(
            MultiModalDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a validator that preserves multi-modal pairing during training validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return MultiModalDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


class MultiModalSegmentationTrainer(MultiModalTrainerMixin, yolo.segment.SegmentationTrainer):
    """Instance-segmentation trainer backed by ``MultiModalDataset``."""

    def get_model(self, cfg: dict | str | None = None, weights: str | Path | None = None, verbose: bool = True):
        """Build a segmentation model with the channel count declared by the paired dataset."""
        model = self.set_model_names_for_load(
            MultiModalSegmentationModel(
                cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1
            )
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a segmentation validator that preserves multi-modal pairing during training validation."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"
        return MultiModalSegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
