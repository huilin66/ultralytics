# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Multi-modal YOLO facade that keeps the standard Ultralytics model lifecycle."""

from __future__ import annotations

from typing import Any

from ultralytics.models.yolo.model import YOLO

from .predict import MultiModalDetectionPredictor, MultiModalSegmentationPredictor
from .tasks import MultiModalDetectionModel, MultiModalSegmentationModel
from .train import MultiModalDetectionTrainer, MultiModalSegmentationTrainer
from .val import MultiModalDetectionValidator, MultiModalSegmentationValidator


class MultiModalYOLO(YOLO):
    """Use pixel-aligned multi-modal datasets with standard YOLO detection and segmentation heads."""

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Return the standard task map with only dataset-facing components replaced."""
        task_map = super().task_map
        task_map["detect"].update(
            {
                "model": MultiModalDetectionModel,
                "trainer": MultiModalDetectionTrainer,
                "validator": MultiModalDetectionValidator,
                "predictor": MultiModalDetectionPredictor,
            }
        )
        task_map["segment"].update(
            {
                "model": MultiModalSegmentationModel,
                "trainer": MultiModalSegmentationTrainer,
                "validator": MultiModalSegmentationValidator,
                "predictor": MultiModalSegmentationPredictor,
            }
        )
        return task_map
