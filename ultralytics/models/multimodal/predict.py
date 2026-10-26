# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Image inference for multi-modal YOLO models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.data.multimodal import Modalities
from ultralytics.data.utils import check_det_dataset
from ultralytics.models import yolo


class MultiModalPredictorMixin:
    """Fuse image-file modalities just before standard predictor preprocessing."""

    def setup_source(self, source, stride=None):
        """Set up the normal source loader and optional path mapper from ``data=multimodal.yaml``."""
        super().setup_source(source, stride)
        self.modalities = None
        if self.args.data:
            data = check_det_dataset(self.args.data)
            if "modalities" in data:
                self.modalities = Modalities(data)
                if self.modalities.channels != self.model.channels:
                    raise ValueError(
                        f"data.yaml declares {self.modalities.channels} channels, "
                        f"but the model expects {self.model.channels}."
                    )

    def preprocess(self, images: torch.Tensor | list[np.ndarray]) -> torch.Tensor:
        """Accept a pre-fused array, or load aligned files when the source contains primary-modality paths."""
        if isinstance(images, torch.Tensor):
            if images.shape[1] != self.model.channels:
                raise ValueError(f"Input tensor has {images.shape[1]} channels, expected {self.model.channels}.")
            return super().preprocess(images)

        if all(image.shape[-1] == self.model.channels for image in images):
            fused = images
        elif self.modalities is not None:
            fused = [self.modalities.load(Path(path)) for path in self.batch[0]]
        else:
            raise ValueError(
                "A multi-modal file source needs data='path/to/multimodal.yaml'. "
                "Alternatively pass pre-fused HWC arrays with the model's channel count."
            )

        images = np.stack(self.pre_transform(fused)).transpose((0, 3, 1, 2))
        images = torch.from_numpy(np.ascontiguousarray(images)).to(self.device)
        images = images.half() if self.model.fp16 else images.float()
        return images / 255


class MultiModalDetectionPredictor(MultiModalPredictorMixin, yolo.detect.DetectionPredictor):
    """Detection predictor that loads N aligned image modalities from a primary file path."""


class MultiModalSegmentationPredictor(MultiModalPredictorMixin, yolo.segment.SegmentationPredictor):
    """Segmentation predictor that loads N aligned image modalities from a primary file path."""
