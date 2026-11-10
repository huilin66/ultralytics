# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Image inference for multi-modal YOLO models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.data.loaders import LoadImagesAndVideos, SourceTypes
from ultralytics.data.multimodal import Modalities
from ultralytics.data.utils import check_det_dataset
from ultralytics.models import yolo


class MultiModalPredictorMixin:
    """Fuse image-file modalities just before standard predictor preprocessing."""

    def setup_source(self, source, stride=None):
        """Set up the normal source loader and optional path mapper from ``data=multimodal.yaml``."""
        super().setup_source(self._as_file_loader(source), stride)
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

    def _as_file_loader(self, source):
        """Keep primary file paths when the caller passes a list of image paths.

        The shared source check treats a list as decoded in-memory images and ``LoadPilAndNumpy`` replaces each
        source name with ``image<i>.jpg``. Companion modalities are addressed relative to the primary image path, so
        a path list must go through the file loader instead.
        """
        if isinstance(source, (list, tuple)) and source and all(isinstance(item, (str, Path)) for item in source):
            loader = LoadImagesAndVideos(
                source,
                batch=self.args.batch,
                vid_stride=self.args.vid_stride,
                channels=getattr(self.model, "channels", 3),
            )
            loader.source_type = SourceTypes()  # load_inference_source reads this attribute for loader instances
            return loader
        return source

    def preprocess(self, images: torch.Tensor | list[np.ndarray]) -> torch.Tensor:
        """Accept a pre-fused array, or load aligned files when the source contains primary-modality paths."""
        if isinstance(images, torch.Tensor):
            if images.shape[1] != self.model.channels:
                raise ValueError(f"Input tensor has {images.shape[1]} channels, expected {self.model.channels}.")
            return super().preprocess(images)

        if all(image.shape[-1] == self.model.channels for image in images):
            fused = images
        elif self.modalities is not None:
            fused = [self.modalities.load(path) for path in self._primary_paths()]
        else:
            raise ValueError(
                "A multi-modal file source needs data='path/to/multimodal.yaml'. "
                "Alternatively pass pre-fused HWC arrays with the model's channel count."
            )

        images = np.stack(self.pre_transform(fused)).transpose((0, 3, 1, 2))
        images = torch.from_numpy(np.ascontiguousarray(images)).to(self.device)
        images = images.half() if self.model.fp16 else images.float()
        return images / 255

    def _primary_paths(self) -> list[Path]:
        """Return the primary image path of every sample in the current batch, or explain why it is unavailable."""
        paths = [Path(path) for path in self.batch[0]]
        if not all(path.is_file() for path in paths):
            raise ValueError(
                "Multi-modal inference pairs each sample by its primary image path, but this batch has no file "
                f"paths (for example '{paths[0]}'). Pass image file paths or a directory, or pass pre-fused "
                f"arrays with {self.model.channels} channels."
            )
        return paths


class MultiModalDetectionPredictor(MultiModalPredictorMixin, yolo.detect.DetectionPredictor):
    """Detection predictor that loads N aligned image modalities from a primary file path."""


class MultiModalSegmentationPredictor(MultiModalPredictorMixin, yolo.segment.SegmentationPredictor):
    """Segmentation predictor that loads N aligned image modalities from a primary file path."""
