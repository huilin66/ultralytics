# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Task models that honor the input-channel layout declared by a multi-modal YAML."""

from __future__ import annotations

from ultralytics.nn.tasks import DetectionModel, SegmentationModel, yaml_model_load

from .fusion import parse_fusion_spec
from .modules import ModalFold, ModalUnfold, register_multimodal_modules

register_multimodal_modules()


class MultiModalModelMixin:
    """Resolve YAML input channels before the standard model performs its stride-inference forward pass."""

    def __init__(self, cfg, ch=None, nc=None, verbose=True):
        """Use the YAML channel count by default and reject mismatches supplied by a data YAML."""
        config = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        configured_channels = config.get("channels")
        if not isinstance(configured_channels, int) or configured_channels < 2:
            raise ValueError("A multi-modal model YAML needs an integer 'channels' value of at least 2.")
        if ch is None:
            ch = configured_channels
        elif ch != configured_channels:
            raise ValueError(f"Model YAML declares {configured_channels} channels, but received {ch}.")
        self.fusion_spec = parse_fusion_spec(config)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        if self.fusion_spec and self.fusion_spec.share_weight:
            modules = tuple(self.model.modules())
            if not any(isinstance(module, ModalFold) for module in modules) or not any(
                isinstance(module, ModalUnfold) for module in modules
            ):
                raise ValueError("share_weight=true requires ModalFold and ModalUnfold around the shared YAML stage.")


class MultiModalDetectionModel(MultiModalModelMixin, DetectionModel):
    """DetectionModel with YAML-owned input-channel initialization."""


class MultiModalSegmentationModel(MultiModalModelMixin, SegmentationModel):
    """SegmentationModel with YAML-owned input-channel initialization."""
