# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YAML leaf modules owned by the multi-modal extension."""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.registry import register_module


class ModalSplit(nn.Module):
    """Split a channel-stacked tensor into the modality sections declared by its YAML model."""

    def __init__(self, sections: list[int] | tuple[int, ...]):
        """Initialize a split with at least two positive channel sections."""
        super().__init__()
        if (
            not isinstance(sections, (list, tuple))
            or len(sections) < 2
            or any(not isinstance(channels, int) or channels < 1 for channels in sections)
        ):
            raise ValueError("ModalSplit sections must be a list of at least two positive integers.")
        self.sections = tuple(sections)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return one tensor per modality after checking the declared total channel count."""
        if x.shape[1] != sum(self.sections):
            raise ValueError(f"ModalSplit expected {sum(self.sections)} channels, got {x.shape[1]}.")
        return torch.split(x, self.sections, dim=1)


class MultiModalFusion(nn.Module):
    """Fuse same-stride modality features with channel concatenation or elementwise addition."""

    multi_input = True

    def __init__(self, channels: list[int] | tuple[int, ...], operator: str = "concat"):
        """Initialize a fusion operation whose output channels are inferred by the YAML parser."""
        super().__init__()
        self.channels = tuple(channels)
        if not isinstance(operator, str):
            raise ValueError("MultiModalFusion operator must be 'concat' or 'add'.")
        self.operator = operator.lower()
        self._validate_channels(self.channels, self.operator)

    @classmethod
    def output_channels(cls, channels: list[int], args: list) -> int:
        """Return the static output channel count used by the standard YAML parser."""
        operator = args[0] if args else "concat"
        operator = operator.lower() if isinstance(operator, str) else operator
        cls._validate_channels(channels, operator)
        return sum(channels) if operator == "concat" else channels[0]

    @staticmethod
    def _validate_channels(channels: list[int] | tuple[int, ...], operator: str) -> None:
        """Validate the static channel contract for a requested fusion operator."""
        if not isinstance(operator, str) or operator not in {"concat", "add"}:
            raise ValueError("MultiModalFusion operator must be 'concat' or 'add'.")
        if len(channels) < 2 or any(not isinstance(channel, int) or channel < 1 for channel in channels):
            raise ValueError("MultiModalFusion requires at least two positive channel inputs.")
        if operator == "add" and len(set(channels)) != 1:
            raise ValueError("MultiModalFusion add requires equal input channel counts; use adapters before fusion.")

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Fuse features after verifying their declared channels and shared spatial geometry."""
        if not isinstance(features, (list, tuple)) or len(features) != len(self.channels):
            raise ValueError(f"MultiModalFusion expected {len(self.channels)} features, got {len(features)}.")
        reference_shape = features[0].shape
        for feature, channels in zip(features, self.channels):
            if feature.ndim != 4 or feature.shape[1] != channels:
                raise ValueError(
                    f"MultiModalFusion expected a 4D feature with {channels} channels, got {tuple(feature.shape)}."
                )
            if feature.shape[0] != reference_shape[0] or feature.shape[2:] != reference_shape[2:]:
                raise ValueError("MultiModalFusion requires matching batch and spatial dimensions.")
        return torch.cat(features, dim=1) if self.operator == "concat" else torch.stack(features).sum(0)


class ModalFold(nn.Module):
    """Fold equally-shaped modality features into the batch dimension for a shared stage."""

    multi_input = True

    def __init__(self, channels: list[int] | tuple[int, ...]):
        """Initialize a fold after validating the adapter-aligned input channels."""
        super().__init__()
        self.channels = tuple(channels)
        self._validate_channels(self.channels)

    @classmethod
    def output_channels(cls, channels: list[int], args: list) -> int:
        """Return the common feature-channel count for the shared stage."""
        cls._validate_channels(channels)
        return channels[0]

    @staticmethod
    def _validate_channels(channels: list[int] | tuple[int, ...]) -> None:
        """Require at least two adapter-aligned feature tensors."""
        if len(channels) < 2 or any(not isinstance(channel, int) or channel < 1 for channel in channels):
            raise ValueError("ModalFold requires at least two positive channel inputs.")
        if len(set(channels)) != 1:
            raise ValueError("ModalFold requires equal input channel counts; use adapters before folding.")

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Stack modality batches after checking shape compatibility."""
        if not isinstance(features, (list, tuple)) or len(features) != len(self.channels):
            raise ValueError(f"ModalFold expected {len(self.channels)} features, got {len(features)}.")
        reference_shape = features[0].shape
        for feature, channels in zip(features, self.channels):
            if feature.ndim != 4 or feature.shape[1] != channels:
                raise ValueError(
                    f"ModalFold expected a 4D feature with {channels} channels, got {tuple(feature.shape)}."
                )
            if feature.shape != reference_shape:
                raise ValueError("ModalFold requires matching batch, channel and spatial dimensions.")
        return torch.cat(features, dim=0)


class ModalUnfold(nn.Module):
    """Recover modality tensors from a batch-folded shared stage."""

    def __init__(self, modalities: int):
        """Initialize an unfold operation for a fixed modality count."""
        super().__init__()
        if not isinstance(modalities, int) or modalities < 2:
            raise ValueError("ModalUnfold modalities must be an integer of at least 2.")
        self.modalities = modalities

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Split a folded batch into one feature tensor per original modality."""
        if x.ndim != 4 or x.shape[0] % self.modalities:
            raise ValueError(f"ModalUnfold expected a 4D batch divisible by {self.modalities}, got {tuple(x.shape)}.")
        return x.chunk(self.modalities, dim=0)


def register_multimodal_modules() -> None:
    """Register this extension's YAML modules without changing the default module namespace."""
    register_module(ModalSplit)
    register_module(MultiModalFusion)
    register_module(ModalFold)
    register_module(ModalUnfold)
