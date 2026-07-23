# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Small YAML-native building blocks for multi-branch vision models."""

from __future__ import annotations

import torch
import torch.nn as nn


class ModalSplit(nn.Module):
    """Split a channel-stacked multi-modal tensor into an arbitrary number of modality tensors."""

    def __init__(self, sections: list[int] | tuple[int, ...]):
        """Initialize a split with positive channel counts whose sum must match the runtime input channels."""
        super().__init__()
        if (
            not isinstance(sections, (list, tuple))
            or len(sections) < 2
            or any(not isinstance(c, int) or c < 1 for c in sections)
        ):
            raise ValueError("ModalSplit sections must be a list of at least two positive integers.")
        self.sections = tuple(sections)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return one tensor per modality while checking that the YAML layout matches the input tensor."""
        if x.shape[1] != sum(self.sections):
            raise ValueError(f"ModalSplit expected {sum(self.sections)} channels, got {x.shape[1]}.")
        return torch.split(x, self.sections, dim=1)
