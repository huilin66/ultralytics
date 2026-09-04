"""Deterministic encoding of label combinations used by the native transforms."""

from dataclasses import dataclass
import json
from typing import Iterable, Tuple

import torch


def normalize_combination(labels: Iterable[int] | str) -> Tuple[int, ...]:
    """Return a sorted, de-duplicated tuple of integer class IDs."""
    if isinstance(labels, str):
        labels = labels.split(",")
    try:
        normalized = tuple(sorted({int(label) for label in labels}))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid multi-label class list: {labels!r}") from exc
    if not normalized:
        raise ValueError("A multi-label object must contain at least one class ID")
    return normalized


@dataclass(frozen=True)
class CombinationCodec:
    """Map observed class combinations to stable scalar transport IDs.

    Transport IDs are only used while running the existing Ultralytics
    instance-oriented augmentation pipeline.  They must never be used as
    indices into the model's ``nc`` class outputs.
    """

    combinations: Tuple[Tuple[int, ...], ...]
    nc: int

    def __post_init__(self):
        combinations = tuple(normalize_combination(combo) for combo in self.combinations)
        if len(set(combinations)) != len(combinations):
            raise ValueError("Combination codec contains duplicate combinations")
        if tuple(sorted(combinations)) != combinations:
            raise ValueError("Combination codec combinations must be sorted deterministically")
        if self.nc <= 0:
            raise ValueError(f"nc must be positive, got {self.nc}")
        invalid = [class_id for combo in combinations for class_id in combo if class_id < 0 or class_id >= self.nc]
        if invalid:
            raise ValueError(f"Combination codec contains class IDs outside [0, {self.nc}): {invalid}")
        object.__setattr__(self, "combinations", combinations)

    @classmethod
    def from_combinations(cls, combinations: Iterable[Iterable[int]], nc: int) -> "CombinationCodec":
        """Build a codec from combinations in deterministic lexicographic order."""
        unique = {normalize_combination(combo) for combo in combinations}
        return cls(tuple(sorted(unique)), int(nc))

    @property
    def combo_to_transport(self) -> dict[Tuple[int, ...], int]:
        """Return the combination-to-transport lookup table."""
        return {combo: index for index, combo in enumerate(self.combinations)}

    def encode(self, labels: Iterable[int] | str) -> int:
        """Encode one class combination as a transport ID."""
        combo = normalize_combination(labels)
        try:
            return self.combo_to_transport[combo]
        except KeyError as exc:
            raise KeyError(f"Combination {combo} was not present when this codec was built") from exc

    def decode(self, transport_id: int) -> Tuple[int, ...]:
        """Decode one transport ID into its class IDs."""
        try:
            transport_id = int(transport_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid transport ID: {transport_id!r}") from exc
        if transport_id < 0 or transport_id >= len(self.combinations):
            raise ValueError(f"Transport ID {transport_id} is outside [0, {len(self.combinations)})")
        return self.combinations[transport_id]

    def to_nhot(self, transport_ids, device=None, dtype=torch.float32) -> torch.Tensor:
        """Decode transport IDs into an ``[N, nc]`` n-hot tensor."""
        ids = torch.as_tensor(transport_ids, device=device)
        ids = ids.reshape(-1)
        if ids.numel() and not torch.equal(ids, ids.round()):
            raise ValueError(f"Transport IDs must be integral, got {ids.tolist()}")
        ids = ids.long()
        if ids.numel() and (ids.min() < 0 or ids.max() >= len(self.combinations)):
            raise ValueError(f"Transport IDs must be in [0, {len(self.combinations)})")

        nhot = torch.zeros((ids.numel(), self.nc), device=ids.device, dtype=dtype)
        for transport_id, combo in enumerate(self.combinations):
            rows = ids == transport_id
            if rows.any():
                row_indices = torch.where(rows)[0]
                class_indices = torch.as_tensor(combo, device=ids.device)
                nhot[row_indices[:, None], class_indices[None, :]] = 1
        return nhot

    def decode_ids(self, transport_ids) -> list[Tuple[int, ...]]:
        """Decode a sequence of transport IDs for diagnostics."""
        ids = torch.as_tensor(transport_ids).reshape(-1).tolist()
        return [self.decode(int(transport_id)) for transport_id in ids]

    def as_json(self) -> str:
        """Serialize the codec metadata for cache/debug output."""
        return json.dumps(
            {"nc": self.nc, "combinations": [list(combo) for combo in self.combinations]},
            sort_keys=True,
        )
