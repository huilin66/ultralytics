# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Adapter contract shared by all five candidate datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..schema import DatasetPlan


class Adapter(ABC):
    """Base class for a raw-dataset adapter.

    An adapter only *discovers* and *describes* raw data; all encoding, splitting,
    validation and writing is done once by :mod:`mdet_convert.pipeline`.
    """

    key: str = ""
    display_name: str = ""

    def __init__(self, raw_root: Path, options: dict | None = None):
        """Store the raw root and adapter options (``--set key=value``)."""
        self.raw_root = Path(raw_root)
        self.options = dict(options or {})

    @abstractmethod
    def plan(self) -> DatasetPlan:
        """Inspect raw files and return the full conversion plan."""

    def option(self, name: str, default=None):
        """Read one adapter option with a default."""
        return self.options.get(name, default)

    def require_dir(self, path: Path) -> Path:
        """Return ``path`` or raise with an actionable message."""
        if not path.is_dir():
            raise FileNotFoundError(f"{self.display_name}: expected directory '{path}' is missing")
        return path


def resolve_official_splits(samples, official: dict[str, str] | None) -> dict[str, str]:
    """Map sample ids to their official split, if the dataset ships one."""
    return {sample.sample_id: official[sample.sample_id] for sample in samples if official and sample.sample_id in official}
