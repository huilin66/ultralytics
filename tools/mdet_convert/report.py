# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Conversion bookkeeping: counters, manifests and the quality report."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def write_json(path: Path, payload) -> None:
    """Write indented UTF-8 JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    """Write a CSV with a stable header even when there are no rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class SplitStats:
    """Per-split accumulators."""

    samples: int = 0
    boxes: int = 0
    empty_labels: int = 0
    excluded: int = 0
    per_class: Counter = field(default_factory=Counter)
    images_with_class: Counter = field(default_factory=Counter)
    box_widths: list[float] = field(default_factory=list)
    box_heights: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        def quantiles(values: list[float]) -> dict:
            if not values:
                return {}
            ordered = sorted(values)
            pick = lambda q: ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]
            return {"min": ordered[0], "p25": pick(0.25), "median": pick(0.5), "p75": pick(0.75), "max": ordered[-1]}

        return {
            "samples": self.samples,
            "boxes": self.boxes,
            "empty_labels": self.empty_labels,
            "excluded": self.excluded,
            "per_class": dict(self.per_class),
            "images_with_class": dict(self.images_with_class),
            "box_width_px": quantiles(self.box_widths),
            "box_height_px": quantiles(self.box_heights),
        }


@dataclass
class ConversionReport:
    """Aggregated conversion state for one dataset."""

    dataset: str
    counts: Counter = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)
    splits: dict[str, SplitStats] = field(default_factory=lambda: defaultdict(SplitStats))
    modality_notes: dict = field(default_factory=dict)
    exclusions: list[dict] = field(default_factory=list)

    def warn(self, message: str) -> None:
        """Record a warning once, preserving order."""
        if message not in self.warnings:
            self.warnings.append(message)

    def exclude(self, sample_id: str, reason: str) -> None:
        """Record why a sample was not converted."""
        self.exclusions.append({"sample_id": sample_id, "reason": reason})

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "counts": dict(self.counts),
            "warnings": self.warnings,
            "splits": {name: stats.as_dict() for name, stats in self.splits.items()},
            "modalities": self.modality_notes,
            "exclusions": len(self.exclusions),
        }
