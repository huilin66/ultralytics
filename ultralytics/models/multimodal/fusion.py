# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Validation for the declarative multi-modal fusion contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FUSION_MODES = frozenset({"IF", "EF", "NIF", "BF", "NF", "HF"})
FUSION_OPERATORS = frozenset({"concat", "add"})
STAGES_BEFORE_FUSION = {
    "IF": (),
    "EF": ("encoder",),
    "NIF": ("encoder",),
    "BF": ("encoder", "nape"),
    "NF": ("encoder", "nape", "neck"),
    "HF": ("encoder", "nape", "neck", "head"),
}


@dataclass(frozen=True)
class FusionSpec:
    """Validated multi-modal fusion metadata stored alongside a model YAML."""

    input_sections: tuple[int, ...]
    mode: str
    operator: str
    fusion_points: tuple[str, ...]
    share_weight: bool
    shared_stages: tuple[str, ...]


def parse_fusion_spec(config: dict[str, Any]) -> FusionSpec | None:
    """Parse and validate the optional ``multimodal`` section of a model YAML dictionary."""
    settings = config.get("multimodal")
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise TypeError("Model YAML multimodal settings must be a mapping.")

    sections = settings.get("input_sections")
    channels = config.get("channels")
    if (
        not isinstance(sections, list)
        or len(sections) < 2
        or any(not isinstance(section, int) or section < 1 for section in sections)
    ):
        raise ValueError("multimodal.input_sections must contain at least two positive integers.")
    if sum(sections) != channels:
        raise ValueError(f"multimodal.input_sections sums to {sum(sections)}, but model channels={channels}.")

    mode = settings.get("fusion")
    if mode not in FUSION_MODES:
        raise ValueError(f"multimodal.fusion must be one of {sorted(FUSION_MODES)}, got {mode!r}.")
    operator = settings.get("operator", "concat")
    if operator not in FUSION_OPERATORS:
        raise ValueError(f"multimodal.operator must be one of {sorted(FUSION_OPERATORS)}, got {operator!r}.")

    points = settings.get("fusion_points", [])
    if not isinstance(points, list) or any(not isinstance(point, str) or not point for point in points):
        raise ValueError("multimodal.fusion_points must be a list of non-empty strings.")
    if mode == "IF" and points:
        raise ValueError("IF has no feature fusion points.")
    if mode != "IF" and not points:
        raise ValueError(f"{mode} requires at least one multimodal.fusion_points entry.")

    share_weight = settings.get("share_weight", False)
    if not isinstance(share_weight, bool):
        raise TypeError("multimodal.share_weight must be a boolean.")
    allowed_stages = STAGES_BEFORE_FUSION[mode]
    stages = settings.get("shared_stages", list(allowed_stages) if share_weight else [])
    if not isinstance(stages, list) or any(not isinstance(stage, str) for stage in stages):
        raise TypeError("multimodal.shared_stages must be a list of strings.")
    if share_weight and mode == "IF":
        raise ValueError("IF has no pre-fusion branches, so share_weight must be false.")
    if not share_weight and stages:
        raise ValueError("multimodal.shared_stages requires share_weight=true.")
    unknown_stages = set(stages).difference(allowed_stages)
    if unknown_stages:
        raise ValueError(f"{mode} may share only {allowed_stages}, got {sorted(unknown_stages)}.")

    return FusionSpec(tuple(sections), mode, operator, tuple(points), share_weight, tuple(stages))
