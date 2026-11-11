# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Resolve a multi-modal model config against the shared ``cfg/mmodels/base.yaml`` contract."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from ultralytics.nn.tasks import yaml_model_load

BASE_YAML = Path(__file__).resolve().parents[2] / "cfg" / "mmodels" / "base.yaml"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` with every key of ``override`` applied recursively."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_multimodal_config(cfg: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Load a model config and fill the shared contract from ``base.yaml``.

    Args:
        cfg (dict | str | Path): A model config dict or a YAML path.

    Returns:
        (dict): The merged config. Merging an already-merged config is idempotent.
    """
    config = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
    base = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8")) or {}
    return deep_merge(base, config)
