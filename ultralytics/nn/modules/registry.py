# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Registry for optional YAML modules supplied by extensions."""

from __future__ import annotations

from typing import Type, TypeVar

from torch import nn

ModuleType = TypeVar("ModuleType", bound=Type[nn.Module])
_MODULES: dict[str, type[nn.Module]] = {}


def register_module(module: ModuleType) -> ModuleType:
    """Register an optional module class for YAML model parsing."""
    if not isinstance(module, type) or not issubclass(module, nn.Module):
        raise TypeError(f"Expected an nn.Module class, but received {module!r}.")
    existing = _MODULES.setdefault(module.__name__, module)
    if existing is not module:
        raise ValueError(f"YAML module name '{module.__name__}' is already registered by {existing!r}.")
    return module


def get_module(name: str) -> type[nn.Module]:
    """Return a registered optional module class by YAML name."""
    try:
        return _MODULES[name]
    except KeyError as e:
        raise KeyError(f"Unknown YAML module '{name}'.") from e
