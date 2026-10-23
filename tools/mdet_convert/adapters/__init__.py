# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Adapter registry keyed by CLI name."""

from .infraparis import InfraParisAdapter
from .mid3k import Mid3kAdapter
from .mm5 import MM5Adapter
from .r_livit import RLiViTAdapter
from .vdt2048 import Vdt2048Adapter

ADAPTERS = {
    "mid3k": Mid3kAdapter,
    "infraparis": InfraParisAdapter,
    "r_livit": RLiViTAdapter,
    "mm5": MM5Adapter,
    "vdt2048": Vdt2048Adapter,
}

__all__ = ["ADAPTERS"]
