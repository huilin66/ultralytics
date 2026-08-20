# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def load_leakage_only_files(list_path_value: str | None) -> frozenset[str] | None:
    """Read and normalize the leakage-only image list once per configured path."""
    if list_path_value is None:
        return None
    if not list_path_value.strip():
        raise ValueError("LEAKAGE_ONLY_LIST must point to a non-empty list file.")

    list_path = Path(list_path_value).expanduser()
    if not list_path.is_file():
        raise FileNotFoundError(f"Leakage-only list file does not exist: {list_path}")

    names = frozenset(
        Path(line).name
        for raw_line in list_path.read_text(encoding="utf-8-sig").splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    )
    if not names:
        raise ValueError(f"Leakage-only list file is empty: {list_path}")
    return names
