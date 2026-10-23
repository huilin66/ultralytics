# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""File-name helpers shared by the adapters and the pipeline."""

from __future__ import annotations

import re
import unicodedata


def slugify(text: str, fallback: str = "sample") -> str:
    """Reduce arbitrary text to an ASCII, path-safe token.

    Ultraalytics decodes images with ``cv2.imread``, which cannot open non-ASCII
    paths on Windows, and zip extraction can leave mojibake in scene names, so
    every emitted id is normalised here.
    """
    decomposed = unicodedata.normalize("NFKD", str(text))
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_text).strip("_")
    return ascii_text or fallback
