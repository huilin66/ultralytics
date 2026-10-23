# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Explicit raw-sensor -> ``uint8`` encoding.

``ultralytics.data.multimodal.Modalities.load`` rejects any modality that is not
``uint8``.  Every conversion therefore performs one documented affine mapping
here and records it in the dataset manifest, instead of letting the dataloader
rescale silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .schema import EncodeSpec


class EncodingError(RuntimeError):
    """Raised when a raw file cannot be mapped to the declared modality contract."""


@dataclass
class EncodeStats:
    """Per-image diagnostics collected while encoding."""

    raw_dtype: str = ""
    raw_shape: tuple = ()
    dropped_alpha: bool = False
    replicated_channels: bool = False
    invalid_fraction: float = 0.0
    saturated_fraction: float = 0.0
    raw_min: float = 0.0
    raw_max: float = 0.0


def read_raw(path: Path) -> np.ndarray:
    """Read an image without any depth/channel conversion applied by the decoder.

    ``cv2.imread`` cannot open non-ASCII paths on Windows (one InfraParis scene is
    named ``Bièvres``), so the file is read through Python and decoded from memory.
    """
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise EncodingError(f"unreadable image: {path} ({error})") from error
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise EncodingError(f"unreadable image: {path}")
    return image


def _drop_alpha(image: np.ndarray, stats: EncodeStats) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 4:
        stats.dropped_alpha = True
        return image[..., :3]
    return image


def _to_single_channel(image: np.ndarray, stats: EncodeStats) -> np.ndarray:
    """Reduce an image to one channel, proving that the extra channels are copies."""
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise EncodingError(f"expected a 2D or 3D array, got shape {image.shape}")
    if image.shape[2] == 1:
        return image[..., 0]
    if image.shape[2] == 3:
        first = image[..., 0]
        if not (np.array_equal(first, image[..., 1]) and np.array_equal(first, image[..., 2])):
            raise EncodingError(
                "three-channel source is not a replicated grayscale image; refusing to guess a color->gray mapping"
            )
        stats.replicated_channels = True
        return first
    raise EncodingError(f"cannot reduce {image.shape[2]} channels to one")


def encode_image(path: Path, spec: EncodeSpec) -> tuple[np.ndarray, EncodeStats]:
    """Load one raw file and return ``(H, W, C)`` ``uint8`` plus diagnostics."""
    raw = read_raw(path)
    stats = EncodeStats(raw_dtype=str(raw.dtype), raw_shape=tuple(raw.shape))
    stats.raw_min, stats.raw_max = float(raw.min()), float(raw.max())

    image = _drop_alpha(raw, stats)

    if spec.kind == "rgb":
        if image.ndim != 3 or image.shape[2] != 3:
            raise EncodingError(f"RGB modality expects three channels, got shape {image.shape}")
        if image.dtype != np.uint8:
            raise EncodingError(f"RGB modality expects uint8, got {image.dtype}")
        return np.ascontiguousarray(image), stats

    if spec.kind not in {"gray", "scaled"}:
        raise EncodingError(f"unknown encode kind '{spec.kind}'")

    gray = _to_single_channel(image, stats)

    if spec.kind == "gray" and gray.dtype == np.uint8:
        return np.ascontiguousarray(gray), stats

    values = gray.astype(np.float32)
    if spec.invalid:
        invalid_mask = np.isin(values, np.asarray(spec.invalid, dtype=np.float32))
    else:
        invalid_mask = None
    if spec.hi <= spec.lo:
        raise EncodingError(f"invalid encode range [{spec.lo}, {spec.hi}]")

    scaled = (values - spec.lo) / (spec.hi - spec.lo) * 255.0
    stats.saturated_fraction = float((scaled >= 255.0).mean())
    np.clip(scaled, 0.0, 255.0, out=scaled)
    out = np.rint(scaled).astype(np.uint8)
    if invalid_mask is not None:
        stats.invalid_fraction = float(invalid_mask.mean())
        out[invalid_mask] = 0
    return np.ascontiguousarray(out), stats


def write_png(path: Path, image: np.ndarray) -> None:
    """Write a PNG deterministically (no implicit dtype conversion)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise EncodingError(f"PNG encode failed for {path}")
    path.write_bytes(buffer.tobytes())


def estimate_range(
    paths: list[Path], decode=lambda p: read_raw(p), low: float = 1.0, high: float = 99.0
) -> tuple[float, float]:
    """Return a robust global (low, high) range sampled from ``paths``.

    The range is global on purpose: per-frame min/max normalisation would destroy
    cross-frame radiometric comparability.
    """
    if not paths:
        raise EncodingError("cannot estimate a range from zero files")
    values = []
    for path in paths:
        array = decode(path)
        values.append(np.percentile(array, [low, high]))
    stacked = np.asarray(values, dtype=np.float64)
    return float(np.median(stacked[:, 0])), float(np.median(stacked[:, 1]))
