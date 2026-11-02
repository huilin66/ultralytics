# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""LiDAR -> camera depth projection shared by sensor-fusion adapters.

The converted datasets must expose every modality at the reference-camera
resolution and in ``uint8``.  This module does the two steps explicitly:

1. project LiDAR points into the reference camera with the shipped intrinsics
   and ``lidar_to_rgb`` extrinsic, keeping the nearest point per pixel
   (Z-buffer);
2. map the resulting metric depth to ``uint16`` millimetres, leaving ``0`` for
   "no LiDAR return", so :mod:`mdet_convert.encode` can apply its documented
   affine map to ``uint8``.

No interpolation or hole filling is performed: a LiDAR scan only covers a small
fraction of the image, and inventing depth in the gaps would misrepresent the
sensor.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_FRAME_LIST = re.compile(r"\d+")


class ProjectionError(RuntimeError):
    """Raised when calibration, timestamps or point clouds cannot be used."""


@dataclass(frozen=True)
class LidarCalibration:
    """RGB intrinsics plus the ``lidar -> rgb`` rigid transform shipped by the dataset."""

    intrinsic: np.ndarray
    lidar_to_rgb: np.ndarray
    width: int
    height: int
    distortion: tuple[float, ...] = ()
    source: str = ""

    def fingerprint(self) -> str:
        """Stable hash of the calibration values, recorded in ``conversion.json``."""
        payload = {
            "intrinsic": self.intrinsic.tolist(),
            "lidar_to_rgb": self.lidar_to_rgb.tolist(),
            "width": self.width,
            "height": self.height,
            "distortion": list(self.distortion),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LidarMatch:
    """One RGB frame matched to the nearest LiDAR scan by timestamp."""

    lidar_sequence: str
    lidar_frame: int
    rgb_timestamp_s: float
    lidar_timestamp_s: float
    delta_s: float


def load_calibration(path: Path) -> LidarCalibration:
    """Read the R-LiViT-style ``calibration_params.json``."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        camera = payload["rgb_camera"]
        intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float64)
        transform = np.asarray(payload["lidar_to_rgb"]["extrinsic_matrix"], dtype=np.float64)
        width = int(camera["width"])
        height = int(camera["height"])
        distortion = tuple(float(v) for v in camera.get("distortion_coefficients", []))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ProjectionError(f"unusable calibration file {path}: {error}") from error
    if intrinsic.shape != (3, 3) or transform.shape != (4, 4):
        raise ProjectionError(
            f"calibration {path} has intrinsic {intrinsic.shape} and extrinsic {transform.shape}; expected (3,3)/(4,4)"
        )
    if width <= 0 or height <= 0:
        raise ProjectionError(f"calibration {path} has invalid image size {width}x{height}")
    if not (np.isfinite(intrinsic).all() and np.isfinite(transform).all()):
        raise ProjectionError(f"calibration {path} contains non-finite values")
    return LidarCalibration(
        intrinsic=intrinsic,
        lidar_to_rgb=transform,
        width=width,
        height=height,
        distortion=distortion,
        source=str(path),
    )


def load_sequence_mapping(path: Path) -> dict[str, dict[int, tuple[str, ...]]]:
    """Parse ``rgb_to_lidar_sequence_mapping.csv`` into ``rgb_seq -> frame -> lidar_seqs``.

    The ``rgb_imgs`` cell is a Python-style list such as ``[02,06,10]``; it is not
    valid JSON because of the leading zeros, so it is parsed as integers.
    """
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ProjectionError(f"unreadable sequence mapping {path}: {error}") from error
    mapping: dict[str, dict[int, tuple[str, ...]]] = {}
    for row in rows:
        rgb_sequence = (row.get("rgbt_seq_id") or "").strip()
        lidar_sequence = (row.get("lidar_seq_id") or "").strip()
        frames = _FRAME_LIST.findall(row.get("rgb_imgs") or "")
        if not rgb_sequence or not lidar_sequence or not frames:
            continue
        for frame in {int(value) for value in frames}:
            mapping.setdefault(rgb_sequence, {}).setdefault(frame, ())
            existing = mapping[rgb_sequence][frame]
            if lidar_sequence not in existing:
                mapping[rgb_sequence][frame] = (*existing, lidar_sequence)
    if not mapping:
        raise ProjectionError(f"sequence mapping {path} contains no usable rows")
    return mapping


def load_lidar_timestamps(root: Path) -> dict[str, np.ndarray]:
    """Read ``timestamps/<seq>.txt`` into ``seq -> (frame_id, seconds)`` arrays."""
    root = Path(root)
    if not root.is_dir():
        raise ProjectionError(f"LiDAR timestamp folder is missing: {root}")
    timestamps: dict[str, np.ndarray] = {}
    for path in sorted(root.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            raise ProjectionError(f"unreadable LiDAR timestamps {path}: {error}") from error
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("frame"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                parts = line.split()
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except (ValueError, IndexError) as error:
                raise ProjectionError(f"bad timestamp row in {path}: '{line}'") from error
        if rows:
            array = np.asarray(rows, dtype=np.float64)
            timestamps[path.stem] = array[np.argsort(array[:, 0])]
    if not timestamps:
        raise ProjectionError(f"no timestamp files under {root}")
    return timestamps


def match_lidar_frame(
    rgb_sequence: str,
    rgb_frame: int,
    rgb_timestamp_s: float,
    mapping: dict[str, dict[int, tuple[str, ...]]],
    timestamps: dict[str, np.ndarray],
    *,
    tolerance_s: float = 0.1,
) -> LidarMatch | None:
    """Return the nearest LiDAR scan for one annotated RGB frame, or ``None``."""
    candidates = mapping.get(rgb_sequence, {}).get(int(rgb_frame), ())
    best: LidarMatch | None = None
    for lidar_sequence in candidates:
        array = timestamps.get(lidar_sequence)
        if array is None or len(array) == 0:
            continue
        index = int(np.searchsorted(array[:, 1], rgb_timestamp_s))
        for candidate in {max(0, index - 1), min(len(array) - 1, index)}:
            lidar_timestamp = float(array[candidate, 1])
            delta = abs(lidar_timestamp - rgb_timestamp_s)
            if best is None or delta < best.delta_s:
                best = LidarMatch(
                    lidar_sequence=lidar_sequence,
                    lidar_frame=int(array[candidate, 0]),
                    rgb_timestamp_s=float(rgb_timestamp_s),
                    lidar_timestamp_s=lidar_timestamp,
                    delta_s=float(delta),
                )
    if best is None or best.delta_s > tolerance_s:
        return None
    return best


def read_lidar_points(path: Path) -> np.ndarray:
    """Read a KITTI-style ``.bin`` scan as ``(N, 4)`` float32 ``x, y, z, intensity``."""
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise ProjectionError(f"unreadable LiDAR scan {path}: {error}") from error
    if not payload:
        raise ProjectionError(f"empty LiDAR scan: {path}")
    if len(payload) % 16 == 0:
        columns = 4
    elif len(payload) % 12 == 0:
        columns = 3
    else:
        raise ProjectionError(f"LiDAR scan {path} has {len(payload)} bytes, not a whole number of float32 x/y/z[/i]")
    points = np.frombuffer(payload, dtype="<f4").reshape(-1, columns)
    if columns == 3:
        points = np.concatenate([points, np.zeros((len(points), 1), np.float32)], axis=1)
    return points


def project_lidar_to_depth(
    points: np.ndarray,
    calibration: LidarCalibration,
    *,
    min_depth_m: float = 0.5,
    max_depth_m: float = 100.0,
) -> tuple[np.ndarray, dict]:
    """Project one LiDAR scan into the reference camera, keeping the nearest hit.

    The release ships RGB/thermal images already undistorted, so the pinhole
    intrinsic ``K`` is applied without the distortion coefficients.  Returns a
    dense ``float32`` ``(H, W)`` depth map in metres with ``0`` for "no return",
    plus diagnostics recorded in the sample manifest.
    """
    if points.ndim != 2 or points.shape[1] < 3:
        raise ProjectionError(f"point cloud must be (N, >=3), got {points.shape}")
    if min_depth_m <= 0 or max_depth_m <= min_depth_m:
        raise ProjectionError(f"invalid depth window [{min_depth_m}, {max_depth_m}] m")

    xyz = np.asarray(points[:, :3], dtype=np.float64)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    stats = {
        "points_total": int(len(points)),
        "points_finite": int(len(xyz)),
        "points_in_front": 0,
        "points_in_image": 0,
        "coverage": 0.0,
        "depth_min_m": None,
        "depth_p50_m": None,
        "depth_p98_m": None,
        "depth_max_m": None,
    }
    depth = np.zeros((calibration.height, calibration.width), dtype=np.float32)
    if len(xyz) == 0:
        return depth, stats

    homogeneous = np.concatenate([xyz, np.ones((len(xyz), 1), np.float64)], axis=1)
    camera = homogeneous @ calibration.lidar_to_rgb.T
    z = camera[:, 2]
    front = (z >= min_depth_m) & (z <= max_depth_m)
    stats["points_in_front"] = int(front.sum())
    if not front.any():
        return depth, stats

    camera = camera[front]
    z = z[front]
    projected = camera[:, :3] @ calibration.intrinsic.T
    uv = projected[:, :2] / projected[:, 2:3]
    inside = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < calibration.width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < calibration.height)
    )
    stats["points_in_image"] = int(inside.sum())
    if not inside.any():
        return depth, stats

    u = np.rint(uv[inside, 0]).astype(np.int64)
    v = np.rint(uv[inside, 1]).astype(np.int64)
    np.clip(u, 0, calibration.width - 1, out=u)
    np.clip(v, 0, calibration.height - 1, out=v)
    z = z[inside].astype(np.float32)

    # Z-buffer: several returns can land on one pixel; keep the nearest one.
    buffer = np.full((calibration.height, calibration.width), np.inf, dtype=np.float32)
    np.minimum.at(buffer, (v, u), z)
    hit = np.isfinite(buffer)
    depth[hit] = buffer[hit]

    stats["coverage"] = float(hit.mean())
    if hit.any():
        values = depth[hit]
        stats["depth_min_m"] = float(values.min())
        stats["depth_p50_m"] = float(np.percentile(values, 50))
        stats["depth_p98_m"] = float(np.percentile(values, 98))
        stats["depth_max_m"] = float(values.max())
    return depth, stats


def depth_to_uint16_mm(depth_m: np.ndarray, *, max_mm: float = 50000.0) -> tuple[np.ndarray, dict]:
    """Convert metric depth to ``uint16`` millimetres; ``0`` stays "no return".

    Values above ``max_mm`` are saturated (and counted) rather than wrapped.  The
    ``uint16`` image is the high-precision source that the pipeline maps to
    ``uint8`` with ``EncodeSpec("scaled", lo=0, hi=max_mm, invalid=(0,))``.
    """
    if max_mm <= 0 or max_mm > 65535:
        raise ProjectionError(f"max_mm must be in (0, 65535], got {max_mm}")
    depth_m = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth_m) & (depth_m > 0)
    millimetres = np.zeros(depth_m.shape, dtype=np.float64)
    millimetres[valid] = np.rint(depth_m[valid].astype(np.float64) * 1000.0)
    saturated = valid & (millimetres > max_mm)
    millimetres[valid] = np.clip(millimetres[valid], 1.0, max_mm)
    output = millimetres.astype(np.uint16)
    stats = {
        "coverage": float(valid.mean()),
        "saturated_fraction_of_valid": float(saturated.sum() / max(1, valid.sum())),
        "saturated_fraction_of_image": float(saturated.mean()),
        "max_mm": float(max_mm),
    }
    return output, stats


def calibration_sha256(path: Path) -> str:
    """SHA-256 of the calibration file, for provenance without copying it."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise ProjectionError(f"unreadable calibration file {path}: {error}") from error
