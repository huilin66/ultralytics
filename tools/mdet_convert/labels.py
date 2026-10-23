# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Box geometry, YOLO label I/O and instance-mask -> box derivation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .schema import Box


def clip_xyxy(xyxy: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    """Clamp a pixel box to the image rectangle."""
    x1, y1, x2, y2 = xyxy
    return (max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2))


def xyxy_to_yolo(
    xyxy: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    """Convert a pixel ``xyxy`` box to normalised ``cx cy w h``."""
    x1, y1, x2, y2 = clip_xyxy(xyxy, width, height)
    return (
        (x1 + x2) / 2.0 / width,
        (y1 + y2) / 2.0 / height,
        max(0.0, (x2 - x1)) / width,
        max(0.0, (y2 - y1)) / height,
    )


def yolo_to_xyxy(
    cx: float, cy: float, bw: float, bh: float, width: int, height: int
) -> tuple[float, float, float, float]:
    """Inverse of :func:`xyxy_to_yolo`, used by round-trip tests."""
    return ((cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height)


def box_is_valid(xyxy, width: int, height: int, min_size: float = 2.0, min_area: float = 4.0) -> bool:
    """Reject degenerate or fully out-of-frame boxes after clipping to the image."""
    x1, y1, x2, y2 = clip_xyxy(xyxy, width, height)
    w, h = x2 - x1, y2 - y1
    if w < min_size or h < min_size or w * h < min_area:
        return False
    if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
        return False
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return 0.0 <= cx < width and 0.0 <= cy < height


def iou(box_a, box_b) -> float:
    """Intersection-over-union of two ``xyxy`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedupe(boxes: list[Box], threshold: float = 0.95) -> tuple[list[Box], int]:
    """Drop near-identical boxes of the same class, keeping the first occurrence."""
    kept: list[Box] = []
    dropped = 0
    for box in boxes:
        duplicate = any(
            other.class_name == box.class_name and iou(other.xyxy, box.xyxy) >= threshold for other in kept
        )
        if duplicate:
            dropped += 1
        else:
            kept.append(box)
    return kept, dropped


def write_label(path: Path, rows: list[tuple[int, float, float, float, float]]) -> None:
    """Write a YOLO detection label file (one class + four normalised floats per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n" for cls, cx, cy, bw, bh in rows)
    path.write_text(body, encoding="utf-8")


def mask_to_boxes(
    instance_mask: np.ndarray,
    class_mask: np.ndarray | None,
    class_names: dict[int, str],
    *,
    min_area: int = 16,
    min_size: float = 2.0,
) -> tuple[list[Box], dict]:
    """Derive one bounding box per instance id from an instance mask.

    The class of an instance is the modal (most frequent) class id of its pixels
    when a class mask is available.  Instances whose class cannot be resolved are
    kept with ``class_name=""`` and ``ignored=True`` so the caller can report them
    instead of silently inventing a label.
    """
    diagnostics = {"instances": 0, "too_small": 0, "class_unknown": 0}
    boxes: list[Box] = []
    instance_mask = instance_mask.astype(np.int32)
    for instance_id in np.unique(instance_mask):
        if instance_id == 0:
            continue
        diagnostics["instances"] += 1
        region = (instance_mask == instance_id).astype(np.uint8)
        area = int(region.sum())
        if area < min_area:
            diagnostics["too_small"] += 1
            continue
        x, y, w, h = cv2.boundingRect(region)
        xyxy = (float(x), float(y), float(x + w), float(y + h))
        if w < min_size or h < min_size:
            diagnostics["too_small"] += 1
            continue
        class_name = ""
        if class_mask is not None:
            values, counts = np.unique(class_mask[region.astype(bool)], return_counts=True)
            class_id = int(values[int(np.argmax(counts))])
            class_name = class_names.get(class_id, "")
            if not class_name:
                diagnostics["class_unknown"] += 1
        boxes.append(Box(class_name=class_name, xyxy=xyxy, source="derived_instance_box",
                         ignored=not class_name))
    return boxes, diagnostics


def binary_mask_to_boxes(
    mask: np.ndarray, *, threshold: int = 127, min_area: int = 64
) -> tuple[list[Box], dict]:
    """Derive *candidate* boxes from a binary saliency mask.

    The result is explicitly ``candidate_only``: a salient region is not a class
    label, and touching objects merge into one component.
    """
    binary = (mask if mask.ndim == 2 else mask[..., 0]) > threshold
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    boxes: list[Box] = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if area < min_area:
            continue
        boxes.append(
            Box(
                class_name="",
                xyxy=(float(x), float(y), float(x + w), float(y + h)),
                source="candidate_only",
                ignored=True,
                note="saliency component; no class information",
            )
        )
    return boxes, {"components": max(0, count - 1), "kept": len(boxes)}
