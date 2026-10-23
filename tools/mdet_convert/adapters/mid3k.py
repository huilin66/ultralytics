# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""MID-3K adapter: RGB + thermal + LiDAR depth, official YOLO person boxes."""

from __future__ import annotations

import datetime as dt
from collections import Counter
from pathlib import Path

import cv2

from ..schema import Box, DatasetPlan, EncodeSpec, ModalitySpec, RawSample
from .base import Adapter

# The LiDAR writes this value where the beam found no return (64.536 m == 0xFC18).
DEPTH_NO_RETURN = 64536.0


def _timestamp(stem: str) -> float | None:
    """Parse the nanosecond epoch embedded in ``000000_1715087159061715111``."""
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1]) / 1e9
    except ValueError:
        return None


def _group_key(stem: str, bucket_minutes: int) -> str:
    """Group samples by collection time so consecutive frames never straddle splits."""
    seconds = _timestamp(stem)
    if seconds is None:
        return stem.split("_")[0]
    moment = dt.datetime.fromtimestamp(seconds)
    return f"{moment:%Y-%m-%dT%H}:{moment.minute // bucket_minutes * bucket_minutes:02d}"


class Mid3kAdapter(Adapter):
    """Convert MID-3K into an ``rgb`` / ``thermal`` / ``depth`` dataset."""

    key = "mid3k"
    display_name = "MID-3K"

    def plan(self) -> DatasetPlan:
        """Discover the four modality folders and pair them by file stem."""
        extract = Path(self.option("extract_root", self.raw_root / "_extracted"))
        roots = {name: extract / f"MID-3K-{name}-main" for name in ("rgb", "thermal", "depth", "intensity")}
        self.require_dir(roots["rgb"] / "images")

        modalities = (
            ModalitySpec("rgb", 3, EncodeSpec("rgb", units="bgr8"), color="bgr"),
            ModalitySpec("thermal", 1, EncodeSpec("gray", units="raw8", note="thermal PNG stores three identical channels")),
            ModalitySpec(
                "depth",
                1,
                EncodeSpec(
                    "scaled",
                    units="mm",
                    lo=0.0,
                    hi=20000.0,
                    invalid=(DEPTH_NO_RETURN,),
                    note="LiDAR depth in millimetres; 64536 is the no-return sentinel, clipped to a 20 m range",
                ),
            ),
        )

        bucket = int(self.option("group_bucket_minutes", 5))
        warnings: list[str] = []
        samples: list[RawSample] = []
        missing = Counter()

        images = sorted((roots["rgb"] / "images").glob("*.png"))
        if not images:
            raise FileNotFoundError(f"no RGB images under {roots['rgb'] / 'images'}")
        probe = cv2.imread(str(images[0]), cv2.IMREAD_UNCHANGED)
        if probe is None:
            raise FileNotFoundError(f"unreadable reference image {images[0]}")
        self._reference_size = (int(probe.shape[1]), int(probe.shape[0]))
        warnings.append(f"reference image size {self._reference_size[0]}x{self._reference_size[1]} taken from {images[0].name}")

        for image in images:
            stem = image.stem
            files = {"rgb": image}
            for name in ("thermal", "depth"):
                candidate = roots[name] / "images" / f"{stem}.png"
                if candidate.is_file():
                    files[name] = candidate
                else:
                    missing[name] += 1

            label = roots["rgb"] / "labels" / f"{stem}.txt"
            boxes = self._read_label(label, warnings, stem)
            samples.append(
                RawSample(
                    sample_id=stem,
                    group=_group_key(stem, bucket),
                    files=files,
                    boxes=boxes,
                    annotation_status="official_box",
                )
            )

        if missing:
            warnings.append(f"unpaired modality files: {dict(missing)}")
        unpaired_splits = len({s.group for s in samples})
        warnings.append(
            f"{unpaired_splits} time groups of {bucket} min derived from file-name timestamps; "
            "no official train/val/test split ships with MID-3K"
        )
        if roots["intensity"].is_dir():
            warnings.append("intensity modality present but excluded: the plan fixes the modality set to rgb/thermal/depth")

        return DatasetPlan(
            key=self.key,
            display_name=self.display_name,
            reference_modality="rgb",
            modalities=modalities,
            classes={"person": 0},
            samples=samples,
            warnings=warnings,
            source={"raw": str(roots["rgb"]), "license": "see https://github.com/kennedyk1/MID-3K"},
        )

    def _read_label(self, path: Path, warnings: list[str], stem: str) -> list[Box]:
        """Read one official YOLO label file (class xc yc w h, normalised)."""
        if not path.is_file():
            warnings.append(f"missing label file for {stem}")
            return []
        boxes: list[Box] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id, cx, cy, bw, bh = int(parts[0]), *(float(v) for v in parts[1:])
            if class_id != 0:
                warnings.append(f"unexpected class id {class_id} in {stem}")
                continue
            # MID-3K ships normalised boxes; convert to pixels in the reference frame
            # and let the pipeline re-normalise after clipping.
            width, height = self._reference_size
            boxes.append(
                Box(
                    class_name="person",
                    xyxy=(
                        (cx - bw / 2) * width,
                        (cy - bh / 2) * height,
                        (cx + bw / 2) * width,
                        (cy + bh / 2) * height,
                    ),
                )
            )
        return boxes
