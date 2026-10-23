# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""InfraParis adapter: RGB (Left) + 8-bit depth, official pedestrian boxes.

The infrared folder is **not** pixel-aligned with the RGB camera (288x382 raw
thermal against 1920x1080 RGB) and the release ships no RGB<->IR homography, so
thermal stays out of the converted dataset unless a calibration is supplied.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..encode import read_raw
from ..schema import Box, DatasetPlan, EncodeSpec, ModalitySpec, RawSample
from .base import Adapter

SPLITS = ("train", "val", "test")


class InfraParisAdapter(Adapter):
    """Convert InfraParis into an ``rgb`` (+ optional ``depth``) dataset."""

    key = "infraparis"
    display_name = "InfraParis"

    def plan(self) -> DatasetPlan:
        """Pair ``Left`` frames with ``Depth`` maps and ``BBox`` JSON annotations."""
        extract = Path(self.option("extract_root", self.raw_root / "_extracted"))
        left_root = self.require_dir(extract / "Left")
        depth_root = extract / "Depth"
        bbox_root = self.require_dir(extract / "BBox")

        modalities = [
            ModalitySpec("rgb", 3, EncodeSpec("rgb", units="bgr8", note="Left camera, alpha channel dropped")),
        ]
        warnings: list[str] = []
        if self.option("include_depth", True) and depth_root.is_dir():
            modalities.append(
                ModalitySpec(
                    "depth",
                    1,
                    EncodeSpec("gray", units="unknown", note="8-bit depth PNG as shipped; physical unit undocumented"),
                )
            )
        else:
            warnings.append("Depth folder missing: converting RGB only")

        thermal_root = extract / "Infra"
        if self.option("include_thermal", False):
            warnings.append(
                "include_thermal was requested but Infra is 288x382 raw thermal against 1920x1080 RGB with no "
                "RGB<->IR calibration in the release; thermal is excluded rather than blindly resized"
            )
        elif thermal_root.is_dir():
            warnings.append(
                "thermal (Infra, 288x382 uint16) is not pixel-aligned with RGB (1920x1080) and has no supplied "
                "homography: alignment_status=blocked, modality omitted"
            )

        samples: list[RawSample] = []
        probe_file = next(iter(sorted(left_root.rglob("*.png"))), None)
        if probe_file is None:
            raise FileNotFoundError(f"no Left frames under {left_root}")
        probe = read_raw(probe_file)
        self._reference_size = (int(probe.shape[1]), int(probe.shape[0]))
        warnings.append(f"reference frame size {self._reference_size[0]}x{self._reference_size[1]} from {probe_file.name}")

        for split in SPLITS:
            split_dir = left_root / split
            if not split_dir.is_dir():
                warnings.append(f"official split '{split}' has no Left folder")
                continue
            for scene_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                scene = scene_dir.name
                for image in sorted(scene_dir.glob("*.png")):
                    stem = image.stem
                    files = {"rgb": image}
                    if len(modalities) > 1:
                        depth = depth_root / split / scene / f"{stem}.png"
                        if depth.is_file():
                            files["depth"] = depth
                    label = bbox_root / split / scene / f"{stem}_BB.json"
                    boxes = self._read_boxes(label, warnings, stem)
                    samples.append(
                        RawSample(
                            sample_id=f"{scene}__{stem}",
                            group=scene,
                            files=files,
                            boxes=boxes,
                            split=split,
                            annotation_status="official_box",
                        )
                    )

        if not samples:
            raise FileNotFoundError(f"no Left/{'|'.join(SPLITS)} frames found under {left_root}")

        return DatasetPlan(
            key=self.key,
            display_name=self.display_name,
            reference_modality="rgb",
            modalities=tuple(modalities),
            classes={"person": 0},
            samples=samples,
            warnings=warnings,
            source={"raw": str(left_root), "license": "CC BY-NC-ND 4.0", "official_splits": True},
        )

    def _read_boxes(self, path: Path, warnings: list[str], stem: str) -> list[Box]:
        """Read Label Studio rectangles (normalised xy) from one ``_BB.json`` file."""
        if not path.is_file():
            warnings.append(f"missing BBox json for {stem}")
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        width, height = self._reference_size
        boxes: list[Box] = []
        for rectangle in payload.get("annotations", {}).get("rectangles", []):
            label = str(rectangle.get("label", "")).strip()
            if label.lower() != "person":
                warnings.append(f"unexpected annotation label '{label}' in {path.name}")
                continue
            points = rectangle.get("coordinates") or []
            if len(points) < 4:
                continue
            xs = [float(p["x"]) * width for p in points]
            ys = [float(p["y"]) * height for p in points]
            boxes.append(Box(class_name="person", xyxy=(min(xs), min(ys), max(xs), max(ys))))
        return boxes
