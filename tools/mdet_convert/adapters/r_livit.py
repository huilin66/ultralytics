# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""R-LiViT adapter: aligned RGB + thermal, Pascal-VOC boxes, official sequence splits."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

from ..schema import Box, DatasetPlan, EncodeSpec, ModalitySpec, RawSample
from .base import Adapter


class RLiViTAdapter(Adapter):
    """Convert the R-LiViT RGB-T release into an ``rgb`` / ``thermal`` dataset."""

    key = "r_livit"
    display_name = "R-LiViT"

    def plan(self) -> DatasetPlan:
        """Walk ``rgb/``, pair ``thermal/`` frames and parse ``annotations/`` XML."""
        extract = Path(self.option("extract_root", self.raw_root / "_extracted"))
        base = self.require_dir(extract / "R-LiViT_RGB-T")
        rgb_root = self.require_dir(base / "rgb")
        thermal_root = base / "thermal"
        annotations_root = self.require_dir(base / "annotations")

        official = self._official_sequences(base)
        modalities = (
            ModalitySpec("rgb", 3, EncodeSpec("rgb", units="bgr8"), color="bgr"),
            ModalitySpec("thermal", 1, EncodeSpec("gray", units="raw8", note="already pixel-aligned with RGB")),
        )
        warnings: list[str] = []
        if not thermal_root.is_dir():
            warnings.append("thermal folder missing: RGB-only conversion")
        if base.joinpath("train.txt").is_file() and not base.joinpath("val.txt").is_file():
            warnings.append("release ships train/test sequences only; validation must be carved out of train groups")

        samples: list[RawSample] = []
        class_names: set[str] = set()
        unreadable = 0

        for sequence_dir in sorted(p for p in rgb_root.iterdir() if p.is_dir()):
            sequence = sequence_dir.name
            for image in sorted(sequence_dir.glob("*.png")):
                stem = image.stem
                files = {"rgb": image}
                thermal = thermal_root / sequence / f"{stem}.png"
                if thermal.is_file():
                    files["thermal"] = thermal

                annotation = annotations_root / sequence / f"{stem}.xml"
                boxes, names, status = self._read_voc(annotation)
                class_names.update(names)
                if status:
                    unreadable += 1
                samples.append(
                    RawSample(
                        sample_id=f"{sequence}__{stem}",
                        group=sequence,
                        files=files,
                        boxes=boxes,
                        split=official.get(sequence),
                        annotation_status="official_box" if not status else status,
                    )
                )

        if unreadable:
            warnings.append(f"{unreadable} frames have missing or unparsable annotations")
        if not samples:
            raise FileNotFoundError(f"no RGB frames under {rgb_root}")

        classes = {name: index for index, name in enumerate(sorted(class_names))}
        warnings.append(f"classes taken verbatim from VOC XML: {sorted(classes)}")
        warnings.append(
            "LiDAR->depth projection is not implemented here: the point clouds need the shipped extrinsics and "
            "timestamp matching, so this dataset is converted as RGB-T only"
        )

        return DatasetPlan(
            key=self.key,
            display_name=self.display_name,
            reference_modality="rgb",
            modalities=tuple(m for m in modalities if (m.name != "thermal" or thermal_root.is_dir())),
            classes=classes,
            samples=samples,
            warnings=warnings,
            source={"raw": str(base), "official_splits": "sequence level"},
        )

    def _official_sequences(self, base: Path) -> dict[str, str]:
        """Map sequence id -> split from ``train.txt`` / ``test.txt``."""
        mapping: dict[str, str] = {}
        for split, filename in (("test", "test.txt"), ("train", "train.txt")):
            path = base / filename
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                sequence = line.strip()
                if sequence:
                    mapping[sequence] = split
        return mapping

    def _read_voc(self, path: Path) -> tuple[list[Box], list[str], str]:
        """Parse one Pascal-VOC annotation file."""
        if not path.is_file():
            return [], [], "missing_annotation"
        try:
            root = ET.fromstring(path.read_text(encoding="utf-8"))
        except ET.ParseError:
            return [], [], "unparsable_annotation"
        boxes, names = [], []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            node = obj.find("bndbox")
            if not name or node is None:
                continue
            boxes.append(
                Box(
                    class_name=name,
                    xyxy=(
                        float(node.findtext("xmin")),
                        float(node.findtext("ymin")),
                        float(node.findtext("xmax")),
                        float(node.findtext("ymax")),
                    ),
                )
            )
            names.append(name)
        return boxes, names, ""
