# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""MM5 adapter: aligned RGB8 / T8 / D16 with instance+class masks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2

from ..labels import mask_to_boxes
from ..schema import DatasetPlan, EncodeSpec, ModalitySpec, RawSample
from .base import Adapter


class MM5Adapter(Adapter):
    """Convert the aligned MM5 release into ``rgb`` / ``thermal`` / ``depth``."""

    key = "mm5"
    display_name = "MM5"

    def plan(self) -> DatasetPlan:
        """Read class mapping, instance masks and the official challenge split lists."""
        extract = Path(self.option("extract_root", self.raw_root / "_extracted"))
        base = self.require_dir(extract / "MM5_ALIGNED")
        rgb_root = self.require_dir(base / "RGB8")

        label_mapping = json.loads((base / "label_mapping.json").read_text(encoding="utf-8"))
        # ANNO_CLASS stores the ids from label_mapping.json; YOLO indexes from zero.
        id_to_name = {int(v): k for k, v in label_mapping.items()}
        classes = {name: index - 1 for name, index in label_mapping.items()}

        modalities = (
            ModalitySpec("rgb", 3, EncodeSpec("rgb", units="bgr8"), color="bgr"),
            ModalitySpec("thermal", 1, EncodeSpec("gray", units="raw8")),
            ModalitySpec(
                "depth",
                1,
                EncodeSpec("scaled", units="mm", lo=0.0, hi=2000.0, note="D16 stored in millimetres"),
            ),
        )

        warnings: list[str] = []
        suffix = str(self.option("split_suffix", "f"))
        include_unlisted = bool(self.option("include_unlisted", False))
        official = self._split_lists(base, suffix, warnings)
        groups = self._groups(base)
        if not official:
            warnings.append("no usable official split lists; the pipeline will build grouped splits from Sequence ids")
        else:
            warnings.append(f"using official challenge split lists list_*_{suffix}.txt")

        instance_dir = base / "ANNO_INST"
        class_dir = base / "ANNO_CLASS"
        samples: list[RawSample] = []
        mask_diagnostics = {"instances": 0, "too_small": 0, "class_unknown": 0}
        unlisted = 0

        for image in sorted(rgb_root.glob("*.png"), key=lambda p: (len(p.stem), p.stem)):
            stem = image.stem
            if official and not include_unlisted and stem not in official:
                # Mixing samples outside a challenge list into that train/val set would
                # leak sequences across splits, so unlisted ids are dropped by default.
                unlisted += 1
                continue
            files = {"rgb": image}
            for name, folder in (("thermal", "T8"), ("depth", "D16")):
                candidate = base / folder / f"{stem}.png"
                if candidate.is_file():
                    files[name] = candidate

            instance_mask = cv2.imread(str(instance_dir / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            class_mask = cv2.imread(str(class_dir / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            if instance_mask is None:
                warnings.append(f"missing instance mask for sample {stem}")
                boxes = []
            else:
                boxes, diagnostics = mask_to_boxes(instance_mask, class_mask, id_to_name)
                for key, value in diagnostics.items():
                    mask_diagnostics[key] += value

            samples.append(
                RawSample(
                    sample_id=stem,
                    group=groups.get(stem, stem),
                    files=files,
                    boxes=boxes,
                    split=official.get(stem),
                    annotation_status="derived_instance_box",
                )
            )

        if not samples:
            raise FileNotFoundError(f"no aligned RGB8 frames under {rgb_root}")
        if unlisted:
            warnings.append(f"{unlisted} samples are not part of list_*_{suffix}.txt and were skipped")
        warnings.append(f"instance mask diagnostics: {mask_diagnostics}")
        warnings.append("boxes are the minimal outer rectangle of each instance mask, not official detection boxes")

        return DatasetPlan(
            key=self.key,
            display_name=self.display_name,
            reference_modality="rgb",
            modalities=modalities,
            classes=classes,
            samples=samples,
            warnings=warnings,
            source={"raw": str(base), "official_splits": f"list_*_{suffix}.txt"},
        )

    def _split_lists(self, base: Path, suffix: str, warnings: list[str]) -> dict[str, str]:
        """Read ``list_train_<suffix>.txt`` / ``list_eval_<suffix>.txt`` into id -> split."""
        mapping: dict[str, str] = {}
        for split, pattern in (("val", f"list_eval_{suffix}.txt"), ("train", f"list_train_{suffix}.txt")):
            path = base / pattern
            if not path.is_file():
                warnings.append(f"split list {pattern} not found")
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                value = line.strip()
                if value:
                    mapping[value] = split
        return mapping

    def _groups(self, base: Path) -> dict[str, str]:
        """Map sample id -> acquisition sequence from ``dataset_meta.csv``."""
        path = base / "dataset_meta.csv"
        if not path.is_file():
            return {}
        groups: dict[str, str] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                sample_id = (row.get("ID") or "").strip()
                if sample_id:
                    groups[sample_id] = f"seq{(row.get('Sequence') or 'unknown').strip()}"
        return groups
