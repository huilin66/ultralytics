# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""VDT-2048 adapter: RGB / thermal / depth with *saliency* masks.

VDT-2048 ships a binary salient/background mask, not instance boxes with
categories.  The derived boxes are therefore written with
``annotation_status=candidate_only`` and one placeholder class: they must not be
reported as multi-class detection accuracy.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from ..labels import binary_mask_to_boxes
from ..schema import DatasetPlan, EncodeSpec, ModalitySpec, RawSample
from .base import Adapter

SPLITS = {"Train": "train", "Test": "test"}


class Vdt2048Adapter(Adapter):
    """Convert VDT-2048 into ``rgb`` / ``thermal`` / ``depth`` with candidate boxes."""

    key = "vdt2048"
    display_name = "VDT-2048"

    def plan(self) -> DatasetPlan:
        """Pair the V/T/D folders and derive candidate boxes from the GT saliency mask."""
        extract = Path(self.option("extract_root", self.raw_root / "_extracted"))
        base = self.require_dir(extract / "VDT-2048 dataset")
        modalities = (
            ModalitySpec("rgb", 3, EncodeSpec("rgb", units="bgr8"), color="bgr"),
            ModalitySpec(
                "thermal",
                3,
                EncodeSpec(
                    "rgb",
                    units="palette_rgb8",
                    note="thermal stored as a 256-colour palette rendering, not a radiometric single channel; "
                    "kept as shipped rather than guessing a gray mapping",
                ),
            ),
            ModalitySpec(
                "depth",
                1,
                EncodeSpec("scaled", units="mm", lo=0.0, hi=3000.0, note="16-bit depth, unit assumed millimetres"),
            ),
        )

        samples: list[RawSample] = []
        warnings: list[str] = ["VDT-2048 GT is a binary saliency mask: boxes are candidate-only with no category"]
        diagnostics = {"components": 0, "kept": 0}

        for raw_split, split in SPLITS.items():
            split_dir = base / raw_split
            if not split_dir.is_dir():
                warnings.append(f"split folder '{raw_split}' is missing")
                continue
            for image in sorted((split_dir / "V").glob("*.png")):
                stem = image.stem
                files = {"rgb": image}
                for name, folder in (("thermal", "T"), ("depth", "D")):
                    candidate = split_dir / folder / f"{stem}.png"
                    if candidate.is_file():
                        files[name] = candidate
                mask = cv2.imread(str(split_dir / "GT" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    warnings.append(f"missing GT mask for {raw_split}/{stem}")
                    boxes = []
                else:
                    boxes, info = binary_mask_to_boxes(mask)
                    for box in boxes:
                        box.class_name = "salient_object"
                        box.ignored = False
                    diagnostics["components"] += info["components"]
                    diagnostics["kept"] += info["kept"]
                samples.append(
                    RawSample(
                        sample_id=f"{split}_{stem}",
                        group=f"{split}_saliency",
                        files=files,
                        boxes=boxes,
                        split=split,
                        annotation_status="candidate_only",
                    )
                )

        if not samples:
            raise FileNotFoundError(f"no V/T/D frames under {base}")
        warnings.append(f"saliency components: {diagnostics}")
        warnings.append(
            "candidate boxes carry one placeholder class 'salient_object'; do not mix them into a multi-class mAP table"
        )

        return DatasetPlan(
            key=self.key,
            display_name=self.display_name,
            reference_modality="rgb",
            modalities=modalities,
            classes={"salient_object": 0},
            samples=samples,
            warnings=warnings,
            source={"raw": str(base), "official_splits": True},
        )
