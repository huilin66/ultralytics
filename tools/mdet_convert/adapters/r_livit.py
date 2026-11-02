# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""R-LiViT adapter: RGB + thermal + LiDAR-derived depth, Pascal-VOC boxes.

The release ships ``calibration_params.json``, a RGB<->LiDAR sequence mapping
and 10 Hz LiDAR scans.  RGB and thermal frames are already undistorted and
pixel-aligned; the third modality is obtained by projecting the nearest LiDAR
scan into the RGB camera with the shipped ``lidar_to_rgb`` extrinsic and a
Z-buffer.  The projection is sparse by design and is never interpolated.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from ..encode import write_png
from ..naming import slugify
from ..projection import (
    LidarCalibration,
    ProjectionError,
    calibration_sha256,
    depth_to_uint16_mm,
    load_calibration,
    load_lidar_timestamps,
    load_sequence_mapping,
    match_lidar_frame,
    project_lidar_to_depth,
    read_lidar_points,
)
from ..schema import Box, DatasetPlan, EncodeSpec, ModalitySpec, RawSample
from .base import Adapter


@dataclass(frozen=True)
class _DepthContext:
    """Everything needed to render and document the LiDAR-derived depth modality."""

    lidar_root: Path
    calibration: LidarCalibration
    calibration_path: Path
    mapping_path: Path
    mapping: dict[str, dict[int, tuple[str, ...]]]
    timestamps: dict[str, object]
    available_scans: set[str]
    min_depth_m: float
    max_depth_m: float
    encode_max_mm: float
    tolerance_s: float


class RLiViTAdapter(Adapter):
    """Convert the R-LiViT RGB-T release and its calibrated LiDAR scans."""

    key = "r_livit"
    display_name = "R-LiViT"

    def plan(self) -> DatasetPlan:
        """Walk ``rgb/``, pair ``thermal/`` and annotations, and index LiDAR scans."""
        extract = Path(self.option("extract_root", self.raw_root / "_extracted"))
        base = self.require_dir(extract / "R-LiViT_RGB-T")
        rgb_root = self.require_dir(base / "rgb")
        thermal_root = base / "thermal"
        annotations_root = self.require_dir(base / "annotations")

        official = self._official_sequences(base)
        warnings: list[str] = []
        if not thermal_root.is_dir():
            warnings.append("thermal folder missing: RGB-only conversion")
        if base.joinpath("train.txt").is_file() and not base.joinpath("val.txt").is_file():
            warnings.append("release ships train/test sequences only; validation must be carved out of train groups")

        depth = None
        if self.option("include_lidar_depth", True):
            depth = self._load_depth_context(extract, warnings)
        else:
            warnings.append("LiDAR depth disabled with --set include_lidar_depth=false")

        modalities: list[ModalitySpec] = [
            ModalitySpec("rgb", 3, EncodeSpec("rgb", units="bgr8"), color="bgr"),
        ]
        if thermal_root.is_dir():
            modalities.append(
                ModalitySpec("thermal", 1, EncodeSpec("gray", units="raw8", note="already pixel-aligned with RGB"))
            )
        if depth is not None:
            modalities.append(
                ModalitySpec(
                    "depth",
                    1,
                    EncodeSpec(
                        "scaled",
                        units="mm",
                        lo=0.0,
                        hi=depth.encode_max_mm,
                        invalid=(0,),
                        note=(
                            "LiDAR returns projected into the undistorted RGB camera with the shipped "
                            "lidar_to_rgb extrinsic; Z-buffer nearest hit; 0 = no return; not interpolated"
                        ),
                    ),
                )
            )

        samples: list[RawSample] = []
        class_names: set[str] = set()
        unreadable = 0
        unmatched_depth = 0
        missing_scans: set[str] = set()
        deltas: list[float] = []

        for sequence_dir in sorted(p for p in rgb_root.iterdir() if p.is_dir()):
            sequence = sequence_dir.name
            for image in sorted(sequence_dir.glob("*.png")):
                stem = image.stem
                files = {"rgb": image}
                thermal = thermal_root / sequence / f"{stem}.png"
                if thermal.is_file():
                    files["thermal"] = thermal

                annotation = annotations_root / sequence / f"{stem}.xml"
                boxes, names, status, rgb_timestamp = self._read_voc(annotation)
                class_names.update(names)
                if status:
                    unreadable += 1
                sample = RawSample(
                    sample_id=f"{sequence}__{stem}",
                    group=sequence,
                    files=files,
                    boxes=boxes,
                    split=official.get(sequence),
                    annotation_status="official_box" if not status else status,
                )

                if depth is not None:
                    match = None
                    if rgb_timestamp is not None:
                        match = match_lidar_frame(
                            sequence,
                            int(stem),
                            rgb_timestamp,
                            depth.mapping,
                            depth.timestamps,
                            tolerance_s=depth.tolerance_s,
                        )
                    if match is None:
                        unmatched_depth += 1
                        warnings.append(f"no LiDAR scan within {depth.tolerance_s * 1000:.0f} ms for {sample.sample_id}")
                    else:
                        scan = f"{match.lidar_sequence}/{match.lidar_frame:06d}"
                        bin_path = depth.lidar_root / "point_clouds" / match.lidar_sequence / f"{match.lidar_frame:06d}.bin"
                        deltas.append(match.delta_s)
                        if scan not in depth.available_scans:
                            missing_scans.add(scan)
                        sample.modality_meta["depth"] = {
                            "source_path": str(bin_path),
                            "lidar_sequence": match.lidar_sequence,
                            "lidar_frame": match.lidar_frame,
                            "rgb_timestamp_s": match.rgb_timestamp_s,
                            "lidar_timestamp_s": match.lidar_timestamp_s,
                            "timestamp_delta_ms": round(match.delta_s * 1000.0, 3),
                            "calibration": str(depth.calibration_path),
                            "calibration_sha256": depth.calibration.fingerprint(),
                            "projection": "lidar_to_rgb pinhole Z-buffer (nearest return); no interpolation",
                            "alignment_status": "calibrated_lidar_to_rgb_projection",
                            "aligned_to": "rgb",
                            "min_depth_m": depth.min_depth_m,
                            "max_depth_m": depth.max_depth_m,
                            "encode_max_mm": depth.encode_max_mm,
                        }
                samples.append(sample)

        if unreadable:
            warnings.append(f"{unreadable} frames have missing or unparsable annotations")
        if not samples:
            raise FileNotFoundError(f"no RGB frames under {rgb_root}")

        if depth is not None:
            if unmatched_depth:
                warnings.append(
                    f"{unmatched_depth} samples have no LiDAR scan within {depth.tolerance_s * 1000:.0f} ms and "
                    "will be excluded because depth is a required modality"
                )
            if missing_scans:
                warnings.append(
                    f"{len(missing_scans)} matched LiDAR scans are absent from {depth.lidar_root}; run "
                    "`python tools/extract_raw.py --dataset r_livit --with-lidar` or extract the referenced .bin files"
                )
            if deltas:
                deltas_sorted = sorted(deltas)
                median_ms = deltas_sorted[len(deltas_sorted) // 2] * 1000.0
                warnings.append(
                    f"LiDAR depth: {len(deltas)} frames matched to scans, |dt| median {median_ms:.2f} ms, "
                    f"max {max(deltas) * 1000.0:.2f} ms; RGB/thermal are shipped undistorted so the pinhole K is "
                    "used without distortion coefficients; depth is sparse (no interpolation)"
                )

        classes = {name: index for index, name in enumerate(sorted(class_names))}
        warnings.append(f"classes taken verbatim from VOC XML: {sorted(classes)}")
        warnings.append("LiDAR 3D labels are not used as 2D detections; boxes remain the official RGB-T VOC boxes")

        source = {"raw": str(base), "official_splits": "sequence level"}
        if depth is not None:
            source.update(
                {
                    "lidar": str(depth.lidar_root),
                    "calibration": str(depth.calibration_path),
                    "calibration_sha256": calibration_sha256(depth.calibration_path),
                    "sequence_mapping": str(depth.mapping_path),
                    "projection": {
                        "extrinsic": "lidar_to_rgb",
                        "intrinsics": "rgb_camera.intrinsic_matrix",
                        "distortion_applied": False,
                        "min_depth_m": depth.min_depth_m,
                        "max_depth_m": depth.max_depth_m,
                        "encode_max_mm": depth.encode_max_mm,
                    },
                }
            )

        plan = DatasetPlan(
            key=self.key,
            display_name=self.display_name,
            reference_modality="rgb",
            modalities=tuple(modalities),
            classes=classes,
            samples=samples,
            warnings=warnings,
            source=source,
            reference_camera="rgb",
        )
        if depth is not None:
            self._depth = depth
            self._plan = plan
            plan.materialize = self._materialize_depth
        return plan

    def _load_depth_context(self, extract: Path, warnings: list[str]) -> _DepthContext | None:
        """Resolve calibration, mapping, timestamps and available scans, or explain what is missing."""
        lidar_root = self._first_dir(
            extract / "R-LiViT_LiDAR",
            extract / "R-LiViT" / "R-LiViT_LiDAR",
        )
        calibration_path = self._first_file(
            extract / "R-LiViT" / "calibration_params.json",
            extract / "calibration_params.json",
        )
        mapping_path = self._first_file(
            extract / "R-LiViT" / "rgb_to_lidar_sequence_mapping.csv",
            extract / "rgb_to_lidar_sequence_mapping.csv",
        )
        missing = [
            name
            for name, value in (
                ("LiDAR root", lidar_root),
                ("calibration_params.json", calibration_path),
                ("rgb_to_lidar_sequence_mapping.csv", mapping_path),
            )
            if value is None
        ]
        if missing:
            warnings.append(
                "LiDAR depth unavailable (missing "
                + ", ".join(missing)
                + "); converting RGB-T only. Run `python tools/extract_raw.py --dataset r_livit --with-lidar`."
            )
            return None
        assert lidar_root is not None and calibration_path is not None and mapping_path is not None
        point_root = lidar_root / "point_clouds"
        if not point_root.is_dir():
            warnings.append(f"LiDAR point_clouds folder missing under {lidar_root}; converting RGB-T only")
            return None
        available = {f"{path.parent.name}/{path.stem}" for path in point_root.rglob("*.bin")}
        if not available:
            warnings.append(f"no LiDAR .bin scans under {point_root}; converting RGB-T only")
            return None

        calibration = load_calibration(calibration_path)
        mapping = load_sequence_mapping(mapping_path)
        timestamps = load_lidar_timestamps(lidar_root / "timestamps")

        try:
            min_depth_m = float(self.option("depth_min_m", 0.5))
            max_depth_m = float(self.option("depth_max_m", 100.0))
            encode_max_m = float(self.option("depth_encode_max_m", 50.0))
            tolerance_ms = float(self.option("lidar_timestamp_tolerance_ms", 100.0))
        except (TypeError, ValueError) as error:
            raise ProjectionError(f"invalid R-LiViT depth option: {error}") from error
        if encode_max_m <= 0 or encode_max_m * 1000.0 > 65535:
            raise ProjectionError("depth_encode_max_m must be in (0, 65.535]")
        if min_depth_m <= 0 or max_depth_m <= min_depth_m:
            raise ProjectionError("require 0 < depth_min_m < depth_max_m")
        if tolerance_ms <= 0:
            raise ProjectionError("lidar_timestamp_tolerance_ms must be positive")

        warnings.append(
            f"LiDAR calibration loaded from {calibration_path.name}: RGB {calibration.width}x{calibration.height}, "
            f"projection window [{min_depth_m}, {max_depth_m}] m, uint8 encoding range 0-{encode_max_m:g} m"
        )
        return _DepthContext(
            lidar_root=lidar_root,
            calibration=calibration,
            calibration_path=calibration_path,
            mapping_path=mapping_path,
            mapping=mapping,
            timestamps=timestamps,
            available_scans=available,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            encode_max_mm=encode_max_m * 1000.0,
            tolerance_s=tolerance_ms / 1000.0,
        )

    def _materialize_depth(
        self,
        selected: list[tuple[RawSample, str]],
        derived_dir: Path,
        workers: int,
    ) -> None:
        """Render the uint16 depth cache and attach it to each sample."""
        context: _DepthContext = self._depth
        tasks = [(sample, split) for sample, split in selected if "depth" in sample.modality_meta]
        skipped = [sample.sample_id for sample, _ in selected if "depth" not in sample.modality_meta]
        if skipped:
            self._plan.warnings.append(
                f"{len(skipped)} samples were not depth-materialised (no LiDAR match) and will be excluded"
            )
        if not tasks:
            return

        def render(item: tuple[RawSample, str]) -> dict:
            sample, split = item
            meta = sample.modality_meta["depth"]
            source = Path(meta["source_path"])
            if not source.is_file():
                raise ProjectionError(
                    f"LiDAR scan {source} for sample {sample.sample_id} is missing; re-run "
                    "`python tools/extract_raw.py --dataset r_livit --with-lidar`"
                )
            points = read_lidar_points(source)
            depth_m, stats = project_lidar_to_depth(
                points,
                context.calibration,
                min_depth_m=context.min_depth_m,
                max_depth_m=context.max_depth_m,
            )
            depth_mm, depth_stats = depth_to_uint16_mm(depth_m, max_mm=context.encode_max_mm)
            target = derived_dir / "depth_uint16" / split / f"{slugify(sample.sample_id)}.png"
            write_png(target, depth_mm)
            sample.files["depth"] = target
            meta.update(
                {
                    "encoded_from": str(target),
                    "points_in_image": stats["points_in_image"],
                    "coverage": round(stats["coverage"], 6),
                    "depth_p50_m": None if stats["depth_p50_m"] is None else round(stats["depth_p50_m"], 3),
                    "depth_p98_m": None if stats["depth_p98_m"] is None else round(stats["depth_p98_m"], 3),
                    "saturated_fraction_of_valid": round(depth_stats["saturated_fraction_of_valid"], 6),
                    "projection_status": "ok",
                }
            )
            return {"sample_id": sample.sample_id, "coverage": stats["coverage"], "depth_stats": depth_stats}

        # Render serially on purpose.  Projecting several large LiDAR scans at
        # once reproducibly crashed this environment (cv2 5.0.0 + Python
        # threads: SIGSEGV or a pool hang after ~500 frames), while the exact
        # same workload is stable when serial.  The later uint8 encoding phase
        # stays threaded; only the derived-sensor stage pays this cost.
        results = [render(item) for item in tasks]

        coverages = sorted(float(result["coverage"]) for result in results)
        summary = {
            "modality": "depth",
            "source": "LiDAR",
            "calibration": str(context.calibration_path),
            "calibration_sha256": calibration_sha256(context.calibration_path),
            "sequence_mapping": str(context.mapping_path),
            "projection": {
                "extrinsic": "lidar_to_rgb",
                "intrinsics": "rgb_camera.intrinsic_matrix",
                "distortion_applied": False,
                "min_depth_m": context.min_depth_m,
                "max_depth_m": context.max_depth_m,
                "encode_max_mm": context.encode_max_mm,
                "z_buffer": True,
                "interpolation": False,
            },
            "samples": len(results),
            "coverage": {
                "min": round(coverages[0], 6),
                "median": round(coverages[len(coverages) // 2], 6),
                "max": round(coverages[-1], 6),
            },
        }
        derived_dir.mkdir(parents=True, exist_ok=True)
        (derived_dir / "depth_projection_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        self._plan.warnings.append(
            f"LiDAR depth cache written for {len(results)} samples under {derived_dir.name}/depth_uint16; "
            f"pixel coverage min/median/max = {coverages[0]:.3%}/{coverages[len(coverages) // 2]:.3%}/"
            f"{coverages[-1]:.3%} (sparse, not interpolated)"
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

    def _read_voc(self, path: Path) -> tuple[list[Box], list[str], str, float | None]:
        """Parse one Pascal-VOC annotation file and its RGB timestamp in seconds."""
        if not path.is_file():
            return [], [], "missing_annotation", None
        try:
            root = ET.fromstring(path.read_text(encoding="utf-8"))
        except ET.ParseError:
            return [], [], "unparsable_annotation", None
        timestamp_text = (root.findtext("timestampRGB") or "").strip()
        try:
            timestamp = float(timestamp_text) / 1e9 if timestamp_text else None
        except ValueError:
            timestamp = None
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
        return boxes, names, "", timestamp

    @staticmethod
    def _first_dir(*candidates: Path) -> Path | None:
        """Return the first existing directory from the candidate paths."""
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return None

    @staticmethod
    def _first_file(*candidates: Path) -> Path | None:
        """Return the first existing file from the candidate paths."""
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None
