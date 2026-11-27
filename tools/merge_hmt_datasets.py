"""Merge the three single-modality HMT datasets into one paired RGB-T dataset.

The RGB slot takes both the RGB camera images and the cube-projection images; the thermal slot takes the
thermal images. A sample without a real thermal (or RGB) companion gets a same-size black image instead,
so the merged set can be loaded by ``MultiModalDataset`` without touching the source directories.

Source datasets are read-only. Everything is written below ``--out``.

Usage::

    python tools/merge_hmt_datasets.py --rgb Y:/dataset/hmt_rgb_update_v6.1 \\
        --thermal Y:/dataset/hmt_t_update_v6.1 --cube Y:/dataset/hmt_cube_update_v6.1 \\
        --out Y:/dataset/hmt_merged_v1 --dry-run
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

UNIFIED_CLASSES = [
    "Corrosion",
    "Efflorescene",
    "Delaminate",
    "Broken",
    "peeling",
    "Hollow Confirmed",
    "Hollow Suspected",
    "Leakage",
]


@dataclass
class Sample:
    """One merged sample: a primary image, its thermal companion (may be missing) and label files."""

    source: str
    stem: str
    split: str
    primary: Path | None
    thermal: Path | None
    labels: list[tuple[str, Path]] = field(default_factory=list)


@dataclass
class Report:
    """Merge statistics written next to the merged dataset."""

    samples: dict[str, dict[str, int]] = field(default_factory=dict)
    boxes: dict[str, int] = field(default_factory=dict)
    black_thermal: dict[str, int] = field(default_factory=dict)
    black_primary: dict[str, int] = field(default_factory=dict)
    size_mismatch: list[list[str]] = field(default_factory=list)
    dropped_orphan_thermal: int = 0


def load_names(dataset_dir: Path) -> list[str]:
    """Read the class names a source dataset declares in its ``dataset.yaml``."""
    config = yaml.safe_load((dataset_dir / "dataset.yaml").read_text(encoding="utf-8"))
    return [str(name) for name in config["names"]]


def class_remap(names: list[str]) -> dict[int, int]:
    """Map a source dataset's class ids onto the unified class list (case-insensitive)."""
    lookup = {name.lower(): index for index, name in enumerate(UNIFIED_CLASSES)}
    mapping = {}
    for index, name in enumerate(names):
        key = name.strip().lower()
        if key not in lookup:
            raise SystemExit(f"Class {name!r} has no slot in {UNIFIED_CLASSES}.")
        mapping[index] = lookup[key]
    return mapping


def read_split(dataset_dir: Path, split: str) -> list[str]:
    """Return image stems listed in a split file; the files hold server-side absolute paths."""
    path = dataset_dir / f"{split}.txt"
    if not path.exists():
        return []
    stems = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            stems.append(Path(line).stem)
    return stems


def index_images(dataset_dir: Path) -> dict[str, Path]:
    """Index a source ``images/`` directory by file stem."""
    return {path.stem: path for path in (dataset_dir / "images").iterdir() if path.is_file()}


def index_labels(dataset_dir: Path) -> dict[str, Path]:
    """Index a source ``labels/`` directory by file stem."""
    return {path.stem: path for path in (dataset_dir / "labels").rglob("*.txt")}


def collect(rgb_dir: Path, t_dir: Path, cube_dir: Path) -> list[Sample]:
    """Pair the three sources into merged samples."""
    rgb_images, t_images, cube_images = index_images(rgb_dir), index_images(t_dir), index_images(cube_dir)
    rgb_labels, t_labels, cube_labels = index_labels(rgb_dir), index_labels(t_dir), index_labels(cube_dir)

    rgb_splits = {split: read_split(rgb_dir, split) for split in ("train", "val")}
    cube_splits = {split: read_split(cube_dir, split) for split in ("train", "val")}
    for name, stems in rgb_splits.items():
        unknown = sorted(set(stems) - set(rgb_images))
        if unknown:
            raise SystemExit(f"RGB {name} lists {len(unknown)} images that do not exist, e.g. {unknown[:3]}")
    for name, stems in cube_splits.items():
        unknown = sorted(set(stems) - set(cube_images))
        if unknown:
            raise SystemExit(f"Cube {name} lists {len(unknown)} images that do not exist, e.g. {unknown[:3]}")

    samples: list[Sample] = []
    for split, stems in rgb_splits.items():
        for stem in stems:
            labels = [
                (source, path)
                for source, path in (("rgb", rgb_labels.get(stem)), ("t", t_labels.get(stem)))
                if path is not None
            ]
            samples.append(Sample("rgb", stem, split, rgb_images[stem], t_images.get(stem), labels))
    for split, stems in cube_splits.items():
        for stem in stems:
            labels = [("cube", cube_labels[stem])] if stem in cube_labels else []
            samples.append(Sample("cube", stem, split, cube_images[stem], None, labels))
    for stem, path in t_images.items():
        if stem not in rgb_images:
            labels = [("t", t_labels[stem])] if stem in t_labels else []
            samples.append(Sample("t", stem, "train", None, path, labels))

    return samples


def assign_split(samples: list[Sample], val_per_source: int) -> None:
    """Keep the source split, or hold out ``val_per_source`` annotated frames per label source for validation."""
    if val_per_source <= 0:
        return
    by_source: dict[str, list[Sample]] = {}
    for sample in samples:
        for source, _ in sample.labels:
            by_source.setdefault(source, []).append(sample)
    held_out: set[tuple[str, str]] = set()
    for source, group in by_source.items():
        ordered = sorted(group, key=lambda item: item.stem)
        count = min(val_per_source, len(ordered))
        picks = {round(index * (len(ordered) - 1) / max(count - 1, 1)) for index in range(count)}
        held_out.update((source, ordered[index].stem) for index in picks)
    for sample in samples:
        if any((source, sample.stem) in held_out for source, _ in sample.labels):
            sample.split = "val"
        else:
            sample.split = "train"


def summarize(samples: list[Sample]) -> Report:
    """Collect per-source, per-split sample counts and black-fill counts."""
    report = Report()
    for sample in samples:
        report.samples.setdefault(sample.source, {}).setdefault(sample.split, 0)
        report.samples[sample.source][sample.split] += 1
        if sample.thermal is None:
            report.black_thermal[sample.source] = report.black_thermal.get(sample.source, 0) + 1
        if sample.primary is None:
            report.black_primary[sample.source] = report.black_primary.get(sample.source, 0) + 1
    return report


def probe_size(path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` without decoding the pixel data."""
    with Image.open(path) as image:
        return image.size


def check_alignment(samples: list[Sample], report: Report) -> None:
    """Record every RGB/Thermal pair whose heights or widths disagree."""
    for sample in samples:
        if sample.primary is None or sample.thermal is None:
            continue
        if probe_size(sample.primary) != probe_size(sample.thermal):
            report.size_mismatch.append(
                [sample.source, sample.stem, str(probe_size(sample.primary)), str(probe_size(sample.thermal))]
            )


def read_primary(path: Path) -> np.ndarray:
    """Read an RGB-slot image as three-channel BGR."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Cannot read {path}")
    return image


def read_thermal(path: Path) -> np.ndarray:
    """Read a thermal image as single-channel grayscale."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"Cannot read {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def write_label(labels: list[tuple[str, Path]], mappings: dict[str, dict[int, int]], target: Path) -> None:
    """Rewrite every YOLO label file of one sample into the unified class space and write one merged file."""
    lines = []
    for source, source_label in labels:
        mapping = mappings[source]
        for line in source_label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            class_id = int(parts[0])
            if class_id not in mapping:
                raise SystemExit(f"{source_label}: class id {class_id} is outside the declared names.")
            lines.append(" ".join([str(mapping[class_id]), *parts[1:]]))
    if lines:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(samples: list[Sample], report: Report, out: Path, rgb_dir: Path, t_dir: Path, cube_dir: Path) -> None:
    """Write the merged images, labels and data.yaml."""
    mappings = {
        "rgb": class_remap(load_names(rgb_dir)),
        "cube": class_remap(load_names(cube_dir)),
        "t": class_remap(load_names(t_dir)),
    }
    box_counts = {name: 0 for name in UNIFIED_CLASSES}
    for sample in samples:
        for source, source_label in sample.labels:
            for line in source_label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts:
                    box_counts[UNIFIED_CLASSES[mappings[source][int(parts[0])]]] += 1
    report.boxes = box_counts

    thermal_root = out / "images" / "thermal"
    rgb_root = out / "images" / "rgb"
    for sample in samples:
        out_stem = f"{sample.source}__{sample.stem}"
        split = sample.split
        rgb_target = rgb_root / split / f"{out_stem}.jpg"
        thermal_target = thermal_root / split / f"{out_stem}.png"
        rgb_target.parent.mkdir(parents=True, exist_ok=True)
        thermal_target.parent.mkdir(parents=True, exist_ok=True)

        if sample.primary is None:
            height, width = read_thermal(sample.thermal).shape[:2]
            primary = np.zeros((height, width, 3), np.uint8)
        else:
            primary = read_primary(sample.primary)
        if sample.thermal is None:
            thermal = np.zeros(primary.shape[:2], np.uint8)
        else:
            thermal = read_thermal(sample.thermal)

        if primary.shape[:2] != thermal.shape[:2]:
            raise SystemExit(
                f"{sample.source}/{sample.stem}: RGB {primary.shape[:2]} and thermal {thermal.shape[:2]} "
                "are not pixel-aligned."
            )
        cv2.imwrite(str(rgb_target), primary, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(thermal_target), thermal)
        write_label(
            sample.labels,
            mappings,
            out / "labels" / "rgb" / split / f"{out_stem}.txt",
        )

    config = {
        "path": str(out),
        "train": "images/rgb/train",
        "val": "images/rgb/val",
        "channels": 4,
        "names": UNIFIED_CLASSES,
        "modalities": [
            {"name": "rgb", "path": "images/rgb", "channels": 3, "suffix": ".jpg"},
            {"name": "thermal", "path": "images/thermal", "channels": 1, "suffix": ".png"},
        ],
    }
    (out / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out / "merge_report.json").write_text(
        json.dumps(
            {
                "samples": report.samples,
                "boxes": report.boxes,
                "black_thermal": report.black_thermal,
                "black_primary": report.black_primary,
                "dropped_orphan_thermal": report.dropped_orphan_thermal,
                "size_mismatch": report.size_mismatch,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Parse arguments, pair the sources and optionally build the merged dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True, help="hmt_rgb_update_v6.1 directory")
    parser.add_argument("--thermal", type=Path, required=True, help="hmt_t_update_v6.1 directory")
    parser.add_argument("--cube", type=Path, required=True, help="hmt_cube_update_v6.1 directory")
    parser.add_argument("--out", type=Path, required=True, help="output directory for the merged dataset")
    parser.add_argument(
        "--val-per-source",
        type=int,
        default=3,
        help="annotated frames held out per label source for validation; 0 keeps the source split",
    )
    parser.add_argument("--dry-run", action="store_true", help="pair and validate only; write nothing")
    parser.add_argument(
        "--drop-orphan-thermal",
        action="store_true",
        help="drop samples that exist only in the thermal source instead of giving them a black primary image",
    )
    args = parser.parse_args()

    samples = collect(args.rgb, args.thermal, args.cube)
    # Split assignment runs on the full set so that removing the orphans cannot move a held-out frame
    # of any other source; the filter is therefore applied afterwards.
    assign_split(samples, args.val_per_source)
    orphans = [sample for sample in samples if sample.primary is None]
    if args.drop_orphan_thermal:
        samples = [sample for sample in samples if sample.primary is not None]
    report = summarize(samples)
    report.dropped_orphan_thermal = len(orphans) if args.drop_orphan_thermal else 0
    check_alignment(samples, report)
    summary = {
        "samples": report.samples,
        "black_thermal": report.black_thermal,
        "black_primary": report.black_primary,
        "dropped_orphan_thermal": report.dropped_orphan_thermal,
        "size_mismatch": len(report.size_mismatch),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if report.size_mismatch:
        print("first mismatches:", report.size_mismatch[:10])
    if args.dry_run:
        return
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; remove it or pick another output directory.")
    build(samples, report, args.out, args.rgb, args.thermal, args.cube)
    print(f"merged dataset written to {args.out}")


if __name__ == "__main__":
    main()
