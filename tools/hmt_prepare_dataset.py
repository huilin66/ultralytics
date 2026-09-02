# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Create immutable, training-ready HMT YOLO datasets.

The source HMT datasets use a one-based object-class convention with an unused
``background`` class at id 0. This tool writes a new dataset tree, compacts
class ids to the standard zero-based YOLO convention, optionally creates
training-only tiles, and writes group-aware split files. It never writes to
the source tree.

Standard validation and statistics are intentionally delegated to
``yolo_data_manager``. This tool owns only HMT-specific operations that need
custom semantics: class remapping, sequence/view grouping, and label-aware
tiling.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Iterable

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - environment error
    raise SystemExit("Pillow is required in common_py312: python -m pip install Pillow") from exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    source_dir: str
    output_dir: str
    names: tuple[str, ...]
    source_to_target: dict[int, int]
    tile: bool
    tile_size: int
    tile_stride: int


@dataclass
class Record:
    image: Path
    source_split: str | None
    group: str = ""
    labels: list[tuple[int, float, float, float, float]] | None = None


SPECS = {
    "t": DatasetSpec(
        key="t",
        source_dir="sua_t",
        output_dir="sua_t_update",
        names=(
            "Hollow Low Risk",
            "Hollow High Risk",
            "Hollow High Risk Line",
            "Leakage High Risk",
            "Temperature Medium Risk",
            "Temperature High Risk",
        ),
        # The original labels are 1..6; class 0 is an unused background slot.
        source_to_target={1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5},
        tile=False,
        tile_size=0,
        tile_stride=0,
    ),
    "rgb": DatasetSpec(
        key="rgb",
        source_dir="sua_rgb",
        output_dir="sua_rgb_update",
        names=(
            "Broken",
            "Corrosion",
            "Delaminated Tile Low Risk",
            "Delaminated Tile High Risk",
            "Efflorescence Low Gray",
            "Efflorescence Low Risk",
            "Efflorescence High Risk",
        ),
        # Rebuild the merge from the raw class names. In particular, the
        # existing merge maps raw Broken Low Risk (id 1) to Efflorescence Low
        # Risk, which is almost certainly an ontology/remap error. Cracked
        # Tile and Spalling are grouped into the available Broken category.
        source_to_target={
            1: 0,  # Broken Low Risk -> Broken
            2: 0,  # Broken High Risk -> Broken
            3: 1,  # Corrosion -> Corrosion
            4: 0,  # Cracked Tile -> Broken
            5: 2,  # Delaminated Tile Low Risk
            6: 3,  # Delaminated Tile High Risk
            7: 4,  # Efflorescence Low Gray
            8: 5,  # Efflorescence Low Risk
            9: 6,  # Efflorescence High Risk
            10: 0,  # Spalling -> Broken
        },
        tile=True,
        tile_size=768,
        tile_stride=512,
    ),
    "cube": DatasetSpec(
        key="cube",
        source_dir="bp_cube",
        output_dir="bp_cube_update",
        names=("Broken", "Efflorescence", "Peeling"),
        source_to_target={1: 0, 2: 1, 3: 2},
        tile=True,
        tile_size=1024,
        tile_stride=768,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SPECS), required=True)
    parser.add_argument("--source-root", type=Path, required=True, help="Original bdd_hmt directory")
    parser.add_argument("--output-root", type=Path, required=True, help="New bdd_hmt_update directory")
    parser.add_argument("--split-mode", choices=("group", "source"), default="group")
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--sequence-group-size", type=int, default=12)
    parser.add_argument("--empty-train-ratio", type=float, default=0.25)
    parser.add_argument("--max-repeat", type=int, default=4)
    parser.add_argument("--min-visibility", type=float, default=0.35)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--tile-stride", type=int, default=None)
    parser.add_argument("--no-tiles", action="store_true", help="Disable training-only tiles")
    parser.add_argument("--force", action="store_true", help="Allow an existing generated update directory")
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"[hmt_prepare] ERROR: {message}")


def ensure_distinct_roots(source: Path, output: Path) -> None:
    source_abs = os.path.normcase(os.path.abspath(str(source)))
    output_abs = os.path.normcase(os.path.abspath(str(output)))
    if source_abs == output_abs:
        fail("source and output are identical; refusing to modify the source dataset")
    try:
        common = os.path.normcase(os.path.commonpath((source_abs, output_abs)))
    except ValueError:
        return
    if common == source_abs:
        fail(f"output is inside source; refusing to modify source data: {output}")


def read_split_names(root: Path, split: str) -> set[str]:
    path = root / f"{split}.txt"
    if not path.exists():
        return set()
    result: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip().strip('"').strip("'")
        if not line:
            continue
        # Source lists may contain Linux-server absolute paths. Only the
        # filename is needed because source images are flat.
        result.add(Path(line.replace("\\", "/")).name.lower())
    return result


def image_files(root: Path) -> list[Path]:
    image_root = root / "images"
    if not image_root.is_dir():
        fail(f"missing images directory: {image_root}")
    return sorted(
        (path for path in image_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )


def read_labels(path: Path, mapping: dict[int, int]) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    labels: list[tuple[int, float, float, float, float]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 5:
            fail(f"invalid label with fewer than 5 fields: {path}:{line_no}")
        try:
            source_class = int(float(fields[0]))
            cx, cy, width, height = (float(value) for value in fields[1:5])
        except ValueError as exc:
            fail(f"non-numeric label: {path}:{line_no} ({exc})")
        if source_class not in mapping:
            fail(f"class id {source_class} is not in the explicit map: {path}:{line_no}")
        if not all(math.isfinite(value) for value in (cx, cy, width, height)):
            fail(f"non-finite label: {path}:{line_no}")
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            fail(f"out-of-range YOLO coordinates: {path}:{line_no}")
        labels.append((mapping[source_class], cx, cy, width, height))
    return labels


def write_labels(path: Path, labels: Iterable[tuple[int, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [f"{cls} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}" for cls, cx, cy, width, height in labels]
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def natural_image_key(path: Path) -> tuple[str, int, str]:
    match = re.search(r"_(\d+)$", path.stem)
    return (path.stem[: match.start()] if match else path.stem, int(match.group(1)) if match else -1, path.name.lower())


def group_key_for_records(records: list[Record], dataset: str, sequence_group_size: int) -> None:
    if dataset == "cube":
        for record in records:
            record.group = re.sub(r"_(?:L|R|B|F)$", "", record.image.stem, flags=re.IGNORECASE)
        return
    ordered = sorted(records, key=lambda record: natural_image_key(record.image))
    for index, record in enumerate(ordered):
        record.group = f"sequence_{index // max(sequence_group_size, 1):04d}"


def class_counts(records: Iterable[Record]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for record in records:
        counts.update(label[0] for label in record.labels or [])
    return counts


def assign_group_splits(records: list[Record], val_ratio: float, test_ratio: float, seed: int) -> dict[str, str]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        fail("val/test ratios must be non-negative and sum to less than 1")
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[record.group].append(record)
    ratios = {"train": 1.0 - val_ratio - test_ratio, "val": val_ratio, "test": test_ratio}
    active_splits = [split for split, ratio in ratios.items() if ratio > 0]
    target_images = {split: len(records) * ratio for split, ratio in ratios.items()}
    total_classes = class_counts(records)
    target_classes = {split: {cls: count * ratios[split] for cls, count in total_classes.items()} for split in ratios}
    current_images: Counter[str] = Counter()
    current_classes: dict[str, Counter[int]] = {split: Counter() for split in ratios}

    rng = random.Random(seed)
    group_names = list(groups)
    rng.shuffle(group_names)
    group_names.sort(
        key=lambda name: (
            -sum(count / math.sqrt(max(total_classes[cls], 1)) for cls, count in class_counts(groups[name]).items()),
            -len(groups[name]),
            name,
        )
    )

    assignments: dict[str, str] = {}
    for group_name in group_names:
        group_records = groups[group_name]
        group_count = class_counts(group_records)
        group_size = len(group_records)
        candidates = active_splits
        underfilled = [
            split
            for split in candidates
            if current_images[split] + group_size <= target_images[split] + max(2.0, group_size * 0.5)
        ]
        if underfilled:
            candidates = underfilled

        def score(split: str) -> float:
            new_images = current_images[split] + group_size
            image_error = ((new_images - target_images[split]) / max(target_images[split], 1.0)) ** 2
            class_error = sum(
                (
                    (current_classes[split][cls] + group_count[cls] - target) / max(target, 1.0)
                )
                ** 2
                for cls, target in target_classes[split].items()
            )
            empty_penalty = 0.05 if not group_count and split != "train" else 0.0
            return image_error * 0.8 + class_error + empty_penalty

        selected = min(candidates, key=lambda split: (score(split), split))
        assignments[group_name] = selected
        current_images[selected] += group_size
        current_classes[selected].update(group_count)

    # Keep rare but sufficiently represented classes visible in both
    # evaluation splits. This only moves an eligible train group and never
    # breaks the sequence/view grouping invariant.
    for split in ("val", "test"):
        if ratios[split] <= 0:
            continue
        for cls in total_classes:
            groups_with_class = [
                name for name, group_records in groups.items() if class_counts(group_records)[cls] > 0
            ]
            if len(groups_with_class) < 2:
                continue
            if any(
                assignments[name] == split and class_counts(groups[name])[cls] > 0 for name in groups_with_class
            ):
                continue
            candidates = [name for name in groups_with_class if assignments[name] == "train"]
            if candidates:
                selected = min(candidates, key=lambda name: (len(groups[name]), name))
                assignments[selected] = split
    # Guarantee a non-empty requested split when the greedy score put every
    # group elsewhere. The source groups are never deleted; only ownership is
    # changed in the generated manifest.
    for split in active_splits:
        if any(value == split for value in assignments.values()):
            continue
        donor_candidates = [name for name, value in assignments.items() if value == "train"]
        if not donor_candidates:
            donor_candidates = list(assignments)
        donor = max(donor_candidates, key=lambda name: len(groups[name]))
        assignments[donor] = split
    return {record.image.name: assignments[record.group] for record in records}


def source_split_assignments(records: list[Record], root: Path) -> dict[str, str]:
    split_sets = {split: read_split_names(root, split) for split in ("train", "val", "test")}
    assignments: dict[str, str] = {}
    for record in records:
        name = record.image.name.lower()
        chosen = next((split for split in ("val", "test", "train") if name in split_sets[split]), "train")
        assignments[record.image.name] = chosen
        record.source_split = chosen
    return assignments


def tile_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if tile_size <= 0 or stride <= 0 or length <= tile_size:
        return [0]
    last = length - tile_size
    positions = list(range(0, last + 1, stride))
    if positions[-1] != last:
        positions.append(last)
    return positions


def labels_to_pixel_boxes(labels: Iterable[tuple[int, float, float, float, float]], width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    return [
        (
            cls,
            (cx - box_width / 2) * width,
            (cy - box_height / 2) * height,
            (cx + box_width / 2) * width,
            (cy + box_height / 2) * height,
        )
        for cls, cx, cy, box_width, box_height in labels
    ]


def make_tiles(
    record: Record,
    output_images: Path,
    output_labels: Path,
    tile_size: int,
    stride: int,
    min_visibility: float,
) -> list[str]:
    if not record.labels:
        return []
    created: list[str] = []
    try:
        with Image.open(record.image) as image:
            source_image = image.convert("RGB")
            width, height = source_image.size
            boxes = labels_to_pixel_boxes(record.labels, width, height)
            for top in tile_positions(height, tile_size, stride):
                for left in tile_positions(width, tile_size, stride):
                    right, bottom = min(left + tile_size, width), min(top + tile_size, height)
                    tile_labels: list[tuple[int, float, float, float, float]] = []
                    for cls, x1, y1, x2, y2 in boxes:
                        original_area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
                        ix1, iy1 = max(x1, left), max(y1, top)
                        ix2, iy2 = min(x2, right), min(y2, bottom)
                        intersection = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
                        if original_area <= 0 or intersection / original_area < min_visibility:
                            continue
                        tile_width, tile_height = right - left, bottom - top
                        tile_labels.append(
                            (
                                cls,
                                ((ix1 + ix2) / 2 - left) / tile_width,
                                ((iy1 + iy2) / 2 - top) / tile_height,
                                (ix2 - ix1) / tile_width,
                                (iy2 - iy1) / tile_height,
                            )
                        )
                    if not tile_labels:
                        continue
                    tile_stem = f"{record.image.stem}__tile_x{left}_y{top}"
                    suffix = record.image.suffix.lower()
                    tile_name = f"{tile_stem}{suffix}"
                    image_path = output_images / tile_name
                    label_path = output_labels / f"{tile_stem}.txt"
                    tile_image = source_image.crop((left, top, right, bottom))
                    save_kwargs = {"quality": 95} if suffix in {".jpg", ".jpeg"} else {}
                    tile_image.save(image_path, **save_kwargs)
                    tile_image.close()
                    write_labels(label_path, tile_labels)
                    created.append(f"images/{tile_name}")
            source_image.close()
    except OSError as exc:
        fail(f"cannot tile image {record.image}: {exc}")
    return created


def relative_image_path(name: str) -> str:
    return f"images/{name}"


def write_lines(path: Path, lines: Iterable[str]) -> None:
    values = list(lines)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def build_balanced_train_list(
    entries: list[tuple[str, list[tuple[int, float, float, float, float]]]],
    empty_ratio: float,
    max_repeat: int,
    seed: int,
) -> tuple[list[str], dict[str, int]]:
    if not 0 <= empty_ratio <= 1:
        fail("empty-train-ratio must be between 0 and 1")
    rng = random.Random(seed)
    empty = [entry for entry in entries if not entry[1]]
    positive = [entry for entry in entries if entry[1]]
    keep_empty_count = min(len(empty), int(math.ceil(len(empty) * empty_ratio)))
    rng.shuffle(empty)
    selected = positive + empty[:keep_empty_count]
    counts = class_counts([Record(image=Path(name), source_split=None, labels=labels) for name, labels in selected])
    max_count = max(counts.values(), default=1)
    result: list[str] = []
    repeat_counts: dict[str, int] = {}
    for name, labels in selected:
        present_counts = [counts[label[0]] for label in labels if label[0] in counts]
        rarest = min(present_counts, default=max_count)
        repeat = max(1, min(max_repeat, int(math.ceil(math.sqrt(max_count / max(rarest, 1))))))
        repeat_counts[name] = repeat
        result.extend([name] * repeat)
    rng.shuffle(result)
    return result, repeat_counts


def write_manifest(
    output: Path,
    spec: DatasetSpec,
    source: Path,
    records: list[Record],
    assignments: dict[str, str],
    train_entries: list[str],
    repeat_counts: dict[str, int],
    tile_entries: list[str],
    args: argparse.Namespace,
) -> None:
    split_stats = {}
    for split in ("train", "val", "test"):
        selected = [record for record in records if assignments[record.image.name] == split]
        split_stats[split] = {
            "images": len(selected),
            "empty_images": sum(not record.labels for record in selected),
            "boxes": sum(len(record.labels or []) for record in selected),
            "class_counts": dict(sorted(class_counts(selected).items())),
        }
    payload = {
        "source_root": str(source),
        "output_root": str(output),
        "dataset": spec.key,
        "class_names": list(spec.names),
        "source_to_target": {str(key): value for key, value in sorted(spec.source_to_target.items())},
        "split_mode": args.split_mode,
        "seed": args.seed,
        "split_grouping": "cube scene prefix / SUA contiguous filename sequence",
        "split_stats": split_stats,
        "train_unique_entries": len(set(train_entries)),
        "train_manifest_entries": len(train_entries),
        "train_oversample_max": max(repeat_counts.values(), default=1),
        "tile_count": len(tile_entries),
        "tile_size": args.tile_size if args.tile_size is not None else spec.tile_size,
        "tile_stride": args.tile_stride if args.tile_stride is not None else spec.tile_stride,
        "empty_train_ratio": args.empty_train_ratio,
        "source_unchanged": True,
    }
    (output / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_dataset(args: argparse.Namespace) -> Path:
    spec = SPECS[args.dataset]
    source = args.source_root / spec.source_dir
    output = args.output_root / spec.output_dir
    ensure_distinct_roots(source, output)
    if not source.is_dir():
        fail(f"source dataset does not exist: {source}")
    if output.exists():
        if not args.force:
            fail(f"output already exists; refusing to overwrite: {output} (use --force only for a generated _update directory)")
        if output.is_file() or output.is_symlink():
            fail(f"output is not a directory: {output}")
        if output.name not in {"sua_t_update", "sua_rgb_update", "bp_cube_update"}:
            fail(f"--force is restricted to known HMT update directory names: {output}")
        shutil.rmtree(output)

    images = image_files(source)
    if not images:
        fail(f"no images found: {source / 'images'}")
    labels_root = source / "labels"
    if not labels_root.is_dir():
        fail(f"missing labels directory: {labels_root}")

    source_splits = {split: read_split_names(source, split) for split in ("train", "val", "test")}
    records: list[Record] = []
    for image in images:
        labels = read_labels(labels_root / f"{image.stem}.txt", spec.source_to_target)
        source_split = next((split for split in ("val", "test", "train") if image.name.lower() in source_splits[split]), None)
        records.append(Record(image=image, source_split=source_split, labels=labels))

    group_key_for_records(records, args.dataset, args.sequence_group_size)
    assignments = source_split_assignments(records, source) if args.split_mode == "source" else assign_group_splits(
        records, args.val_ratio, args.test_ratio, args.seed
    )

    output_images = output / "images"
    output_labels = output / "labels"
    output_reports = output / "reports"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    output_reports.mkdir(parents=True, exist_ok=True)
    for record in records:
        shutil.copy2(record.image, output_images / record.image.name)
        write_labels(output_labels / f"{record.image.stem}.txt", record.labels or [])

    entry_labels: dict[str, list[tuple[int, float, float, float, float]]] = {
        relative_image_path(record.image.name): record.labels or []
        for record in records
        if assignments[record.image.name] == "train"
    }
    tile_entries: list[str] = []
    if spec.tile and not args.no_tiles:
        tile_size = args.tile_size or spec.tile_size
        tile_stride = args.tile_stride or spec.tile_stride
        for record in records:
            if assignments[record.image.name] != "train" or not record.labels:
                continue
            created = make_tiles(record, output_images, output_labels, tile_size, tile_stride, args.min_visibility)
            tile_entries.extend(created)
            for entry in created:
                label_path = output_labels / f"{Path(entry).stem}.txt"
                identity_map = {index: index for index in range(len(spec.names))}
                entry_labels[entry] = read_labels(label_path, identity_map)

    balanced_entries, repeat_counts = build_balanced_train_list(
        list(entry_labels.items()), args.empty_train_ratio, max(args.max_repeat, 1), args.seed
    )
    split_entries = {
        split: [relative_image_path(record.image.name) for record in records if assignments[record.image.name] == split]
        for split in ("train", "val", "test")
    }
    write_lines(output / "train.txt", split_entries["train"])
    write_lines(output / "train_balanced.txt", balanced_entries)
    write_lines(output / "val.txt", split_entries["val"])
    write_lines(output / "test.txt", split_entries["test"])
    write_lines(output / "all.txt", [relative_image_path(record.image.name) for record in records])
    for split in ("train", "val", "test"):
        write_lines(
            output / f"source_{split}.txt",
            [relative_image_path(record.image.name) for record in records if record.source_split == split],
        )
    write_lines(output / "class.txt", spec.names)
    (output / "label_map.json").write_text(
        json.dumps(
            {
                "source_classes_are_one_based": True,
                "source_to_target": {str(key): value for key, value in sorted(spec.source_to_target.items())},
                "names": {str(index): name for index, name in enumerate(spec.names)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_manifest(output, spec, source, records, assignments, balanced_entries, repeat_counts, tile_entries, args)
    return output


def main() -> int:
    args = parse_args()
    output = prepare_dataset(args)
    print(f"[hmt_prepare] created {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
