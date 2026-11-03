# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build an Ultralytics image-classification dataset from cropped images.

The source is expected to be a single directory whose immediate subdirectories
are the class names, e.g.::

    crop/
      circle/     Nanjing_B_ladybug_panoramic_005207_1.jpg
      octagon/    ...
      polygon/    ...
      square/     ...
      triangle/   ...

The output is the layout consumed by ``yolo classify train data=<out>``::

    <out>/
      train/<class>/<image>
      val/<class>/<image>
      test/<class>/<image>       # only when a test split is assigned
      classification_metadata.json

Splits are assigned per *source image*, not per crop.  A crop filename must end
with ``_<instance>`` (all filenames in this release do), so all crops cut from
the same frame stay in the same split.  When the detection dataset ships
``train.txt``/``val.txt`` (``test.txt`` may be empty), those official image
splits are reused; otherwise a deterministic hash split over source images is
used.

Examples:
    python tools/build_cls_dataset.py \\
        --src 'Y:/dataset/WHU_Infra2D_cube_yolo_shape_core/ydm_vis/crop' \\
        --out 'Y:/dataset/WHU_Infra2D_cube_yolo_shape_core/ydm_vis/crop_split' \\
        --split-dir 'Y:/dataset/WHU_Infra2D_cube_yolo_shape_core' --overwrite

    python tools/build_cls_dataset.py --src crop --out crop_split --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, type=Path, help="crop root; immediate subdirectories are classes")
    parser.add_argument("--out", required=True, type=Path, help="output root passed as data= to yolo classify")
    parser.add_argument(
        "--split-dir",
        type=Path,
        help="directory containing train.txt/val.txt/test.txt with source image paths (optional)",
    )
    parser.add_argument("--ratios", default="0.8,0.1,0.1", help="train,val,test ratios for the hash fallback split")
    parser.add_argument("--seed", default="mdet", help="seed mixed into the hash fallback split")
    parser.add_argument("--mode", choices=("copy", "symlink", "hardlink"), default="copy", help="how files are placed")
    parser.add_argument("--overwrite", action="store_true", help="delete existing train/val/test before writing")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without writing files")
    return parser.parse_args()


def parse_ratios(text: str) -> tuple[float, float, float]:
    """Parse and normalise the ``train,val,test`` ratio triple."""
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 3 or any(v < 0 for v in parts) or sum(parts) <= 0:
        raise SystemExit("--ratios expects three non-negative floats summing to > 0, e.g. 0.8,0.1,0.1")
    total = sum(parts)
    return tuple(v / total for v in parts)  # type: ignore[return-value]


def read_split_lists(split_dir: Path) -> dict[str, str]:
    """Read ``train.txt``/``val.txt``/``test.txt`` and return ``source_stem -> split``.

    Lines are image paths; only the file stem is kept.  A stem assigned to more
    than one split is a hard error: it would leak the same frame into two splits.
    """
    mapping: dict[str, str] = {}
    conflicts: list[tuple[str, str, str]] = []
    found = 0
    for split in SPLIT_NAMES:
        list_path = split_dir / f"{split}.txt"
        if not list_path.is_file():
            continue
        found += 1
        for line in list_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            stem = Path(line).stem
            if not stem:
                continue
            previous = mapping.get(stem)
            if previous and previous != split:
                conflicts.append((stem, previous, split))
            else:
                mapping[stem] = split
    if not found:
        raise SystemExit(f"no train.txt/val.txt/test.txt found under {split_dir}")
    if conflicts:
        examples = ", ".join(f"{stem} ({a}|{b})" for stem, a, b in conflicts[:5])
        raise SystemExit(f"{len(conflicts)} source images appear in more than one split, e.g. {examples}")
    return mapping


def source_stem(stem: str) -> str:
    """Strip the trailing ``_<instance>`` from a crop filename stem."""
    base, _sep, index = stem.rpartition("_")
    if not base or not index.isdigit():
        raise ValueError(f"crop filename '{stem}' does not end with _<instance>")
    return base


def collect_samples(src: Path) -> list[tuple[str, Path, str]]:
    """Return ``(class_name, image_path, source_stem)`` for every crop image."""
    classes = sorted(p.name for p in src.iterdir() if p.is_dir())
    if not classes:
        raise SystemExit(f"no class subdirectories found under {src}")
    samples: list[tuple[str, Path, str]] = []
    for class_name in classes:
        files = sorted(
            p for p in (src / class_name).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not files:
            raise SystemExit(f"class '{class_name}' contains no images")
        for path in files:
            try:
                samples.append((class_name, path, source_stem(path.stem)))
            except ValueError as error:
                raise SystemExit(str(error)) from error
    return samples


def hash_split(source: str, ratios: tuple[float, float, float], seed: str) -> str:
    """Deterministically map one source image to a split from a hash of its stem."""
    digest = hashlib.sha256(f"{seed}:{source}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    train, val, _test = ratios
    if value < train:
        return "train"
    if value < train + val:
        return "val"
    return "test"


def assign_splits(
    samples: list[tuple[str, Path, str]],
    split_map: dict[str, str] | None,
    ratios: tuple[float, float, float],
    seed: str,
) -> tuple[list[tuple[str, Path, str, str]], list[str]]:
    """Attach a split to every sample and return warnings."""
    warnings: list[str] = []
    assigned: list[tuple[str, Path, str, str]] = []
    unassigned: Counter[str] = Counter()
    for class_name, path, source in samples:
        split = split_map.get(source) if split_map is not None else None
        if split is None:
            unassigned[source] += 1
            split = hash_split(source, ratios, seed)
        assigned.append((class_name, path, source, split))
    if unassigned:
        warnings.append(
            f"{len(unassigned)} source images ({sum(unassigned.values())} crops) are absent from the split "
            f"lists and were assigned by the hash fallback, e.g. {', '.join(list(unassigned)[:3])}"
        )
    return assigned, warnings


def hash_file(path: Path) -> str:
    """Return the SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def copy_hashing(src: Path, dst: Path) -> str:
    """Copy a file while hashing it, returning the SHA-256 of the content."""
    digest = hashlib.sha256()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as source, dst.open("wb") as target:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
            target.write(chunk)
    shutil.copystat(src, dst)
    return digest.hexdigest()


def place_file(src: Path, dst: Path, mode: str) -> str:
    """Copy/link one image into the output tree and return its SHA-256."""
    if mode == "copy":
        return copy_hashing(src, dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    else:
        os.link(src, dst)
    return hash_file(src)


def clear_output(out: Path) -> None:
    """Remove the split trees written by a previous run."""
    for split in SPLIT_NAMES:
        shutil.rmtree(out / split, ignore_errors=True)
    metadata = out / "classification_metadata.json"
    if metadata.exists():
        metadata.unlink()


def main() -> int:
    """Build the dataset and print a summary."""
    args = parse_args()
    src = args.src.resolve()
    out = args.out.resolve()
    if not src.is_dir():
        raise SystemExit(f"source directory not found: {src}")
    if out == src:
        raise SystemExit("--out must differ from --src")

    ratios = parse_ratios(args.ratios)
    split_map = read_split_lists(args.split_dir.resolve()) if args.split_dir else None
    split_protocol = (
        "official detection split reused at source-image level"
        if split_map is not None
        else f"deterministic hash split over source images (ratios={ratios}, seed={args.seed})"
    )

    samples = collect_samples(src)
    assigned, warnings = assign_splits(samples, split_map, ratios, args.seed)
    classes = sorted({class_name for class_name, *_ in assigned})
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLIT_NAMES}
    sources: dict[str, set[str]] = {split: set() for split in SPLIT_NAMES}
    for class_name, _path, source, split in assigned:
        counts[split][class_name] += 1
        sources[split].add(source)

    print(f"[plan] {len(samples)} crops in {len(classes)} classes: {', '.join(classes)}")
    print(f"[plan] split protocol: {split_protocol}")
    for split in SPLIT_NAMES:
        if counts[split]:
            detail = ", ".join(f"{name}={counts[split][name]}" for name in classes)
            print(f"[plan] {split}: {sum(counts[split].values())} crops / {len(sources[split])} source images ({detail})")

    # Class directories must exist in every split: Ultralytics derives the class
    # index from the train folder and errors when val/test has a different count.
    for split in ("val", "test"):
        missing = [name for name in classes if counts[split][name] == 0]
        if counts[split] and missing:
            raise SystemExit(f"split '{split}' has no images for {missing}; every split must contain every class")
    if not counts["val"]:
        warnings.append("no validation split assigned; Ultralytics will fall back to test or fail validate()")
    for name in classes:
        if counts["train"][name] < 10:
            warnings.append(f"class '{name}' has only {counts['train'][name]} training crops")

    if args.dry_run:
        print("[dry-run] nothing written")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    if any((out / split).exists() for split in SPLIT_NAMES):
        if not args.overwrite:
            raise SystemExit(f"{out} already contains split folders; pass --overwrite to rebuild")
        clear_output(out)

    hashes: dict[str, dict[str, list[str]]] = {split: defaultdict(list) for split in SPLIT_NAMES}
    for index, (class_name, path, _source, split) in enumerate(assigned, start=1):
        dst = out / split / class_name / path.name
        if dst.exists():
            raise SystemExit(f"duplicate output path: {dst}")
        hashes[split][place_file(path, dst, args.mode)].append(f"{split}/{class_name}/{path.name}")
        if index % 1000 == 0 or index == len(assigned):
            print(f"[build] {index}/{len(assigned)} files placed", flush=True)

    # Exact-content duplicates are the only leakage that can be proven from the
    # files alone; near-duplicate frames need a perceptual hash and are not checked.
    seen: dict[str, str] = {}
    cross_split: list[str] = []
    for split in SPLIT_NAMES:
        for digest, names in hashes[split].items():
            if digest in seen and seen[digest] != split:
                cross_split.append(", ".join(names))
            else:
                seen[digest] = split
    if cross_split:
        warnings.append(
            f"{len(cross_split)} crop file(s) are byte-identical across splits, e.g. {cross_split[0]}"
        )

    metadata = {
        "task": "classify",
        "source": str(src),
        "root": str(out),
        "mode": args.mode,
        "classes": classes,
        "split_protocol": split_protocol,
        "split_lists": str(args.split_dir.resolve()) if args.split_dir else None,
        "counts": {split: dict(counts[split]) for split in SPLIT_NAMES if counts[split]},
        "source_images": {split: len(sources[split]) for split in SPLIT_NAMES if counts[split]},
        "cross_split_duplicates": len(cross_split),
        "warnings": warnings,
    }
    (out / "classification_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {out}")
    print(f"[yaml] not required; train with: yolo classify train data={out}")
    for warning in warnings:
        print(f"  ! {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
