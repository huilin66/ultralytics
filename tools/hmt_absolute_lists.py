# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert YOLO image-list files to absolute image paths in a copied dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


LIST_NAMES = ("train.txt", "train_balanced.txt", "val.txt", "test.txt", "all.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def absolute_image_path(dataset_root: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (dataset_root / path).resolve())


def convert_file(dataset_root: Path, path: Path, dry_run: bool) -> int:
    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    converted = [absolute_image_path(dataset_root, line.strip()) for line in lines if line.strip()]
    if not dry_run:
        path.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")
    return sum(old.strip() != new for old, new in zip(lines, converted))


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset root not found: {dataset_root}")
    files = [dataset_root / name for name in LIST_NAMES]
    files.extend(sorted(dataset_root.glob("source_*.txt")))
    changed = sum(convert_file(dataset_root, path, args.dry_run) for path in files)
    action = "would convert" if args.dry_run else "converted"
    print(f"[hmt_absolute_lists] {action} {changed} entries under {dataset_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
