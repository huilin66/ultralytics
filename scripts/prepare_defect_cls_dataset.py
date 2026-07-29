"""Prepare or validate an Ultralytics classification dataset for defects.

Supported inputs:
- Directory with class subfolders.
- CSV file with columns ``image_path`` and ``label``.

By default this script only inventories the dataset. Pass ``--copy-files`` to
materialize a train/val folder structure.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def clean_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in label.strip())


def rows_from_class_dirs(root: Path) -> list[dict[str, str]]:
    rows = []
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = class_dir.name
        for image in class_dir.rglob("*"):
            if image.is_file() and image.suffix.lower() in IMAGE_EXTS:
                rows.append({"image_path": str(image), "label": label})
    return rows


def rows_from_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"image_path", "label"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return [{"image_path": row["image_path"], "label": row["label"]} for row in rows]


def split_rows(rows: list[dict[str, str]], val_ratio: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)

    train_rows, val_rows = [], []
    for label_rows in grouped.values():
        rng.shuffle(label_rows)
        if len(label_rows) == 1:
            train_rows.extend(label_rows)
            continue
        n_val = max(1, int(round(len(label_rows) * val_ratio)))
        val_rows.extend(label_rows[:n_val])
        train_rows.extend(label_rows[n_val:])
    return train_rows, val_rows


def copy_split(rows: list[dict[str, str]], output_dir: Path, split: str) -> int:
    copied = 0
    for i, row in enumerate(rows):
        src = Path(row["image_path"])
        if not src.exists():
            continue
        label = clean_label(row["label"])
        dst_dir = output_dir / split / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{src.stem}_{i:07d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        copied += 1
    return copied


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a defect classification dataset.")
    parser.add_argument("--source", required=True, help="Class-folder root or CSV with image_path,label.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--copy-files", action="store_true", help="Actually copy images into train/val class folders.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    output_dir = Path(args.output_dir)

    if not source.exists():
        raise FileNotFoundError(f"source not found: {source}")

    rows = rows_from_csv(source) if source.is_file() else rows_from_class_dirs(source)
    rows = [row for row in rows if row["label"].strip()]
    train_rows, val_rows = split_rows(rows, args.val_ratio, args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "all.csv", rows)
    write_rows(output_dir / "train.csv", train_rows)
    write_rows(output_dir / "val.csv", val_rows)

    counts = Counter(row["label"] for row in rows)
    with (output_dir / "class_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "count"])
        writer.writeheader()
        for label, count in sorted(counts.items()):
            writer.writerow({"label": label, "count": count})

    copied_train = copied_val = 0
    if args.copy_files:
        copied_train = copy_split(train_rows, output_dir, "train")
        copied_val = copy_split(val_rows, output_dir, "val")

    print(f"[ok] total images: {len(rows)}")
    print(f"[ok] classes: {len(counts)}")
    print(f"[ok] train/val: {len(train_rows)}/{len(val_rows)}")
    if args.copy_files:
        print(f"[ok] copied train/val: {copied_train}/{copied_val}")
    else:
        print("[dry-run] no files copied; pass --copy-files to materialize the classification dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
