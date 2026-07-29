"""Export a YOLO dataset skeleton from image index, SAM labels, and pseudo labels."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import yaml


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_pseudo_labels(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    by_crop = {row.get("crop_path", ""): row for row in rows if row.get("crop_path")}
    by_instance = {
        (row.get("image_path", ""), row.get("instance_index", "")): row
        for row in rows
        if row.get("image_path") and row.get("instance_index")
    }
    return {"by_crop": by_crop, "by_instance": by_instance, "rows": rows}


def safe_stem(row: dict[str, str], image_path: Path) -> str:
    image_id = row.get("image_id") or image_path.stem
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in image_id)


def find_label_file(labels_root: Path, image_path: Path, row: dict[str, str]) -> Path | None:
    candidates = [
        labels_root / f"{image_path.stem}.txt",
        labels_root / "labels" / f"{image_path.stem}.txt",
        labels_root / "runs" / safe_stem(row, image_path) / "labels" / f"{image_path.stem}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def copy_label(label_file: Path, output_label: Path) -> tuple[str, int]:
    output_label.parent.mkdir(parents=True, exist_ok=True)
    lines = [line for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    output_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return "copied_yolo_txt", len(lines)


def export_json_mask(mask_file: Path, label_file: Path) -> tuple[str, int]:
    data = json.loads(mask_file.read_text(encoding="utf-8"))
    polygons = data.get("polygons") or data.get("segments") or []
    label_file.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with label_file.open("w", encoding="utf-8") as f:
        for polygon in polygons:
            class_id = int(polygon.get("class_id", 0)) if isinstance(polygon, dict) else 0
            points = polygon.get("points", polygon) if isinstance(polygon, dict) else polygon
            flat = []
            for point in points:
                flat.extend([float(point[0]), float(point[1])])
            if flat:
                f.write(" ".join([str(class_id), *[f"{v:.6f}" for v in flat]]) + "\n")
                n += 1
    return "converted_json_polygons", n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO annotation dataset skeleton.")
    parser.add_argument("--image-index", required=True)
    parser.add_argument("--labels-dir", required=True, help="Flat labels dir or SAM batch output dir.")
    parser.add_argument("--pseudo-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--copy-images", action="store_true", help="Copy images into output/images instead of writing image list.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_index = Path(args.image_index)
    labels_root = Path(args.labels_dir)
    output_dir = Path(args.output_dir)

    rows = read_csv(image_index)
    pseudo_labels = load_pseudo_labels(Path(args.pseudo_labels))
    labels_dir = output_dir / "labels"
    images_dir = output_dir / "images"
    review_rows = []
    image_list = []

    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        image_path = Path(row["image_path"])
        image_list.append(str(image_path if not args.copy_images else images_dir / image_path.name))
        if args.copy_images and image_path.exists():
            shutil.copy2(image_path, images_dir / image_path.name)

        label_file = find_label_file(labels_root, image_path, row)
        output_label = labels_dir / f"{image_path.stem}.txt"
        if label_file is None:
            review_rows.append(
                {"image_path": str(image_path), "status": "missing_label", "label_file": "", "instances": 0}
            )
            continue
        if label_file.suffix.lower() == ".txt":
            status, instances = copy_label(label_file, output_label)
        elif label_file.suffix.lower() == ".json":
            status, instances = export_json_mask(label_file, output_label)
        else:
            status, instances = "unsupported_label_format", 0
        review_rows.append(
            {"image_path": str(image_path), "status": status, "label_file": str(label_file), "instances": instances}
        )

    (output_dir / "images.txt").write_text("\n".join(image_list), encoding="utf-8")
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images.txt",
        "val": "images.txt",
        "names": {0: "billboard"},
        "pseudo_labels": str(Path(args.pseudo_labels)),
        "pseudo_label_count": len(pseudo_labels["rows"]),
    }
    (output_dir / "billboard_seg.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    with (output_dir / "export_review.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "status", "label_file", "instances"])
        writer.writeheader()
        writer.writerows(review_rows)

    print(f"[ok] wrote YOLO dataset skeleton: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
