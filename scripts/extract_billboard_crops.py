"""Extract billboard crops from YOLO segmentation labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageOps


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_yolo_seg_line(line: str, width: int, height: int) -> tuple[int, list[tuple[float, float]]] | None:
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    class_id = int(float(parts[0]))
    coords = [float(x) for x in parts[1:]]
    if len(coords) % 2 != 0:
        coords = coords[:-1]
    points = []
    for x, y in zip(coords[0::2], coords[1::2]):
        points.append((x * width, y * height))
    return class_id, points


def polygon_bbox(points: list[tuple[float, float]], width: int, height: int, padding: float) -> tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(width, int(x2 + pad_x) + 1),
        min(height, int(y2 + pad_y) + 1),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop billboard instances from YOLO segmentation labels.")
    parser.add_argument("--image-index", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--padding", type=float, default=0.08, help="Relative bbox padding around polygon bbox.")
    parser.add_argument("--min-size", type=int, default=8, help="Minimum crop width and height.")
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_index(Path(args.image_index))
    if args.max_images:
        rows = rows[: args.max_images]

    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    crops_dir = output_dir / "images"
    crops_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "crop_manifest.csv"
    review_path = output_dir / "crop_review.csv"
    manifest_rows = []
    review_rows = []

    for row in rows:
        image_path = Path(row["image_path"])
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not image_path.exists():
            review_rows.append({"image_path": str(image_path), "label_path": str(label_path), "status": "missing_image"})
            continue
        if not label_path.exists():
            review_rows.append({"image_path": str(image_path), "label_path": str(label_path), "status": "missing_label"})
            continue

        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            width, height = im.size
            lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                review_rows.append({"image_path": str(image_path), "label_path": str(label_path), "status": "empty_label"})
                continue

            for instance_idx, line in enumerate(lines):
                parsed = parse_yolo_seg_line(line, width, height)
                if parsed is None:
                    review_rows.append(
                        {"image_path": str(image_path), "label_path": str(label_path), "status": "bad_label_line"}
                    )
                    continue
                class_id, points = parsed
                x1, y1, x2, y2 = polygon_bbox(points, width, height, args.padding)
                if x2 - x1 < args.min_size or y2 - y1 < args.min_size:
                    review_rows.append(
                        {"image_path": str(image_path), "label_path": str(label_path), "status": "crop_too_small"}
                    )
                    continue
                crop_name = f"{image_path.stem}_{instance_idx:03d}.jpg"
                crop_path = crops_dir / crop_name
                im.crop((x1, y1, x2, y2)).save(crop_path, quality=95)
                manifest_rows.append(
                    {
                        "crop_id": crop_path.stem,
                        "crop_path": str(crop_path),
                        "image_id": row.get("image_id", ""),
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "instance_index": instance_idx,
                        "class_id": class_id,
                        "bbox_xyxy": f"{x1},{y1},{x2},{y2}",
                        "width": x2 - x1,
                        "height": y2 - y1,
                    }
                )

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "crop_id",
                "crop_path",
                "image_id",
                "image_path",
                "label_path",
                "instance_index",
                "class_id",
                "bbox_xyxy",
                "width",
                "height",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    with review_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label_path", "status"])
        writer.writeheader()
        writer.writerows(review_rows)

    print(f"[ok] crops: {len(manifest_rows)}")
    print(f"[ok] wrote crop manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
