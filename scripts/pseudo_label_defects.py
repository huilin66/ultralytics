"""Generate defect pseudo labels from billboard crops with a trained classifier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def read_crop_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run defect classifier over billboard crops and write pseudo labels.")
    parser.add_argument("--crops-dir", required=True, help="Directory with billboard crop images.")
    parser.add_argument("--crop-manifest", default="", help="Optional crop_manifest.csv from extract_billboard_crops.py.")
    parser.add_argument("--weights", required=True, help="Trained Ultralytics classification weights.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--conf", type=float, default=0.5, help="Minimum confidence for accepted pseudo labels.")
    parser.add_argument("--max-images", type=int, default=0, help="Optional limit for testing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    crops_dir = Path(args.crops_dir)
    weights = Path(args.weights)
    output = Path(args.output)

    if not crops_dir.exists():
        raise FileNotFoundError(f"crops dir not found: {crops_dir}")
    if not weights.exists():
        raise FileNotFoundError(f"classifier weights not found: {weights}")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    if args.crop_manifest:
        manifest_rows = read_crop_manifest(Path(args.crop_manifest))
        images = [Path(row["crop_path"]) for row in manifest_rows]
    else:
        images = list(iter_images(crops_dir))
        manifest_rows = [{"crop_path": str(path), "crop_id": path.stem} for path in images]

    if args.max_images:
        images = images[: args.max_images]
        manifest_rows = manifest_rows[: args.max_images]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "crop_id",
                "crop_path",
                "image_id",
                "image_path",
                "instance_index",
                "bbox_xyxy",
                "pseudo_class",
                "confidence",
                "accepted",
                "probs_json",
            ],
        )
        writer.writeheader()
        for image, row in zip(images, manifest_rows):
            result = model.predict(str(image), verbose=False)[0]
            probs = result.probs
            top1 = int(probs.top1)
            conf = float(probs.top1conf)
            names = result.names
            writer.writerow(
                {
                    "crop_id": row.get("crop_id", image.stem),
                    "crop_path": str(image),
                    "image_id": row.get("image_id", ""),
                    "image_path": row.get("image_path", ""),
                    "instance_index": row.get("instance_index", ""),
                    "bbox_xyxy": row.get("bbox_xyxy", ""),
                    "pseudo_class": names.get(top1, str(top1)),
                    "confidence": conf,
                    "accepted": int(conf >= args.conf),
                    "probs_json": json.dumps({names.get(i, str(i)): float(v) for i, v in enumerate(probs.data)}),
                }
            )

    print(f"[ok] wrote pseudo labels: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
