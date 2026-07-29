"""Batch wrapper for the external SAM3 text-to-mask script.

The wrapper reads ``raw_image_index.csv`` from ``annotation_pipeline.py`` and
calls the reference ``sam3_text2mask.py`` one image at a time. It keeps each
image's raw SAM3 outputs in a per-image folder and also copies YOLO labels into
a flat ``labels/`` directory for downstream crop extraction and dataset export.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_stem(row: dict[str, str], image_path: Path) -> str:
    image_id = row.get("image_id") or image_path.stem
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in image_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3 text-to-mask over an image index.")
    parser.add_argument("--image-index", required=True, help="CSV created by annotation_pipeline.py.")
    parser.add_argument("--output-dir", required=True, help="SAM batch output directory.")
    parser.add_argument("--sam-script", required=True, help="Path to sam3_text2mask.py.")
    parser.add_argument("--model", required=True, help="Path to sam3.pt.")
    parser.add_argument("--prompt", default="billboard signboard advertising board", help="SAM3 text prompt.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-mask-area", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-images", type=int, default=0, help="Optional limit.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip images with an existing flat label txt.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_index = Path(args.image_index)
    output_dir = Path(args.output_dir)
    sam_script = Path(args.sam_script)
    model = Path(args.model)

    if not image_index.exists():
        raise FileNotFoundError(f"image index not found: {image_index}")
    if not sam_script.exists():
        raise FileNotFoundError(f"SAM3 script not found: {sam_script}")
    if not model.exists():
        raise FileNotFoundError(f"SAM3 model not found: {model}")

    labels_dir = output_dir / "labels"
    runs_dir = output_dir / "runs"
    labels_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    rows = read_index(image_index)
    if args.max_images:
        rows = rows[: args.max_images]

    manifest_path = output_dir / "sam3_batch_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "image_path", "run_dir", "label_path", "status", "returncode"],
        )
        writer.writeheader()
        for row in rows:
            image_path = Path(row["image_path"])
            stem = safe_stem(row, image_path)
            run_dir = runs_dir / stem
            flat_label = labels_dir / f"{image_path.stem}.txt"

            if args.skip_existing and flat_label.exists():
                writer.writerow(
                    {
                        "image_id": row.get("image_id", ""),
                        "image_path": str(image_path),
                        "run_dir": str(run_dir),
                        "label_path": str(flat_label),
                        "status": "skipped_existing",
                        "returncode": 0,
                    }
                )
                continue

            cmd = [
                sys.executable,
                str(sam_script),
                "--image",
                str(image_path),
                "--model",
                str(model),
                "--output-dir",
                str(run_dir),
                "--prompt",
                args.prompt,
                "--conf",
                str(args.conf),
                "--mask-threshold",
                str(args.mask_threshold),
                "--min-mask-area",
                str(args.min_mask_area),
                "--device",
                args.device,
            ]

            proc = subprocess.run(cmd, check=False)
            produced_label = run_dir / "labels" / f"{image_path.stem}.txt"
            status = "ok" if proc.returncode == 0 else "failed"
            if produced_label.exists():
                shutil.copy2(produced_label, flat_label)
            elif proc.returncode == 0:
                flat_label.touch()
                status = "ok_empty"

            writer.writerow(
                {
                    "image_id": row.get("image_id", ""),
                    "image_path": str(image_path),
                    "run_dir": str(run_dir),
                    "label_path": str(flat_label),
                    "status": status,
                    "returncode": proc.returncode,
                }
            )

            if proc.returncode != 0:
                print(f"[warning] SAM3 failed for {image_path} with return code {proc.returncode}", file=sys.stderr)

    print(f"[ok] wrote SAM3 batch manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
