"""Generate reproducible EigenCAM visualizations for two detector checkpoints.

The script reuses the local heatmap implementation and does not depend on
``yolo_data_manager``.  It writes one directory per model and a horizontal
side-by-side comparison for every input image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

# ``run_experiment.sh`` executes this file from the repository root, but
# Python places ``scripts/`` first on ``sys.path`` for a file-based launch.
# Add the repository root explicitly so the local heatmap implementation is
# always used instead of requiring an installed package.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from heatmap_grad_acm import yolov8_heatmap


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def normalize_device(device: str) -> str:
    """Accept both Ultralytics-style ``0`` and PyTorch-style ``cuda:0``."""
    value = str(device).strip()
    if value.isdigit():
        return f"cuda:{value}"
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mayolo-weight", required=True, type=Path)
    parser.add_argument("--yolov10-weight", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path, help="An image file or a directory of images.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/experiments/E3_final_test/heatmaps_eigencam"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[10, 12, 14, 16, 18],
        help="Backbone/head layer indices used by EigenCAM.",
    )
    parser.add_argument("--conf", type=float, default=0.2, help="Detection confidence threshold.")
    parser.add_argument("--ratio", type=float, default=0.02, help="Fraction of top detections used by CAM.")
    parser.add_argument("--show-box", action="store_true", help="Draw detection boxes and labels.")
    parser.add_argument("--renormalize", action="store_true", help="Renormalize CAM responses inside boxes.")
    return parser.parse_args()


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        images = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if images:
            return images
    raise FileNotFoundError(f"No supported images found: {source}")


def generate_model_outputs(
    label: str,
    weight: Path,
    images: list[Path],
    output: Path,
    args: argparse.Namespace,
) -> dict[str, Path]:
    model_output = output / label
    model_output.mkdir(parents=True, exist_ok=True)
    device = normalize_device(args.device)
    heatmap = yolov8_heatmap(
        weight=str(weight),
        device=device,
        method="EigenCAM",
        layer=args.layers,
        backward_type="class",
        conf_threshold=args.conf,
        ratio=args.ratio,
        show_box=args.show_box,
        renormalize=args.renormalize,
    )

    generated: dict[str, Path] = {}
    try:
        for image in images:
            destination = model_output / f"{image.stem}_EigenCAM.png"
            heatmap.process(str(image), str(destination))
            if destination.exists():
                generated[image.name] = destination
    finally:
        activations_and_grads = getattr(heatmap.method, "activations_and_grads", None)
        if activations_and_grads is not None:
            activations_and_grads.release()
    return generated


def make_side_by_side(
    images: list[Path],
    mayolo_outputs: dict[str, Path],
    yolov10_outputs: dict[str, Path],
    output: Path,
) -> None:
    side_by_side = output / "side_by_side"
    side_by_side.mkdir(parents=True, exist_ok=True)
    for image in images:
        left_path = mayolo_outputs.get(image.name)
        right_path = yolov10_outputs.get(image.name)
        if left_path is None or right_path is None:
            print(f"[warning] Missing EigenCAM output for {image.name}; skip comparison")
            continue
        left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left is None or right is None:
            print(f"[warning] Cannot read EigenCAM output for {image.name}; skip comparison")
            continue
        if left.shape[0] != right.shape[0]:
            right = cv2.resize(right, (right.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
        comparison = np.concatenate((left, right), axis=1)
        destination = side_by_side / f"{image.stem}_MAYOLOx_vs_YOLOv10x_EigenCAM.png"
        if not cv2.imwrite(str(destination), comparison):
            raise OSError(f"Failed to write {destination}")


def main() -> None:
    args = parse_args()
    args.device = normalize_device(args.device)
    for path, name in (
        (args.mayolo_weight, "MAYOLO weight"),
        (args.yolov10_weight, "YOLOv10 weight"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")

    images = collect_images(args.images)
    args.output.mkdir(parents=True, exist_ok=True)
    mayolo_outputs = generate_model_outputs("MAYOLOx", args.mayolo_weight, images, args.output, args)
    yolov10_outputs = generate_model_outputs("YOLOv10x", args.yolov10_weight, images, args.output, args)
    make_side_by_side(images, mayolo_outputs, yolov10_outputs, args.output)

    metadata = {
        "method": "EigenCAM",
        "mayolo_weight": str(args.mayolo_weight),
        "yolov10_weight": str(args.yolov10_weight),
        "images": str(args.images),
        "device": args.device,
        "layers": args.layers,
        "backward_type": "class",
        "confidence": args.conf,
        "ratio": args.ratio,
        "show_box": args.show_box,
        "renormalize": args.renormalize,
    }
    (args.output / "eigencam_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] EigenCAM outputs: {args.output}")


if __name__ == "__main__":
    main()
