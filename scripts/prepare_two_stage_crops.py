"""Generate detector-aligned crop images and multi-label sidecar targets.

The source dataset uses the repository's mdetect annotation format. One crop
is produced per physical object. The crop sidecar contains the positive
attribute IDs only; the detector's object class remains in metadata.jsonl and
is expected to be supplied by the first stage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from attribute_dataset_utils import (
    DatasetSpec,
    ensure_output_dir,
    label_path_for_image,
    load_dataset_spec,
    parse_mdet_label_file,
    write_json,
    write_yaml,
)


def _relative_image_path(image: Path, image_root: Path) -> Path:
    """Return a stable relative source path."""

    try:
        relative = image.relative_to(image_root)
    except ValueError:
        relative = Path(image.name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Image path cannot be represented relative to {image_root}: {image}")
    return relative


def _crop_box(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    pad: float,
) -> tuple[int, int, int, int]:
    """Convert normalized xywh to a padded, clipped PIL crop box."""

    x_center, y_center, box_width, box_height = bbox
    x_width = box_width * width
    y_height = box_height * height
    pad_x = x_width * pad
    pad_y = y_height * pad
    left = max(0, math.floor((x_center * width) - x_width / 2 - pad_x))
    top = max(0, math.floor((y_center * height) - y_height / 2 - pad_y))
    right = min(width, math.ceil((x_center * width) + x_width / 2 + pad_x))
    bottom = min(height, math.ceil((y_center * height) + y_height / 2 + pad_y))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid crop box from normalized bbox {bbox}: {(left, top, right, bottom)}")
    return left, top, right, bottom


def _crop_name(relative_source: Path, object_index: int) -> Path:
    """Create a collision-resistant relative crop filename."""

    return relative_source.with_name(f"{relative_source.stem}__obj{object_index:04d}.png")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Generate crop images, sidecar labels, metadata, and classifier YAML."""

    if args.pad < 0:
        raise ValueError("--pad must be non-negative")
    spec = load_dataset_spec(args.data, image_root=args.image_root, labels_root=args.labels_root)
    output = ensure_output_dir(args.output, exist_ok=args.exist_ok)
    metadata: list[dict[str, Any]] = []
    split_counts: dict[str, dict[str, int]] = {}

    for split, images in spec.splits.items():
        if not images:
            continue
        crop_count = 0
        object_count = 0
        for image_path in images:
            source_label = label_path_for_image(image_path, spec.image_root, spec.labels_root)
            objects = parse_mdet_label_file(source_label, expected_attributes=len(spec.attribute_names))
            with Image.open(image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                width, height = image.size
                relative_source = _relative_image_path(image_path, spec.image_root)
                for object_index, obj in enumerate(objects):
                    crop_box = _crop_box(obj.bbox, width, height, args.pad)
                    crop_relative = _crop_name(relative_source, object_index)
                    crop_path = output / "images" / split / crop_relative
                    label_path = output / "labels" / split / crop_relative.with_suffix(".txt")
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    image.crop(crop_box).save(crop_path, format="PNG")
                    positive_attributes = [
                        index for index, value in enumerate(obj.attributes) if value > args.positive_threshold
                    ]
                    label_path.write_text(
                        ",".join(str(index) for index in positive_attributes) + "\n",
                        encoding="utf-8",
                    )
                    metadata.append(
                        {
                            "split": split,
                            "crop": str(crop_path),
                            "label": str(label_path),
                            "source_image": str(image_path),
                            "source_label": str(source_label),
                            "object_index": object_index,
                            "object_class": obj.object_class,
                            "object_class_name": (
                                spec.object_names[obj.object_class]
                                if 0 <= obj.object_class < len(spec.object_names)
                                else str(obj.object_class)
                            ),
                            "source_bbox_xywh": list(obj.bbox),
                            "crop_box_xyxy_pixels": list(crop_box),
                            "attribute_ids": positive_attributes,
                            "attribute_names": [
                                spec.attribute_names[index] for index in positive_attributes
                            ],
                            "pad_fraction": args.pad,
                        }
                    )
                    crop_count += 1
            object_count += len(objects)
        split_counts[split] = {"source_images": len(images), "objects": object_count, "crops": crop_count}

    yaml_data: dict[str, Any] = {
        "path": str(output),
        **{split: f"images/{split}" for split in split_counts},
        "labels": "labels",
        "names": {index: name for index, name in enumerate(spec.attribute_names)},
        "nc": len(spec.attribute_names),
        "threshold": args.threshold,
        "source_format": "mdet_object_crops",
        "source_dataset": str(spec.yaml_path),
    }
    write_yaml(output / "data.yaml", yaml_data)
    metadata_path = output / "metadata.jsonl"
    metadata_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in metadata),
        encoding="utf-8",
    )
    manifest = {
        "source_dataset": str(spec.yaml_path),
        "source_root": str(spec.root),
        "source_image_root": str(spec.image_root),
        "source_labels_root": str(spec.labels_root) if spec.labels_root else None,
        "output": str(output),
        "pad_fraction": args.pad,
        "positive_threshold": args.positive_threshold,
        "classifier_threshold": args.threshold,
        "attribute_names": list(spec.attribute_names),
        "splits": split_counts,
        "metadata": str(metadata_path),
    }
    write_json(output / "conversion_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="source mdetect dataset YAML")
    parser.add_argument("--output", required=True, help="output crop dataset directory")
    parser.add_argument("--image-root", default=None, help="optional source image root override")
    parser.add_argument("--labels-root", default=None, help="optional source label root override")
    parser.add_argument("--pad", type=float, default=0.10, help="padding as a fraction of each bbox size")
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5, help="classifier threshold in generated YAML")
    parser.add_argument("--exist-ok", action="store_true", help="reuse a non-empty generated directory")
    return parser


def main() -> None:
    """Run crop preparation."""

    args = build_parser().parse_args()
    manifest = prepare(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

