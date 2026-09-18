"""Convert mdetect labels into one-box, n-hot multi-label detection data.

The source format is the repository's mdetect format:

    object_class attribute_count attr_0 ... attr_N-1 x_center y_center width height

The default output preserves both the object class and positive attributes in
one label combination. For a two-class, ten-attribute dataset the resulting
label space therefore has twelve labels. Each physical object is written
exactly once; no duplicate boxes are created for individual attributes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from attribute_dataset_utils import (
    DatasetSpec,
    copy_or_link,
    ensure_output_dir,
    label_path_for_image,
    load_dataset_spec,
    parse_mdet_label_file,
    write_json,
    write_yaml,
)


def _relative_image_path(image: Path, image_root: Path) -> Path:
    """Return a stable relative image path for the generated dataset."""

    try:
        relative = image.relative_to(image_root)
    except ValueError:
        relative = Path(image.name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Image path cannot be represented relative to {image_root}: {image}")
    return relative


def _build_names(spec: DatasetSpec, *, attributes_only: bool, empty_label_name: str | None) -> list[str]:
    names = [] if attributes_only else [f"object_{name}" for name in spec.object_names]
    names.extend(f"attribute_{name}" for name in spec.attribute_names)
    if attributes_only and empty_label_name:
        names.append(empty_label_name)
    return names


def _object_label_ids(
    object_class: int,
    attributes: tuple[float, ...],
    *,
    object_count: int,
    attributes_only: bool,
    positive_threshold: float,
    empty_label_id: int | None,
) -> list[int]:
    """Map one mdetect object to one multi-label class combination."""

    ids = [] if attributes_only else [object_class]
    attribute_offset = 0 if attributes_only else object_count
    ids.extend(attribute_offset + index for index, value in enumerate(attributes) if value > positive_threshold)
    if not ids:
        if empty_label_id is None:
            raise ValueError(
                "An object has no positive labels in attributes-only mode. "
                "Use the default combined mode or pass --empty-label-name."
            )
        ids.append(empty_label_id)
    return sorted(set(ids))


def convert(args: argparse.Namespace) -> dict[str, Any]:
    """Convert all requested dataset splits and return a summary."""

    spec = load_dataset_spec(args.data, image_root=args.image_root, labels_root=args.labels_root)
    output = ensure_output_dir(args.output, exist_ok=args.exist_ok)
    attributes_only = bool(args.attributes_only)
    names = _build_names(spec, attributes_only=attributes_only, empty_label_name=args.empty_label_name)
    empty_label_id = len(names) - 1 if attributes_only and args.empty_label_name else None
    split_counts: dict[str, dict[str, int]] = {}
    output_splits: dict[str, str] = {}

    for split, images in spec.splits.items():
        if not images:
            continue
        split_entries: list[str] = []
        object_count = 0
        label_count = 0
        for image in images:
            relative = _relative_image_path(image, spec.image_root)
            destination_image = output / "images" / split / relative
            destination_label = output / "labels" / split / relative.with_suffix(".txt")
            copy_or_link(image, destination_image, args.image_mode)
            source_label = label_path_for_image(image, spec.image_root, spec.labels_root)
            objects = parse_mdet_label_file(source_label, expected_attributes=len(spec.attribute_names))
            lines = []
            for obj in objects:
                ids = _object_label_ids(
                    obj.object_class,
                    obj.attributes,
                    object_count=len(spec.object_names),
                    attributes_only=attributes_only,
                    positive_threshold=args.positive_threshold,
                    empty_label_id=empty_label_id,
                )
                if any(class_id < 0 or class_id >= len(names) for class_id in ids):
                    raise ValueError(f"Generated class ID outside output names for {source_label}:{obj.line_number}")
                lines.append(
                    ",".join(str(class_id) for class_id in ids)
                    + " "
                    + " ".join(f"{value:.8f}" for value in obj.bbox)
                )
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            destination_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            split_entries.append(str(destination_image))
            object_count += len(objects)
            label_count += sum(1 for line in lines if line)

        split_file = output / f"{split}.txt"
        split_file.write_text("\n".join(split_entries) + "\n", encoding="utf-8")
        output_splits[split] = f"{split}.txt"
        split_counts[split] = {"images": len(images), "objects": object_count, "label_rows": label_count}

    yaml_data: dict[str, Any] = {
        "path": str(output),
        **output_splits,
        "names": {index: name for index, name in enumerate(names)},
        "nc": len(names),
        "source_format": "mdet_to_one_box_nhot",
        "source_dataset": str(spec.yaml_path),
    }
    write_yaml(output / "data.yaml", yaml_data)
    manifest = {
        "source_dataset": str(spec.yaml_path),
        "source_root": str(spec.root),
        "source_image_root": str(spec.image_root),
        "source_labels_root": str(spec.labels_root) if spec.labels_root else None,
        "output": str(output),
        "mode": "attributes_only" if attributes_only else "object_class_plus_attributes",
        "positive_threshold": args.positive_threshold,
        "image_mode": args.image_mode,
        "names": yaml_data["names"],
        "splits": split_counts,
    }
    write_json(output / "conversion_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="source mdetect dataset YAML")
    parser.add_argument("--output", required=True, help="output dataset directory")
    parser.add_argument("--image-root", default=None, help="optional source image root override")
    parser.add_argument("--labels-root", default=None, help="optional source label root override")
    parser.add_argument("--image-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--attributes-only", action="store_true", help="omit object classes from the n-hot space")
    parser.add_argument(
        "--empty-label-name",
        default=None,
        help="label used for all-negative objects in attributes-only mode",
    )
    parser.add_argument("--positive-threshold", type=float, default=0.5)
    parser.add_argument("--exist-ok", action="store_true", help="reuse a non-empty generated directory")
    return parser


def main() -> None:
    """Run the converter."""

    args = build_parser().parse_args()
    manifest = convert(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

