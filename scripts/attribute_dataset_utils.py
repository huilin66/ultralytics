"""Shared utilities for converting mdetect annotations into comparison datasets.

The mdetect annotation format used by this repository is:

    object_class attribute_count attr_0 ... attr_N-1 x_center y_center width height

The helpers in this module deliberately do not import the Ultralytics runtime.
They can therefore be used from a clean data-preparation environment before a
training environment is started.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class MDetObject:
    """One physical object parsed from an mdetect label row."""

    object_class: int
    attributes: tuple[float, ...]
    bbox: tuple[float, float, float, float]
    line_number: int


@dataclass(frozen=True)
class DatasetSpec:
    """Resolved source dataset metadata and image lists."""

    yaml_path: Path
    root: Path
    raw: Mapping[str, Any]
    object_names: tuple[str, ...]
    attribute_names: tuple[str, ...]
    splits: Mapping[str, tuple[Path, ...]]
    image_root: Path
    labels_root: Path | None


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping with a useful error for malformed files."""

    yaml_path = Path(path).expanduser().resolve()
    with yaml_path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {yaml_path}")
    return value


def _resolve_path(value: str | Path, root: Path, base_dir: Path) -> Path:
    """Resolve a path relative to the dataset root, then the YAML directory."""

    candidate = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if candidate.is_absolute():
        return candidate
    root_candidate = root / candidate
    base_candidate = base_dir / candidate
    if root_candidate.exists() or not base_candidate.exists():
        return root_candidate
    return base_candidate


def _resolve_split_entries(value: Any, root: Path, base_dir: Path) -> tuple[Path, ...]:
    """Resolve a YAML split entry, text list, directory, or explicit list."""

    if value is None:
        return tuple()
    values: list[Any]
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        candidate = _resolve_path(value, root, base_dir)
        if candidate.is_file() and candidate.suffix.lower() == ".txt":
            values = [line.split("#", 1)[0].strip() for line in candidate.read_text(encoding="utf-8").splitlines()]
        elif candidate.is_dir():
            values = sorted(
                path for path in candidate.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        else:
            values = [value]

    paths: list[Path] = []
    seen: set[Path] = set()
    for item in values:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        path = _resolve_path(text, root, base_dir)
        if not path.is_file():
            raise FileNotFoundError(f"Image listed by dataset split does not exist: {path}")
        path = path.resolve()
        if path not in seen:
            paths.append(path)
            seen.add(path)
    return tuple(paths)


def _normalize_names(value: Any, field: str) -> tuple[str, ...]:
    """Normalize YAML names represented as a list or integer-keyed mapping."""

    if isinstance(value, Mapping):
        pairs = sorted(((int(key), str(name)) for key, name in value.items()), key=lambda item: item[0])
        if [key for key, _ in pairs] != list(range(len(pairs))):
            raise ValueError(f"{field} IDs must be contiguous from 0: {pairs}")
        return tuple(name for _, name in pairs)
    if isinstance(value, (list, tuple)):
        return tuple(str(name) for name in value)
    raise ValueError(f"Dataset YAML must define {field!r} as a list or mapping")


def _is_relative_to(path: Path, root: Path) -> bool:
    """Backport Path.is_relative_to for older Python versions."""

    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def infer_image_root(images: Sequence[Path], dataset_root: Path) -> Path:
    """Find the directory used to preserve relative image paths in outputs."""

    candidate = dataset_root / "images"
    if images and all(_is_relative_to(path, candidate) for path in images):
        return candidate
    if not images:
        return candidate
    return Path(os.path.commonpath([str(path.parent) for path in images]))


def infer_labels_root(image_root: Path, images: Sequence[Path]) -> Path | None:
    """Infer a sibling labels directory from an image directory."""

    if image_root.name.lower() == "images":
        candidate = image_root.with_name("labels")
        if candidate.exists() or images:
            return candidate
    parts = image_root.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            return Path(*parts[:index], "labels", *parts[index + 1 :])
    return None


def load_dataset_spec(
    dataset_yaml: str | Path,
    *,
    image_root: str | Path | None = None,
    labels_root: str | Path | None = None,
    splits: Iterable[str] = ("train", "val", "test"),
) -> DatasetSpec:
    """Load a source mdetect dataset and resolve its image and label locations."""

    yaml_path = Path(dataset_yaml).expanduser().resolve()
    raw = load_yaml(yaml_path)
    root_value = raw.get("path", "")
    root = _resolve_path(root_value, yaml_path.parent, yaml_path.parent).resolve() if root_value else yaml_path.parent
    object_names = _normalize_names(raw.get("names"), "names")
    attributes = raw.get("attributes")
    if not isinstance(attributes, Mapping) or not attributes:
        raise ValueError("Source dataset YAML must define a non-empty 'attributes' mapping")
    attribute_names = tuple(str(name) for name in attributes)

    resolved_splits = {
        split: _resolve_split_entries(raw.get(split), root, yaml_path.parent)
        for split in splits
        if raw.get(split) is not None
    }
    all_images = tuple(path for split in resolved_splits.values() for path in split)
    resolved_image_root = Path(image_root).expanduser().resolve() if image_root else infer_image_root(all_images, root)
    resolved_labels_root = (
        Path(labels_root).expanduser().resolve() if labels_root else infer_labels_root(resolved_image_root, all_images)
    )
    return DatasetSpec(
        yaml_path=yaml_path,
        root=root,
        raw=raw,
        object_names=object_names,
        attribute_names=attribute_names,
        splits=resolved_splits,
        image_root=resolved_image_root,
        labels_root=resolved_labels_root,
    )


def label_path_for_image(image_path: Path, image_root: Path, labels_root: Path | None) -> Path:
    """Map an image path to its sidecar label path."""

    if labels_root is not None:
        try:
            relative = image_path.relative_to(image_root)
        except ValueError:
            relative = Path(image_path.name)
        primary = labels_root / relative.with_suffix(".txt")
        if primary.exists():
            return primary
        flat = labels_root / f"{image_path.stem}.txt"
        return flat if flat.exists() else primary

    sidecar = image_path.with_suffix(".txt")
    if sidecar.exists():
        return sidecar
    return sidecar


def parse_mdet_label_file(label_path: str | Path, *, expected_attributes: int | None = None) -> list[MDetObject]:
    """Parse the repository's class + attr_count + attrs + bbox rows."""

    path = Path(label_path)
    if not path.is_file():
        return []
    objects: list[MDetObject] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = raw_line.split()
        if not fields:
            continue
        if len(fields) < 6:
            raise ValueError(f"{path}:{line_number}: expected class, attributes, and bbox fields")
        try:
            object_class = int(float(fields[0]))
            attribute_count = int(float(fields[1]))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: class and attribute count must be integers") from exc
        if expected_attributes is not None and attribute_count != expected_attributes:
            raise ValueError(
                f"{path}:{line_number}: declared {attribute_count} attributes, "
                f"but the dataset YAML defines {expected_attributes}"
            )
        expected_fields = 2 + attribute_count + 4
        if len(fields) != expected_fields:
            raise ValueError(
                f"{path}:{line_number}: expected {expected_fields} fields for {attribute_count} attributes, "
                f"got {len(fields)}"
            )
        try:
            attributes = tuple(float(value) for value in fields[2 : 2 + attribute_count])
            bbox = tuple(float(value) for value in fields[2 + attribute_count :])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: attributes and bbox must be numeric") from exc
        if len(bbox) != 4 or any(value < 0 or value > 1 for value in bbox):
            raise ValueError(f"{path}:{line_number}: bbox must contain four normalized values in [0, 1]")
        if any(value < 0 for value in attributes):
            raise ValueError(f"{path}:{line_number}: attribute values must be non-negative")
        objects.append(MDetObject(object_class, attributes, bbox, line_number))
    return objects


def ensure_output_dir(path: str | Path, *, exist_ok: bool = False) -> Path:
    """Create an output directory without deleting an existing experiment."""

    output = Path(path).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    if output.exists() and any(output.iterdir()) and not exist_ok:
        raise FileExistsError(f"Output directory is not empty: {output}; pass --exist-ok to reuse it")
    output.mkdir(parents=True, exist_ok=True)
    return output


def copy_or_link(source: Path, destination: Path, mode: str = "symlink") -> None:
    """Materialize an image as a symlink or copy."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    if mode != "symlink":
        raise ValueError(f"Unsupported image materialization mode: {mode}")
    try:
        destination.symlink_to(source)
    except OSError:
        # Windows and some mounted filesystems disallow symlinks. Falling back
        # to a copy keeps the converter usable without changing annotations.
        shutil.copy2(source, destination)


def write_yaml(path: str | Path, data: Mapping[str, Any]) -> None:
    """Write a reproducible YAML mapping."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, allow_unicode=True, sort_keys=False)


def write_json(path: str | Path, data: Any) -> None:
    """Write UTF-8 JSON with stable formatting."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

