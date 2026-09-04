"""Image-level multi-label classification data loading.

This module is deliberately separate from :mod:`multilabel_yolo.dataset`.
The detector keeps one physical box per object, whereas this dataset has one
multi-hot target per image (or per cropped detection result).  No directory
name is treated as a class because a single image may contain several labels.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import cv2
import torch
from PIL import Image

from ultralytics.data.augment import classify_augmentations, classify_transforms
from ultralytics.data.build import build_dataloader
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import yaml_load


class MultiLabelClassificationLabelError(ValueError):
    """Raised when an image-level multi-label annotation is invalid."""


def _as_names(names, nc=None):
    """Normalize class names to the dictionary convention used by Ultralytics."""
    if isinstance(names, dict):
        names = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, (list, tuple)):
        names = {i: str(v) for i, v in enumerate(names)}
    else:
        raise ValueError("Multi-label classification data YAML must define 'names' as a list or dictionary")

    if nc is None:
        nc = len(names)
    nc = int(nc)
    if nc <= 0:
        raise ValueError(f"Multi-label classification requires at least one class, got nc={nc}")
    expected = set(range(nc))
    if set(names) != expected:
        raise ValueError(f"Classification names must contain exactly class IDs 0..{nc - 1}, got {sorted(names)}")
    return names, nc


def _resolve_path(value, root, base_dir):
    """Resolve a path relative to the dataset root, then the YAML directory."""
    path = Path(value)
    if path.is_absolute():
        return path
    root_candidate = root / path
    return root_candidate if root_candidate.exists() or not (base_dir / path).exists() else base_dir / path


def _split_label_root(label_root, split):
    """Resolve a label root supplied as either ``labels`` or ``labels/train``."""
    if label_root is None:
        return None
    label_root = Path(label_root)
    return label_root if label_root.name.lower() == split.lower() else label_root / split


def load_multilabel_classification_data(data):
    """Load and validate an image-level multi-label dataset YAML.

    The supported schema is intentionally small and explicit::

        path: /dataset/root
        train: images/train
        val: images/val
        labels: labels
        names: [red, blue, damaged]
        threshold: 0.5

    ``labels/<split>/<image-stem>.txt`` contains comma- or whitespace-separated
    zero-based class IDs, for example ``0,2``.  An empty file means that the
    image is a valid all-negative example.  A missing file is an error so that
    incomplete annotations cannot silently become negative samples.
    """
    data_file = None
    if isinstance(data, (str, Path)):
        data_file = Path(data).resolve()
        raw = deepcopy(yaml_load(data_file))
        base_dir = data_file.parent
    elif isinstance(data, dict):
        raw = deepcopy(data)
        base_dir = Path.cwd()
    else:
        raise TypeError(f"data must be a YAML path or dictionary, got {type(data).__name__}")

    dataset_root_value = raw.get("path", "")
    dataset_root = Path(dataset_root_value) if dataset_root_value else base_dir
    if not dataset_root.is_absolute():
        dataset_root = (base_dir / dataset_root).resolve()
    else:
        dataset_root = dataset_root.resolve()

    names, nc = _as_names(raw.get("names"), raw.get("nc"))
    if "train" not in raw:
        raise ValueError("Multi-label classification data YAML must define a 'train' split")

    split_paths = {}
    for split in ("train", "val", "test"):
        if raw.get(split) is not None:
            split_paths[split] = _resolve_path(raw[split], dataset_root, base_dir).resolve()
    if "val" not in split_paths:
        split_paths["val"] = split_paths.get("test", split_paths["train"])
    if "test" not in split_paths:
        split_paths["test"] = split_paths["val"]

    labels_value = raw.get("labels", "labels")
    label_roots = {}
    for split in ("train", "val", "test"):
        value = labels_value.get(split) if isinstance(labels_value, dict) else labels_value
        if value is None:
            label_roots[split] = None
            continue
        label_root = _resolve_path(value, dataset_root, base_dir).resolve()
        label_roots[split] = _split_label_root(label_root, split)

    threshold = raw.get("threshold", raw.get("multilabel_threshold", 0.5))
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Classification threshold must be in [0, 1], got {threshold}")

    result = {
        **raw,
        "path": dataset_root,
        "train": split_paths["train"],
        "val": split_paths["val"],
        "test": split_paths["test"],
        "labels": label_roots,
        "names": names,
        "nc": nc,
        "threshold": threshold,
        "yaml_file": data_file,
    }
    return result


def parse_image_multilabel_label_file(label_file, nc):
    """Parse one image-level label sidecar into sorted, unique class IDs."""
    label_file = Path(label_file)
    if not label_file.exists():
        raise MultiLabelClassificationLabelError(f"Missing image-level label file: {label_file}")

    text = label_file.read_text(encoding="utf-8").strip()
    if not text:
        return tuple()

    # Supporting commas and line breaks makes the format easy to generate,
    # while still rejecting bbox-like rows and non-integer class values.
    tokens = text.replace(",", " ").split()
    class_ids = []
    for token in tokens:
        try:
            value = int(token)
        except (TypeError, ValueError) as exc:
            raise MultiLabelClassificationLabelError(
                f"{label_file}: expected integer class IDs separated by commas or whitespace, got {token!r}"
            ) from exc
        if value < 0 or value >= int(nc):
            raise MultiLabelClassificationLabelError(
                f"{label_file}: class ID {value} is outside the valid range [0, {int(nc)})"
            )
        class_ids.append(value)
    return tuple(sorted(set(class_ids)))


def resolve_image_multilabel_label(image_file, image_root, label_root):
    """Map an image path to its sidecar while preserving nested directories."""
    image_file = Path(image_file)
    image_root = Path(image_root)
    relative = image_file.relative_to(image_root)
    candidates = []
    if label_root is not None:
        candidates.append(Path(label_root) / relative.with_suffix(".txt"))
        candidates.append(Path(label_root) / f"{image_file.stem}.txt")

    # A same-directory fallback is useful for small crop datasets and keeps
    # the loader usable when a caller does not want a separate labels tree.
    candidates.append(image_file.with_suffix(".txt"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


class MultiLabelClassificationDataset(torch.utils.data.Dataset):
    """Image dataset returning one n-hot vector per image."""

    def __init__(self, root, args, nc, label_root=None, augment=False, prefix="", names=None):
        self.root = Path(root).resolve()
        self.args = args
        self.nc = int(nc)
        self.names = names or {i: str(i) for i in range(self.nc)}
        self.prefix = f"{prefix}: " if prefix and not str(prefix).endswith(" ") else str(prefix)
        if not self.root.is_dir():
            raise FileNotFoundError(f"{self.prefix}image directory not found: {self.root}")

        self.im_files = sorted(
            path for path in self.root.rglob("*") if path.is_file() and path.suffix[1:].lower() in IMG_FORMATS
        )
        if not self.im_files:
            raise FileNotFoundError(f"{self.prefix}no images found in {self.root}")
        if augment and float(getattr(args, "fraction", 1.0)) < 1.0:
            count = max(1, round(len(self.im_files) * float(args.fraction)))
            self.im_files = self.im_files[:count]

        self.label_files = [resolve_image_multilabel_label(path, self.root, label_root) for path in self.im_files]
        self.class_ids = [parse_image_multilabel_label_file(path, self.nc) for path in self.label_files]
        self.targets = torch.zeros((len(self.im_files), self.nc), dtype=torch.float32)
        for row, ids in enumerate(self.class_ids):
            if ids:
                self.targets[row, list(ids)] = 1.0

        if augment:
            self.torch_transforms = classify_augmentations(
                size=int(args.imgsz),
                scale=(max(0.08, 1.0 - float(getattr(args, "scale", 0.5))), 1.0),
                hflip=float(getattr(args, "fliplr", 0.5)),
                vflip=float(getattr(args, "flipud", 0.0)),
                auto_augment=getattr(args, "auto_augment", None),
                hsv_h=float(getattr(args, "hsv_h", 0.015)),
                hsv_s=float(getattr(args, "hsv_s", 0.4)),
                hsv_v=float(getattr(args, "hsv_v", 0.4)),
                erasing=float(getattr(args, "erasing", 0.0)),
            )
        else:
            self.torch_transforms = classify_transforms(
                size=int(args.imgsz), crop_fraction=float(getattr(args, "crop_fraction", 1.0))
            )

        self.samples = list(zip(self.im_files, self.class_ids))

    def __len__(self):
        """Return the number of images."""
        return len(self.im_files)

    def __getitem__(self, index):
        """Load and transform one image without changing its n-hot target."""
        im_file = self.im_files[index]
        image = cv2.imread(str(im_file))
        if image is None:
            raise FileNotFoundError(f"{self.prefix}failed to read image: {im_file}")
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return {
            "img": self.torch_transforms(image),
            "cls_nhot": self.targets[index].clone(),
            # Keep the native key as a compatibility alias.  It is never
            # interpreted as a scalar class by the custom loss or validator.
            "cls": self.targets[index].clone(),
            "im_file": str(im_file),
        }

    @staticmethod
    def collate_fn(batch):
        """Stack images and n-hot targets while retaining source paths."""
        return {
            "img": torch.stack([sample["img"] for sample in batch]),
            "cls_nhot": torch.stack([sample["cls_nhot"] for sample in batch]),
            "cls": torch.stack([sample["cls"] for sample in batch]),
            "im_file": [sample["im_file"] for sample in batch],
        }

    def class_counts(self):
        """Return image-level positive counts for each class."""
        return self.targets.sum(0)


def build_multilabel_classification_dataloader(dataset, batch_size, workers, shuffle=False, rank=-1):
    """Build the repository's standard dataloader for this dataset."""
    return build_dataloader(dataset, batch_size, workers, shuffle=shuffle, rank=rank)


__all__ = (
    "MultiLabelClassificationDataset",
    "MultiLabelClassificationLabelError",
    "build_multilabel_classification_dataloader",
    "load_multilabel_classification_data",
    "parse_image_multilabel_label_file",
    "resolve_image_multilabel_label",
)
