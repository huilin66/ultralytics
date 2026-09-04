"""Dataset support for one-box, n-hot-label YOLO annotations."""

import hashlib
import json
import os
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import (
    FORMATS_HELP_MSG,
    HELP_URL,
    IMG_FORMATS,
    exif_size,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
)
from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM
from ultralytics.utils import LOGGER

from .codec import CombinationCodec, normalize_combination


MULTILABEL_CACHE_VERSION = "multilabel_yolo_v1"


class MultiLabelLabelError(ValueError):
    """Raised for a label-format error that must not be silently cached as background."""


def parse_multilabel_label_file(label_file, nc):
    """Parse one YOLO label file without expanding multi-label boxes.

    Returns:
        tuple[list[tuple[int, ...]], np.ndarray, int]: Label combinations,
        ``xywh`` boxes, and the number of duplicate physical rows removed.
    """
    label_file = Path(label_file)
    if not label_file.is_file():
        return [], np.zeros((0, 4), dtype=np.float32), 0

    combinations, bboxes = [], []
    with label_file.open(encoding="utf-8") as file:
        lines = [line.split() for line in file.read().strip().splitlines() if line.strip()]

    seen = set()
    duplicates = 0
    for line_number, fields in enumerate(lines, start=1):
        if len(fields) != 5:
            raise MultiLabelLabelError(
                f"{label_file}: line {line_number} must contain '<class[,class...]> x y w h', "
                f"got {len(fields)} fields"
            )
        try:
            combo = normalize_combination(fields[0])
        except ValueError as exc:
            raise MultiLabelLabelError(
                f"{label_file}: invalid class list on line {line_number}: {fields[0]!r}"
            ) from exc
        invalid = [class_id for class_id in combo if class_id < 0 or class_id >= int(nc)]
        if invalid:
            raise MultiLabelLabelError(
                f"{label_file}: class IDs {invalid} on line {line_number} are outside [0, {int(nc)})"
            )
        try:
            bbox = np.asarray([float(value) for value in fields[1:]], dtype=np.float32)
        except ValueError as exc:
            raise MultiLabelLabelError(f"{label_file}: invalid bbox on line {line_number}") from exc
        if not np.isfinite(bbox).all():
            raise MultiLabelLabelError(f"{label_file}: bbox on line {line_number} contains NaN or infinity")
        if (bbox < 0).any() or (bbox > 1).any():
            raise MultiLabelLabelError(f"{label_file}: bbox on line {line_number} is not normalized to [0, 1]")
        if bbox[2] <= 0 or bbox[3] <= 0:
            raise MultiLabelLabelError(f"{label_file}: bbox width and height must be positive on line {line_number}")

        key = (combo, tuple(float(value) for value in bbox))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        combinations.append(combo)
        bboxes.append(bbox)

    return combinations, np.asarray(bboxes, dtype=np.float32).reshape(-1, 4), duplicates


def _verify_multilabel_image_label(args):
    """Verify one image/label pair for the custom cache builder."""
    im_file, lb_file, prefix, nc = args
    missing, found, empty, corrupt = 0, 0, 0, 0
    message = ""
    try:
        from PIL import Image

        with Image.open(im_file) as image:
            image.verify()
            shape = exif_size(image)
            shape = (shape[1], shape[0])
            if not (shape[0] > 9 and shape[1] > 9):
                raise ValueError(f"image size {shape} <10 pixels")
            if image.format is None or image.format.lower() not in IMG_FORMATS:
                raise ValueError(f"invalid image format {image.format}. {FORMATS_HELP_MSG}")

        if os.path.isfile(lb_file):
            found = 1
            combinations, bboxes, duplicates = parse_multilabel_label_file(lb_file, nc)
            if not combinations:
                empty = 1
            if duplicates:
                message = f"{prefix}WARNING ⚠️ {im_file}: {duplicates} duplicate labels removed"
        else:
            missing = 1
            combinations, bboxes = [], np.zeros((0, 4), dtype=np.float32)

        return {
            "im_file": im_file,
            "shape": shape,
            "combinations": combinations,
            "bboxes": bboxes,
            "missing": missing,
            "found": found,
            "empty": empty,
            "corrupt": corrupt,
            "message": message,
        }
    except MultiLabelLabelError:
        raise
    except Exception as exc:
        corrupt = 1
        return {
            "im_file": None,
            "shape": None,
            "combinations": [],
            "bboxes": np.zeros((0, 4), dtype=np.float32),
            "missing": missing,
            "found": found,
            "empty": empty,
            "corrupt": corrupt,
            "message": f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {exc}",
        }


class MultiLabelYOLODataset(YOLODataset):
    """YOLO dataset retaining one physical box for each n-hot label vector."""

    cache_version = MULTILABEL_CACHE_VERSION

    def __init__(self, *args, data=None, task="detect", **kwargs):
        if task != "detect":
            raise ValueError(f"MultiLabelYOLODataset only supports task='detect', got {task!r}")
        self.data = data
        self.nc = int(data["nc"] if data and "nc" in data else len(data["names"]))
        self.codec = None
        super().__init__(*args, data=data, task="detect", **kwargs)

    @property
    def cache_path(self):
        """Return a cache path that cannot collide with native single-label caches."""
        return Path(self.label_files[0]).parent.with_suffix(".multilabel.cache")

    def cache_identity(self, file_hash=None):
        """Return the identity used to reject incompatible caches."""
        names = self.data.get("names", {})
        if isinstance(names, dict):
            names = [names[key] for key in sorted(names, key=lambda key: int(key))]
        payload = {
            "cache_version": self.cache_version,
            "parser": "comma-separated-classes-v1",
            "nc": self.nc,
            "names": names,
            "file_hash": file_hash if file_hash is not None else get_hash(self.label_files + self.im_files),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def cache_labels(self, path=None):
        """Build and save the versioned multi-label cache."""
        path = Path(path or self.cache_path)
        cache = {"labels": []}
        missing = found = empty = corrupt = 0
        messages = []
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        file_hash = get_hash(self.label_files + self.im_files)
        all_combinations = []

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                _verify_multilabel_image_label,
                zip(self.im_files, self.label_files, repeat(self.prefix), repeat(self.nc)),
            )
            pbar = TQDM(results, desc=desc, total=len(self.im_files))
            for record in pbar:
                missing += record["missing"]
                found += record["found"]
                empty += record["empty"]
                corrupt += record["corrupt"]
                if record["im_file"]:
                    cache["labels"].append(record)
                    all_combinations.extend(record["combinations"])
                if record["message"]:
                    messages.append(record["message"])
                pbar.desc = f"{desc} {found} labels, {missing + empty} backgrounds, {corrupt} corrupt"
            pbar.close()

        if messages:
            LOGGER.info("\n".join(messages))
        if not found:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")

        codec = CombinationCodec.from_combinations(all_combinations, self.nc)
        labels = []
        for record in cache["labels"]:
            transport_ids = np.asarray([codec.encode(combo) for combo in record["combinations"]], dtype=np.float32)
            labels.append(
                {
                    "im_file": record["im_file"],
                    "shape": record["shape"],
                    "cls": transport_ids.reshape(-1, 1),
                    "bboxes": record["bboxes"],
                    "segments": [],
                    "keypoints": None,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )

        cache = {
            "labels": labels,
            "combinations": [list(combo) for combo in codec.combinations],
            "codec": codec.as_json(),
            "hash": file_hash,
            "cache_identity": self.cache_identity(file_hash),
            "results": (found, missing, empty, corrupt, len(self.im_files)),
            "msgs": messages,
            "version": self.cache_version,
        }
        save_dataset_cache_file(self.prefix, path, cache, self.cache_version)
        self.codec = codec
        return cache

    def get_labels(self):
        """Load only the custom cache and rebuild it when metadata is incompatible."""
        self.label_files = img2label_paths(self.im_files)
        self.check_files()
        path = self.cache_path
        file_hash = get_hash(self.label_files + self.im_files)
        identity = self.cache_identity(file_hash)
        try:
            cache, exists = load_dataset_cache_file(path), True
            assert cache["version"] == self.cache_version
            assert cache["hash"] == file_hash
            assert cache["cache_identity"] == identity
            codec = CombinationCodec.from_combinations(cache["combinations"], self.nc)
        except (AssertionError, AttributeError, EOFError, FileNotFoundError, KeyError, OSError, TypeError, ValueError):
            cache, exists = self.cache_labels(path), False
            codec = self.codec

        results = cache.pop("results")
        found, missing, empty, corrupt, total = results
        if exists and LOCAL_RANK in {-1, 0}:
            desc = f"Scanning {path}... {found} labels, {missing + empty} backgrounds, {corrupt} corrupt"
            TQDM(None, desc=self.prefix + desc, total=total, initial=total)
            if cache.get("msgs"):
                LOGGER.info("\n".join(cache["msgs"]))

        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {path}, training may not work correctly. {HELP_URL}")
        self.im_files = [label["im_file"] for label in labels]
        self.codec = codec

        if sum(len(label["cls"]) for label in labels) == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found in {path}, training may not work correctly. {HELP_URL}")
        for label in labels:
            if len(label["cls"]) != len(label["bboxes"]):
                raise ValueError(f"{label['im_file']}: physical bbox count and transport-ID count differ")
        return labels

    def update_labels(self, include_class=None):
        """Reject scalar-class filtering that would corrupt transport IDs."""
        if include_class is not None:
            raise NotImplementedError("classes filtering is not supported in multi-label mode")
        if self.single_cls:
            raise NotImplementedError("single_cls is not supported in multi-label mode")
        super().update_labels(include_class=None)

    def __getitem__(self, index):
        """Apply native transforms, then reconstruct the aligned n-hot labels."""
        labels = self.transforms(self.get_image_and_label(index))
        transport_ids = labels["cls"].reshape(-1)
        labels["cls_nhot"] = self.codec.to_nhot(transport_ids, dtype=torch.float32)
        if labels["cls_nhot"].shape[0] != labels["bboxes"].shape[0]:
            raise RuntimeError(f"{labels['im_file']}: transformed bbox and n-hot counts differ")
        return labels

    def get_class_counts(self):
        """Count real class occurrences from n-hot combinations, not transport IDs."""
        counts = np.zeros(self.nc, dtype=np.int64)
        for label in self.labels:
            for combo in self.codec.decode_ids(label["cls"].reshape(-1)):
                counts[list(combo)] += 1
        return counts

    @staticmethod
    def collate_fn(batch):
        """Collate physical boxes and n-hot labels without expanding either."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(sample.values()) for sample in batch]))
        for index, key in enumerate(keys):
            value = values[index]
            if key == "img":
                value = torch.stack(value, 0)
            if key in {"masks", "keypoints", "bboxes", "cls", "segments", "obb", "cls_nhot"}:
                value = torch.cat(value, 0)
            new_batch[key] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for index in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][index] += index
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch
