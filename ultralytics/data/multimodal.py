# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Pixel-aligned multi-modal datasets for YOLO detection and instance segmentation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ultralytics.data.augment import Albumentations, Format, RandomHSV
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM
from ultralytics.utils.patches import imread


@dataclass(frozen=True)
class Modality:
    """Describe one image modality in a pixel-aligned sample."""

    name: str
    root: Path
    channels: int
    suffix: str | None = None
    color: str | None = None


class Modalities:
    """Map a primary image path to its aligned modality files and load their channel stack."""

    def __init__(self, data: dict[str, Any]):
        """Initialize modality metadata from a dataset YAML dictionary.

        Args:
            data (dict[str, Any]): Dataset configuration containing a ``modalities`` list and resolved ``path``.
        """
        specs = data.get("modalities")
        if not isinstance(specs, list) or len(specs) < 2:
            raise ValueError("Multi-modal data requires a 'modalities' list with at least two entries.")

        dataset_root = Path(data.get("path", "")).resolve()
        modalities = []
        names = set()
        for spec in specs:
            if not isinstance(spec, dict):
                raise TypeError("Each 'modalities' entry must be a mapping.")
            name, path, channels = spec.get("name"), spec.get("path"), spec.get("channels")
            if not isinstance(name, str) or not name:
                raise ValueError("Each modality needs a non-empty 'name'.")
            if name in names:
                raise ValueError(f"Duplicate modality name '{name}'.")
            if not isinstance(path, str) or not path:
                raise ValueError(f"Modality '{name}' needs a relative or absolute 'path'.")
            if not isinstance(channels, int) or channels < 1:
                raise ValueError(f"Modality '{name}' needs a positive integer 'channels'.")
            root = Path(path)
            root = root if root.is_absolute() else dataset_root / root
            suffix = spec.get("suffix")
            if suffix is not None and (not isinstance(suffix, str) or not suffix.startswith(".")):
                raise ValueError(f"Modality '{name}' suffix must start with '.', for example '.png'.")
            color = spec.get("color")
            if color not in {None, "bgr"}:
                raise ValueError(f"Modality '{name}' color must be 'bgr' when specified.")
            modalities.append(Modality(name, root.resolve(), channels, suffix, color))
            names.add(name)

        self.items = tuple(modalities)
        self.channels = sum(item.channels for item in self.items)
        configured_channels = data.get("channels", self.channels)
        if configured_channels != self.channels:
            raise ValueError(
                f"data.yaml channels={configured_channels} does not match the modality total ({self.channels})."
            )

    @property
    def primary_root(self) -> Path:
        """Return the root directory of the primary modality."""
        return self.items[0].root

    @property
    def bgr_slices(self) -> tuple[slice, ...]:
        """Return channel slices that may receive BGR HSV augmentation."""
        start, slices = 0, []
        for item in self.items:
            if item.color == "bgr":
                if item.channels != 3:
                    raise ValueError(f"Modality '{item.name}' uses color: bgr but does not have three channels.")
                slices.append(slice(start, start + item.channels))
            start += item.channels
        return tuple(slices)

    def paths(self, primary_path: str | Path) -> tuple[Path, ...]:
        """Return all modality paths for a primary-modality image path.

        The primary path must be nested under the first modality's ``path``. All other modalities preserve the relative
        path below their own root, optionally replacing the file extension with their configured ``suffix``.
        """
        primary_path = Path(primary_path).resolve()
        try:
            relative = primary_path.relative_to(self.primary_root)
        except ValueError as e:
            raise ValueError(
                f"Primary image '{primary_path}' is not under the first modality path '{self.primary_root}'."
            ) from e
        return tuple(item.root / relative.with_suffix(item.suffix or relative.suffix) for item in self.items)

    def load(self, primary_path: str | Path) -> np.ndarray:
        """Read, validate and concatenate the aligned modality images into one HWC array."""
        images = []
        shape = None
        for item, path in zip(self.items, self.paths(primary_path)):
            image = imread(str(path), flags=cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(f"Missing or unreadable {item.name} modality for '{primary_path}': '{path}'.")
            if image.ndim == 2:
                image = image[..., None]
            if image.ndim != 3 or image.shape[2] != item.channels:
                found = image.shape[2] if image.ndim == 3 else image.shape
                raise ValueError(
                    f"Modality '{item.name}' for '{primary_path}' has {found} channels, expected {item.channels}."
                )
            if image.dtype != np.uint8:
                raise TypeError(
                    f"Modality '{item.name}' for '{primary_path}' has dtype {image.dtype}; convert all modalities "
                    "to uint8 before YOLO preprocessing."
                )
            if shape is None:
                shape = image.shape[:2]
            elif image.shape[:2] != shape:
                raise ValueError(
                    f"Modalities for '{primary_path}' are not pixel-aligned: expected shape {shape}, "
                    f"got {image.shape[:2]} for '{path}'."
                )
            images.append(image)
        return np.concatenate(images, axis=2)


class MultiModalHSV(RandomHSV):
    """Apply one HSV sample to explicitly marked BGR modality groups without touching other sensors."""

    def __init__(self, *args, bgr_slices: tuple[slice, ...] = (), **kwargs):
        """Initialize the transform with channel groups selected by ``color: bgr`` in data YAML."""
        super().__init__(*args, **kwargs)
        self.bgr_slices = bgr_slices

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Augment only configured BGR groups using the same sampled HSV lookup tables."""
        image = labels["img"]
        if not self.bgr_slices or not (self.hgain or self.sgain or self.vgain):
            return labels
        if image.dtype != np.uint8:
            raise TypeError("HSV augmentation requires uint8 multi-modal images.")

        gains = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain]
        values = np.arange(256, dtype=gains.dtype)
        hue = ((values + gains[0] * 180) % 180).astype(np.uint8)
        saturation = np.clip(values * (gains[1] + 1), 0, 255).astype(np.uint8)
        value = np.clip(values * (gains[2] + 1), 0, 255).astype(np.uint8)
        saturation[0] = 0
        for channels in self.bgr_slices:
            bgr = np.ascontiguousarray(image[..., channels])
            h, s, v = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
            hsv = cv2.merge((cv2.LUT(h, hue), cv2.LUT(s, saturation), cv2.LUT(v, value)))
            image[..., channels] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return labels


class MultiModalDataset(YOLODataset):
    """Load N pixel-aligned modalities as one tensor while reusing the standard synchronized YOLO transforms."""

    def __init__(self, *args, data: dict[str, Any] | None = None, **kwargs):
        """Initialize the dataset and validate that its declared channels equal the modality-channel total."""
        if data is None:
            raise ValueError("MultiModalDataset requires a dataset YAML dictionary.")
        self.modalities = Modalities(data)
        super().__init__(*args, data=data, **kwargs)

    def get_labels(self) -> list[dict]:
        """Load normal YOLO labels and establish a private, fused-image disk-cache namespace."""
        labels = super().get_labels()
        self._cache_files = [Path(path).with_suffix(f".mm{self.modalities.channels}.npy") for path in self.im_files]
        return labels

    def _resize(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize every modality group independently with the same output geometry."""
        start, resized = 0, []
        for modality in self.modalities.items:
            part = cv2.resize(
                image[..., start : start + modality.channels], (width, height), interpolation=cv2.INTER_LINEAR
            )
            resized.append(part[..., None] if part.ndim == 2 else part)
            start += modality.channels
        return np.concatenate(resized, axis=2)

    def load_image(
        self, i: int, rect_mode: bool = True, resize_short: bool = False
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Load a fused sample and resize every modality with the same geometry."""
        image = self.ims[i]
        if image is None:
            cache_file = self._cache_files[i]
            if cache_file.exists():
                try:
                    image = np.load(cache_file, allow_pickle=False)
                    if image.ndim != 3 or image.shape[2] != self.channels:
                        raise ValueError(f"expected {self.channels} channels")
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}Removing invalid multi-modal cache {cache_file}: {e}")
                    cache_file.unlink(missing_ok=True)
                    image = None
            if image is None:
                image = self.modalities.load(self.im_files[i])

            h0, w0 = image.shape[:2]
            if rect_mode:
                if resize_short:
                    ratio = self.imgsz / min(h0, w0)
                    if ratio != 1:
                        width, height = (
                            (math.ceil(w0 * ratio), self.imgsz)
                            if h0 < w0
                            else (self.imgsz, math.ceil(h0 * ratio))
                        )
                        image = self._resize(image, width, height)
                else:
                    ratio = self.imgsz / max(h0, w0)
                    if ratio != 1:
                        width, height = min(math.ceil(w0 * ratio), self.imgsz), min(math.ceil(h0 * ratio), self.imgsz)
                        image = self._resize(image, width, height)
            elif not (h0 == w0 == self.imgsz):
                image = self._resize(image, self.imgsz, self.imgsz)

            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = image, (h0, w0), image.shape[:2]
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length and self.cache != "ram":
                    j = self.buffer.pop(0)
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
            return image, (h0, w0), image.shape[:2]
        return image, self.im_hw0[i], self.im_hw[i]

    def cache_images_to_disk(self, i: int) -> None:
        """Save the fused, pre-resize image rather than only the primary modality."""
        cache_file = self._cache_files[i]
        if not cache_file.exists():
            try:
                np.save(cache_file, self.modalities.load(self.im_files[i]), allow_pickle=False)
            except Exception as e:
                cache_file.unlink(missing_ok=True)
                LOGGER.warning(f"{self.prefix}WARNING ⚠️ Failed to cache multi-modal image {cache_file}: {e}")

    def cache_images(self) -> None:
        """Cache fused samples in RAM or in the private multi-modal disk-cache namespace."""
        total_bytes, gigabyte = 0, 1 << 30
        loader = self.cache_images_to_disk if self.cache == "disk" else self.load_image
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(loader, range(self.ni))
            progress = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, loaded in progress:
                if self.cache == "disk":
                    total_bytes += self._cache_files[i].stat().st_size
                else:
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = loaded
                    total_bytes += self.ims[i].nbytes
                storage = "Disk" if self.cache == "disk" else "RAM"
                progress.desc = f"{self.prefix}Caching images ({total_bytes / gigabyte:.1f}GB {storage})"
            progress.close()

    def check_cache_disk(self, safety_margin: float = 0.5) -> bool:
        """Estimate disk requirements from fused samples before enabling disk caching."""
        import shutil

        total_bytes, gigabyte = 0, 1 << 30
        samples = min(self.ni, 30)
        for index in random.choices(range(self.ni), k=samples):
            total_bytes += self.modalities.load(self.im_files[index]).nbytes
        required = total_bytes * self.ni / samples * (1 + safety_margin)
        total, _used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if required > free:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{required / gigabyte:.1f}GB disk space required, "
                f"but only {free / gigabyte:.1f}/{total / gigabyte:.1f}GB is free; disabling cache='disk'."
            )
            return False
        return True

    def check_cache_ram(self, safety_margin: float = 0.5) -> bool:
        """Estimate RAM requirements from fused samples before enabling RAM caching."""
        total_bytes, gigabyte = 0, 1 << 30
        samples = min(self.ni, 30)
        for index in random.choices(range(self.ni), k=samples):
            image = self.modalities.load(self.im_files[index])
            ratio = self.imgsz / max(image.shape[:2])
            total_bytes += image.nbytes * ratio**2
        required = total_bytes * self.ni / samples * (1 + safety_margin)
        memory = __import__("psutil").virtual_memory()
        if required > memory.available:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{required / gigabyte:.1f}GB RAM required, "
                f"but only {memory.available / gigabyte:.1f}/{memory.total / gigabyte:.1f}GB is available; "
                "disabling cache='ram'."
            )
            return False
        return True

    def build_transforms(self, hyp):
        """Reuse YOLO's synchronized geometry/mixing transforms and replace only color augmentation policy."""
        transforms = super().build_transforms(hyp)
        for i, transform in enumerate(transforms.transforms):
            if isinstance(transform, Albumentations):
                transform.p = 0.0  # Generic Albumentations assumes a normal image, not a declared modality layout.
            elif isinstance(transform, RandomHSV):
                transforms.transforms[i] = MultiModalHSV(
                    hgain=transform.hgain,
                    sgain=transform.sgain,
                    vgain=transform.vgain,
                    bgr_slices=self.modalities.bgr_slices,
                )
            elif isinstance(transform, Format):
                # The base formatter reverses a whole three-channel image. A multi-modal channel stack must preserve
                # its declared ordering, including the uncommon case where all modalities sum to three channels.
                transform.bgr = 1.0
        return transforms


def build_multimodal_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32):
    """Build a MultiModalDataset using the standard YOLO dataset argument policy."""
    return MultiModalDataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=f"{mode}: ",
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )
