"""Generate deterministic offline robustness variants for mdetect test data.

The generator creates one materialized dataset per condition/severity.  Every
model therefore receives exactly the same transformed pixels and annotations.
The output manifest records the seed, source image hashes, transformation
parameters, and the generated dataset YAML paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml

from attribute_dataset_utils import label_path_for_image, load_dataset_spec, parse_mdet_label_file


LEVELS = ("mild", "moderate", "severe")
CORE_CONDITIONS = (
    "low_light",
    "overexposure",
    "local_shadow",
    "local_glare",
    "gaussian_blur",
    "motion_blur",
    "local_occlusion",
    "perspective",
    "affine",
    "scale",
    "translate",
)
OPTIONAL_CONDITIONS = ("gaussian_noise", "fog", "rain")
ALL_CONDITIONS = CORE_CONDITIONS + OPTIONAL_CONDITIONS

LEVEL_PARAMS: dict[str, tuple[float, float, float]] = {
    "low_light": (0.75, 0.50, 0.30),
    "overexposure": (1.25, 1.60, 2.00),
    "local_shadow": (0.25, 0.50, 0.75),
    "local_glare": (0.25, 0.50, 0.75),
    "gaussian_blur": (1.0, 2.0, 4.0),
    "motion_blur": (5.0, 10.0, 20.0),
    "local_occlusion": (0.05, 0.15, 0.30),
    "gaussian_noise": (0.02, 0.05, 0.10),
    "fog": (0.10, 0.25, 0.40),
    "rain": (0.15, 0.30, 0.45),
    "perspective": (0.02, 0.05, 0.10),
    "affine": (0.03, 0.07, 0.12),
    "scale": (0.85, 0.70, 0.55),
    "translate": (0.02, 0.05, 0.10),
}


def _stable_rng(seed: int, image_id: str, condition: str) -> np.random.Generator:
    """Create a cross-process deterministic RNG without Python hash randomization."""

    key = f"{seed}:{image_id}:{condition}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    value = int.from_bytes(digest[:8], "little") % (2**32)
    return np.random.default_rng(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _level_index(level: str) -> int:
    return LEVELS.index(level)


def _soft_ellipse_mask(height: int, width: int, area_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Create a deterministic soft ellipse mask in [0, 1]."""

    aspect = float(rng.uniform(0.65, 1.55))
    ellipse_area = max(1.0, area_ratio * height * width)
    semi_x = math.sqrt(ellipse_area * aspect / math.pi)
    semi_y = math.sqrt(ellipse_area / (math.pi * aspect))
    center_x = int(rng.uniform(0.20, 0.80) * width)
    center_y = int(rng.uniform(0.20, 0.80) * height)
    angle = float(rng.uniform(0.0, 180.0))

    mask = np.zeros((height, width), dtype=np.float32)
    cv2.ellipse(
        mask,
        (center_x, center_y),
        (max(1, int(semi_x)), max(1, int(semi_y))),
        angle,
        0.0,
        360.0,
        1.0,
        thickness=-1,
    )
    sigma = max(1.0, min(height, width) * 0.02)
    kernel = max(3, int(round(sigma * 6)) | 1)
    mask = cv2.GaussianBlur(mask, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    maximum = float(mask.max())
    return mask / maximum if maximum > 0 else mask


def _random_occlusion_mask(height: int, width: int, area_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Create a deterministic rectangular cutout mask."""

    aspect = float(rng.uniform(0.6, 1.8))
    box_area = max(1.0, area_ratio * height * width)
    box_width = min(width, max(1, int(round(math.sqrt(box_area * aspect)))))
    box_height = min(height, max(1, int(round(box_area / box_width))))
    x0 = int(rng.integers(0, max(1, width - box_width + 1)))
    y0 = int(rng.integers(0, max(1, height - box_height + 1)))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y0 : y0 + box_height, x0 : x0 + box_width] = 1
    return mask


def _warp(image: np.ndarray, matrix: np.ndarray, *, perspective: bool) -> np.ndarray:
    height, width = image.shape[:2]
    if perspective:
        return cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    return cv2.warpAffine(
        image,
        matrix[:2],
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _apply_variant(
    image: np.ndarray,
    condition: str,
    level: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply one deterministic variant and return image plus optional bbox homography."""

    height, width = image.shape[:2]
    index = _level_index(level)
    parameter = LEVEL_PARAMS[condition][index]
    image_float = image.astype(np.float32) / 255.0
    transform: np.ndarray | None = None

    if condition in {"low_light", "overexposure"}:
        output = np.clip(image_float * parameter, 0.0, 1.0)
        return (output * 255.0 + 0.5).astype(np.uint8), transform

    if condition in {"local_shadow", "local_glare"}:
        area_ratio = (0.15, 0.30, 0.45)[index]
        mask = _soft_ellipse_mask(height, width, area_ratio, rng)
        mask = mask * parameter
        if condition == "local_shadow":
            output = image_float * (1.0 - mask[..., None])
        else:
            output = image_float * (1.0 - mask[..., None]) + mask[..., None]
        return (np.clip(output, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), transform

    if condition == "gaussian_blur":
        sigma = parameter
        kernel = max(3, int(round(sigma * 6)) | 1)
        return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma, sigmaY=sigma), transform

    if condition == "motion_blur":
        length = max(1, int(round(parameter)))
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0 / length
        angle = float(rng.uniform(0.0, 180.0))
        center = (length / 2.0 - 0.5, length / 2.0 - 0.5)
        rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation, (length, length))
        total = float(kernel.sum())
        if total > 0:
            kernel /= total
        return cv2.filter2D(image, -1, kernel), transform

    if condition == "local_occlusion":
        mask = _random_occlusion_mask(height, width, parameter, rng)
        output = image.copy()
        output[mask.astype(bool)] = 0
        return output, transform

    if condition == "gaussian_noise":
        noise = rng.normal(0.0, parameter, size=image_float.shape).astype(np.float32)
        output = np.clip(image_float + noise, 0.0, 1.0)
        return (output * 255.0 + 0.5).astype(np.uint8), transform

    if condition == "fog":
        output = image_float * (1.0 - parameter) + parameter
        return (np.clip(output, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), transform

    if condition == "rain":
        output = image.astype(np.float32)
        overlay = np.zeros_like(output)
        count = max(12, int(round(height * width * 0.00005)))
        angle = float(rng.uniform(-25.0, 25.0))
        radians = math.radians(angle)
        length = max(8, int(round(min(height, width) * 0.08)))
        for _ in range(count):
            x0 = int(rng.integers(0, width))
            y0 = int(rng.integers(0, height))
            x1 = int(round(x0 + math.sin(radians) * length))
            y1 = int(round(y0 + math.cos(radians) * length))
            cv2.line(overlay, (x0, y0), (x1, y1), (255, 255, 255), 1, cv2.LINE_AA)
        output = output * (1.0 - parameter) + overlay * parameter
        return np.clip(output, 0.0, 255.0).astype(np.uint8), transform

    if condition == "perspective":
        source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        offsets = rng.uniform(-parameter, parameter, size=(4, 2))
        destination = (source + offsets * np.float32([width, height])).astype(np.float32)
        transform = cv2.getPerspectiveTransform(source, destination)
        return _warp(image, transform, perspective=True), transform

    if condition == "affine":
        sign = -1.0 if rng.random() < 0.5 else 1.0
        angle = sign * (3.0, 7.0, 12.0)[index]
        center = (width / 2.0, height / 2.0)
        rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotation_h = np.vstack([rotation, [0.0, 0.0, 1.0]])
        shear = sign * parameter
        shear_h = np.array(
            [[1.0, shear, -shear * height / 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        transform = shear_h @ rotation_h
        return _warp(image, transform, perspective=True), transform

    if condition == "scale":
        center_x, center_y = width / 2.0, height / 2.0
        transform = np.array(
            [[parameter, 0.0, (1.0 - parameter) * center_x], [0.0, parameter, (1.0 - parameter) * center_y], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return _warp(image, transform, perspective=False), transform

    if condition == "translate":
        dx = float(rng.uniform(-parameter, parameter) * width)
        dy = float(rng.uniform(-parameter, parameter) * height)
        transform = np.array(
            [[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return _warp(image, transform, perspective=False), transform

    raise ValueError(f"Unsupported condition: {condition}")


def _transform_bbox(
    bbox: tuple[float, float, float, float],
    transform: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    """Transform a normalized xywh box and clip it to the output image."""

    cx, cy, bw, bh = bbox
    x0, y0 = (cx - bw / 2.0) * width, (cy - bh / 2.0) * height
    x1, y1 = (cx + bw / 2.0) * width, (cy + bh / 2.0) * height
    corners = np.float32([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]])
    transformed = cv2.perspectiveTransform(corners, transform).reshape(-1, 2)
    new_x0 = float(np.clip(transformed[:, 0].min(), 0.0, width - 1.0))
    new_y0 = float(np.clip(transformed[:, 1].min(), 0.0, height - 1.0))
    new_x1 = float(np.clip(transformed[:, 0].max(), 0.0, width - 1.0))
    new_y1 = float(np.clip(transformed[:, 1].max(), 0.0, height - 1.0))
    if new_x1 <= new_x0 or new_y1 <= new_y0:
        return None
    return (
        ((new_x0 + new_x1) / 2.0) / width,
        ((new_y0 + new_y1) / 2.0) / height,
        (new_x1 - new_x0) / width,
        (new_y1 - new_y0) / height,
    )


def _format_number(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".") or "0"


def _write_label(path: Path, objects: Iterable[Any], transform: np.ndarray | None, width: int, height: int) -> int:
    rows: list[str] = []
    for obj in objects:
        bbox = obj.bbox
        if transform is not None:
            bbox = _transform_bbox(bbox, transform, width, height)
            if bbox is None:
                continue
        fields = [str(obj.object_class), str(len(obj.attributes))]
        fields.extend(_format_number(value) for value in obj.attributes)
        fields.extend(_format_number(value) for value in bbox)
        rows.append(" ".join(fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def _dataset_yaml(path: Path, root: Path, source_raw: dict[str, Any]) -> None:
    data = {
        "path": str(root.resolve()),
        "train": "test.txt",
        "val": "test.txt",
        "test": "test.txt",
        "names": source_raw["names"],
        "attributes": source_raw["attributes"],
    }
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic offline mdet robustness variants")
    parser.add_argument("--data", required=True, help="Source mdet dataset YAML")
    parser.add_argument("--output", required=True, help="Output directory for generated variants")
    parser.add_argument("--split", default="test", choices=("test",), help="Only test split is supported")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-optional", action="store_true", help="Include Gaussian noise, fog, and rain")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse an existing output directory")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_yaml = Path(args.data).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.exist_ok:
        raise FileExistsError(f"Output directory is not empty: {output}; use --exist-ok to reuse it")
    output.mkdir(parents=True, exist_ok=True)

    spec = load_dataset_spec(source_yaml, splits=("test",))
    images = list(spec.splits.get("test", ()))
    if not images:
        raise RuntimeError(f"No test images found in {source_yaml}")
    conditions = list(CORE_CONDITIONS)
    if args.include_optional:
        conditions.extend(OPTIONAL_CONDITIONS)

    source_records: list[dict[str, Any]] = []
    for image_path in images:
        try:
            source_id = image_path.relative_to(spec.root).as_posix()
        except ValueError:
            source_id = image_path.name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read source image: {image_path}")
        source_records.append(
            {
                "source_id": source_id,
                "source_path": str(image_path),
                "sha256": _sha256(image_path),
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
            }
        )

    variants: list[dict[str, Any]] = []
    for condition in conditions:
        for level in LEVELS:
            variant_id = f"{condition}_{level}"
            variant_root = output / "datasets" / variant_id
            images_root = variant_root / "images"
            labels_root = variant_root / "labels"
            images_root.mkdir(parents=True, exist_ok=True)
            labels_root.mkdir(parents=True, exist_ok=True)
            test_entries: list[str] = []
            label_counts: list[int] = []
            for index, image_path in enumerate(images):
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Could not read source image: {image_path}")
                source_id = source_records[index]["source_id"]
                rng = _stable_rng(args.seed, source_id, condition)
                transformed, transform = _apply_variant(image, condition, level, rng)
                output_name = f"{index:05d}_{image_path.stem}.png"
                output_image = images_root / output_name
                if not cv2.imwrite(str(output_image), transformed):
                    raise RuntimeError(f"Could not write generated image: {output_image}")
                source_label = label_path_for_image(image_path, spec.image_root, spec.labels_root)
                objects = parse_mdet_label_file(source_label, expected_attributes=len(spec.attribute_names))
                output_label = labels_root / f"{Path(output_name).stem}.txt"
                label_counts.append(_write_label(output_label, objects, transform, image.shape[1], image.shape[0]))
                test_entries.append(f"images/{output_name}")

            (variant_root / "test.txt").write_text("\n".join(test_entries) + "\n", encoding="utf-8")
            data_yaml = variant_root / "data.yaml"
            _dataset_yaml(data_yaml, variant_root, dict(spec.raw))
            variants.append(
                {
                    "variant": variant_id,
                    "condition": condition,
                    "severity": level,
                    "data_yaml": str(data_yaml),
                    "image_count": len(test_entries),
                    "label_object_count": int(sum(label_counts)),
                }
            )
            print(f"[generated] {variant_id}: {len(test_entries)} images, {sum(label_counts)} objects", flush=True)

    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "source_yaml": str(source_yaml),
        "source_root": str(spec.root),
        "split": args.split,
        "image_count": len(images),
        "conditions": conditions,
        "levels": list(LEVELS),
        "source_images": source_records,
        "variants": variants,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[manifest] {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
