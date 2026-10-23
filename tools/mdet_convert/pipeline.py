# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""One conversion pipeline for every dataset.

``raw tree -> adapter.plan() -> validate pair -> encode uint8 -> YOLO labels ->
``<name>_mdet`` tree -> manifest + quality report``.
"""

from __future__ import annotations

import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .encode import EncodeStats, EncodingError, encode_image, write_png
from .labels import box_is_valid, dedupe, write_label, xyxy_to_yolo
from .naming import slugify
from .report import ConversionReport, write_csv, write_json
from .schema import DatasetPlan, RawSample, SPLITS, format_yaml
from .splits import assign_grouped_splits, groups_from_samples

CONVERTER_VERSION = "1.0.0"


def git_sha(repo: Path) -> str:
    """Best-effort repository revision, for the conversion manifest."""
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _resolve_splits(
    plan: DatasetPlan, ratios: tuple[float, float, float], seed: str
) -> tuple[dict[str, str], str]:
    """Return ``(sample_id -> split, protocol)``.

    Official splits win.  When a release ships train/test but no validation set,
    the validation part is carved out of **train groups only** so the frozen test
    set is never touched; when no official split exists at all, the whole dataset
    is split by group.
    """
    official = {s.sample_id: s.split for s in plan.samples if s.split}
    protocol = "official"

    if not official:
        sizes = groups_from_samples(plan.samples)
        group_split = assign_grouped_splits(sizes, ratios=ratios, seed=seed)
        plan.warnings.append(f"no official split: grouped split over {len(sizes)} groups")
        split_of = {sample.sample_id: group_split[sample.group] for sample in plan.samples}
        protocol = "grouped"
    else:
        split_of = dict(official)
        missing = [sample for sample in plan.samples if sample.sample_id not in official]
        if missing:
            plan.warnings.append(f"{len(missing)} samples have no official split; assigned to train")
            for sample in missing:
                split_of[sample.sample_id] = "train"

        if "val" not in set(split_of.values()):
            train_groups = groups_from_samples(
                [sample for sample in plan.samples if split_of[sample.sample_id] == "train"]
            )
            if len(train_groups) >= 2:
                val_fraction = ratios[1] / max(1e-9, ratios[0] + ratios[1])
                group_split = assign_grouped_splits(
                    train_groups, ratios=(1.0 - val_fraction, val_fraction, 0.0), seed=seed
                )
                for sample in plan.samples:
                    if split_of[sample.sample_id] == "train" and group_split.get(sample.group) == "val":
                        split_of[sample.sample_id] = "val"
                plan.warnings.append(
                    f"release has no validation split: {val_fraction:.0%} of train groups carved out as val; "
                    "official test groups stay frozen"
                )
                protocol = "official_train_carved_val"
            else:
                plan.warnings.append("not enough train groups to carve a validation split")

    counts = Counter(split_of.values())
    plan.splits = {split: counts[split] for split in SPLITS if counts[split]}
    return split_of, protocol


def _safe_id(sample: RawSample) -> str:
    """Return an ASCII, filesystem-safe sample id.

    The dataset loader reads images with ``cv2.imread``, which cannot open
    non-ASCII paths on Windows, so every emitted file name has to be ASCII
    (InfraParis ships a scene called ``Bièvres``).
    """
    return slugify(sample.sample_id)


def _encode_sample(sample: RawSample, plan: DatasetPlan) -> tuple[dict[str, np.ndarray], dict[str, EncodeStats]]:
    """Encode every declared modality and assert pixel alignment."""
    arrays: dict[str, np.ndarray] = {}
    stats: dict[str, EncodeStats] = {}
    for modality in plan.modalities:
        path = sample.files.get(modality.name)
        if path is None:
            if modality.required:
                raise EncodingError(f"missing required modality '{modality.name}'")
            continue
        array, info = encode_image(path, modality.encode)
        arrays[modality.name], stats[modality.name] = array, info

    reference = arrays[plan.reference_modality]
    for name, array in arrays.items():
        if array.shape[:2] != reference.shape[:2]:
            raise EncodingError(
                f"modality '{name}' is {array.shape[:2]} but reference '{plan.reference_modality}' is "
                f"{reference.shape[:2]}; refusing to resize without calibration"
            )
    return arrays, stats


def _draw_preview(arrays: dict[str, np.ndarray], boxes, width: int = 320) -> np.ndarray:
    """Build a side-by-side RGB / thermal / depth panel with the reference boxes drawn."""
    panels = []
    for name in ("rgb", "thermal", "depth"):
        if name not in arrays:
            continue
        image = arrays[name]
        if image.ndim == 2:
            panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            panel = cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2BGR)
        else:
            panel = image[..., :3].copy()
        for box in boxes:
            x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy)
            cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 2)
        scale = width / panel.shape[1]
        panels.append(cv2.resize(panel, (width, max(1, int(panel.shape[0] * scale)))))
    return np.hstack(panels) if panels else np.zeros((1, 1, 3), np.uint8)


def convert(
    plan: DatasetPlan,
    out_root: Path,
    *,
    repo: Path,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: str = "mdet",
    limit: int | None = None,
    only_splits: set[str] | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    previews: int = 6,
) -> Path:
    """Run the conversion and return the produced dataset directory."""
    dataset_dir = Path(out_root) / f"{plan.display_name}_mdet"
    report = ConversionReport(dataset=plan.display_name)
    for warning in plan.warnings:
        report.warn(warning)

    split_of, split_protocol = _resolve_splits(plan, ratios, seed)
    if dry_run:
        counts = Counter(split_of.values())
        report.counts.update({"samples": len(plan.samples), **{f"split_{k}": v for k, v in counts.items()}})
        return dataset_dir

    if dataset_dir.exists() and not overwrite:
        existing = [p for p in dataset_dir.rglob("*") if p.is_file()]
        if existing:
            raise FileExistsError(
                f"{dataset_dir} already contains {len(existing)} files; pass --overwrite to rebuild it"
            )

    classes = dict(plan.classes)
    records: list[dict] = []
    split_manifests: list[dict] = []
    samples_by_split: dict[str, list[dict]] = {split: [] for split in SPLITS}
    processed = 0

    for sample in plan.samples:
        split = split_of[sample.sample_id]
        if only_splits and split not in only_splits:
            continue
        if limit is not None and processed >= limit:
            break

        try:
            arrays, encode_stats = _encode_sample(sample, plan)
        except EncodingError as error:
            report.exclude(sample.sample_id, str(error))
            continue

        boxes = [box for box in sample.boxes if not box.ignored]
        boxes, dropped_duplicates = dedupe(boxes)
        height, width = arrays[plan.reference_modality].shape[:2]

        rows, kept_boxes = [], []
        for box in boxes:
            if box.class_name not in classes:
                report.exclude(sample.sample_id, f"unknown class '{box.class_name}'")
                continue
            if not box_is_valid(box.xyxy, width, height):
                continue
            rows.append((classes[box.class_name], *xyxy_to_yolo(box.xyxy, width, height)))
            kept_boxes.append(box)

        sample_id = _safe_id(sample)
        for modality in plan.modalities:
            if modality.name not in arrays:
                continue
            write_png(dataset_dir / "images" / modality.name / split / f"{sample_id}.png", arrays[modality.name])
        write_label(dataset_dir / "labels" / plan.reference_modality / split / f"{sample_id}.txt", rows)

        stats = report.splits[split]
        stats.samples += 1
        stats.boxes += len(rows)
        stats.empty_labels += 0 if rows else 1
        for box in kept_boxes:
            stats.per_class[box.class_name] += 1
            stats.box_widths.append(box.xyxy[2] - box.xyxy[0])
            stats.box_heights.append(box.xyxy[3] - box.xyxy[1])
        for name in {box.class_name for box in kept_boxes}:
            stats.images_with_class[name] += 1

        record = {
            "sample_id": sample_id,
            "split": split,
            "group": sample.group,
            "annotation_status": sample.annotation_status,
            "reference": plan.reference_modality,
            "width": width,
            "height": height,
            "boxes": len(rows),
            "dropped_duplicate_boxes": dropped_duplicates,
            "modalities": {
                modality.name: {
                    "source": str(sample.files[modality.name]),
                    "dtype": encode_stats[modality.name].raw_dtype,
                    "shape": list(encode_stats[modality.name].raw_shape),
                    "channels": int(arrays[modality.name].shape[2]) if arrays[modality.name].ndim == 3 else 1,
                    "encoding": modality.encode.describe(),
                    "units": modality.encode.units,
                    "invalid_fraction": round(encode_stats[modality.name].invalid_fraction, 6),
                    "saturated_fraction": round(encode_stats[modality.name].saturated_fraction, 6),
                    "dropped_alpha": encode_stats[modality.name].dropped_alpha,
                }
                for modality in plan.modalities
                if modality.name in arrays
            },
        }
        records.append(record)
        samples_by_split[split].append(record)
        split_manifests.append(
            {"sample_id": sample_id, "split": split, "group": sample.group, "boxes": len(rows),
             "source": str(sample.files[plan.reference_modality])}
        )
        processed += 1
        if processed % 200 == 0:
            print(f"  [convert] {processed} samples written", flush=True)

    report.counts["samples"] = processed
    report.counts["excluded"] = len(report.exclusions)

    # --- manifests ---------------------------------------------------------
    meta = dataset_dir / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    with (meta / "samples.jsonl").open("w", encoding="utf-8") as handle:
        import json

        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(meta / "classes.json", {"names": classes, "source": plan.display_name})
    write_csv(meta / "split_manifest.csv", split_manifests, ["sample_id", "split", "group", "boxes", "source"])
    write_csv(meta / "exclusions.csv", report.exclusions, ["sample_id", "reason"])
    write_json(meta / "quality_report.json", report.as_dict())
    write_json(
        meta / "conversion.json",
        {
            "converter": "tools/convert_dataset.py",
            "version": CONVERTER_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": plan.display_name,
            "repo_git_sha": git_sha(repo),
            "source": plan.source,
            "splits": plan.splits,
            "split_protocol": split_protocol,
            "seed": seed,
            "ratios": ratios,
            "channels": plan.channels,
            "classes": classes,
            "modalities": [dict(m.yaml_entry(), encoding=m.encode.describe()) for m in plan.modalities],
            "samples": processed,
            "warnings": report.warnings,
        },
    )
    (dataset_dir / "data.yaml").write_text(format_yaml(plan, dataset_dir.absolute()), encoding="utf-8")

    # --- previews ----------------------------------------------------------
    if previews:
        preview_dir = dataset_dir / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for split, entries in samples_by_split.items():
            for index, record in enumerate(entries[:previews]):
                sample = next(s for s in plan.samples if _safe_id(s) == record["sample_id"])
                try:
                    arrays, _ = _encode_sample(sample, plan)
                except EncodingError:
                    continue
                panel = _draw_preview(arrays, [b for b in sample.boxes if not b.ignored])
                cv2.imwrite(str(preview_dir / f"{split}_{index:03d}_{record['sample_id']}.jpg"), panel)

    return dataset_dir
