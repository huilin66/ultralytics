#!/usr/bin/env python3
"""
End-to-end pipeline:
YOLO (train.txt / val.txt)
 -> COCO (via FiftyOne, per split)
 -> COCO slice (via SAHI, per split)
 -> YOLO sliced (per split)

Author: ChatGPT
"""

import shutil
import subprocess
from pathlib import Path

from sahi.utils.coco import Coco, export_coco_as_yolov5


# ====================== 配置区（你只需要改这里） ======================

YOLO_DIR = Path("/scrinvme/huilin/bdd/collected_data/HMT_data/dataset/rgb_selected_3_p12")

WORK_DIR = Path("/scrinvme/huilin/bdd/collected_data/HMT_data/dataset/_work_slice")
COCO_DIR = WORK_DIR / "coco"
COCO_SLICED_DIR = WORK_DIR / "coco_sliced"
YOLO_SLICED_DIR = WORK_DIR / "yolo_sliced"

SLICE_SIZE = 512          # tile 尺寸
OVERLAP = None            # 旧版 SAHI 不支持 overlap，填 None

SPLITS = ["train", "val"]


# =====================================================================


def run(cmd):
    print("\n>>", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def fiftyone_yolo_to_coco(split: str):
    """YOLO -> COCO (single split)"""
    out_dir = COCO_DIR / split
    clean_dir(out_dir)

    run([
        "fiftyone", "convert",
        "--input-dir", str(YOLO_DIR),
        "--input-type", "fiftyone.types.YOLOv5Dataset",
        "--output-dir", str(out_dir),
        "--output-type", "fiftyone.types.COCODetectionDataset",
        "--split", split,
        "--overwrite",
    ])

    images_dir = out_dir / "data"
    ann_json = out_dir / "labels.json"

    if not ann_json.exists():
        raise FileNotFoundError(f"[{split}] COCO json not found: {ann_json}")

    return images_dir, ann_json


def sahi_slice_coco(split: str, images_dir: Path, ann_json: Path):
    """COCO -> sliced COCO"""
    out_dir = COCO_SLICED_DIR / split
    clean_dir(out_dir)

    cmd = [
        "sahi", "coco", "slice",
        "--image_dir", str(images_dir),
        "--dataset_json_path", str(ann_json),
        "--output_dir", str(out_dir),
        "--slice_size", str(SLICE_SIZE),
    ]

    # 如果你之后升级 SAHI，可以打开 overlap
    if OVERLAP is not None:
        cmd += [
            "--overlap_height_ratio", str(OVERLAP),
            "--overlap_width_ratio", str(OVERLAP),
        ]

    run(cmd)

    # 自动找 slice 后的 json
    candidates = [
        out_dir / "annotations.json",
        out_dir / "labels.json",
        out_dir / "coco.json",
        out_dir / "dataset.json",
    ]
    sliced_json = next((p for p in candidates if p.exists()), None)
    if sliced_json is None:
        raise FileNotFoundError(f"[{split}] sliced COCO json not found")

    sliced_images = out_dir / "images"
    if not sliced_images.exists():
        alt = out_dir / "data"
        if alt.exists():
            sliced_images = alt
        else:
            raise FileNotFoundError(f"[{split}] sliced images dir not found")

    return sliced_images, sliced_json


def coco_to_yolo(split: str, images_dir: Path, ann_json: Path):
    """sliced COCO -> YOLO"""
    out_dir = YOLO_SLICED_DIR / split
    clean_dir(out_dir)

    coco = Coco.from_coco_dict_or_path(str(ann_json))
    export_coco_as_yolov5(
        coco=coco,
        image_dir=str(images_dir),
        output_dir=str(out_dir),
    )


def main():
    print("==== YOLO -> COCO -> SAHI slice -> YOLO (per split) ====")

    for d in [WORK_DIR, COCO_DIR, COCO_SLICED_DIR, YOLO_SLICED_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        print(f"\n========== Processing split: {split} ==========")

        # 1) YOLO -> COCO
        coco_images, coco_json = fiftyone_yolo_to_coco(split)

        # 2) COCO -> sliced COCO
        sliced_images, sliced_json = sahi_slice_coco(split, coco_images, coco_json)

        # 3) sliced COCO -> YOLO
        coco_to_yolo(split, sliced_images, sliced_json)

    print("\n✅ DONE")
    print(f"Final YOLO sliced dataset at:\n{YOLO_SLICED_DIR}")


if __name__ == "__main__":
    main()
