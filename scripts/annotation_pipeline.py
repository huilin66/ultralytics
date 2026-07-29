"""Billboard annotation pipeline scaffold.

Safe by default: indexes a small image sample and writes command manifests. It
runs no SAM inference, classifier training, pseudo-labeling, or dataset export
unless explicit --execute-* flags are provided.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class PipelineConfig:
    raw_root: str
    work_dir: str
    history_data: str
    classifier_train_data: str
    sam_reference: str
    sam_script: str
    sam_model: str
    sam_prompt: str
    sam_device: str
    max_images: int
    classifier_model: str
    classifier_epochs: int
    classifier_imgsz: int
    confidence_threshold: float
    execute_sam: bool
    execute_crop: bool
    execute_prepare_cls: bool
    execute_training: bool
    execute_pseudo_labeling: bool
    execute_export: bool
    copy_classifier_files: bool


def iter_images(root: Path, max_images: int = 0) -> Iterable[Path]:
    count = 0
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            yield path
            count += 1
            if max_images and count >= max_images:
                return


def write_image_index(raw_root: Path, work_dir: Path, max_images: int) -> list[Path]:
    rows = []
    images = list(iter_images(raw_root, max_images=max_images))
    index_path = work_dir / "raw_image_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "image_path", "relative_path", "suffix", "size_bytes"],
        )
        writer.writeheader()
        for i, image_path in enumerate(images):
            try:
                size_bytes = image_path.stat().st_size
            except OSError:
                size_bytes = -1
            row = {
                "image_id": f"img_{i:08d}",
                "image_path": str(image_path),
                "relative_path": str(image_path.relative_to(raw_root)),
                "suffix": image_path.suffix.lower(),
                "size_bytes": size_bytes,
            }
            writer.writerow(row)
            rows.append(row)
    return images


def write_manifest(work_dir: Path, config: PipelineConfig, images: list[Path]) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(config),
        "sample_count": len(images),
        "outputs": {
            "raw_image_index": str(work_dir / "raw_image_index.csv"),
            "sam_output_dir": str(work_dir / "sam3"),
            "sam_labels_dir": str(work_dir / "sam3" / "labels"),
            "billboard_crops_dir": str(work_dir / "billboard_crops"),
            "crop_manifest": str(work_dir / "billboard_crops" / "crop_manifest.csv"),
            "classifier_data_dir": str(work_dir / "defect_cls_data"),
            "classifier_runs_dir": str(work_dir / "defect_classifier"),
            "pseudo_labels": str(work_dir / "pseudo_labels.csv"),
            "yolo_dataset_dir": str(work_dir / "yolo_dataset"),
        },
    }
    (work_dir / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_sam_command(config: PipelineConfig, work_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("sam3_batch_segment.py")),
        "--image-index",
        str(work_dir / "raw_image_index.csv"),
        "--output-dir",
        str(work_dir / "sam3"),
        "--sam-script",
        config.sam_script,
        "--model",
        config.sam_model,
        "--prompt",
        config.sam_prompt,
        "--device",
        config.sam_device,
        "--max-images",
        str(config.max_images),
        "--skip-existing",
    ]


def build_crop_command(work_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("extract_billboard_crops.py")),
        "--image-index",
        str(work_dir / "raw_image_index.csv"),
        "--labels-dir",
        str(work_dir / "sam3" / "labels"),
        "--output-dir",
        str(work_dir / "billboard_crops"),
    ]


def build_prepare_cls_command(config: PipelineConfig, work_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("prepare_defect_cls_dataset.py")),
        "--source",
        config.history_data,
        "--output-dir",
        str(work_dir / "defect_cls_data"),
    ]
    if config.copy_classifier_files:
        cmd.append("--copy-files")
    return cmd


def classifier_data_arg(config: PipelineConfig, work_dir: Path) -> str:
    if config.classifier_train_data:
        return config.classifier_train_data
    prepared = work_dir / "defect_cls_data"
    return str(prepared if prepared.exists() else Path(config.history_data))


def build_classifier_training_command(config: PipelineConfig, work_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "ultralytics",
        "classify",
        "train",
        f"model={config.classifier_model}",
        f"data={classifier_data_arg(config, work_dir)}",
        f"epochs={config.classifier_epochs}",
        f"imgsz={config.classifier_imgsz}",
        f"project={work_dir / 'defect_classifier'}",
        "name=defect_cls",
    ]


def build_pseudo_label_command(config: PipelineConfig, work_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("pseudo_label_defects.py")),
        "--crops-dir",
        str(work_dir / "billboard_crops" / "images"),
        "--crop-manifest",
        str(work_dir / "billboard_crops" / "crop_manifest.csv"),
        "--weights",
        str(work_dir / "defect_classifier" / "defect_cls" / "weights" / "best.pt"),
        "--output",
        str(work_dir / "pseudo_labels.csv"),
        "--conf",
        str(config.confidence_threshold),
    ]


def build_export_command(work_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("export_billboard_yolo_labels.py")),
        "--image-index",
        str(work_dir / "raw_image_index.csv"),
        "--labels-dir",
        str(work_dir / "sam3"),
        "--pseudo-labels",
        str(work_dir / "pseudo_labels.csv"),
        "--output-dir",
        str(work_dir / "yolo_dataset"),
    ]


def write_command_manifest(work_dir: Path, commands: dict[str, list[str]]) -> None:
    lines = []
    for name, command in commands.items():
        lines.append(f"## {name}")
        lines.append(" ".join(f'\"{part}\"' if " " in part else part for part in command))
        lines.append("")
    (work_dir / "commands.md").write_text("\n".join(lines), encoding="utf-8")


def run_command(name: str, command: list[str]) -> None:
    print(f"[execute] {name}: {' '.join(command)}")
    subprocess.run(command, check=True)


def ensure_dirs(work_dir: Path) -> None:
    for name in ["sam3", "sam3/labels", "billboard_crops", "defect_cls_data", "defect_classifier", "yolo_dataset"]:
        (work_dir / name).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe scaffold for billboard annotation and defect-label preparation.")
    parser.add_argument("--raw-root", default=r"\\10.22.50.44\individualdata\VMMS\2024-09-19_HyD_collect")
    parser.add_argument("--work-dir", default="runs/annotation_pipeline")
    parser.add_argument("--history-data", default=r"\\158.132.186.40\isds\huilin\isds\back up\final_data")
    parser.add_argument("--classifier-train-data", default="", help="Optional ready Ultralytics classify dataset root.")
    parser.add_argument("--sam-reference", default=r"E:\repository\dataset_tools\llm_tools\vllm")
    parser.add_argument("--sam-script", default=r"E:\repository\dataset_tools\llm_tools\vllm\sam3_text2mask.py")
    parser.add_argument("--sam-model", default=r"E:\repository\dataset_tools\llm_tools\vllm\sam3.pt")
    parser.add_argument("--sam-prompt", default="billboard signboard advertising board shop sign")
    parser.add_argument("--sam-device", default="cuda:0")
    parser.add_argument("--max-images", type=int, default=20, help="Maximum images for smoke testing. Use 0 for all.")
    parser.add_argument("--classifier-model", default="yolo11n-cls.pt")
    parser.add_argument("--classifier-epochs", type=int, default=50)
    parser.add_argument("--classifier-imgsz", type=int, default=224)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--execute-sam", action="store_true")
    parser.add_argument("--execute-crop", action="store_true")
    parser.add_argument("--execute-prepare-cls", action="store_true")
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--execute-pseudo-labeling", action="store_true")
    parser.add_argument("--execute-export", action="store_true")
    parser.add_argument("--copy-classifier-files", action="store_true", help="Used with --execute-prepare-cls to copy history images.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PipelineConfig(**vars(args))
    raw_root = Path(args.raw_root)
    work_dir = Path(args.work_dir)

    if not raw_root.exists():
        print(f"[error] raw root does not exist or is not accessible: {raw_root}", file=sys.stderr)
        return 2

    work_dir.mkdir(parents=True, exist_ok=True)
    ensure_dirs(work_dir)
    images = write_image_index(raw_root, work_dir, args.max_images)

    commands = {
        "sam3_billboard_masks": build_sam_command(config, work_dir),
        "extract_billboard_crops": build_crop_command(work_dir),
        "prepare_defect_cls_dataset": build_prepare_cls_command(config, work_dir),
        "defect_classifier_training": build_classifier_training_command(config, work_dir),
        "pseudo_label_defects": build_pseudo_label_command(config, work_dir),
        "export_yolo_labels": build_export_command(work_dir),
    }
    write_manifest(work_dir, config, images)
    write_command_manifest(work_dir, commands)

    print(f"[ok] indexed {len(images)} image(s)")
    print(f"[ok] work dir: {work_dir}")
    print(f"[ok] command manifest: {work_dir / 'commands.md'}")

    if args.execute_sam:
        run_command("sam3_billboard_masks", commands["sam3_billboard_masks"])
    if args.execute_crop:
        run_command("extract_billboard_crops", commands["extract_billboard_crops"])
    if args.execute_prepare_cls:
        run_command("prepare_defect_cls_dataset", commands["prepare_defect_cls_dataset"])
    if args.execute_training:
        run_command("defect_classifier_training", commands["defect_classifier_training"])
    if args.execute_pseudo_labeling:
        run_command("pseudo_label_defects", commands["pseudo_label_defects"])
    if args.execute_export:
        run_command("export_yolo_labels", commands["export_yolo_labels"])

    if not any(
        [
            args.execute_sam,
            args.execute_crop,
            args.execute_prepare_cls,
            args.execute_training,
            args.execute_pseudo_labeling,
            args.execute_export,
        ]
    ):
        print("[dry-run] no heavy process was started; pass --execute-* flags to run individual stages")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
