"""Train the final MAYOLO multi-scale variants.

MAYOLO is defined in this project as YOLOv10 + GIA-v2.5.7 + Cross-GIN
margin-residual + HO.  The existing ``mayolov*.yaml`` files are native
MAYOLO detector configurations and do not contain the selected GIA/GCA
modules, so this launcher derives one final configuration from each E3
YOLOv10 mdet scale before calling the standard two-stage trainer.

The launcher is intentionally sequential per process.  Run one process for
GPU 0 (for example ``n s m``) and another for GPU 1 (``l b``); each process
uses the requested thread limits and never launches two models on the same
GPU at once.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# These are the same YOLOv10 compound scales used by the E3 mdet YAMLs.
SCALES: Dict[str, str] = {
    "n": "[0.33, 0.25, 1024]",
    "s": "[0.33, 0.50, 1024]",
    "m": "[0.67, 0.75, 768]",
    "b": "[0.67, 1.00, 512]",
    "l": "[1.00, 1.00, 512]",
    "x": "[1.00, 1.25, 512]",
}

_MATRIX_PLACEHOLDER = "/nfsv4/23039356r/data/billboard/data0806_m/yolo_rgb_detection5_10_c/co_occurrence_matrix_train.csv"


def _build_final_config(size: str, output_dir: Path) -> Path:
    """Derive a GIA + FGA source YAML for one YOLOv10 scale."""
    source = PROJECT_ROOT / f"ultralytics/cfg/models/experiments/yolov10{size}-mdetect.yaml"
    if not source.is_file():
        raise FileNotFoundError(f"Missing YOLOv10 E3 source config: {source}")

    text = source.read_text(encoding="utf-8")
    scale_marker = f"  {size}:"
    if scale_marker not in text:
        raise ValueError(f"{source} does not contain the expected {size!r} scale")

    # Add GIA-v2 at the two backbone downsampling positions used by the
    # selected YOLOv10x final configuration.  The head downsampling SCDown
    # remains an ordinary feature path.
    updated = text.replace(
        "  - [-1, 1, SCDown, [512, 3, 2]] # 5-P4/16",
        "  - [-1, 1, SCDown, [512, 3, 2, True, True, True]] # 5-P4/16 + GIA-v2",
        1,
    )
    if updated == text:
        raise ValueError(f"Could not insert GIA-v2 at P4 into {source}")
    text = updated

    updated = text.replace(
        "  - [-1, 1, SCDown, [1024, 3, 2]] # 7-P5/32",
        "  - [-1, 1, SCDown, [1024, 3, 2, True, True, True]] # 7-P5/32 + GIA-v2",
        1,
    )
    if updated == text:
        raise ValueError(f"Could not insert GIA-v2 at P5 into {source}")
    text = updated

    # Keep the fixed-graph token in the source so the normal launcher can
    # materialize the selected GIN operator with --gnn-type gin.  The matrix
    # path is replaced by --com-path on the training machine.
    old_head = "[nc, na, nal, [False, None, None]]"
    new_head = (
        "[nc, na, nal, [False, None, 'fga_margin_residual', False, "
        f"{_MATRIX_PLACEHOLDER}]]"
    )
    if text.count(old_head) != 1:
        raise ValueError(f"Could not locate the standard mdet head in {source}")
    text = text.replace(old_head, new_head, 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"yolov10{size}_MAYOLO_GIA_Cross_GIN.yaml"
    target.write_text(text, encoding="utf-8")
    return target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train final MAYOLO multi-scale variants")
    parser.add_argument("--sizes", nargs="+", choices=tuple(SCALES), required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--data", default="ultralytics/cfg/mayolo_r1/mayolo_v3.yaml")
    parser.add_argument("--project", default="runs/experiments/E3_versions")
    parser.add_argument("--label", default="E3_MAYOLO_final")
    parser.add_argument("--stage1-epochs", type=int, default=100)
    parser.add_argument("--stage2-epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--w4", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hsv-h", type=float, default=0.0)
    parser.add_argument("--hsv-s", type=float, default=0.2)
    parser.add_argument("--hsv-v", type=float, default=0.2)
    parser.add_argument(
        "--com-path",
        default="/localnvme/data/billboard/mayolo_v3/co_occurrence_matrix_train.csv",
    )
    parser.add_argument("--pretrain-dir", default=None)
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip completed size runs (use --no-skip-existing to retrain them)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _run_one(args: argparse.Namespace, size: str, generated_dir: Path) -> int:
    config = _build_final_config(size, generated_dir)
    variant = f"mayolo{size}"
    pretrain = f"yolov10{size}.pt"
    if args.pretrain_dir:
        pretrain = str(Path(args.pretrain_dir) / pretrain)

    command: List[str] = [
        sys.executable,
        "scripts/train_mdet_experiments.py",
        "gca-structure",
        "--label",
        args.label,
        "--data",
        args.data,
        "--project",
        args.project,
        "--stage1-epochs",
        str(args.stage1_epochs),
        "--stage2-epochs",
        str(args.stage2_epochs),
        "--batch",
        str(args.batch),
        "--imgsz",
        str(args.imgsz),
        "--workers",
        str(args.workers),
        "--w4",
        str(args.w4),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--hsv-h",
        str(args.hsv_h),
        "--hsv-s",
        str(args.hsv_s),
        "--hsv-v",
        str(args.hsv_v),
        "--variant",
        f"{variant}={config}",
        "--pretrain-map",
        f"{variant}={pretrain}",
        "--gnn-type",
        "gin",
        "--com-path",
        args.com_path,
    ]
    if args.skip_existing:
        command.append("--skip-existing")
    if args.dry_run:
        command.append("--dry-run")

    print(f"[MAYOLO {size}] device={args.device} pretrain={pretrain}", flush=True)
    print("[command] " + " ".join(str(part) for part in command), flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=_thread_limited_environment())
    if completed.returncode:
        print(f"[MAYOLO failed] size={size}, returncode={completed.returncode}", file=sys.stderr, flush=True)
    return completed.returncode


def _thread_limited_environment() -> Dict[str, str]:
    """Return the requested CPU-thread limits for every child trainer."""
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "4"
    environment.setdefault("PYTHONUNBUFFERED", "1")
    return environment


def main() -> int:
    args = _build_parser().parse_args()
    generated_dir = PROJECT_ROOT / args.project / "_generated_mayolo_configs"
    failed = []
    for size in args.sizes:
        if _run_one(args, size, generated_dir):
            failed.append(size)
    if failed:
        print(f"MAYOLO failed sizes: {' '.join(failed)}", file=sys.stderr)
        return 1
    print(f"MAYOLO completed sizes: {' '.join(args.sizes)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
