"""Run a short GPU-memory smoke test for mdet model variants.

The test intentionally uses the real mdet training entry point from
``mayolo_r1.py``.  It runs each selected model for two epochs, records the
peak CUDA memory, and continues with the remaining models when one model is
missing a checkpoint or runs out of memory.  This script only covers mdet;
segmentation models are not included.

Example (Linux/bash):

    python scripts/gpu_memory_smoke_test.py \
      --data path/to/billboard_mdet.yaml \
      --batch 16 --imgsz 640 --device 0 \
      --project runs/gpu_memory_smoke

For the E2.1 GIA checks, select the ``e2_1_*`` names and set ``--epochs 1``.

Use ``--pretrain-map NAME=CHECKPOINT`` when a checkpoint is not in the
standard Ultralytics search path.  The default model list contains the
largest configured variant of every mdet family currently used by the
project: YOLOv8x, YOLOv9e, YOLOv10x, YOLOv11x, YOLOv12x, YOLOv13x, YOLO26x,
MAYOLOx, and RT-DETR-L.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cli_compat import add_bool_argument


MAX_MDET_MODELS = {
    "yolov8x": {
        "config": "ultralytics/cfg/models/experiments/yolov8x-mdetect.yaml",
        "pretrain": "yolov8x.pt",
        "network": "yolo",
    },
    # YOLOv9's largest configured variant is e, not x.
    "yolov9e": {
        "config": "ultralytics/cfg/models/experiments/yolov9e-mdetect.yaml",
        "pretrain": "yolov9e.pt",
        "network": "yolo",
    },
    "yolov10x": {
        "config": "ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml",
        "pretrain": "yolov10x.pt",
        "network": "yolo",
    },
    "yolov11x": {
        "config": "ultralytics/cfg/models/experiments/yolov11x-mdetect.yaml",
        "pretrain": "yolo11x.pt",
        "network": "yolo",
    },
    "yolov12x": {
        "config": "ultralytics/cfg/models/experiments/yolov12x-mdetect.yaml",
        "pretrain": "yolo12x.pt",
        "network": "yolo",
    },
    "yolov13x": {
        "config": "ultralytics/cfg/models/experiments/yolov13x-mdetect.yaml",
        "pretrain": "yolov13x.pt",
        "network": "yolo",
    },
    "yolov26x": {
        "config": "ultralytics/cfg/models/experiments/yolov26x-mdetect.yaml",
        "pretrain": "yolo26x.pt",
        "network": "yolo",
    },
    "mayolox": {
        "config": "ultralytics/cfg/models/mayolo/mayolovx.yaml",
        "pretrain": "yolov10x.pt",
        "network": "yolo",
    },
    "rtdetr-l": {
        "config": "ultralytics/cfg/models/rt-detr/rtdetr-l-md.yaml",
        "pretrain": "rtdetr-l.pt",
        "network": "rtdetr",
    },
}

E21_GIA_RES_MODELS = {
    "e2_1_gia5_res": {
        "config": "ultralytics/cfg/models/exp_ablation/yolov10x_GIA_5_Res.yaml",
        "pretrain": "yolov10x.pt",
        "network": "yolo",
    },
    "e2_1_gia7_res": {
        "config": "ultralytics/cfg/models/exp_ablation/yolov10x_GIA_7_Res.yaml",
        "pretrain": "yolov10x.pt",
        "network": "yolo",
    },
    "e2_1_gia8_res": {
        "config": "ultralytics/cfg/models/exp_ablation/yolov10x_GIA_8_Res.yaml",
        "pretrain": "yolov10x.pt",
        "network": "yolo",
    },
    "e2_1_gia9_res": {
        "config": "ultralytics/cfg/models/exp_ablation/yolov10x_GIA_9_Res.yaml",
        "pretrain": "yolov10x.pt",
        "network": "yolo",
    },
    "e2_1_gia10_res": {
        "config": "ultralytics/cfg/models/exp_ablation/yolov10x_GIA_10_Res.yaml",
        "pretrain": "yolov10x.pt",
        "network": "yolo",
    },
    "e2_1_gia5_7_res": {
        "config": "ultralytics/cfg/models/exp_ablation/yolov10x_GIA_5_7_Res.yaml",
        "pretrain": "yolov10x.pt",
        "network": "yolo",
    },
}

E21_GIA_V2_MODELS = {
    "e2_1_gia_v2_7": {
        "config": "ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_7.yaml",
        "pretrain": "yolov10x.pt",
        "network": "yolo",
    },
    "e2_1_gia_v2_8": {
        "config": "ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_8.yaml",
        "pretrain": "yolov10x.pt",
        "network": "yolo",
    },
    "e2_1_gia_v2_9": {
        "config": "ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_9.yaml",
        "pretrain": "yolov10x.pt",
        "network": "yolo",
    },
    "e2_1_gia_v2_10": {
        "config": "ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_10.yaml",
        "pretrain": "yolov10x.pt",
        "network": "yolo",
    },
    "e2_1_gia_v2_5_7": {
        "config": "ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_5_7.yaml",
        "pretrain": "yolov10x.pt",
        "network": "yolo",
    },
}

SMOKE_MODELS = {**MAX_MDET_MODELS, **E21_GIA_RES_MODELS, **E21_GIA_V2_MODELS}

SUMMARY_FIELDS = (
    "model",
    "config",
    "pretrain",
    "network",
    "epochs",
    "batch",
    "imgsz",
    "device",
    "seed",
    "status",
    "elapsed_seconds",
    "max_memory_allocated_gib",
    "max_memory_reserved_gib",
    "memory_free_before_gib",
    "memory_total_gib",
    "best",
    "error",
)


def _parse_key_value(items: Sequence[str]) -> Dict[str, str]:
    """Parse repeated ``NAME=VALUE`` options."""
    result: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--pretrain-map expects NAME=CHECKPOINT, got: {item!r}")
        name, value = (part.strip() for part in item.split("=", 1))
        if not name or not value:
            raise ValueError(f"--pretrain-map expects non-empty NAME and CHECKPOINT, got: {item!r}")
        if name in result:
            raise ValueError(f"Duplicate checkpoint mapping for {name!r}")
        result[name] = value
    return result


def _resolve_device(device: str):
    """Validate a single CUDA device and return its torch device/index."""
    import torch

    value = str(device).strip().lower()
    if value == "cuda":
        index = torch.cuda.current_device()
    elif value.startswith("cuda:"):
        index = int(value.split(":", 1)[1])
    elif value.isdigit():
        index = int(value)
    else:
        raise ValueError("GPU memory smoke test requires one CUDA device, e.g. --device 0")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current Python environment")
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {index} is unavailable; count={torch.cuda.device_count()}")
    return torch.device(f"cuda:{index}"), index


def _gib(value: int) -> float:
    """Convert bytes to GiB for human-readable summaries."""
    return round(float(value) / (1024**3), 3)


def _reset_memory(torch, device) -> None:
    """Clear cached allocations and reset the per-run CUDA peak counters."""
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def _memory_snapshot(torch, device) -> Dict[str, float]:
    """Return the peak allocated and reserved CUDA memory for this run."""
    return {
        "max_memory_allocated_gib": _gib(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_gib": _gib(torch.cuda.max_memory_reserved(device)),
    }


def _write_record(path: Path, record: Dict[str, object]) -> None:
    """Append one JSONL record, creating its parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _report_path(project: Path, stem: str) -> Path:
    """Return a new report path without overwriting a previous smoke test."""
    path = project / f"{stem}.csv"
    if not path.exists():
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project / f"{stem}_{stamp}.csv"


def _run_one(args, name: str, spec: Dict[str, str], checkpoint: str, torch, device, index: int) -> Dict[str, object]:
    """Run one short mdet smoke test and return its memory record."""
    from mayolo_r1 import myolo_train
    from ultralytics import RTDETR, YOLO

    network = RTDETR if spec["network"] == "rtdetr" else YOLO
    started = time.perf_counter()
    record: Dict[str, object] = {
        "model": name,
        "config": spec["config"],
        "pretrain": checkpoint,
        "network": spec["network"],
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "seed": args.seed,
        "status": "started",
        "elapsed_seconds": "",
        "max_memory_allocated_gib": "",
        "max_memory_reserved_gib": "",
        "memory_free_before_gib": "",
        "memory_total_gib": "",
        "best": "",
        "error": "",
    }

    print(f"\n=== GPU smoke: {name} | {args.epochs} epochs | batch={args.batch} | imgsz={args.imgsz} ===")
    _reset_memory(torch, device)
    free, total = torch.cuda.mem_get_info(index)
    record["memory_free_before_gib"] = _gib(free)
    record["memory_total_gib"] = _gib(total)

    try:
        best = myolo_train(
            spec["config"],
            pretrain_path=checkpoint,
            network=network,
            auto_optim=args.auto_optim,
            data=args.data,
            device=args.device,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            seed=args.seed,
            amp=args.amp,
            val=args.val,
            patience=args.epochs,
            mdet=args.w4,
            project=args.project,
            name=f"{args.name}_{name}",
            exist_ok=args.exist_ok,
        )
        record["status"] = "finished"
        record["best"] = str(best) if best else ""
    except torch.cuda.OutOfMemoryError as error:
        record["status"] = "oom"
        record["error"] = repr(error)
        print(f"[OOM] {name}: {error}")
    except Exception as error:  # Continue the matrix so one missing weight does not hide other results.
        record["status"] = "failed"
        record["error"] = repr(error)
        print(f"[failed] {name}: {error}")
    finally:
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        record.update(_memory_snapshot(torch, device))
        record["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        print(
            f"[{record['status']}] {name}: "
            f"peak_allocated={record['max_memory_allocated_gib']} GiB, "
            f"peak_reserved={record['max_memory_reserved_gib']} GiB"
        )
        gc.collect()
        torch.cuda.empty_cache()
    return record


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a short GPU-memory smoke test for mdet model variants"
    )
    parser.add_argument("--data", required=True, help="mdet dataset YAML")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(SMOKE_MODELS),
        default=list(MAX_MDET_MODELS),
        help="models to test (default: every largest configured mdet variant)",
    )
    parser.add_argument("--pretrain-map", action="append", default=[], metavar="NAME=CHECKPOINT")
    parser.add_argument("--epochs", type=int, default=2, help="epochs per model (default: 2)")
    parser.add_argument("--batch", type=int, default=16, help="batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0", help="one CUDA device, e.g. 0 or cuda:0")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument("--project", default="runs/gpu_memory_smoke")
    parser.add_argument("--name", default="gpu_smoke")
    parser.add_argument("--w4", type=float, default=0.5, help="mdet/attribute loss gain")
    add_bool_argument(parser, "--auto-optim", default=False)
    add_bool_argument(parser, "--amp", default=True)
    add_bool_argument(parser, "--val", default=True, help="run validation after each epoch")
    add_bool_argument(parser, "--exist-ok", default=False)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse options and run the selected GPU smoke tests."""
    args = _build_parser().parse_args(argv)
    if args.epochs < 1:
        raise ValueError("--epochs must be positive; the smoke-test default is 2")
    if args.batch < 1 or args.imgsz < 1 or args.workers < 0:
        raise ValueError("--batch and --imgsz must be positive, and --workers cannot be negative")
    if args.w4 < 0:
        raise ValueError("--w4 must be non-negative")

    checkpoints = _parse_key_value(args.pretrain_map)
    unknown = set(checkpoints) - set(SMOKE_MODELS)
    if unknown:
        raise ValueError(f"Unknown --pretrain-map model(s): {', '.join(sorted(unknown))}")

    import torch

    device, index = _resolve_device(args.device)
    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)
    summary_path = _report_path(project, "gpu_memory_smoke_summary")
    jsonl_path = summary_path.with_suffix(".jsonl")

    print(f"CUDA device: {torch.cuda.get_device_name(index)} ({_gib(torch.cuda.get_device_properties(index).total_memory)} GiB)")
    print(f"Models: {', '.join(args.models)}")
    print(f"Each model: {args.epochs} epochs, batch={args.batch}, imgsz={args.imgsz}, amp={args.amp}")

    records = []
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for name in args.models:
            spec = SMOKE_MODELS[name]
            checkpoint = checkpoints.get(name, spec["pretrain"])
            record = _run_one(args, name, spec, checkpoint, torch, device, index)
            records.append(record)
            writer.writerow(record)
            file.flush()
            _write_record(jsonl_path, record)

    successful = [
        record
        for record in records
        if record["status"] == "finished" and record["max_memory_reserved_gib"] != ""
    ]
    if successful:
        ranked = sorted(
            successful,
            key=lambda record: float(record["max_memory_reserved_gib"]),
            reverse=True,
        )
        print(f"\nPeak reserved-memory ranking (batch={args.batch}, imgsz={args.imgsz}):")
        for rank, record in enumerate(ranked, start=1):
            print(
                f"{rank}. {record['model']}: "
                f"{record['max_memory_reserved_gib']} GiB reserved / "
                f"{record['max_memory_allocated_gib']} GiB allocated"
            )
    print(f"\nSummary CSV: {summary_path}")
    print(f"Summary JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()
