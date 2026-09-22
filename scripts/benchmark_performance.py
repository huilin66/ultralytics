"""Benchmark model efficiency and inference performance.

This script deliberately separates model-forward timing from optional
dataset-level timing.  The former is reproducible with synthetic tensors and
works for the detector variants used in this project; the latter reuses the
project's Ultralytics evaluation loop and records its preprocessing,
inference, and postprocessing timings.

Examples (not executed by this file):

    python scripts/benchmark_performance.py \
        --weights path/to/yolov10x.pt path/to/mayolox.pt \
        --labels YOLOv10x MAYOLOx \
        --device 0 --batch 1 --precision fp16 \
        --output runs/experiments/performance/summary.csv

    python scripts/benchmark_performance.py \
        --weights path/to/model.pt --data path/to/data.yaml \
        --profile-dataset --device 0

The script writes a CSV and a JSON file.  It does not train models and does
not modify checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, select_device


CSV_FIELDS = (
    "model",
    "weight",
    "task",
    "head_mode",
    "device",
    "device_name",
    "precision",
    "fused",
    "batch",
    "imgsz",
    "warmup",
    "iterations",
    "parameters_m",
    "gflops",
    "weight_mib",
    "latency_batch_mean_ms",
    "latency_image_mean_ms",
    "latency_image_std_ms",
    "latency_image_p50_ms",
    "latency_image_p95_ms",
    "fps",
    "throughput_images_s",
    "peak_vram_allocated_gib",
    "peak_vram_reserved_gib",
    "dataset_preprocess_ms",
    "dataset_inference_ms",
    "dataset_loss_ms",
    "dataset_postprocess_ms",
    "dataset_latency_ms",
    "dataset_fps",
)


def _load_object(spec: str) -> Any:
    """Load an object specified as ``module.submodule:attribute``."""

    if ":" not in spec:
        raise ValueError(f"Expected MODULE:ATTRIBUTE, got {spec!r}")
    module_name, attribute = spec.split(":", maxsplit=1)
    return getattr(importlib.import_module(module_name), attribute)


def _weight_size_mib(weight: str) -> float | None:
    """Return a local checkpoint size in MiB when available."""

    path = Path(weight)
    if not path.is_file():
        return None
    return round(path.stat().st_size / 2**20, 3)


def _device_name(device: torch.device) -> str:
    """Return a stable human-readable device name."""

    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return str(device)


def _parameter_count(model: torch.nn.Module) -> float:
    """Return total parameter count in millions."""

    return round(sum(parameter.numel() for parameter in model.parameters()) / 1e6, 6)


def _set_one2many(model: YOLO) -> None:
    """Switch an mdetect model to its one-to-many inference head."""

    detector = getattr(model, "model", None)
    layers = getattr(detector, "model", None)
    if layers is None or not layers:
        raise RuntimeError("Could not locate the mdetect model head")
    switch = getattr(layers[-1], "use_one2many_head", None)
    if switch is None:
        raise RuntimeError("Checkpoint does not expose use_one2many_head()")
    switch()


def _time_forward(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> tuple[list[float], float | None, float | None]:
    """Measure forward latency and peak CUDA memory.

    Returned latency values are batch latencies in milliseconds.  CUDA events
    are used on GPU so host scheduling overhead is not included in the timing.
    """

    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    latencies: list[float] = []
    for _ in range(iterations):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.inference_mode():
                model(inputs)
            end.record()
            end.synchronize()
            latencies.append(float(start.elapsed_time(end)))
        else:
            start_time = time.perf_counter()
            with torch.inference_mode():
                model(inputs)
            latencies.append((time.perf_counter() - start_time) * 1000.0)

    if device.type != "cuda":
        return latencies, None, None

    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device) / 2**30
    reserved = torch.cuda.max_memory_reserved(device) / 2**30
    return latencies, round(allocated, 6), round(reserved, 6)


def _dataset_speed(model: YOLO, args: argparse.Namespace, label: str) -> dict[str, float | None]:
    """Run the existing dataset loop and extract its timing fields."""

    validator = _load_object(args.validator) if args.validator else None
    kwargs: dict[str, Any] = {
        "data": args.data,
        "split": args.split,
        "device": args.device,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "plots": False,
        "project": args.dataset_project,
        "name": f"{label}_speed",
    }
    if validator is not None:
        kwargs["validator"] = validator

    metrics = model.val(**kwargs)
    speed = getattr(metrics, "speed", {}) or {}
    preprocess = float(speed.get("preprocess", 0.0))
    inference = float(speed.get("inference", 0.0))
    loss = float(speed.get("loss", 0.0))
    postprocess = float(speed.get("postprocess", 0.0))
    deployment_latency = preprocess + inference + postprocess
    return {
        "dataset_preprocess_ms": round(preprocess, 6),
        "dataset_inference_ms": round(inference, 6),
        "dataset_loss_ms": round(loss, 6),
        "dataset_postprocess_ms": round(postprocess, 6),
        "dataset_latency_ms": round(deployment_latency, 6),
        "dataset_fps": round(1000.0 / deployment_latency, 6) if deployment_latency > 0 else None,
    }


def benchmark_one(weight: str, label: str, args: argparse.Namespace) -> dict[str, Any]:
    """Benchmark one checkpoint and return one serializable result row."""

    device = select_device(args.device, verbose=False)
    task_kwargs = {"task": args.task} if args.task else {}
    yolo = YOLO(weight, **task_kwargs)
    if args.head_mode == "one2many":
        _set_one2many(yolo)
    if args.fuse:
        yolo.fuse()
    network = yolo.model.to(device).eval()

    if args.precision == "fp16":
        if device.type != "cuda":
            raise ValueError("--precision fp16 requires a CUDA device")
        network.half()
    else:
        network.float()

    inputs = torch.rand((args.batch, 3, args.imgsz, args.imgsz), device=device)
    if args.precision == "fp16":
        inputs = inputs.half()

    try:
        gflops = float(get_flops(network, imgsz=args.imgsz))
    except Exception:
        gflops = 0.0

    latencies, peak_allocated, peak_reserved = _time_forward(
        network, inputs, device, args.warmup, args.iterations
    )
    batch_mean = float(np.mean(latencies))
    image_latencies = np.asarray(latencies, dtype=np.float64) / args.batch
    image_mean = float(np.mean(image_latencies))

    row: dict[str, Any] = {
        "model": label,
        "weight": weight,
        "task": getattr(yolo, "task", args.task or "auto"),
        "head_mode": args.head_mode,
        "device": str(device),
        "device_name": _device_name(device),
        "precision": args.precision,
        "fused": bool(args.fuse),
        "batch": args.batch,
        "imgsz": args.imgsz,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "parameters_m": _parameter_count(network),
        "gflops": round(gflops, 6),
        "weight_mib": _weight_size_mib(weight),
        "latency_batch_mean_ms": round(batch_mean, 6),
        "latency_image_mean_ms": round(image_mean, 6),
        "latency_image_std_ms": round(float(np.std(image_latencies)), 6),
        "latency_image_p50_ms": round(float(np.percentile(image_latencies, 50)), 6),
        "latency_image_p95_ms": round(float(np.percentile(image_latencies, 95)), 6),
        "fps": round(1000.0 / image_mean, 6) if image_mean > 0 else None,
        "throughput_images_s": round(args.batch * 1000.0 / batch_mean, 6) if batch_mean > 0 else None,
        "peak_vram_allocated_gib": peak_allocated,
        "peak_vram_reserved_gib": peak_reserved,
        "dataset_preprocess_ms": None,
        "dataset_inference_ms": None,
        "dataset_loss_ms": None,
        "dataset_postprocess_ms": None,
        "dataset_latency_ms": None,
        "dataset_fps": None,
    }

    if args.profile_dataset:
        row.update(_dataset_speed(yolo, args, label))
    return row


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs="+", required=True, help="One or more checkpoint paths")
    parser.add_argument("--labels", nargs="*", help="Optional labels aligned with --weights")
    parser.add_argument("--task", default=None, help="Optional Ultralytics task override, e.g. detect or mdetect")
    parser.add_argument("--head-mode", choices=("native", "one2many"), default="native")
    parser.add_argument("--device", default="0", help="CUDA device, CPU, or device string")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--fuse", action="store_true", help="Fuse Conv-BN layers before benchmarking")
    parser.add_argument("--profile-dataset", action="store_true", help="Also record dataset-loop timing")
    parser.add_argument("--data", default=None, help="Dataset YAML used with --profile-dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validator", default=None, help="Optional validator class as MODULE:ATTRIBUTE")
    parser.add_argument("--dataset-project", default="runs/experiments/performance_profile")
    parser.add_argument("--output", default="runs/experiments/performance_profile/summary.csv")
    return parser


def main() -> None:
    """Run the requested performance benchmark."""

    args = build_parser().parse_args()
    if args.batch < 1 or args.warmup < 0 or args.iterations < 1:
        raise ValueError("batch must be >= 1, warmup must be >= 0, and iterations must be >= 1")
    if args.profile_dataset and not args.data:
        raise ValueError("--profile-dataset requires --data")
    if args.labels and len(args.labels) not in (0, len(args.weights)):
        raise ValueError("--labels must have the same number of values as --weights")

    labels = args.labels if args.labels else [Path(weight).stem for weight in args.weights]
    rows = [benchmark_one(weight, label, args) for weight, label in zip(args.weights, labels)]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    json_output = output.with_suffix(".json")
    json_output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] {output}")
    print(f"[summary] {json_output}")


if __name__ == "__main__":
    main()
