# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train one seed of the detector-family sweep: RGB-only against RGB+T.

The sweep asks whether the dual-modality gain survives a change of detector family. Every
family contributes two cells with the identical recipe used by E1.5/E2: an RGB-only
baseline (the stock detector, ``data-rgb.yaml``) and an RGB+T baseline (two modality
backbones fused at P3/P4/P5 with BF, ``data-rgb-t.yaml``). Single and dual cells share the
family's own COCO checkpoint, so the only difference between them is the second modality.

Each process owns one GPU, so three processes give seeds 0/1/2 in parallel.

Examples:
    CUDA_VISIBLE_DEVICES=0 python tools/run_backbone_sweep.py --seed 0 --items yolov8x-rgb yolov8x-rgb-t
    CUDA_VISIBLE_DEVICES=1 python tools/run_backbone_sweep.py --seed 0 --items yolo11x-rgb yolo11x-rgb-t --epochs 1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch

from ultralytics import __version__
from ultralytics.cfg import get_save_dir
from ultralytics.models.multimodal import MultiModalYOLO
from ultralytics.models.multimodal.pretrained import load_coco_pretrained

# family -> the largest scale it ships, its stock YAML (donor graph) and its two-modality template.
FAMILIES = {
    "yolov8x": {"stock": "v8/yolov8x.yaml", "ckpt": "yolov8x.pt", "dual": "yolov8x-mm2-31-bf.yaml"},
    "yolov9e": {"stock": "v9/yolov9e.yaml", "ckpt": "yolov9e.pt", "dual": "yolov9e-mm2-31-bf.yaml"},
    "yolov10x": {"stock": "v10/yolov10x.yaml", "ckpt": "yolov10x.pt", "dual": "yolov10x-mm2-31-bf.yaml"},
    "yolo11x": {"stock": "11/yolo11x.yaml", "ckpt": "yolo11x.pt", "dual": "yolo11x-mm2-31-bf.yaml"},
    "yolo12x": {"stock": "12/yolo12x.yaml", "ckpt": "yolo12x.pt", "dual": "yolo12x-mm2-31-bf.yaml"},
    "yolo26x": {"stock": "26/yolo26x.yaml", "ckpt": "yolo26x.pt", "dual": "yolo26x-mm2-31-bf.yaml"},
}
RUNS = {f"{family}-{kind}": {"family": family, "kind": kind} for family in FAMILIES for kind in ("rgb", "rgb-t")}
DATA = {"rgb": "data-rgb.yaml", "rgb-t": "data-rgb-t.yaml"}


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", nargs="+", default=list(RUNS), choices=list(RUNS), help="cells to train")
    parser.add_argument("--seed", type=int, required=True, help="training seed")
    parser.add_argument(
        "--data-root", default="/srv/nas/home/dataset/R-LiViT_mdet", help="directory of the dataset YAMLs"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=4, help="physical batch; nbs=64 keeps the effective batch fixed")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=1.0, help="fraction of the training split (smoke tests)")
    parser.add_argument("--device", default="0")
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=0.5)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--project", default=str(REPO / "runs" / "backbone_sweep"))
    parser.add_argument("--overwrite", action="store_true", help="rerun even when a finished run exists")
    return parser.parse_args(argv)


def build_model(item: str):
    """Build one cell with its COCO initialization.

    The dual cell re-parses the released checkpoint through this repository's stock YAML before the
    transfer, so donor and target share one set of layer declarations; a checkpoint saved by an
    older release declares the same modules with different channel caps and would not match.
    """
    from ultralytics import YOLO

    spec = FAMILIES[RUNS[item]["family"]]
    if RUNS[item]["kind"] == "rgb":
        return YOLO(spec["ckpt"], task="detect")

    stock = REPO / "ultralytics" / "cfg" / "models" / spec["stock"]
    donor = YOLO(str(stock), task="detect").load(spec["ckpt"])
    cfg = REPO / "ultralytics" / "cfg" / "mmodels" / spec["dual"]
    model = MultiModalYOLO(str(cfg), task="detect")
    report = load_coco_pretrained(model, donor, verbose=False)
    print(f"[transfer] {item}: {report.summary()}", flush=True)
    return model


def _digest(*paths: Path) -> str:
    """Return a short sha256 over the given files, in order."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def describe(item: str, args: argparse.Namespace, data: Path) -> dict:
    """Record what determines this cell's numbers, so a later config change cannot pass unnoticed."""
    spec = FAMILIES[RUNS[item]["family"]]
    stock = REPO / "ultralytics" / "cfg" / "models" / spec["stock"]
    base = REPO / "ultralytics" / "cfg" / "mmodels" / "base.yaml"
    template = REPO / "ultralytics" / "cfg" / "mmodels" / spec["dual"] if RUNS[item]["kind"] == "rgb-t" else stock
    checkpoint = Path(spec["ckpt"])
    images = Path(args.data_root) / "images"
    return {
        "item": item,
        "template": template.name,
        "template_sha": _digest(template, base),
        "data_yaml": str(data),
        "data_sha": _digest(data),
        "images": {split: len(list((images / "rgb" / split).iterdir())) for split in ("train", "val")},
        "pretrained": {"file": checkpoint.name, "bytes": checkpoint.stat().st_size if checkpoint.is_file() else None},
        "recipe": {
            key: getattr(args, key)
            for key in ("epochs", "batch", "imgsz", "lr0", "lrf", "warmup_epochs", "close_mosaic", "seed")
        },
        "env": {"python": platform.python_version(), "torch": torch.__version__, "ultralytics": __version__},
    }


def pin_run(run_dir: Path, item: str, args: argparse.Namespace, data: Path) -> dict:
    """Write this run's provenance and refuse to continue when anything that decides its numbers changed."""
    record = describe(item, args, data)
    path = run_dir / "provenance.json"
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        for key in ("template_sha", "data_sha", "recipe"):
            if previous.get(key) != record[key]:
                raise SystemExit(
                    f"[conflict] {run_dir.name}: {key} changed since this run was trained "
                    f"({previous.get(key)} -> {record[key]}). "
                    "Train into a new name or delete the stale run; never mix two recipes in one directory."
                )
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def completed_epochs(run_dir: Path) -> int:
    """Return how many epochs a run directory has recorded, 0 when absent."""
    path = run_dir / "results.csv"
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return len([row for row in csv.DictReader(handle)])


def read_results(run_dir: Path) -> list[dict]:
    """Read results.csv with stripped header and value whitespace."""
    with (run_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [{k.strip(): (v or "").strip() for k, v in row.items()} for row in reader]


def run_one(args: argparse.Namespace, item: str) -> dict:
    """Train one cell for one seed and return its metrics row."""
    data = str(Path(args.data_root) / DATA[RUNS[item]["kind"]])
    if not Path(data).is_file():
        raise SystemExit(f"missing dataset: {data}")
    name = f"{item}-seed{args.seed}"
    run_dir = Path(
        get_save_dir(SimpleNamespace(project=args.project, name=name, task="detect", mode="train", exist_ok=True))
    )
    pin_run(run_dir, item, args, Path(data))
    done = completed_epochs(run_dir)
    if done >= args.epochs and (run_dir / "weights" / "best.pt").is_file() and not args.overwrite:
        last = read_results(run_dir)[-1]
        print(f"[skip] {name}: {done} epochs, mAP50-95={last.get('metrics/mAP50-95(B)', '?')}", flush=True)
        return {"item": item, "status": "skipped", **last}

    model = build_model(item)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    model.train(
        data=data,
        task="detect",
        mode="train",
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        fraction=args.fraction,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=True,
        warmup_epochs=args.warmup_epochs,
        close_mosaic=args.close_mosaic,
        seed=args.seed,
        deterministic=True,
        val=True,
        plots=False,
        save=True,
        project=args.project,
        name=name,
        exist_ok=True,
    )
    elapsed = time.time() - started
    rows = read_results(run_dir)
    last = rows[-1]
    row = {
        "item": item,
        "status": "ok",
        "seed": args.seed,
        "epochs": len(rows),
        "seconds": round(elapsed, 1),
        "epoch_seconds": round(elapsed / max(len(rows), 1), 1),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        **{k: v for k, v in last.items() if k.startswith(("metrics/", "train/", "val/"))},
    }
    (run_dir / "sweep.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(
        f"[done] {name}: {row['epoch_seconds']}s/epoch, {row['peak_gb']}GB, mAP50-95={last.get('metrics/mAP50-95(B)')}",
        flush=True,
    )
    return row


def main(argv=None) -> int:
    """Run the requested cells for one seed."""
    args = parse_args(argv)
    Path(args.project).mkdir(parents=True, exist_ok=True)
    rows = []
    for item in args.items:
        try:
            rows.append(run_one(args, item))
        except RuntimeError as error:  # keep the remaining cells alive after an OOM
            if "out of memory" not in str(error).lower():
                raise
            print(f"[oom] {item}-seed{args.seed}: {error}", flush=True)
            rows.append({"item": item, "status": "oom", "seed": args.seed})
    # One seed may be split over several GPUs, so the summary is keyed by the cells this process owned.
    summary = Path(args.project) / f"summary-seed{args.seed}-{rows[0]['item'] if rows else 'empty'}.json"
    summary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[summary] {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
