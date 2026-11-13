# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run one seed of the E2 baseline matrix (R-LiViT, B0-B5 + B8 + C1).

Each process owns one GPU and runs every new condition for its seed, so three
processes give seeds 0/1/2 in parallel. B6 (IF) and B7 (BF) reuse the published
E1.5 runs and are intentionally absent here; the summarizer joins them in.

Every condition trains with the identical E1.5 recipe: COCO transfer, AdamW at
lr0=1e-3, cosine decay, 200 epochs, imgsz 640, physical batch 4 with nbs=64.

Examples:
    CUDA_VISIBLE_DEVICES=0 python tools/run_e2_baselines.py --seed 0
    CUDA_VISIBLE_DEVICES=1 python tools/run_e2_baselines.py --seed 1 --items b1-thermal b2-depth
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ultralytics.cfg import get_save_dir  # noqa: E402
from ultralytics.models.multimodal import MultiModalYOLO  # noqa: E402
from ultralytics.models.multimodal.pretrained import load_coco_pretrained  # noqa: E402

# Every condition except B0 starts from the same COCO checkpoint transferred into its template.
RUNS = {
    "b0-rgb": {"model": "official", "data": "data-rgb.yaml"},
    "b1-thermal": {"model": "yolo26x-mm1-if.yaml", "data": "data-thermal.yaml"},
    "b2-depth": {"model": "yolo26x-mm1-if.yaml", "data": "data-depth.yaml"},
    "b3-rgb-t": {"model": "yolo26x-mm2-31-if.yaml", "data": "data-rgb-t.yaml"},
    "b4-rgb-d": {"model": "yolo26x-mm2-31-if.yaml", "data": "data-rgb-d.yaml"},
    "b5-t-d": {"model": "yolo26x-mm2-11-if.yaml", "data": "data-t-d.yaml"},
    "b8-gated": {"model": "yolo26x-mm3-bf-gated.yaml", "data": "data.yaml"},
    "c1-conv": {"model": "yolo26x-mm3-bf-conv.yaml", "data": "data.yaml"},
}


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", nargs="+", default=list(RUNS), choices=list(RUNS), help="conditions to train")
    parser.add_argument("--seed", type=int, required=True, help="training seed")
    parser.add_argument(
        "--data-root", default="/srv/nas/home/dataset/R-LiViT_mdet", help="directory of the E2 data YAMLs"
    )
    parser.add_argument("--pretrained", default="yolo26x.pt", help="COCO checkpoint; empty disables transfer")
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
    parser.add_argument("--project", default=str(REPO / "runs" / "e2_rlivit"))
    parser.add_argument("--overwrite", action="store_true", help="rerun even when a finished run exists")
    return parser.parse_args(argv)


def completed_epochs(run_dir: Path) -> int:
    """Return how many epochs a run directory has recorded, 0 when absent."""
    path = run_dir / "results.csv"
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)]
    return len(rows)


def read_results(run_dir: Path) -> list[dict]:
    """Read results.csv with stripped header and value whitespace."""
    path = run_dir / "results.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [{k.strip(): (v or "").strip() for k, v in row.items()} for row in reader]


def build_model(args: argparse.Namespace, item: str):
    """Build one condition and apply its COCO initialization."""
    template = RUNS[item]["model"]
    if template == "official":
        from ultralytics import YOLO

        return YOLO(args.pretrained or "yolo26x.yaml", task="detect")
    cfg = REPO / "ultralytics" / "cfg" / "mmodels" / template
    if not cfg.is_file():
        raise SystemExit(f"missing config: {cfg}")
    model = MultiModalYOLO(str(cfg), task="detect")
    if args.pretrained:
        report = load_coco_pretrained(model, args.pretrained, verbose=False)
        print(f"[transfer] {item}-seed{args.seed}: {report.summary()}", flush=True)
    return model


def run_one(args: argparse.Namespace, item: str) -> dict:
    """Train one condition for one seed and return its metrics row."""
    data = str(Path(args.data_root) / RUNS[item]["data"])
    if not Path(data).is_file():
        raise SystemExit(f"missing dataset: {data}")
    name = f"{item}-seed{args.seed}"
    # Resolve exactly as BaseTrainer does, so a relative --project also lands under runs/<task>/.
    run_dir = Path(
        get_save_dir(SimpleNamespace(project=args.project, name=name, task="detect", mode="train", exist_ok=True))
    )
    done = completed_epochs(run_dir)
    if done >= args.epochs and (run_dir / "weights" / "best.pt").is_file() and not args.overwrite:
        rows = read_results(run_dir)
        last = rows[-1]
        print(f"[skip] {name}: {done} epochs, mAP50-95={last.get('metrics/mAP50-95(B)', '?')}", flush=True)
        return {"item": item, "status": "skipped", **last}

    import torch

    model = build_model(args, item)
    overrides = {
        "data": data,
        "task": "detect",
        "mode": "train",
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "fraction": args.fraction,
        "optimizer": "AdamW",
        "lr0": args.lr0,
        "lrf": args.lrf,
        "cos_lr": True,
        "warmup_epochs": args.warmup_epochs,
        "close_mosaic": args.close_mosaic,
        "seed": args.seed,
        "deterministic": True,
        "val": True,
        "plots": False,
        "save": True,
        "project": args.project,
        "name": name,
        "exist_ok": True,
    }

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    # pretrained stays at its default so train() reuses the module carrying the transferred weights.
    model.train(**overrides)
    elapsed = time.time() - started
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    rows = read_results(run_dir)
    last = rows[-1]
    row = {
        "item": item,
        "status": "ok",
        "seed": args.seed,
        "epochs": len(rows),
        "seconds": round(elapsed, 1),
        "epoch_seconds": round(elapsed / max(len(rows), 1), 1),
        "peak_gb": round(peak_gb, 2),
        **{k: v for k, v in last.items() if k.startswith(("metrics/", "train/", "val/"))},
    }
    (run_dir / "e2.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(
        f"[done] {name}: {row['epoch_seconds']}s/epoch, {row['peak_gb']}GB, {last.get('metrics/mAP50-95(B)')}",
        flush=True,
    )
    return row


def main(argv=None) -> int:
    """Run the requested conditions for one seed."""
    args = parse_args(argv)
    Path(args.project).mkdir(parents=True, exist_ok=True)
    rows = []
    for item in args.items:
        try:
            rows.append(run_one(args, item))
        except RuntimeError as error:  # keep the remaining conditions alive after an OOM
            if "out of memory" not in str(error).lower():
                raise
            print(f"[oom] {item}-seed{args.seed}: {error}", flush=True)
            rows.append({"item": item, "status": "oom", "seed": args.seed})
    summary = Path(args.project) / f"seed{args.seed}_summary.json"
    summary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[summary] {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
