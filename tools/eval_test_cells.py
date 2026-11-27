# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Evaluate ``weights/best.pt`` of the E2 branch-ablation cells on the frozen test split.

Each cell reads a different data YAML, because the modality view is part of the condition:
``b0-rgb`` reads three channels, ``b3-rgb-t`` four, and so on. The test split is touched once
per checkpoint and the record is written to ``<project>/test_eval/<item>-seed<k>-best.json``.

Examples:
    python tools/eval_test_cells.py --project runs/e2_rlivit --items b3-rgb-t b4-rgb-d b5-t-d
    python tools/eval_test_cells.py --project runs/e2_rlivit --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

KEYS = ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")
DATA = {
    "b0-rgb": "data-rgb.yaml",
    "b1-thermal": "data-thermal.yaml",
    "b2-depth": "data-depth.yaml",
    "b3-rgb-t": "data-rgb-t.yaml",
    "b4-rgb-d": "data-rgb-d.yaml",
    "b5-t-d": "data-t-d.yaml",
}


def data_yaml(item: str) -> str:
    """Return the dataset view a cell was trained on.

    The explicit table covers the E2 cells; sweep cells are named ``<family>-rgb`` or ``<family>-rgb-t``,
    so their modality view follows from the suffix.
    """
    if item in DATA:
        return DATA[item]
    for suffix, name in (("-rgb-t", "data-rgb-t.yaml"), ("-rgb", "data-rgb.yaml")):
        if item.endswith(suffix):
            return name
    raise SystemExit(f"no dataset view known for {item}")


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=str(REPO / "runs" / "e2_rlivit"))
    parser.add_argument("--items", nargs="+", default=list(DATA), help="cells to evaluate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--data-root", default="/srv/nas/home/dataset/R-LiViT_mdet")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--overwrite", action="store_true", help="re-evaluate a checkpoint that already has a record")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Evaluate every requested checkpoint and persist one JSON per checkpoint."""
    args = parse_args(argv)
    from ultralytics import YOLO
    from ultralytics.models.multimodal import MultiModalYOLO

    project = Path(args.project)
    out_dir = project / "test_eval"
    out_dir.mkdir(exist_ok=True)
    failed = 0
    for item in args.items:
        data = str(Path(args.data_root) / data_yaml(item))
        for seed in args.seeds:
            run_dir = project / f"{item}-seed{seed}"
            ckpt = run_dir / "weights" / "best.pt"
            target = out_dir / f"{item}-seed{seed}-best.json"
            if not ckpt.is_file():
                print(f"[skip] {run_dir.name}: no weights/best.pt", flush=True)
                continue
            if target.is_file() and not args.overwrite:
                print(f"[skip] {target.name}: exists", flush=True)
                continue
            started = time.time()
            try:
                facade = YOLO if item == "b0-rgb" or (item not in DATA and item.endswith("-rgb")) else MultiModalYOLO
                model = facade(str(ckpt), task="detect")
                results = model.val(
                    data=data,
                    split="test",
                    imgsz=args.imgsz,
                    batch=args.batch,
                    device=args.device,
                    plots=False,
                    verbose=False,
                )
                record = {
                    "item": item,
                    "seed": seed,
                    "checkpoint": "best.pt",
                    "seconds": round(time.time() - started, 1),
                }
                record.update({key: float(results.results_dict[key]) for key in KEYS})
                record["fitness"] = record["metrics/mAP50-95(B)"]
                record["per_class_mAP50-95"] = {
                    str(cls): round(float(value), 4) for cls, value in enumerate(results.box.maps)
                }
            except Exception as error:  # noqa: BLE001 - keep evaluating the remaining checkpoints
                failed += 1
                print(f"[error] {run_dir.name}: {error}", flush=True)
                continue
            target.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
            print(f"[test] {item}-seed{seed}: mAP50-95={record['metrics/mAP50-95(B)']:.4f}", flush=True)
            del model, results
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
