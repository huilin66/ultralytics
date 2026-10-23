# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Evaluate an HMT clean-v1 checkpoint and dump per-class metrics as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="dataset.yaml (absolute list files are fine)")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        conf=args.conf,
        plots=False,
        verbose=False,
    )
    names = metrics.names if hasattr(metrics, "names") else {}
    box = metrics.box
    per_class = []
    ap_class = getattr(box, "ap_class_index", [])
    for i, cls_id in enumerate(ap_class):
        cls_id = int(cls_id)
        record = {
            "class_id": cls_id,
            "class": names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id),
            "precision": round(float(box.p[i]), 4),
            "recall": round(float(box.r[i]), 4),
            "ap50": round(float(box.ap50[i]), 4),
            "ap50_95": round(float(box.ap[i]), 4),
        }
        per_class.append(record)
    result = {
        "weights": str(args.weights),
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "overall": {
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "map50": round(float(box.map50), 4),
            "map50_95": round(float(box.map), 4),
        },
        "per_class": per_class,
        "instances": int(sum(box.nc_i) if hasattr(box, "nc_i") else 0),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
