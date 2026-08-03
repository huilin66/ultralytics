# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train the prepared HMT datasets with one reproducible command."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


DATASETS = {
    "t": {
        "yaml": "hmt_t_update.yaml",
        "name": "hmt_t_update-yolov8x",
        "epochs": 240,
        "imgsz": 640,
        "batch": -1,
    },
    "rgb": {
        "yaml": "hmt_rgb_update.yaml",
        "name": "hmt_rgb_update-yolov8x",
        "epochs": 240,
        "imgsz": 768,
        "batch": -1,
    },
    "cube": {
        "yaml": "hmt_bp_cube_update.yaml",
        "name": "hmt_bp_cube_update-yolov8x",
        "epochs": 300,
        "imgsz": 1024,
        "batch": -1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("yolov8x.pt"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    for key, defaults in DATASETS.items():
        parser.add_argument(f"--epochs-{key}", type=int, default=defaults["epochs"])
        parser.add_argument(f"--imgsz-{key}", type=int, default=defaults["imgsz"])
        parser.add_argument(f"--batch-{key}", type=int, default=defaults["batch"])
    return parser.parse_args()


def resolve_model(model: Path, repo_root: Path) -> Path:
    if model.is_absolute() and model.exists():
        return model
    candidates = [model, repo_root / model, repo_root / model.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    # Let Ultralytics resolve a model name such as yolov8x.pt/download it if
    # the user intentionally supplied one that is not local.
    return model


def train_one(args: argparse.Namespace, key: str) -> None:
    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from ultralytics import YOLO

    settings = DATASETS[key]
    config = repo_root / "ultralytics" / "cfg" / "datasets_hmt" / settings["yaml"]
    if not config.exists():
        raise FileNotFoundError(f"dataset config not found: {config}")
    if not args.data_root.exists():
        raise FileNotFoundError(f"prepared data root not found: {args.data_root}")
    model_path = resolve_model(args.model, repo_root)
    print(f"[hmt_train] dataset={key} model={model_path} data={config}")
    model = YOLO(str(model_path))
    model.train(
        data=str(config),
        epochs=getattr(args, f"epochs_{key}"),
        imgsz=getattr(args, f"imgsz_{key}"),
        batch=getattr(args, f"batch_{key}"),
        device=args.device,
        workers=args.workers,
        project=str(args.run_root),
        name=settings["name"],
        exist_ok=False,
        pretrained=True,
        optimizer="auto",
        cos_lr=True,
        close_mosaic=20,
        patience=80,
        save=True,
        save_period=10,
        plots=True,
        cache=False,
        amp=True,
        seed=args.seed,
        deterministic=True,
        verbose=True,
    )


def main() -> int:
    args = parse_args()
    keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for key in keys:
        train_one(args, key)
    print("[hmt_train] requested HMT training jobs completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
