# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train HMT datasets using absolute image-list files derived from YAML."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml


DATASETS = {
    "t": {"yaml": "hmt_t_update.yaml", "name": "hmt_t_update-yolov8x", "epochs": 240, "imgsz": 640, "batch": -1},
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
    for candidate in (model, repo_root / model, repo_root / model.name):
        if candidate.exists():
            return candidate.resolve()
    return model


def dataset_root(data: dict, config: Path) -> Path:
    value = data.get("path") or config.parent
    root = Path(value)
    return (config.parent / root).resolve() if not root.is_absolute() else root.resolve()


def absolute_list(value: str | list[str], root: Path, out: Path) -> str:
    source = value[0] if isinstance(value, list) and len(value) == 1 else value
    if isinstance(source, str) and source.lower().endswith(".txt"):
        list_path = Path(source)
        list_path = list_path if list_path.is_absolute() else root / list_path
        if not list_path.is_file():
            raise FileNotFoundError(f"image list not found: {list_path}")
        lines = list_path.read_text(encoding="utf-8", errors="replace").splitlines()
    elif isinstance(value, list):
        lines = [str(item) for item in value]
    else:
        lines = [str(value)]
    output_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        path = Path(line)
        output_lines.append(str(path if path.is_absolute() else (root / path).resolve()))
    out.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")
    return str(out.resolve())


def runtime_config(config: Path, run_root: Path, key: str) -> Path:
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    root = dataset_root(data, config)
    workspace = run_root / "hmt_absolute_lists" / key
    workspace.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        if data.get(split):
            data[split] = absolute_list(data[split], root, workspace / f"{split}.txt")
    data["path"] = str(root)
    output = workspace / "dataset.yaml"
    output.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return output


def train_one(args: argparse.Namespace, key: str) -> None:
    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from ultralytics import YOLO

    settings = DATASETS[key]
    config = repo_root / "ultralytics" / "cfg" / "datasets_hmt" / settings["yaml"]
    if not config.exists():
        raise FileNotFoundError(f"dataset config not found: {config}")
    args.run_root.mkdir(parents=True, exist_ok=True)
    data_config = runtime_config(config, args.run_root, key)
    model_path = resolve_model(args.model, repo_root)
    print(f"[hmt_train] dataset={key} model={model_path} data={data_config}")
    model = YOLO(str(model_path))
    model.train(
        data=str(data_config),
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
    for key in DATASETS if args.dataset == "all" else (args.dataset,):
        train_one(args, key)
    print("[hmt_train] requested HMT training jobs completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
