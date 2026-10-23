# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train HMT clean datasets (bdd_hmt_clean_v1) using absolute image-list files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import yaml


DATASETS = {
    "sua_t": {
        "yaml": "hmt_clean_v1_sua_t.yaml",
        "name": "hmt_clean_v1_sua_t-yolov8x",
        "epochs": 200,
        "imgsz": 640,
        "batch": -1,
    },
    "sua_rgb": {
        "yaml": "hmt_clean_v1_sua_rgb.yaml",
        "name": "hmt_clean_v1_sua_rgb-yolov8x",
        "epochs": 200,
        "imgsz": 1280,
        "batch": -1,
    },
    "bp_cube": {
        "yaml": "hmt_clean_v1_bp_cube.yaml",
        "name": "hmt_clean_v1_bp_cube-yolov8x",
        "epochs": 300,
        "imgsz": 960,
        "batch": -1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("yolov8x.pt"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory that contains sua_t/ sua_rgb/ bp_cube/; overrides the yaml `path` field.",
    )
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=80)
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


def resolve_data_root(args: argparse.Namespace) -> Path | None:
    """Return the override dataset root, if any (CLI wins over env var)."""
    if args.data_root is not None:
        return args.data_root
    env_root = os.environ.get("HMT_DATA_ROOT")
    return Path(env_root) if env_root else None


def dataset_root(data: dict, config: Path, override: Path | None, key: str) -> Path:
    if override is not None:
        return (override / key).resolve()
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
        if path.is_absolute() and path.is_file():
            output_lines.append(str(path))
            continue
        # Portable fallback: POSIX-absolute lists copied from another OS, or a
        # plain file name, resolve against <root>/images when that exists.
        fallback = root / "images" / path.name
        if fallback.is_file():
            output_lines.append(str(fallback.resolve()))
        else:
            output_lines.append(str(path if path.is_absolute() else (root / path).resolve()))
    out.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")
    return str(out.resolve())


def runtime_config(config: Path, run_root: Path, key: str, data_root: Path | None) -> Path:
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    root = dataset_root(data, config, data_root, key)
    workspace = run_root / "hmt_absolute_lists" / key
    workspace.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        if data.get(split):
            data[split] = absolute_list(data[split], root, workspace / f"{split}.txt")
    data["path"] = str(root)
    output = workspace / "dataset.yaml"
    output.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[hmt_train] dataset_root={root}")
    return output


def train_one(args: argparse.Namespace, key: str, data_root: Path | None) -> None:
    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from ultralytics import YOLO

    settings = DATASETS[key]
    config = repo_root / "ultralytics" / "cfg" / "datasets_hmt" / settings["yaml"]
    if not config.exists():
        raise FileNotFoundError(f"dataset config not found: {config}")
    args.run_root.mkdir(parents=True, exist_ok=True)
    data_config = runtime_config(config, args.run_root, key, data_root)
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
        patience=args.patience,
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
    data_root = resolve_data_root(args)
    if data_root is not None:
        print(f"[hmt_train] data_root override: {data_root}")
    for key in DATASETS if args.dataset == "all" else (args.dataset,):
        train_one(args, key, data_root)
    print("[hmt_train] requested HMT training jobs completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
