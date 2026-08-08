# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train YOLO and pixel-aligned multi-modal YOLO models on the SUA HMT dataset.

The source dataset stores image names in ``train.txt``/``val.txt`` and keeps RGB, thermal and labels in sibling
directories. This launcher creates a small local staging view containing directory links and manifests; the original
dataset is never copied or modified. Single-modal runs use the normal ``YOLO`` facade, while RGBT runs use the
``MultiModalYOLO`` facade and the repository's static YOLOv8x MM2 fusion templates.

Examples:
    # Train the default YOLO11 size sweep on RGB and thermal data, plus all six RGBT fusion templates.
    python tools/train_sua_yolo.py --modalities rgb,t,rgbt --pretrained auto

    # Train one single-modal and one multi-modal experiment.
    python tools/train_sua_yolo.py --modalities rgb --models yolo11n.yaml
    python tools/train_sua_yolo.py --modalities rgbt --multimodal-models yolov8x-mm2-bf.yaml --pretrained yolov8x.pt

    # Expand the complete supported single-modal detector sweep.
    python tools/train_sua_yolo.py --modalities rgb,t --models all --epochs 100
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo_multimodal_base import close_tee_log, tee_log_to_run_dir  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.models.multimodal import MultiModalYOLO  # noqa: E402
from ultralytics.models.multimodal.pretrained import load_coco_pretrained  # noqa: E402


DEFAULT_DATA_ROOT = Path(
    r"\\158.132.186.40\isds\huilin\bdd\collected_data\20260211_HMT_data_all\datasets"
)
DEFAULT_PROJECT = REPO_ROOT / "runs" / "sua"

DATASET_DIRS = {
    "rgb": "sua_dataset_rgb",
    "t": "sua_dataset_t",
    "rgbt": "sua_dataset_rgbt",
}
MULTIMODAL_MODELS = tuple(f"yolov8x-mm2-{mode}.yaml" for mode in ("if", "ef", "nif", "bf", "nf", "hf"))
DEFAULT_SINGLE_MODELS = tuple(f"yolo11{size}.yaml" for size in "nsmlx")
ALL_SINGLE_MODELS = (
    *(f"yolov8{size}.yaml" for size in "nsmlx"),
    *(f"yolov9{size}.yaml" for size in ("t", "s", "m", "c", "e")),
    *(f"yolov10{size}.yaml" for size in ("n", "s", "m", "b", "l", "x")),
    *(f"yolo11{size}.yaml" for size in "nsmlx"),
    *(f"yolo12{size}.yaml" for size in "nsmlx"),
    *(f"yolo26{size}.yaml" for size in "nsmlx"),
)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class DatasetView:
    """Local metadata view consumed by Ultralytics' standard dataset loaders."""

    modality: str
    source_root: Path
    stage_root: Path
    data_yaml: Path
    names: tuple[str, ...]
    channels: int
    train_count: int
    val_count: int


def _split_csv(value: str) -> list[str]:
    """Split a comma-separated CLI argument while preserving user order."""
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_lines(path: Path) -> list[str]:
    """Read non-empty UTF-8 lines from a dataset metadata file."""
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset file does not exist: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _read_classes(root: Path) -> tuple[str, ...]:
    """Read the dataset-local class taxonomy without reordering or renaming it."""
    names = tuple(_read_lines(root / "classes.txt"))
    if not names:
        raise ValueError(f"No classes found in {root / 'classes.txt'}")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate class names found in {root / 'classes.txt'}")
    return names


def _image_for_stem(directory: Path, stem: str) -> Path:
    """Find one image file for a metadata-list entry, accepting common image suffixes."""
    token = Path(stem.replace("\\", "/")).name
    candidate = directory / token
    if candidate.suffix.lower() in IMAGE_SUFFIXES and candidate.is_file():
        return candidate
    stem = Path(token).stem
    matches = [directory / f"{stem}{suffix}" for suffix in IMAGE_SUFFIXES if (directory / f"{stem}{suffix}").is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one image for '{stem}' under {directory}, found {matches}")
    return matches[0]


def _same_path(left: Path, right: Path) -> bool:
    """Compare paths after resolving directory links without requiring the target to be local."""
    return os.path.normcase(os.path.realpath(str(left))) == os.path.normcase(os.path.realpath(str(right)))


def _ensure_directory_link(link: Path, target: Path) -> None:
    """Create or reuse a directory symlink/junction used by the local staging view."""
    target = Path(os.path.abspath(str(target)))
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if _same_path(link, target):
            return
        raise FileExistsError(f"Staging path already exists and points elsewhere: {link}")

    try:
        os.symlink(str(target), str(link), target_is_directory=True)
        return
    except (OSError, NotImplementedError) as error:
        if os.name != "nt":
            raise RuntimeError(f"Cannot create directory link {link} -> {target}") from error

        # A junction avoids administrator privileges for local targets. UNC targets require a directory symlink.
        flag = "/D" if str(target).startswith("\\\\") else "/J"
        result = subprocess.run(
            ["cmd", "/c", "mklink", flag, str(link), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"Cannot create staging link {link} -> {target}. Enable Windows Developer Mode or create the "
                f"link manually. mklink output: {detail}"
            ) from error


def _write_if_changed(path: Path, content: str) -> None:
    """Write generated metadata only when its content changed."""
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _prepare_dataset_view(modality: str, data_root: Path, work_dir: Path) -> DatasetView:
    """Create a manifest/YAML view for one SUA modality without copying source images."""
    if modality not in DATASET_DIRS:
        raise ValueError(f"Unknown modality '{modality}'. Choose from {sorted(DATASET_DIRS)}.")

    source_root = Path(data_root).expanduser() / DATASET_DIRS[modality]
    if not source_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {source_root}")

    names = _read_classes(source_root)
    primary_source = source_root / ("rgb" if modality != "t" else "t")
    thermal_source = source_root / "t"
    labels_source = source_root / "labels"
    for directory in (primary_source, labels_source):
        if not directory.is_dir():
            raise FileNotFoundError(f"Required dataset directory does not exist: {directory}")
    if modality == "rgbt" and not thermal_source.is_dir():
        raise FileNotFoundError(f"Required thermal directory does not exist: {thermal_source}")

    stage_root = Path(work_dir) / "staging" / modality
    stage_root.mkdir(parents=True, exist_ok=True)
    _ensure_directory_link(stage_root / "images", primary_source)
    _ensure_directory_link(stage_root / "labels", labels_source)
    if modality == "rgbt":
        _ensure_directory_link(stage_root / "t", thermal_source)

    manifests: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        split_file = source_root / f"{split}.txt"
        if not split_file.is_file():
            continue
        entries = []
        for item in _read_lines(split_file):
            primary_image = _image_for_stem(primary_source, item)
            if modality == "rgbt":
                _image_for_stem(thermal_source, primary_image.stem)
            entries.append(f"images/{primary_image.name}")
        manifests[split] = entries
        _write_if_changed(stage_root / f"{split}.txt", "\n".join(entries) + "\n")

    if "train" not in manifests or "val" not in manifests:
        raise FileNotFoundError(f"Both train.txt and val.txt are required under {source_root}")

    data: dict[str, Any] = {
        "path": str(stage_root),
        "train": "train.txt",
        "val": "val.txt",
        "channels": 6 if modality == "rgbt" else 3,
        "names": {index: name for index, name in enumerate(names)},
    }
    if "test" in manifests:
        data["test"] = "test.txt"
    if modality == "rgbt":
        data["modalities"] = [
            {"name": "rgb", "path": "images", "channels": 3, "color": "bgr"},
            {"name": "thermal", "path": "t", "channels": 3},
        ]

    data_yaml = stage_root / "data.yaml"
    yaml_content = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    _write_if_changed(data_yaml, yaml_content)
    manifest = {
        "modality": modality,
        "source_root": str(source_root),
        "stage_root": str(stage_root),
        "data_yaml": str(data_yaml),
        "channels": data["channels"],
        "names": list(names),
        "splits": {key: len(value) for key, value in manifests.items()},
    }
    _write_if_changed(stage_root / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return DatasetView(
        modality=modality,
        source_root=source_root,
        stage_root=stage_root,
        data_yaml=data_yaml,
        names=names,
        channels=int(data["channels"]),
        train_count=len(manifests["train"]),
        val_count=len(manifests["val"]),
    )


def _resolve_model_spec(spec: str, multimodal: bool) -> str | Path:
    """Resolve repository-local multimodal YAMLs while leaving official YOLO aliases intact."""
    path = Path(spec)
    if path.is_file():
        return path
    if multimodal:
        candidate = REPO_ROOT / "ultralytics" / "cfg" / "mmodels" / path.name
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Multi-modal model YAML does not exist: {candidate}")
    if path.suffix in {".yaml", ".yml", ".pt", ".onnx", ".engine", ".torchscript"}:
        return spec
    return f"{spec}.yaml"


def _expand_models(value: str, multimodal: bool) -> list[str]:
    """Expand a model preset or return the user-provided model list."""
    tokens = _split_csv(value)
    if not tokens:
        raise ValueError("At least one model must be supplied.")
    if "all" in {token.lower() for token in tokens}:
        return list(MULTIMODAL_MODELS if multimodal else ALL_SINGLE_MODELS)
    if multimodal and any(token.lower() == "mm2" for token in tokens):
        return list(MULTIMODAL_MODELS)
    return tokens


def _train_kwargs(args: argparse.Namespace, data_yaml: Path, run_name: str) -> dict[str, Any]:
    """Build common Ultralytics training arguments."""
    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "patience": args.patience,
        "project": str(args.project),
        "name": run_name,
        "val": True,
        "exist_ok": args.exist_ok,
    }
    if args.device is not None:
        kwargs["device"] = args.device
    if args.cache == "ram":
        kwargs["cache"] = "ram"
    if args.optimizer != "auto":
        kwargs["optimizer"] = args.optimizer
    return kwargs


def _single_pretrained(spec: str, value: str | None) -> str | Path | None:
    """Resolve a single-modal pretraining argument, including the per-model auto alias."""
    if not value:
        return None
    if value.lower() == "auto":
        return f"{Path(spec).stem}.pt"
    return value


def _with_console_log(model, enabled: bool) -> None:
    """Register the same run-directory console logging used by demo_multimodal_det.py."""
    if enabled:
        model.add_callback("on_train_start", tee_log_to_run_dir)
        model.add_callback("on_train_end", close_tee_log)


def _train_single(model_spec: str, view: DatasetView, args: argparse.Namespace) -> None:
    """Train one standard YOLO detector on RGB or thermal images."""
    model = YOLO(_resolve_model_spec(model_spec, multimodal=False))
    run_name = args.name or f"{view.modality}-{Path(model_spec).stem}"
    _with_console_log(model, args.console_log)
    kwargs = _train_kwargs(args, view.data_yaml, run_name)
    pretrained = _single_pretrained(model_spec, args.pretrained)
    kwargs["pretrained"] = pretrained if pretrained else False
    try:
        model.train(**kwargs)
    finally:
        if args.console_log:
            close_tee_log(getattr(model, "trainer", None))


def _multimodal_pretrained(value: str | None) -> str | Path | None:
    """Resolve the COCO source checkpoint used by YAML-declared multimodal transfer plans."""
    if not value:
        return None
    return "yolov8x.pt" if value.lower() == "auto" else value


def _train_multimodal(model_spec: str, view: DatasetView, args: argparse.Namespace) -> None:
    """Train one static YOLOv8x MM2 detector on the paired RGBT view."""
    model_path = _resolve_model_spec(model_spec, multimodal=True)
    model = MultiModalYOLO(model_path, task="detect")
    pretrained = _multimodal_pretrained(args.pretrained)
    run_name = args.name or f"{view.modality}-{Path(model_spec).stem}"
    train_kwargs = _train_kwargs(args, view.data_yaml, run_name)
    if pretrained:
        load_coco_pretrained(model, pretrained)
        initial_weights = args.work_dir / "initial_weights" / f"{run_name}.pt"
        initial_weights.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(initial_weights))
        train_kwargs["pretrained"] = str(initial_weights)
    else:
        train_kwargs["pretrained"] = False
    _with_console_log(model, args.console_log)
    try:
        model.train(**train_kwargs)
    finally:
        if args.console_log:
            close_tee_log(getattr(model, "trainer", None))


def _run_non_training(mode: str, view: DatasetView, args: argparse.Namespace, multimodal: bool) -> None:
    """Run validation or prediction for one already-trained checkpoint."""
    if not args.weights:
        raise ValueError(f"--weights is required for --mode {mode}")
    model = MultiModalYOLO(args.weights, task="detect") if multimodal else YOLO(args.weights)
    common = {"data": str(view.data_yaml), "batch": args.batch}
    if args.device is not None:
        common["device"] = args.device
    if mode == "val":
        model.val(**common)
    elif mode == "predict":
        if not args.source:
            raise ValueError("--source is required for --mode predict")
        model.predict(source=str(args.source), **common)
    else:
        raise ValueError(f"Unsupported non-training mode: {mode}")


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("train", "val", "predict"), default="train")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--modalities", default="rgb,t,rgbt", help="Comma-separated subset of rgb,t,rgbt.")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_SINGLE_MODELS),
        help="Single-modal models, or 'all' for YOLOv8/9/10/11/12/26 detector sizes.",
    )
    parser.add_argument(
        "--multimodal-models",
        default="mm2",
        help="RGBT models, 'mm2' for all six YOLOv8x fusion templates, or a comma-separated list.",
    )
    parser.add_argument("--weights", type=Path, help="Checkpoint for val/predict.")
    parser.add_argument("--source", type=Path, help="Prediction source; for RGBT use a staged primary image path.")
    parser.add_argument("--pretrained", help="Single-modal checkpoint, 'auto', or multimodal COCO source checkpoint.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0 or 0,1; default lets Ultralytics choose.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_PROJECT / "metadata")
    parser.add_argument("--name", help="Use one explicit run name; omit to derive one per modality/model.")
    parser.add_argument("--cache", choices=("none", "ram"), default="none")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--no-console-log", dest="console_log", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Prepare metadata and print planned runs without training.")
    return parser


def main() -> None:
    """Prepare SUA views and execute the requested experiment matrix."""
    args = _build_parser().parse_args()
    modalities = _split_csv(args.modalities)
    invalid = set(modalities).difference(DATASET_DIRS)
    if invalid:
        raise ValueError(f"Unknown modalities {sorted(invalid)}; choose from {sorted(DATASET_DIRS)}.")
    if args.mode != "train" and len(modalities) != 1:
        raise ValueError("val/predict requires exactly one modality.")

    single_models = _expand_models(args.models, multimodal=False)
    multimodal_models = _expand_models(args.multimodal_models, multimodal=True)
    args.data_root = args.data_root.expanduser()
    args.project = args.project.expanduser()
    args.work_dir = args.work_dir.expanduser()
    args.project.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    for modality in modalities:
        view = _prepare_dataset_view(modality, args.data_root, args.work_dir)
        if modality == "rgbt":
            planned_models = multimodal_models
            is_multimodal = True
        else:
            planned_models = single_models
            is_multimodal = False
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "modality": modality,
                    "data": str(view.data_yaml),
                    "channels": view.channels,
                    "classes": len(view.names),
                    "train_images": view.train_count,
                    "val_images": view.val_count,
                    "models": planned_models,
                },
                ensure_ascii=False,
            )
        )
        if args.dry_run:
            continue
        if args.mode == "train":
            for model_spec in planned_models:
                if is_multimodal:
                    _train_multimodal(model_spec, view, args)
                else:
                    _train_single(model_spec, view, args)
        else:
            if len(planned_models) != 1:
                raise ValueError(f"{args.mode} requires exactly one model; got {planned_models}")
            _run_non_training(args.mode, view, args, is_multimodal)


if __name__ == "__main__":
    main()
