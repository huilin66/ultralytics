"""Reproducible training launcher for the mdet experiment matrix.

This launcher deliberately uses :func:`mayolo_r1.myolo_train_full`, so every
mdet run follows the project's two-stage protocol: ``100 + 100`` epochs by
default, with the second stage retaining the attribute head.  It does not
touch segmentation code.

The experiment plan calls the attribute-loss coefficient ``w4``.  The current
Ultralytics configuration exposes that coefficient as ``mdet``; therefore
this script maps ``--w4`` to the model argument ``mdet`` and never forwards an
unknown ``w4`` argument to Ultralytics.

Examples (PowerShell):

    python scripts/train_mdet_experiments.py w4 `
        --data path/to/billboard_mdet.yaml `
        --model ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml `
        --pretrain yolov10x.pt `
        --w4-values 0.25 0.5 1.0

    python scripts/train_mdet_experiments.py variants `
        --label E2_1_GIA_position `
        --data path/to/billboard_mdet.yaml `
        --pretrain yolov10x.pt `
        --variant baseline=ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml `
        --variant gia_p3=path/to/yolov10x_gia_p3.yaml `
        --w4 0.5

For GCA configurations copied from a Linux training machine, pass
``--com-path`` to replace the embedded ``/nfsv4/...co_occurrence_matrix*.csv``
path in a generated copy.  The source YAML is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _slug(value: object) -> str:
    """Make a value safe to use as a run name."""
    text = str(value).strip()
    text = text.replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    return text.strip("_") or "run"


def _parse_key_value(items: Sequence[str], option: str) -> Dict[str, str]:
    """Parse repeated ``name=value`` command-line options."""
    result: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{option} expects NAME=VALUE, got: {item!r}")
        key, value = item.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key or not value:
            raise ValueError(f"{option} expects non-empty NAME and VALUE, got: {item!r}")
        if key in result:
            raise ValueError(f"Duplicate name {key!r} in {option}")
        result[key] = value
    return result


def _resolve_config(config: str) -> Path:
    """Resolve a YAML path without relying on the current working directory."""
    raw = Path(config).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend(
            [
                PROJECT_ROOT / raw,
                PROJECT_ROOT / "ultralytics" / "cfg" / "models" / raw,
                PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "experiments" / raw,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    formatted = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Model YAML not found. Checked:\n  {formatted}")


def _materialize_config(config: str, com_path: Optional[str], project: str) -> str:
    """Return a runnable config, optionally replacing an inaccessible GCA CSV.

    The ablation YAMLs in ``exp_ablation0107`` contain an absolute Linux path
    to the co-occurrence matrix.  Replacing it in a generated copy keeps the
    experiment reproducible and avoids changing the checked-in configuration.
    """
    source = _resolve_config(config)
    text = source.read_text(encoding="utf-8")
    if "/nfsv4/" not in text:
        return str(source)

    if not com_path:
        raise ValueError(
            f"{source} contains an /nfsv4/ path. Pass --com-path pointing to the "
            "local co-occurrence matrix CSV."
        )
    matrix = Path(com_path).expanduser()
    if not matrix.is_file():
        raise FileNotFoundError(f"Co-occurrence matrix not found: {matrix}")
    if matrix.suffix.lower() != ".csv":
        raise ValueError(f"--com-path must point to a CSV file, got: {matrix}")

    # The original path is a YAML list item and may or may not be quoted.
    pattern = re.compile(
        r"(?P<quote>['\"]?)/nfsv4/[^,\]\s'\"]*co_occurrence_matrix"
        r"[^,\]\s'\"]*\.csv(?P=quote)"
    )
    replacement = repr(matrix.resolve().as_posix())
    updated, count = pattern.subn(replacement, text)
    if count == 0:
        raise ValueError(
            f"{source} contains /nfsv4/ but no co_occurrence_matrix*.csv entry "
            "could be replaced safely."
        )

    generated_dir = Path(project) / "_generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(updated.encode("utf-8")).hexdigest()[:10]
    target = generated_dir / f"{source.stem}_{digest}.yaml"
    if not target.exists():
        target.write_text(updated, encoding="utf-8")
    return str(target.resolve())


def _add_train_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by all mdet training experiments."""
    parser.add_argument("--data", required=True, help="mdet dataset YAML")
    parser.add_argument("--project", default="runs/experiments", help="output root")
    parser.add_argument("--stage1-epochs", type=int, default=100)
    parser.add_argument("--stage2-epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0", help="CUDA index, cpu, or device string")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--w4", type=float, default=0.5, help="attribute loss gain; mapped to mdet")
    parser.add_argument("--close-mosaic", type=int, default=None)
    parser.add_argument("--auto-optim", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--com-path",
        default=None,
        help="local GCA co-occurrence CSV; replaces an /nfsv4 path in a generated YAML copy",
    )
    parser.add_argument("--dry-run", action="store_true", help="print runs without training")


def _add_variant_arguments(parser: argparse.ArgumentParser, default_network: str = "yolo") -> None:
    """Add arguments for a set of named model/config variants."""
    _add_train_arguments(parser)
    parser.add_argument("--label", default=None, help="experiment label used in run names")
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        metavar="NAME=CONFIG_YAML",
        help="repeat for every ablation/model variant",
    )
    parser.add_argument(
        "--pretrain",
        default=None,
        help="common pretrained checkpoint; overridden by --pretrain-map",
    )
    parser.add_argument(
        "--pretrain-map",
        action="append",
        default=[],
        metavar="NAME=CHECKPOINT",
        help="variant-specific checkpoint, repeat as needed",
    )
    parser.add_argument("--network", choices=("yolo", "rtdetr"), default=default_network)


def _record_path(project: str) -> Path:
    """Return the append-only manifest path for a project."""
    path = Path(project) / "experiment_manifest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _append_manifest(project: str, record: Dict[str, object]) -> None:
    """Record the exact resolved configuration of a requested run."""
    with _record_path(project).open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _training_kwargs(args: argparse.Namespace, w4: float, seed: int) -> Dict[str, object]:
    """Build kwargs accepted by the existing mdet trainer."""
    kwargs: Dict[str, object] = {
        "data": args.data,
        "device": args.device,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "seed": seed,
        "amp": args.amp,
        "exist_ok": args.exist_ok,
        # Current code names the paper's w4 coefficient `mdet`.
        "mdet": float(w4),
    }
    if args.close_mosaic is not None:
        kwargs["close_mosaic"] = args.close_mosaic
    return kwargs


def _train_one(
    args: argparse.Namespace,
    *,
    label: str,
    variant_name: str,
    config: str,
    pretrain: str,
    w4: float,
    seed: int,
    network_name: str,
) -> Optional[str]:
    """Train one two-stage mdet run and return its best checkpoint."""
    if "seg" in Path(config).stem.lower() or "segment" in Path(config).stem.lower():
        raise ValueError(f"Segmentation config is outside this launcher: {config}")
    if args.stage1_epochs < 1 or args.stage2_epochs < 1:
        raise ValueError("Both stage epoch counts must be positive")
    if w4 < 0:
        raise ValueError("w4/mdet must be non-negative")

    resolved_config = _materialize_config(config, args.com_path, args.project)
    run_base = f"{_slug(label)}_{_slug(variant_name)}_w4_{_slug(w4)}_seed_{seed}"
    record: Dict[str, object] = {
        "label": label,
        "variant": variant_name,
        "config": resolved_config,
        "pretrain": pretrain,
        "network": network_name,
        "w4": float(w4),
        "ultralytics_argument": {"mdet": float(w4)},
        "seed": seed,
        "stage1_epochs": args.stage1_epochs,
        "stage2_epochs": args.stage2_epochs,
        "data": args.data,
        "project": args.project,
        "stage1_name": f"{run_base}_stage1",
        "stage2_name": f"{run_base}_stage2",
        "status": "dry-run" if args.dry_run else "started",
    }
    _append_manifest(args.project, record)

    print(json.dumps(record, ensure_ascii=False, indent=2))
    if args.dry_run:
        return None

    # Keep imports lazy so --help/--dry-run can validate the experiment matrix
    # on a machine that does not have the GPU environment installed.
    from mayolo_r1 import myolo_train_full
    from ultralytics import RTDETR, YOLO

    network = RTDETR if network_name == "rtdetr" else YOLO
    kwargs = _training_kwargs(args, w4, seed)
    try:
        best = myolo_train_full(
            resolved_config,
            pretrain_path=pretrain,
            network=network,
            auto_optim=args.auto_optim,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            stage1_name=f"{run_base}_stage1",
            stage2_name=f"{run_base}_stage2",
            project=args.project,
            **kwargs,
        )
    except Exception as error:
        failed = dict(record)
        failed.update({"status": "failed", "error": repr(error)})
        _append_manifest(args.project, failed)
        raise
    finished = dict(record)
    finished.update({"status": "finished", "best": str(best) if best else None})
    _append_manifest(args.project, finished)
    print(f"[finished] {run_base}: best={best}")
    return str(best) if best else None


def _run_w4(args: argparse.Namespace) -> None:
    """Run E1: w4 sensitivity."""
    for value in args.w4_values:
        _train_one(
            args,
            label=args.label,
            variant_name="base",
            config=args.model,
            pretrain=args.pretrain,
            w4=float(value),
            seed=args.seed,
            network_name="yolo",
        )


def _run_variants(args: argparse.Namespace) -> None:
    """Run ablation, architecture-size, or RT-DETR variant experiments."""
    variants = _parse_key_value(args.variant, "--variant")
    checkpoints = _parse_key_value(args.pretrain_map, "--pretrain-map")
    label = args.label or args.experiment
    for name, config in variants.items():
        pretrain = checkpoints.get(name, args.pretrain)
        if not pretrain:
            raise ValueError(
                f"No checkpoint for variant {name!r}. Pass --pretrain or --pretrain-map {name}=..."
            )
        _train_one(
            args,
            label=label,
            variant_name=name,
            config=config,
            pretrain=pretrain,
            w4=args.w4,
            seed=args.seed,
            network_name=args.network,
        )


def _run_stability(args: argparse.Namespace) -> None:
    """Run E7: repeated seeds for each requested YOLO/MAYOLO variant."""
    variants = _parse_key_value(args.variant, "--variant")
    checkpoints = _parse_key_value(args.pretrain_map, "--pretrain-map")
    label = args.label or "E7_stability"
    for seed in args.seeds:
        for name, config in variants.items():
            pretrain = checkpoints.get(name, args.pretrain)
            if not pretrain:
                raise ValueError(
                    f"No checkpoint for variant {name!r}. Pass --pretrain or --pretrain-map {name}=..."
                )
            _train_one(
                args,
                label=label,
                variant_name=name,
                config=config,
                pretrain=pretrain,
                w4=args.w4,
                seed=int(seed),
                network_name=args.network,
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the planned mdet experiments with the project's 100+100 protocol"
    )
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    w4 = subparsers.add_parser("w4", help="E1: attribute-loss coefficient sensitivity")
    _add_train_arguments(w4)
    w4.add_argument("--label", default="E1_w4")
    w4.add_argument("--model", required=True, help="one mdet model YAML")
    w4.add_argument("--pretrain", required=True, help="pretrained detector checkpoint")
    w4.add_argument("--w4-values", nargs="+", type=float, default=[0.25, 0.5, 1.0])

    for name, help_text, default_network in (
        ("variants", "Run named ablation/model variants", "yolo"),
        ("gia-position", "E2.1: GIA position ablation", "yolo"),
        ("gca-structure", "E2.2: GCA structure ablation", "yolo"),
        ("gia-gca", "E2.3: joint GIA/GCA ablation", "yolo"),
        ("ho", "E2.4: train the HO checkpoint for inference comparison", "yolo"),
        ("versions", "E3: YOLOv8-YOLOv13/MAYOLO sizes", "yolo"),
        ("rtdetr", "E4: RT-DETR attribute detector sizes", "rtdetr"),
    ):
        variant_parser = subparsers.add_parser(name, help=help_text)
        _add_variant_arguments(variant_parser, default_network=default_network)

    stability = subparsers.add_parser("stability", help="E7: repeated seeds for YOLOv10x/MAYOLOx")
    _add_variant_arguments(stability, default_network="yolo")
    stability.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse arguments and execute the selected experiment."""
    args = _build_parser().parse_args(argv)
    if args.experiment == "w4":
        _run_w4(args)
    elif args.experiment == "stability":
        _run_stability(args)
    else:
        _run_variants(args)


if __name__ == "__main__":
    main()
