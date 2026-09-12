"""Reproducible training launcher for the mdet experiment matrix.

The launcher supports the normal two-stage protocol and two schedule studies:

* ``stage1-sweep`` independently trains stage 1 for several epoch budgets;
* ``stage2-sweep`` independently trains stage 2 from one fixed stage-1
  checkpoint for several epoch budgets.
* ``gca-stage2`` compares GCA/GNN variants from one fixed stage-1 checkpoint;
* ``prior-stage2`` compares fixed co-occurrence-prior attention heads from one
  fixed stage-1 checkpoint;
* ``hsv-ablation`` independently trains stage 1 for three HSV augmentation
  settings at a fixed epoch budget.
* variant experiments accept ``--stage1-only`` for position/structure tests
  that intentionally stop after stage 1.

The schedule studies deliberately do not reuse intermediate checkpoints from a
longer run, because the learning-rate and augmentation schedules depend on the
configured total epoch count.  It does not touch segmentation code.

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

    python scripts/train_mdet_experiments.py gia-position `
        --label E2_1_GIA_position `
        --data path/to/billboard_mdet.yaml `
        --pretrain yolov10x.pt `
        --stage1-only `
        --variant gia5=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_5.yaml `
        --variant gia5_7=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_5_7.yaml `
        --variant gia5_7_res=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_5_7_Res.yaml

    python scripts/train_mdet_experiments.py stage1-sweep `
        --data path/to/billboard_mdet.yaml `
        --model ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml `
        --pretrain yolov10x.pt `
        --stage1-values 100 200 300 400 500

    python scripts/train_mdet_experiments.py stage2-sweep `
        --data path/to/billboard_mdet.yaml `
        --model ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml `
        --stage1-checkpoint runs/experiments/E0_stage1_sweep/E0_stage1_stage1_100_w4_0p5_seed_0/weights/best.pt `
        --stage1-epochs 100 `
        --stage2-values 50 100 150 200

    python scripts/train_mdet_experiments.py gca-stage2 `
        --data path/to/billboard_mdet.yaml `
        --stage1-checkpoint runs/experiments/E1_w4/.../weights/best.pt `
        --variant baseline=ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml `
        --variant gcn=ultralytics/cfg/models/exp_ablation/yolov10x_GCN.yaml `
        --variant gat=ultralytics/cfg/models/exp_ablation/yolov10x_GAT_learned.yaml `
        --variant graphsage=ultralytics/cfg/models/exp_ablation/yolov10x_GraphSAGE.yaml `
        --variant gin=ultralytics/cfg/models/exp_ablation/yolov10x_GIN.yaml `
        --gnn-types gca gcn gat graphsage gin `
        --com-path path/to/co_occurrence_matrix_train.csv

    python scripts/train_mdet_experiments.py versions `
        --data path/to/billboard_mdet.yaml `
        --include-yolo26 `
        --yolo26-sizes n s m l x

For GCA configurations copied from a Linux training machine, pass
``--com-path`` to replace the embedded ``/nfsv4/...co_occurrence_matrix*.csv``
path in a generated copy.  The source YAML is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

YOLO26_MDET_CONFIGS = {
    f"yolov26{size}": f"ultralytics/cfg/models/experiments/yolov26{size}-mdetect.yaml"
    for size in "nsmlx"
}

# Keep the previously tested configuration fixed for the HSV ablation, even
# though the project-wide default below is now the reduced setting.
HSV_ABLATION_CURRENT = (0.015, 0.7, 0.4)
HSV_ABLATION_REDUCED = (0.0, 0.2, 0.2)
HSV_ABLATION_DISABLED = (0.0, 0.0, 0.0)
DEFAULT_STAGE1_EPOCHS = 100
DEFAULT_STAGE2_EPOCHS = 100

from scripts.cli_compat import add_bool_argument


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


def _materialize_config(
    config: str,
    com_path: Optional[str],
    project: str,
    gnn_type: Optional[str] = None,
    feature_gain: Optional[float] = None,
    prior_type: Optional[str] = None,
    prior_conditional: bool = False,
) -> str:
    """Return a runnable config with optional graph substitutions.

    The ablation YAMLs in ``exp_ablation`` contain an absolute Linux path
    to the co-occurrence matrix.  Replacing it in a generated copy keeps the
    experiment reproducible and avoids changing the checked-in configuration.
    For the 5x5 GCA study, the checked-in YAML keeps the ``com_gca_*`` token
    and this function materializes the requested GNN-specific token in the
    per-project generated copy.  Feature-graph YAMLs may receive an optional
    trailing residual gain in the same generated copy, so the source YAML
    remains a stable baseline with the default gain of 1.0.  The co-occurrence
    prior study similarly materializes one of the non-GNN ``com_prior_*``
    heads and can select the transposed conditional-matrix interpretation.
    """
    source = _resolve_config(config)
    text = source.read_text(encoding="utf-8")
    updated = text
    changed = False

    if gnn_type is not None:
        valid_gnn_types = {"gca", "gcn", "gat", "graphsage", "gin"}
        if gnn_type not in valid_gnn_types:
            raise ValueError(f"Unsupported --gnn-types value: {gnn_type!r}")

        # The five GCA variants are defined once in YAML. Replace only the
        # operator token, preserving the selected structural variant:
        # com_gca_context_residual -> com_gcn_context_residual, etc.
        gnn_pattern = re.compile(
            r"(?P<quote>['\"]?)com_gca_"
            r"(?P<variant>context|adaptive|twohop|conv_adapter)_residual"
            r"(?P=quote)"
        )
        updated, count = gnn_pattern.subn(
            lambda match: (
                f"{match.group('quote')}com_{gnn_type}_"
                f"{match.group('variant')}_residual{match.group('quote')}"
            ),
            updated,
        )
        if count == 0:
            raise ValueError(
                f"{source} does not contain one of the materializable "
                "com_gca_{context,adaptive,twohop,conv_adapter}_residual tokens."
            )
        changed = True

    if feature_gain is not None:
        feature_gain = float(feature_gain)
        if not math.isfinite(feature_gain) or feature_gain < 0:
            raise ValueError(
                f"--feature-gain must be a finite non-negative number, got {feature_gain!r}"
            )

        # Feature graph params are [sep, c4, gat, retrain, com_path]. Add or
        # replace the optional sixth value without touching unrelated YAMLs.
        number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        feature_pattern = re.compile(
            r"(?P<prefix>['\"]feature_(?:gca|gcn|gat|graphsage|gin|local)_"
            r"(?:cross|conditional)['\"]\s*,\s*(?:False|false)\s*,\s*[^,\]\n]+)"
            rf"(?:\s*,\s*{number})?(?P<close>\s*\])"
        )
        replacement = rf"\g<prefix>, {feature_gain:g}\g<close>"
        updated, count = feature_pattern.subn(replacement, updated)
        if count == 0:
            raise ValueError(
                f"{source} does not contain a feature graph params list with a "
                "feature_gain insertion point."
            )
        changed = True

    if prior_type is not None:
        valid_prior_types = {
            "bias",
            "channel",
            "spatial",
            "moe",
            "texture",
            "logit_blend",
            "logit_bias",
            "logit_mlp",
            "cross_attention",
            "dynamic_gate",
        }
        if prior_type not in valid_prior_types:
            raise ValueError(f"Unsupported --prior-types value: {prior_type!r}")

        prior_pattern = re.compile(
            r'''(?P<quote>['"]?)com_prior_(?:bias|channel|spatial|moe|texture|'''
            r'''logit_blend|logit_bias|logit_mlp|cross_attention|dynamic_gate)'''
            r'''(?:_conditional)?(?P=quote)'''
        )
        suffix = "_conditional" if prior_conditional else ""
        replacement = rf"\g<quote>com_prior_{prior_type}{suffix}\g<quote>"
        updated, count = prior_pattern.subn(replacement, updated)
        if count == 0:
            raise ValueError(
                f"{source} does not contain a materializable com_prior_* token."
            )
        changed = True

    if "/nfsv4/" in updated:
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
        matrix_pattern = re.compile(
            r"(?P<quote>['\"]?)/nfsv4/[^,\]\s'\"]*co_occurrence_matrix"
            r"[^,\]\s'\"]*\.csv(?P=quote)"
        )
        replacement = repr(matrix.resolve().as_posix())
        updated, count = matrix_pattern.subn(replacement, updated)
        if count == 0:
            raise ValueError(
                f"{source} contains /nfsv4/ but no co_occurrence_matrix*.csv entry "
                "could be replaced safely."
            )
        changed = True

    if not changed:
        return str(source)

    generated_dir = Path(project) / "_generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(updated.encode("utf-8")).hexdigest()[:10]
    target = generated_dir / f"{source.stem}_{digest}.yaml"
    if not target.exists():
        target.write_text(updated, encoding="utf-8")
    return str(target.resolve())


def _add_common_train_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by all mdet training experiments."""
    parser.add_argument("--data", required=True, help="mdet dataset YAML")
    parser.add_argument("--project", default="runs/experiments", help="output root")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0", help="CUDA index, cpu, or device string")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--w4", type=float, default=0.5, help="attribute loss gain; mapped to mdet")
    parser.add_argument(
        "--feature-gain",
        type=float,
        default=None,
        help=(
            "multiplier for the feature-graph residual correction; only applies "
            "to feature_* configs and is stored in the generated YAML"
        ),
    )
    parser.add_argument("--hsv-h", type=float, default=0.0, help="HSV hue augmentation gain")
    parser.add_argument("--hsv-s", type=float, default=0.2, help="HSV saturation augmentation gain")
    parser.add_argument("--hsv-v", type=float, default=0.2, help="HSV value/brightness augmentation gain")
    parser.add_argument("--close-mosaic", type=int, default=None)
    add_bool_argument(parser, "--auto-optim", default=False)
    add_bool_argument(parser, "--amp", default=True)
    add_bool_argument(parser, "--exist-ok", default=False)
    parser.add_argument(
        "--com-path",
        default=None,
        help="local GCA co-occurrence CSV; replaces an /nfsv4 path in a generated YAML copy",
    )
    parser.add_argument("--dry-run", action="store_true", help="print runs without training")


def _add_train_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the standard two-stage mdet protocol."""
    _add_common_train_arguments(parser)
    parser.add_argument("--stage1-epochs", type=int, default=DEFAULT_STAGE1_EPOCHS)
    parser.add_argument("--stage2-epochs", type=int, default=DEFAULT_STAGE2_EPOCHS)


def _add_variant_arguments(
    parser: argparse.ArgumentParser, default_network: str = "yolo", require_variant: bool = True
) -> None:
    """Add arguments for a set of named model/config variants."""
    _add_train_arguments(parser)
    parser.add_argument("--label", default=None, help="experiment label used in run names")
    parser.add_argument(
        "--variant",
        action="append",
        required=require_variant,
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
    add_bool_argument(
        parser,
        "--stage1-only",
        default=False,
        help="train each variant for stage 1 only; skip the stage-2 fine-tuning pass",
    )


def _record_path(project: str) -> Path:
    """Return the append-only manifest path for a project."""
    path = Path(project) / "experiment_manifest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _append_manifest(project: str, record: Dict[str, object]) -> None:
    """Record the exact resolved configuration of a requested run."""
    with _record_path(project).open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _get_hsv_values(
    args: argparse.Namespace, override: Optional[Sequence[float]] = None
) -> tuple[float, float, float]:
    """Return validated HSV gains, optionally using an ablation override."""
    values = tuple(
        float(x) for x in (override if override is not None else (args.hsv_h, args.hsv_s, args.hsv_v))
    )
    if len(values) != 3 or any(value < 0 for value in values):
        raise ValueError(f"HSV gains must contain three non-negative values, got: {values}")
    return values


def _training_kwargs(
    args: argparse.Namespace,
    w4: float,
    seed: int,
    hsv: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Build kwargs accepted by the existing mdet trainer."""
    hsv_h, hsv_s, hsv_v = _get_hsv_values(args, hsv)
    kwargs: Dict[str, object] = {
        "data": args.data,
        "device": args.device,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "seed": seed,
        "amp": args.amp,
        "exist_ok": args.exist_ok,
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
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

    hsv_h, hsv_s, hsv_v = _get_hsv_values(args)
    resolved_config = _materialize_config(
        config, args.com_path, args.project, feature_gain=args.feature_gain
    )
    run_base = f"{_slug(label)}_{_slug(variant_name)}_w4_{_slug(w4)}_seed_{seed}"
    record: Dict[str, object] = {
        "label": label,
        "variant": variant_name,
        "config": resolved_config,
        "pretrain": pretrain,
        "network": network_name,
        "w4": float(w4),
        "feature_gain": None if args.feature_gain is None else float(args.feature_gain),
        "ultralytics_argument": {"mdet": float(w4)},
        "seed": seed,
        "stage1_epochs": args.stage1_epochs,
        "stage2_epochs": args.stage2_epochs,
        "data": args.data,
        "project": args.project,
        "stage1_name": f"{run_base}_stage1",
        "stage2_name": f"{run_base}_stage2",
        "hsv": {"h": hsv_h, "s": hsv_s, "v": hsv_v},
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


def _validate_epoch_values(values: Sequence[int], option: str) -> Sequence[int]:
    """Validate an epoch-budget list and preserve the requested order."""
    if not values or any(value < 1 for value in values):
        raise ValueError(f"{option} must contain only positive integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{option} must not contain duplicate values")
    return values


def _train_direct_stage(
    args: argparse.Namespace,
    *,
    label: str,
    variant_name: str,
    config: str,
    pretrain: str,
    w4: float,
    seed: int,
    epochs: int,
    retrain: bool,
    stage1_epochs: int,
    run_name: str,
    hsv: Optional[Sequence[float]] = None,
    network_name: str = "yolo",
    gnn_type: Optional[str] = None,
    prior_type: Optional[str] = None,
    prior_conditional: bool = False,
) -> Optional[str]:
    """Run one independent stage-only experiment and record its provenance."""
    if "seg" in Path(config).stem.lower() or "segment" in Path(config).stem.lower():
        raise ValueError(f"Segmentation config is outside this launcher: {config}")
    if epochs < 1 or w4 < 0:
        raise ValueError("epochs must be positive and w4/mdet must be non-negative")
    if retrain and not args.dry_run and not Path(pretrain).expanduser().is_file():
        raise FileNotFoundError(f"Stage1 checkpoint not found: {pretrain}")

    hsv_h, hsv_s, hsv_v = _get_hsv_values(args, hsv)
    resolved_config = _materialize_config(
        config,
        args.com_path,
        args.project,
        gnn_type=gnn_type,
        feature_gain=args.feature_gain,
        prior_type=prior_type,
        prior_conditional=prior_conditional,
    )
    record: Dict[str, object] = {
        "protocol": "stage2_only" if retrain else "stage1_only",
        "label": label,
        "variant": variant_name,
        "gnn_type": gnn_type,
        "prior_type": prior_type,
        "prior_conditional": prior_conditional if prior_type is not None else None,
        "config": resolved_config,
        "pretrain": pretrain,
        "stage1_checkpoint": pretrain if retrain else None,
        "network": network_name,
        "w4": float(w4),
        "feature_gain": None if args.feature_gain is None else float(args.feature_gain),
        "ultralytics_argument": {"mdet": float(w4)},
        "seed": seed,
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": epochs if retrain else 0,
        "data": args.data,
        "project": args.project,
        "run_name": run_name,
        "hsv": {"h": hsv_h, "s": hsv_s, "v": hsv_v},
        "status": "dry-run" if args.dry_run else "started",
    }
    _append_manifest(args.project, record)

    print(json.dumps(record, ensure_ascii=False, indent=2))
    if args.dry_run:
        return None

    from mayolo_r1 import myolo_train
    from ultralytics import RTDETR, YOLO

    train_kwargs = _training_kwargs(args, w4, seed, hsv=hsv)
    # Match myolo_train_full's semantics: stage 1 uses the trainer default
    # patience, while the original stage-2 path explicitly uses its own
    # epoch budget as patience.
    if retrain:
        train_kwargs["patience"] = epochs
    try:
        best = myolo_train(
            resolved_config,
            pretrain_path=pretrain,
            network=RTDETR if network_name == "rtdetr" else YOLO,
            auto_optim=args.auto_optim,
            retrain=retrain,
            epochs=epochs,
            name=run_name,
            project=args.project,
            **train_kwargs,
        )
    except Exception as error:
        failed = dict(record)
        failed.update({"status": "failed", "error": repr(error)})
        _append_manifest(args.project, failed)
        raise

    finished = dict(record)
    finished.update({"status": "finished", "best": str(best) if best else None})
    _append_manifest(args.project, finished)
    print(f"[finished] {run_name}: best={best}")
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


def _run_stage1_sweep(args: argparse.Namespace) -> None:
    """Run independent stage-1-only jobs for each requested epoch budget."""
    values = _validate_epoch_values(args.stage1_values, "--stage1-values")
    for epochs in values:
        run_name = (
            f"{_slug(args.label)}_stage1_{epochs}_w4_{_slug(args.w4)}_seed_{args.seed}"
        )
        _train_direct_stage(
            args,
            label=args.label,
            variant_name=f"stage1_{epochs}",
            config=args.model,
            pretrain=args.pretrain,
            w4=args.w4,
            seed=args.seed,
            epochs=int(epochs),
            retrain=False,
            stage1_epochs=int(epochs),
            run_name=run_name,
        )


def _run_hsv_ablation(args: argparse.Namespace) -> None:
    """Run E0.2: compare current, disabled, and reduced HSV augmentation."""
    _validate_epoch_values([args.epochs], "--epochs")
    variants = (
        ("current", HSV_ABLATION_CURRENT),
        ("disabled", HSV_ABLATION_DISABLED),
        ("reduced", HSV_ABLATION_REDUCED),
    )
    for variant_name, hsv in variants:
        run_name = (
            f"{_slug(args.label)}_{variant_name}_stage1_{args.epochs}"
            f"_w4_{_slug(args.w4)}_seed_{args.seed}"
        )
        _train_direct_stage(
            args,
            label=args.label,
            variant_name=f"{variant_name}_hsv",
            config=args.model,
            pretrain=args.pretrain,
            w4=args.w4,
            seed=args.seed,
            epochs=int(args.epochs),
            retrain=False,
            stage1_epochs=int(args.epochs),
            run_name=run_name,
            hsv=hsv,
        )


def _run_stage2_sweep(args: argparse.Namespace) -> None:
    """Run independent stage-2-only jobs from one fixed stage-1 checkpoint."""
    _validate_epoch_values([args.stage1_epochs], "--stage1-epochs")
    values = _validate_epoch_values(args.stage2_values, "--stage2-values")
    for epochs in values:
        run_name = (
            f"{_slug(args.label)}_stage1_{args.stage1_epochs}_stage2_{epochs}"
            f"_w4_{_slug(args.w4)}_seed_{args.seed}"
        )
        _train_direct_stage(
            args,
            label=args.label,
            variant_name=f"stage1_{args.stage1_epochs}_stage2_{epochs}",
            config=args.model,
            pretrain=args.stage1_checkpoint,
            w4=args.w4,
            seed=args.seed,
            epochs=int(epochs),
            retrain=True,
            stage1_epochs=args.stage1_epochs,
            run_name=run_name,
        )


def _run_gca_stage2(args: argparse.Namespace) -> None:
    """Run the GCA/GNN structure comparison from one fixed stage-1 checkpoint.

    Every variant receives the same stage-1 checkpoint and is trained only in
    stage 2.  ``retrain=True`` uses the existing mdet freeze policy, which
    keeps the object-detection branches fixed and leaves the attribute/GNN
    branch trainable.  This deliberately does not toggle the YOLOv10
    one-to-one/one-to-many head; HO remains a separate experiment.
    """
    _validate_epoch_values([args.stage1_epochs], "--stage1-epochs")
    _validate_epoch_values([args.stage2_epochs], "--stage2-epochs")
    variants = _parse_key_value(args.variant, "--variant")
    requested_gnn_types = getattr(args, "gnn_types", None)
    # No --gnn-types keeps the historical behavior and uses the operator
    # already encoded by each YAML. Supplying five values creates the full
    # structural-variant x GNN-operator matrix.
    gnn_types = requested_gnn_types or [None]
    for gnn_type in gnn_types:
        for name, config in variants.items():
            combo_name = f"{gnn_type}_{name}" if gnn_type else name
            run_name = (
                f"{_slug(args.label)}_{_slug(combo_name)}_stage1_{args.stage1_epochs}"
                f"_stage2_{args.stage2_epochs}_w4_{_slug(args.w4)}_seed_{args.seed}"
            )
            _train_direct_stage(
                args,
                label=args.label,
                variant_name=combo_name,
                config=config,
                pretrain=args.stage1_checkpoint,
                w4=args.w4,
                seed=args.seed,
                epochs=args.stage2_epochs,
                retrain=True,
                stage1_epochs=args.stage1_epochs,
                run_name=run_name,
                network_name="yolo",
                gnn_type=gnn_type,
            )


def _run_prior_stage2(args: argparse.Namespace) -> None:
    """Run head-only co-occurrence prior structures from one fixed checkpoint."""
    _validate_epoch_values([args.stage1_epochs], "--stage1-epochs")
    _validate_epoch_values([args.stage2_epochs], "--stage2-epochs")
    for matrix_mode in args.matrix_modes:
        conditional = matrix_mode == "conditional"
        for prior_type in args.prior_types:
            variant_name = f"{matrix_mode}_{prior_type}"
            run_name = (
                f"{_slug(args.label)}_{_slug(variant_name)}_stage1_{args.stage1_epochs}"
                f"_stage2_{args.stage2_epochs}_w4_{_slug(args.w4)}_seed_{args.seed}"
            )
            _train_direct_stage(
                args,
                label=args.label,
                variant_name=variant_name,
                config=args.model,
                pretrain=args.stage1_checkpoint,
                w4=args.w4,
                seed=args.seed,
                epochs=args.stage2_epochs,
                retrain=True,
                stage1_epochs=args.stage1_epochs,
                run_name=run_name,
                network_name="yolo",
                prior_type=prior_type,
                prior_conditional=conditional,
            )


def _run_variants(args: argparse.Namespace) -> None:
    """Run ablation, architecture-size, or RT-DETR variant experiments."""
    variants = _parse_key_value(args.variant or [], "--variant")
    checkpoints = _parse_key_value(args.pretrain_map, "--pretrain-map")
    if getattr(args, "include_yolo26", False):
        sizes = args.yolo26_sizes or tuple("nsmlx")
        for size in sizes:
            name = f"yolov26{size}"
            if name not in variants:
                variants[name] = YOLO26_MDET_CONFIGS[name]
    if not variants:
        raise ValueError("No variants specified. Pass --variant NAME=CONFIG_YAML or --include-yolo26")

    label = args.label or args.experiment
    for name, config in variants.items():
        pretrain = checkpoints.get(name, args.pretrain)
        if not pretrain and name in YOLO26_MDET_CONFIGS:
            pretrain = f"yolo26{name[-1]}.pt"
        if not pretrain:
            raise ValueError(
                f"No checkpoint for variant {name!r}. Pass --pretrain or --pretrain-map {name}=..."
            )
        if args.stage1_only:
            run_name = (
                f"{_slug(label)}_{_slug(name)}_stage1_{args.stage1_epochs}"
                f"_w4_{_slug(args.w4)}_seed_{args.seed}"
            )
            _train_direct_stage(
                args,
                label=label,
                variant_name=name,
                config=config,
                pretrain=pretrain,
                w4=args.w4,
                seed=args.seed,
                epochs=args.stage1_epochs,
                retrain=False,
                stage1_epochs=args.stage1_epochs,
                run_name=run_name,
                network_name=args.network,
            )
        else:
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
        description="Train the planned mdet experiments and independent stage schedule studies"
    )
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    w4 = subparsers.add_parser("w4", help="E1: attribute-loss coefficient sensitivity")
    _add_train_arguments(w4)
    w4.add_argument("--label", default="E1_w4")
    w4.add_argument("--model", required=True, help="one mdet model YAML")
    w4.add_argument("--pretrain", required=True, help="pretrained detector checkpoint")
    w4.add_argument("--w4-values", nargs="+", type=float, default=[0.25, 0.5, 1.0])

    stage1 = subparsers.add_parser(
        "stage1-sweep", help="E0.1: independent stage-1-only epoch sensitivity"
    )
    _add_common_train_arguments(stage1)
    stage1.add_argument("--label", default="E0_stage1")
    stage1.add_argument("--model", required=True, help="one mdet model YAML")
    stage1.add_argument("--pretrain", required=True, help="pretrained detector checkpoint")
    stage1.add_argument(
        "--stage1-values",
        nargs="+",
        type=int,
        default=[100, 200, 300, 400, 500],
        help="independent stage-1 epoch budgets",
    )

    hsv = subparsers.add_parser(
        "hsv-ablation", help="E0.2: fixed-epoch HSV augmentation ablation"
    )
    _add_common_train_arguments(hsv)
    hsv.add_argument("--label", default="E0_hsv")
    hsv.add_argument("--model", required=True, help="one mdet model YAML")
    hsv.add_argument("--pretrain", required=True, help="pretrained detector checkpoint")
    hsv.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="stage-1 epoch budget for each HSV variant (default: 100)",
    )

    stage2 = subparsers.add_parser(
        "stage2-sweep", help="E0.3: independent stage-2 epoch sensitivity"
    )
    _add_common_train_arguments(stage2)
    stage2.add_argument("--label", default="E0_stage2")
    stage2.add_argument("--model", required=True, help="same mdet model YAML used by stage 1")
    stage2.add_argument(
        "--stage1-checkpoint",
        required=True,
        help="one fixed stage-1 best.pt used to initialize every stage-2 run",
    )
    stage2.add_argument(
        "--stage1-epochs",
        required=True,
        type=int,
        help="stage-1 epoch budget that produced --stage1-checkpoint",
    )
    stage2.add_argument(
        "--stage2-values",
        nargs="+",
        type=int,
        default=[50, 100, 150, 200],
        help="independent stage-2 epoch budgets",
    )

    gca_stage2 = subparsers.add_parser(
        "gca-stage2",
        help="E2.2: stage-2-only GCA-variant x GNN comparison",
    )
    _add_common_train_arguments(gca_stage2)
    gca_stage2.add_argument("--label", default="E2_2_GCA_stage2")
    gca_stage2.add_argument(
        "--stage1-checkpoint",
        required=True,
        help="one fixed baseline stage-1 best.pt used to initialize every variant",
    )
    gca_stage2.add_argument(
        "--stage1-epochs",
        type=int,
        default=DEFAULT_STAGE1_EPOCHS,
        help="stage-1 epoch budget that produced --stage1-checkpoint",
    )
    gca_stage2.add_argument(
        "--stage2-epochs",
        type=int,
        default=DEFAULT_STAGE2_EPOCHS,
        help="stage-2 epoch budget for every GCA/GNN variant",
    )
    gca_stage2.add_argument(
        "--variant",
        action="append",
        required=True,
        metavar="NAME=CONFIG_YAML",
        help="repeat for each structural GCA variant; combine with --gnn-types for a matrix",
    )
    gca_stage2.add_argument(
        "--gnn-types",
        nargs="+",
        choices=("gca", "gcn", "gat", "graphsage", "gin"),
        default=None,
        metavar="GNN",
        help=(
            "materialize each selected GNN operator for every structural variant; "
            "omit to keep the operator encoded in each YAML"
        ),
    )

    prior_stage2 = subparsers.add_parser(
        "prior-stage2",
        help="E2.6/E2.7: head-only fixed co-occurrence-prior structure comparison",
    )
    _add_common_train_arguments(prior_stage2)
    prior_stage2.add_argument("--label", default="E2_6_prior_head")
    prior_stage2.add_argument(
        "--model",
        default="ultralytics/cfg/models/exp_ablation/yolov10x_com_prior.yaml",
        help="co-occurrence-prior model YAML",
    )
    prior_stage2.add_argument(
        "--stage1-checkpoint",
        required=True,
        help="one fixed baseline stage-1 best.pt used to initialize every variant",
    )
    prior_stage2.add_argument(
        "--stage1-epochs",
        type=int,
        default=DEFAULT_STAGE1_EPOCHS,
        help="stage-1 epoch budget that produced --stage1-checkpoint",
    )
    prior_stage2.add_argument(
        "--stage2-epochs",
        type=int,
        default=DEFAULT_STAGE2_EPOCHS,
        help="stage-2 epoch budget for every prior-head variant",
    )
    prior_stage2.add_argument(
        "--prior-types",
        nargs="+",
        choices=(
            "bias",
            "channel",
            "spatial",
            "moe",
            "texture",
            "logit_blend",
            "logit_bias",
            "logit_mlp",
            "cross_attention",
            "dynamic_gate",
        ),
        default=["bias", "channel", "spatial", "moe", "texture"],
        help="prior heads to materialize; default runs the original five",
    )
    prior_stage2.add_argument(
        "--matrix-modes",
        nargs="+",
        choices=("cross", "conditional"),
        default=["cross"],
        help="matrix interpretation; conditional transposes the stored CSV",
    )

    for name, help_text, default_network in (
        ("variants", "Run named ablation/model variants", "yolo"),
        ("gia-position", "E2.1: GIA position ablation", "yolo"),
        ("gca-structure", "E2.2: GCA structure ablation", "yolo"),
        ("gia-gca", "E2.3: joint GIA/GCA ablation", "yolo"),
        ("ho", "E2.4: train the HO checkpoint for inference comparison", "yolo"),
        ("versions", "E3: YOLOv8-YOLOv13/YOLO26/MAYOLO sizes", "yolo"),
        ("rtdetr", "E4: RT-DETR attribute detector sizes", "rtdetr"),
    ):
        variant_parser = subparsers.add_parser(name, help=help_text)
        _add_variant_arguments(variant_parser, default_network=default_network, require_variant=name != "versions")
        if name == "versions":
            variant_parser.add_argument(
                "--include-yolo26",
                action="store_true",
                help="append YOLO26 mdet n/s/m/l/x configs; missing checkpoints default to yolo26*.pt",
            )
            variant_parser.add_argument(
                "--yolo26-sizes",
                nargs="+",
                choices=tuple("nsmlx"),
                default=None,
                help="YOLO26 sizes used with --include-yolo26 (default: n s m l x)",
            )

    stability = subparsers.add_parser("stability", help="E7: repeated seeds for YOLOv10x/MAYOLOx")
    _add_variant_arguments(stability, default_network="yolo")
    stability.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse arguments and execute the selected experiment."""
    args = _build_parser().parse_args(argv)
    if args.experiment == "w4":
        _run_w4(args)
    elif args.experiment == "stage1-sweep":
        _run_stage1_sweep(args)
    elif args.experiment == "hsv-ablation":
        _run_hsv_ablation(args)
    elif args.experiment == "stage2-sweep":
        _run_stage2_sweep(args)
    elif args.experiment == "gca-stage2":
        _run_gca_stage2(args)
    elif args.experiment == "prior-stage2":
        _run_prior_stage2(args)
    elif args.experiment == "stability":
        _run_stability(args)
    else:
        _run_variants(args)


if __name__ == "__main__":
    main()
