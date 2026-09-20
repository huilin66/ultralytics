"""Evaluate multiple mdetect checkpoints on an offline robustness manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METRIC_KEYS = (
    ("mAP50_test", "metrics/mAP50(B)"),
    ("mAP50-95_test", "metrics/mAP50-95(B)"),
    ("OA_test", "metrics/OA(A)"),
    ("F1_macro_test", "metrics/f1_macro(A)"),
    ("F1_macro_global_test", "metrics/f1_macro_global(A)"),
    ("P_macro_test", "metrics/P_macro(A)"),
    ("R_macro_test", "metrics/R_macro(A)"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate seed-0 mdet models on deterministic robustness variants")
    parser.add_argument("--data", required=True, help="Clean mdet dataset YAML")
    parser.add_argument("--manifest", required=True, help="Robustness manifest.json")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="LABEL=WEIGHTS[::native|one2many]; repeat for every model",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", default="seed0")
    parser.add_argument("--resume", action="store_true", help="Skip model/variant pairs already in summary.csv")
    return parser


def _parse_model(spec: str) -> tuple[str, str, str]:
    model_spec, separator, mode = spec.partition("::")
    label, equals, weights = model_spec.partition("=")
    if not equals or not label or not weights:
        raise ValueError(f"Invalid model specification: {spec!r}")
    mode = mode or "native"
    if mode not in {"native", "one2many"}:
        raise ValueError(f"Unsupported head mode: {mode}")
    return label, weights, mode


def _as_float(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def _set_one2many(model: Any) -> None:
    detector = getattr(model, "model", None)
    layers = getattr(detector, "model", None)
    if not layers:
        raise RuntimeError("Could not locate the mdetect model head")
    switch = getattr(layers[-1], "use_one2many_head", None)
    if switch is None:
        raise RuntimeError("Checkpoint does not expose use_one2many_head()")
    switch()


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "mode",
        "variant",
        "condition",
        "severity",
        "weight",
        "data",
        "seed",
        *[key for key, _ in METRIC_KEYS],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_existing(path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    if not path.is_file():
        return [], set()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keys = {(row.get("model", ""), row.get("variant", "")) for row in rows}
    return rows, keys


def _evaluate(
    model: Any,
    *,
    label: str,
    mode: str,
    weights: str,
    variant: dict[str, Any],
    args: argparse.Namespace,
    eval_project: Path,
) -> dict[str, Any]:
    from ultralytics.utils import LOGGER

    variant_id = str(variant["variant"])
    data_yaml = str(variant["data_yaml"])
    eval_name = f"{args.name}_{label}_{variant_id}".replace("/", "_").replace(" ", "_")
    kwargs: dict[str, Any] = {
        "data": data_yaml if variant_id != "clean" else args.data,
        "split": "test",
        "device": args.device,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "conf": args.conf,
        "iou": args.iou,
        "seed": args.seed,
        "deterministic": True,
        "project": str(eval_project),
        "name": eval_name,
        "plots": False,
        "verbose": False,
        "save_json": False,
    }
    if kwargs["conf"] is None:
        kwargs.pop("conf")
    LOGGER.info(f"[robustness] model={label}, variant={variant_id}, mode={mode}")
    metrics = model.val(**kwargs)
    values = metrics.results_dict
    row: dict[str, Any] = {
        "model": label,
        "mode": mode,
        "variant": variant_id,
        "condition": variant["condition"],
        "severity": variant["severity"],
        "weight": weights,
        "data": kwargs["data"],
        "seed": args.seed,
    }
    for output_key, metrics_key in METRIC_KEYS:
        row[output_key] = _as_float(values[metrics_key])
    return row


def main() -> None:
    args = _parser().parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    variants = [
        {
            "variant": "clean",
            "condition": "clean",
            "severity": "clean",
            "data_yaml": args.data,
        },
        *manifest["variants"],
    ]
    models = [_parse_model(spec) for spec in args.model]
    output_dir = Path(args.project).expanduser().resolve() / args.name
    summary_path = output_dir / "summary.csv"
    rows, completed = _load_existing(summary_path) if args.resume else ([], set())
    eval_project = output_dir / "eval_runs"

    # Import and load each checkpoint once.  The model is reused across all
    # variants, while each validation call receives a fresh deterministic loader.
    from ultralytics import YOLO

    for label, weights, mode in models:
        model = YOLO(weights, task="mdetect")
        if mode == "one2many":
            _set_one2many(model)
        for variant in variants:
            key = (label, str(variant["variant"]))
            if key in completed:
                print(f"[skip] {label} {variant['variant']}", flush=True)
                continue
            row = _evaluate(
                model,
                label=label,
                mode=mode,
                weights=weights,
                variant=variant,
                args=args,
                eval_project=eval_project,
            )
            rows = [existing for existing in rows if (existing.get("model"), existing.get("variant")) != key]
            rows.append(row)
            _write_summary(summary_path, rows)
            print(
                f"[done] {label} {variant['variant']} "
                f"mAP50={row['mAP50_test']:.6f} F1={row['F1_macro_test']:.6f}",
                flush=True,
            )

    run_info = {
        "seed": args.seed,
        "manifest": str(manifest_path),
        "clean_data": str(Path(args.data).expanduser().resolve()),
        "models": [{"label": label, "weights": weights, "mode": mode} for label, weights, mode in models],
        "variant_count": len(variants),
        "summary": str(summary_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[summary] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
