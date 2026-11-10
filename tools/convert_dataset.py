# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert one raw multi-modal dataset into ``<dataset>_mdet``.

Per-dataset defaults live in ``configs/datasets/<dataset>.yaml`` and are loaded
automatically; CLI flags override the file, and ``--set`` overrides its
``options`` block.

Examples:
    python tools/convert_dataset.py --dataset mid3k --dry-run
    python tools/convert_dataset.py --dataset mid3k
    python tools/convert_dataset.py --config configs/datasets/r_livit.yaml --strict
    python tools/convert_dataset.py --dataset r_livit --limit 200 --overwrite
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
CONFIG_DIR = REPO_ROOT / "configs" / "datasets"
sys.path.insert(0, str(TOOLS_DIR))

from mdet_convert.adapters import ADAPTERS  # noqa: E402
from mdet_convert.pipeline import convert  # noqa: E402


def load_config(path: Path) -> dict:
    """Load one YAML conversion config."""
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"config must be a mapping: {path}")
    return payload


def resolve_config(args: argparse.Namespace) -> tuple[str, dict, Path | None]:
    """Return ``(dataset, config, config_path)`` from ``--config`` or auto-discovery."""
    config_path = Path(args.config) if args.config else None
    if config_path is None and args.dataset:
        candidate = CONFIG_DIR / f"{args.dataset}.yaml"
        config_path = candidate if candidate.is_file() else None
    if config_path is not None and not config_path.is_file():
        raise SystemExit(f"config not found: {config_path}")
    config = load_config(config_path) if config_path else {}
    dataset = args.dataset or config.get("dataset")
    if not dataset:
        raise SystemExit("either --dataset or --config with a 'dataset' key is required")
    if dataset not in ADAPTERS:
        raise SystemExit(f"unknown dataset '{dataset}', choices are {sorted(ADAPTERS)}")
    if args.dataset and config.get("dataset") and config["dataset"] != args.dataset:
        raise SystemExit(f"config dataset '{config['dataset']}' does not match --dataset '{args.dataset}'")
    return dataset, config, config_path


def pick(args: argparse.Namespace, config: dict, name: str, default):
    """Resolve a value with precedence CLI > config file > built-in default."""
    value = getattr(args, name)
    return config.get(name, default) if value is None else value


def strict_failures(warnings: list[str], allow: list[str]) -> list[str]:
    """Return warnings that are not covered by a ``--strict-allow`` substring."""
    tokens = [token.lower() for token in allow if token]
    return [warning for warning in warnings if not any(token in warning.lower() for token in tokens)]


def parse_pairs(values: list[str]) -> dict:
    """Parse ``--set key=value`` pairs into a dict, coercing ints and floats."""
    options = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got '{item}'")
        key, value = item.split("=", 1)
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                continue
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            value = value.lower() == "true"
        options[key.strip()] = value
    return options


def parse_ratios(text: str) -> tuple[float, float, float]:
    """Parse the ``train,val,test`` ratio triple."""
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 3:
        raise SystemExit("--ratios expects three comma-separated floats, e.g. 0.8,0.1,0.1")
    return tuple(parts)  # type: ignore[return-value]


def parse_splits(text: str | None) -> set[str] | None:
    """Parse a comma-separated split filter."""
    if not text:
        return None
    allowed = {"train", "val", "test"}
    values = {part.strip() for part in text.split(",") if part.strip()}
    unknown = values - allowed
    if unknown:
        raise SystemExit(f"unknown splits: {sorted(unknown)}")
    return values


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(ADAPTERS), help="raw dataset adapter (optional with --config)")
    parser.add_argument("--config", help="per-dataset YAML (defaults to configs/datasets/<dataset>.yaml)")
    parser.add_argument("--raw-root", default=None, help=r"root that holds the raw dataset folders (default: Y:\dataset)")
    parser.add_argument("--out-root", default=None, help="where <dataset>_mdet is written (default: --raw-root)")
    parser.add_argument("--limit", type=int, default=None, help="convert only the first N samples (smoke tests)")
    parser.add_argument("--splits", default=None, help="comma-separated split filter, e.g. train,val")
    parser.add_argument("--ratios", default=None, help="train,val,test fractions for grouped splits (default: 0.8,0.1,0.1)")
    parser.add_argument("--seed", default=None, help="seed for deterministic grouped splits (default: mdet)")
    parser.add_argument("--previews", type=int, default=None, help="preview panels per split, 0 disables (default: 6)")
    parser.add_argument("--workers", type=int, default=None, help="threads for encoding (default: min(4, cpu_count))")
    parser.add_argument("--set", action="append", default=[], metavar="K=V", help="adapter option override")
    parser.add_argument("--dry-run", action="store_true", help="report the plan without writing anything")
    parser.add_argument("--overwrite", action="store_true", help="delete and rebuild an existing <dataset>_mdet folder")
    parser.add_argument("--allow-existing", action="store_true", help="write into a non-empty folder without clearing it")
    parser.add_argument(
        "--no-sample-level-val",
        action="store_true",
        help="refuse to carve val from individual samples when the train part has a single group",
    )
    parser.add_argument("--strict", action="store_true", help="exit non-zero when any warning is reported")
    parser.add_argument(
        "--strict-allow",
        action="append",
        default=[],
        metavar="SUBSTR",
        help="warning substring exempt from --strict (repeatable)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one adapter end to end."""
    args = build_parser().parse_args(argv)
    dataset, config, config_path = resolve_config(args)
    if config_path is not None:
        print(f"[config] {config_path}")

    raw_root = Path(pick(args, config, "raw_root", r"Y:\dataset"))
    out_root = Path(pick(args, config, "out_root", str(raw_root)))
    ratios = parse_ratios(pick(args, config, "ratios", "0.8,0.1,0.1"))
    seed = pick(args, config, "seed", "mdet")
    previews = int(pick(args, config, "previews", 6))
    workers = int(pick(args, config, "workers", min(4, os.cpu_count() or 1)))
    limit = pick(args, config, "limit", None)
    limit = int(limit) if limit is not None else None
    only_splits = parse_splits(pick(args, config, "splits", None))
    options = dict(config.get("options") or {})
    options.update(parse_pairs(args.set))
    strict = bool(args.strict or config.get("strict", False))
    allow = list(config.get("strict_allow") or []) + list(args.strict_allow)

    adapter = ADAPTERS[dataset](raw_root, options)
    started = time.time()
    plan = adapter.plan()
    initial_warnings = list(plan.warnings)
    print(f"[plan] {plan.display_name}: {len(plan.samples)} samples, {len(plan.modalities)} modalities, {plan.channels} channels")
    for warning in initial_warnings:
        print(f"  ! {warning}")

    dataset_dir = convert(
        plan,
        out_root,
        repo=REPO_ROOT,
        ratios=ratios,
        seed=seed,
        limit=limit,
        only_splits=only_splits,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        allow_existing=args.allow_existing,
        sample_level_val=not args.no_sample_level_val,
        workers=max(1, workers),
        previews=previews,
    )
    for warning in plan.warnings:
        if warning not in initial_warnings:
            print(f"  ! {warning}")

    elapsed = time.time() - started
    if args.dry_run:
        print(f"[dry-run] splits: {plan.splits}")
        print(f"[dry-run] would write {dataset_dir} ({elapsed:.1f}s)")
    else:
        print(f"[done] {dataset_dir} in {elapsed:.1f}s")
        yaml_path = dataset_dir / "data.yaml"
        if yaml_path.is_file():
            print(f"[yaml] {yaml_path}")
        report = dataset_dir / "metadata" / "quality_report.json"
        if report.is_file():
            import json

            payload = json.loads(report.read_text(encoding="utf-8"))
            print(f"[splits] {payload.get('splits')}")
            if payload.get("warnings"):
                print("[warnings]")
                for warning in payload["warnings"]:
                    print(f"  ! {warning}")

    warnings = list(dict.fromkeys(plan.warnings))
    if strict:
        failures = strict_failures(warnings, allow)
        if failures:
            print(f"[strict] {len(failures)} warning(s) not covered by --strict-allow")
            for warning in failures:
                print(f"  ! {warning}")
            return 1
        print("[strict] no blocking warnings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
