# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert one raw multi-modal dataset into ``<dataset>_mdet``.

Examples:
    python tools/convert_dataset.py --dataset mid3k --dry-run
    python tools/convert_dataset.py --dataset mid3k
    python tools/convert_dataset.py --dataset r_livit --limit 200 --overwrite
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

from mdet_convert.adapters import ADAPTERS  # noqa: E402
from mdet_convert.pipeline import convert  # noqa: E402


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
    parser.add_argument("--dataset", required=True, choices=sorted(ADAPTERS), help="raw dataset adapter to run")
    parser.add_argument("--raw-root", default=r"Y:\dataset", help="root that holds the raw dataset folders")
    parser.add_argument("--out-root", default=None, help="where <dataset>_mdet is written (defaults to --raw-root)")
    parser.add_argument("--limit", type=int, default=None, help="convert only the first N samples (smoke tests)")
    parser.add_argument("--splits", default=None, help="comma-separated split filter, e.g. train,val")
    parser.add_argument("--ratios", default="0.8,0.1,0.1", help="train,val,test fractions for grouped splits")
    parser.add_argument("--seed", default="mdet", help="seed for deterministic grouped splits")
    parser.add_argument("--previews", type=int, default=6, help="preview panels per split (0 disables)")
    parser.add_argument("--set", action="append", default=[], metavar="K=V", help="adapter option override")
    parser.add_argument("--dry-run", action="store_true", help="report the plan without writing anything")
    parser.add_argument("--overwrite", action="store_true", help="rebuild an existing <dataset>_mdet folder")
    parser.add_argument("--allow-existing", action="store_true", help="write into a non-empty folder without clearing it")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one adapter end to end."""
    args = build_parser().parse_args(argv)
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root) if args.out_root else raw_root

    adapter = ADAPTERS[args.dataset](raw_root, parse_pairs(args.set))
    started = time.time()
    plan = adapter.plan()
    print(f"[plan] {plan.display_name}: {len(plan.samples)} samples, {len(plan.modalities)} modalities, {plan.channels} channels")
    for warning in plan.warnings:
        print(f"  ! {warning}")

    if args.overwrite and not args.dry_run:
        target = out_root / f"{plan.display_name}_mdet"
        if target.exists():
            import shutil

            print(f"[clean] removing {target}")
            shutil.rmtree(target)

    dataset_dir = convert(
        plan,
        out_root,
        repo=REPO_ROOT,
        ratios=parse_ratios(args.ratios),
        seed=args.seed,
        limit=args.limit,
        only_splits=parse_splits(args.splits),
        dry_run=args.dry_run,
        overwrite=args.overwrite or args.allow_existing,
        previews=args.previews,
    )

    elapsed = time.time() - started
    if args.dry_run:
        print(f"[dry-run] would write {dataset_dir} ({elapsed:.1f}s)")
        return 0

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
