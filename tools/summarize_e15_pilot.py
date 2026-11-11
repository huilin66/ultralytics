# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Aggregate E1.5 fusion-position pilot runs into one comparison table.

Reads ``<project>/seed*_summary.json`` (or per-run ``pilot.json`` when a process
is still running) and reports mean/std of the final-epoch validation metrics per
fusion mode, plus parameter counts so throughput differences are interpretable.

Example:
    python tools/summarize_e15_pilot.py --project runs/e15_rlivit
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
MODES = ("if", "ef", "nif", "bf", "nf", "hf")
KEYS = ("metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/precision(B)", "metrics/recall(B)")


def load_rows(project: Path) -> dict[str, dict[int, dict]]:
    """Collect the last recorded row per (mode, seed)."""
    rows: dict[str, dict[int, dict]] = {}
    for path in sorted(project.glob("seed*_summary.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row.get("status") == "ok" and "mode" in row:
                rows.setdefault(row["mode"], {})[int(row["seed"])] = row
    for path in sorted(project.glob("*-seed*/pilot.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") == "ok":
            rows.setdefault(row["mode"], {}).setdefault(int(row["seed"]), row)
    return rows


def param_counts(model: str) -> dict[str, float]:
    """Count parameters of each fusion-mode model in millions."""
    counts = {}
    try:
        from ultralytics.models.multimodal import MultiModalYOLO

        for mode in MODES:
            cfg = REPO / "ultralytics" / "cfg" / "mmodels" / f"{model}-mm3-{mode}.yaml"
            if cfg.is_file():
                facade = MultiModalYOLO(str(cfg), task="detect")
                counts[mode] = sum(p.numel() for p in facade.model.parameters()) / 1e6
                del facade
    except Exception as error:  # table is still useful without parameter counts
        print(f"[warn] parameter counts unavailable: {error}", file=sys.stderr)
    return counts


def main(argv=None) -> int:
    """Print the per-mode aggregate table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=str(REPO / "runs" / "e15_rlivit"))
    parser.add_argument("--model", default="yolo26x", help="only for parameter counts; empty skips them")
    args = parser.parse_args(argv)

    project = Path(args.project)
    rows = load_rows(project)
    counts = param_counts(args.model) if args.model else {}
    header = f"{'mode':<5} {'runs':>4} {'params(M)':>9} {'s/epoch':>8} {'peakGB':>7} " + " ".join(
        f"{k.split('/')[1]:>16}" for k in KEYS
    )
    print(header)
    print("-" * len(header))
    for mode in MODES:
        runs = rows.get(mode, {})
        if not runs:
            continue
        cells = []
        for key in KEYS:
            values = [float(run[key]) for run in runs.values() if run.get(key) not in (None, "")]
            cells.append(
                f"{statistics.mean(values):.4f}±{statistics.stdev(values):.4f}"
                if len(values) > 1
                else (f"{values[0]:.4f}" if values else "n/a").rjust(16)
            )
        seconds = statistics.mean(float(run.get("epoch_seconds", 0) or 0) for run in runs.values())
        peak = max(float(run.get("peak_gb", 0) or 0) for run in runs.values())
        params = f"{counts[mode]:.1f}" if mode in counts else "n/a"
        print(
            f"{mode:<5} {len(runs):>4} {params:>9} {seconds:>8.1f} {peak:>7.2f} " + " ".join(c.rjust(16) for c in cells)
        )
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
