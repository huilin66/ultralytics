# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Aggregate the E2 baseline matrix into one comparison table.

Reads ``<project>/seed*_summary.json`` (or per-run ``e2.json`` while a process
is still running), joins B6/B7 from the published E1.5 runs, and reports
mean/std of the final-epoch validation metrics per condition plus parameters.

Example:
    python tools/summarize_e2_baselines.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ITEMS = ("b0-rgb", "b1-thermal", "b2-depth", "b3-rgb-t", "b4-rgb-d", "b5-t-d", "b6-if", "b7-bf", "b8-gated", "c1-conv")
KEYS = ("metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/precision(B)", "metrics/recall(B)")
# B6/B7 reuse the E1.5 runs: item -> fusion mode recorded in that project.
E15_REUSE = {"b6-if": "if", "b7-bf": "bf"}


def load_rows(project: Path) -> dict[str, dict[int, dict]]:
    """Collect the last recorded row per (item, seed) from summaries or per-run JSON files."""
    rows: dict[str, dict[int, dict]] = {}
    for path in sorted(project.glob("seed*_summary.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row.get("status") in {"ok", "skipped"} and "item" in row:
                rows.setdefault(row["item"], {})[int(row["seed"])] = row
    for path in sorted(project.glob("*-seed*/e2.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") == "ok" and "item" in row:
            rows.setdefault(row["item"], {}).setdefault(int(row["seed"]), row)
    return rows


def load_e15_rows(project: Path) -> dict[str, dict[int, dict]]:
    """Collect the reused E1.5 rows keyed by the E2 item name."""
    rows: dict[str, dict[int, dict]] = {}
    for path in sorted(project.glob("seed*_summary.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row.get("status") != "ok":
                continue
            for item, mode in E15_REUSE.items():
                if row.get("mode") == mode:
                    rows.setdefault(item, {})[int(row["seed"])] = row
    return rows


def param_counts() -> dict[str, float]:
    """Count parameters of every condition in millions."""
    counts = {}
    try:
        from ultralytics import YOLO
        from ultralytics.models.multimodal import MultiModalYOLO

        templates = {
            "b0-rgb": "official",
            "b1-thermal": "yolo26x-mm1-if.yaml",
            "b2-depth": "yolo26x-mm1-if.yaml",
            "b3-rgb-t": "yolo26x-mm2-31-if.yaml",
            "b4-rgb-d": "yolo26x-mm2-31-if.yaml",
            "b5-t-d": "yolo26x-mm2-11-if.yaml",
            "b6-if": "yolo26x-mm3-if.yaml",
            "b7-bf": "yolo26x-mm3-bf.yaml",
            "b8-gated": "yolo26x-mm3-bf-gated.yaml",
            "c1-conv": "yolo26x-mm3-bf-conv.yaml",
        }
        for item in ITEMS:
            template = templates[item]
            if template == "official":
                facade = YOLO("yolo26x.yaml", task="detect")
            else:
                facade = MultiModalYOLO(str(REPO / "ultralytics" / "cfg" / "mmodels" / template), task="detect")
            counts[item] = sum(p.numel() for p in facade.model.parameters()) / 1e6
            del facade
    except Exception as error:  # table is still useful without parameter counts
        print(f"[warn] parameter counts unavailable: {error}", file=sys.stderr)
    return counts


def main(argv=None) -> int:
    """Print the per-condition aggregate table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=str(REPO / "runs" / "e2_rlivit"))
    parser.add_argument("--e15-project", default=str(REPO / "runs" / "e15_rlivit_b4_200ep"))
    parser.add_argument("--no-params", action="store_true", help="skip building every model just for parameter counts")
    args = parser.parse_args(argv)

    rows = load_rows(Path(args.project))
    for item, item_rows in load_e15_rows(Path(args.e15_project)).items():
        for seed, row in item_rows.items():  # new runs win if both exist
            rows.setdefault(item, {}).setdefault(seed, row)
    counts = {} if args.no_params else param_counts()
    header = f"{'item':<10} {'runs':>4} {'params(M)':>9} {'s/epoch':>8} {'peakGB':>7} " + " ".join(
        f"{k.split('/')[1]:>16}" for k in KEYS
    )
    print(header)
    print("-" * len(header))
    for item in ITEMS:
        item_rows = rows.get(item, {})
        if not item_rows:
            continue
        cells = []
        for key in KEYS:
            values = [float(run[key]) for run in item_rows.values() if run.get(key) not in (None, "")]
            cells.append(
                f"{statistics.mean(values):.4f}±{statistics.stdev(values):.4f}"
                if len(values) > 1
                else (f"{values[0]:.4f}" if values else "n/a").rjust(16)
            )
        seconds = statistics.mean(float(run.get("epoch_seconds", 0) or 0) for run in item_rows.values())
        peak = max(float(run.get("peak_gb", 0) or 0) for run in item_rows.values())
        params = f"{counts[item]:.1f}" if item in counts else "n/a"
        print(
            f"{item:<10} {len(item_rows):>4} {params:>9} {seconds:>8.1f} {peak:>7.2f} "
            + " ".join(c.rjust(16) for c in cells)
        )
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
