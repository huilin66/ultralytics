# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Summarise every generated ``<dataset>_mdet`` folder into reviewable reports.

The tool only reads the conversion metadata written by ``tools/convert_dataset.py``
and the files on disk; it never re-parses the raw releases and never imports
``ultralytics``.

Examples:
    python tools/summarize_datasets.py --root Y:/dataset
    python tools/summarize_datasets.py --root Y:/dataset --datasets MID-3K_mdet MM5_mdet
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

METADATA_FILES = ("conversion.json", "quality_report.json", "classes.json", "samples.jsonl")


def load_json(path: Path, default):
    """Load JSON if present, otherwise return the default."""
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_samples(path: Path) -> list[dict]:
    """Read ``metadata/samples.jsonl`` line by line."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Write a CSV with a stable column order, even with no rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def dataset_dirs(root: Path, names: list[str] | None) -> list[Path]:
    """Return the ``_mdet`` folders to summarise."""
    if names:
        return [root / name for name in names]
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name.endswith("_mdet"))


def modality_aggregates(samples: list[dict]) -> dict[str, dict]:
    """Aggregate per-sample modality diagnostics into per-modality statistics."""
    aggregate: dict[str, dict] = defaultdict(
        lambda: {"invalid": [], "saturated": [], "statuses": set(), "units": set(), "dtype": set()}
    )
    for sample in samples:
        for name, info in sample.get("modalities", {}).items():
            entry = aggregate[name]
            entry["invalid"].append(float(info.get("invalid_fraction", 0.0)))
            entry["saturated"].append(float(info.get("saturated_fraction", 0.0)))
            entry["statuses"].add(str(info.get("alignment_status", "unknown")))
            entry["units"].add(str(info.get("units", "")))
            entry["dtype"].add(str(info.get("dtype", "")))
    return aggregate


def summarize_one(dataset_dir: Path) -> dict:
    """Collect every report row for one dataset folder."""
    conversion = load_json(dataset_dir / "metadata" / "conversion.json", {})
    quality = load_json(dataset_dir / "metadata" / "quality_report.json", {})
    classes = load_json(dataset_dir / "metadata" / "classes.json", {}).get("names", {})
    samples = load_samples(dataset_dir / "metadata" / "samples.jsonl")
    splits = conversion.get("splits", {})
    modalities = conversion.get("modalities", [])

    annotation_status = Counter(sample.get("annotation_status", "unknown") for sample in samples)
    groups = {sample.get("group") for sample in samples if sample.get("group")}
    boxes = sum(int(stats.get("boxes", 0)) for stats in quality.get("splits", {}).values())
    empty = sum(int(stats.get("empty_labels", 0)) for stats in quality.get("splits", {}).values())

    group_splits: dict[str, set[str]] = defaultdict(set)
    group_samples = Counter()
    for sample in samples:
        group = sample.get("group")
        if group:
            group_splits[group].add(str(sample.get("split")))
            group_samples[group] += 1
    leakage_rows = [
        {
            "dataset": conversion.get("dataset", dataset_dir.name),
            "group": group,
            "splits": "|".join(sorted(splits)),
            "samples": group_samples[group],
        }
        for group, splits in sorted(group_splits.items())
        if len(splits) > 1
    ]

    files_on_disk = {}
    for modality in modalities:
        name = modality["name"]
        files_on_disk[name] = sum(
            len(list((dataset_dir / "images" / name / split).glob("*.png")))
            for split in splits
            if (dataset_dir / "images" / name / split).is_dir()
        )

    summary = {
        "dataset": conversion.get("dataset", dataset_dir.name),
        "folder": dataset_dir.name,
        "samples": conversion.get("samples", len(samples)),
        "train": splits.get("train", 0),
        "val": splits.get("val", 0),
        "test": splits.get("test", 0),
        "groups": len(groups),
        "classes": len(classes),
        "modalities": "+".join(m["name"] for m in modalities),
        "channels": conversion.get("channels", ""),
        "boxes": boxes,
        "empty_labels": empty,
        "excluded": quality.get("exclusions", 0),
        "annotation_status": ";".join(f"{k}={v}" for k, v in sorted(annotation_status.items())),
        "split_protocol": conversion.get("split_protocol", ""),
        "reference_camera": conversion.get("reference_camera", ""),
        "leaked_groups": len(leakage_rows),
        "warnings": len(quality.get("warnings", [])),
    }

    class_rows = []
    for split, stats in sorted(quality.get("splits", {}).items()):
        per_class = stats.get("per_class", {})
        images_with_class = stats.get("images_with_class", {})
        for class_name, instances in sorted(per_class.items(), key=lambda kv: int(classes.get(kv[0], 10**9))):
            class_rows.append(
                {
                    "dataset": summary["dataset"],
                    "split": split,
                    "class_id": classes.get(class_name, ""),
                    "class": class_name,
                    "instances": instances,
                    "images": images_with_class.get(class_name, 0),
                }
            )

    aggregate = modality_aggregates(samples)
    modality_rows = []
    for modality in modalities:
        name = modality["name"]
        stats = aggregate.get(name, {})
        invalid = stats.get("invalid") or [0.0]
        saturated = stats.get("saturated") or [0.0]
        encoding = modality.get("encoding", {})
        modality_rows.append(
            {
                "dataset": summary["dataset"],
                "modality": name,
                "channels": modality.get("channels", ""),
                "files": files_on_disk.get(name, 0),
                "units": encoding.get("units", ""),
                "encoding_kind": encoding.get("kind", ""),
                "scale": encoding.get("scale", ""),
                "invalid_value": "|".join(str(v) for v in encoding.get("invalid", [])),
                "raw_dtype": "|".join(sorted(stats.get("dtype", set()))),
                "mean_invalid_fraction": round(sum(invalid) / len(invalid), 6),
                "max_invalid_fraction": round(max(invalid), 6),
                "mean_saturated_fraction": round(sum(saturated) / len(saturated), 6),
                "alignment_status": "|".join(sorted(stats.get("statuses", set()))),
                "reference_camera": conversion.get("reference_camera", ""),
            }
        )

    return {
        "summary": summary,
        "classes": class_rows,
        "modalities": modality_rows,
        "leakage": leakage_rows,
        "conversion": conversion,
        "quality": quality,
    }


def render_markdown(results: list[dict]) -> str:
    """Render a human-readable selection report with the facts only."""
    lines = [
        "# Dataset selection report (auto-generated)",
        "",
        "This file is generated by `tools/summarize_datasets.py`. It records what",
        "was converted; it does **not** choose the final benchmark. Fill the",
        "`human_decision` table after reviewing the raw data and licenses.",
        "",
        "| dataset | samples | train/val/test | groups | classes | modalities | boxes | empty labels | annotation status | split protocol |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        s = result["summary"]
        lines.append(
            f"| {s['dataset']} | {s['samples']} | {s['train']}/{s['val']}/{s['test']} | {s['groups']} | "
            f"{s['classes']} | {s['modalities']} | {s['boxes']} | {s['empty_labels']} | "
            f"{s['annotation_status']} | {s['split_protocol']} |"
        )
    lines += ["", "## Warnings and blockers", ""]
    for result in results:
        warnings = result["quality"].get("warnings", [])
        if not warnings:
            continue
        lines.append(f"### {result['summary']['dataset']}")
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    lines += ["", "## Split leakage (same group in more than one split)", ""]
    leaked = False
    for result in results:
        for row in result["leakage"]:
            leaked = True
            lines.append(f"- {row['dataset']}: group `{row['group']}` appears in {row['splits']} ({row['samples']} samples)")
    if not leaked:
        lines.append("- none detected from the recorded group ids; file-hash/perceptual-hash checks are still required.")
    lines += [
        "",
        "## Human decision (fill in)",
        "",
        "| dataset | three-modality status | annotation type | class coverage | alignment | test protocol | license | expected role | final decision |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        s = result["summary"]
        lines.append(f"| {s['dataset']} |  | {s['annotation_status']} |  |  | {s['split_protocol']} |  |  |  |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Write the CSV/Markdown reports for every generated dataset."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=r"Y:\dataset", help="root that holds the <dataset>_mdet folders")
    parser.add_argument("--out", default="reports/data_statistics", help="report output directory")
    parser.add_argument("--datasets", nargs="*", default=None, help="only these folder names")
    args = parser.parse_args(argv)

    root = Path(args.root)
    out = Path(args.out)
    results = []
    for dataset_dir in dataset_dirs(root, args.datasets):
        if not (dataset_dir / "metadata" / "conversion.json").is_file():
            print(f"[skip] {dataset_dir}: no metadata/conversion.json")
            continue
        result = summarize_one(dataset_dir)
        results.append(result)
        s = result["summary"]
        print(
            f"[summary] {s['dataset']}: {s['samples']} samples "
            f"({s['train']}/{s['val']}/{s['test']}), {s['boxes']} boxes, {s['classes']} classes"
        )

    if not results:
        print("[error] no <dataset>_mdet folders with metadata found")
        return 1

    summaries = [result["summary"] for result in results]
    classes = [row for result in results for row in result["classes"]]
    modalities = [row for result in results for row in result["modalities"]]
    leakage = [row for result in results for row in result["leakage"]]

    write_csv(out / "dataset_summary.csv", summaries, list(summaries[0].keys()))
    write_csv(
        out / "class_distribution.csv",
        classes,
        ["dataset", "split", "class_id", "class", "instances", "images"],
    )
    write_csv(
        out / "modality_quality.csv",
        modalities,
        [
            "dataset", "modality", "channels", "files", "units", "encoding_kind", "scale",
            "invalid_value", "raw_dtype", "mean_invalid_fraction", "max_invalid_fraction",
            "mean_saturated_fraction", "alignment_status", "reference_camera",
        ],
    )
    write_csv(out / "split_leakage_report.csv", leakage, ["dataset", "group", "splits", "samples"])
    (out / "dataset_selection_report.md").parent.mkdir(parents=True, exist_ok=True)
    (out / "dataset_selection_report.md").write_text(render_markdown(results), encoding="utf-8")
    print(f"[done] reports written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
