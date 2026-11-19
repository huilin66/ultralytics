"""Aggregate end-to-end evaluator CSV rows by model/experiment label.

The evaluator writes one row per checkpoint.  This small utility keeps the
aggregation explicit and reproducible: groups are formed by removing the final
``_seedN`` suffix, and standard deviations are sample standard deviations.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import mean, stdev


DEFAULT_METRICS = (
    "mAP50_test",
    "mAP50-95_test",
    "F1_macro_conditional_test",
    "F1_micro_conditional_test",
    "detection_recall_e2e",
    "Precision_e2e",
    "Recall_e2e",
    "F1_macro_e2e",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Input evaluator summary CSV")
    parser.add_argument("--output", required=True, help="Aggregated CSV output")
    parser.add_argument("--markdown", default=None, help="Optional Markdown table output")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exact model label to exclude; may be supplied more than once",
    )
    parser.add_argument(
        "--group-regex",
        default=r"^(.*)_seed\d+$",
        help="Regex with one capture group used to derive the aggregation key",
    )
    parser.add_argument("--metric", action="append", dest="metrics", default=None)
    return parser


def _read_rows(paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def main() -> None:
    args = _parser().parse_args()
    rows = _read_rows(args.input)
    excluded = set(args.exclude)
    pattern = re.compile(args.group_regex)
    metrics = tuple(args.metrics or DEFAULT_METRICS)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("model", "") in excluded:
            continue
        match = pattern.match(row.get("model", ""))
        key = match.group(1) if match else row.get("model", "")
        grouped.setdefault(key, []).append(row)

    output_rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        members = grouped[key]
        output: dict[str, object] = {"group": key, "n": len(members)}
        for metric in metrics:
            values = [float(row[metric]) for row in members if row.get(metric, "") != ""]
            output[f"{metric}_mean"] = mean(values) if values else 0.0
            output[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        output_rows.append(output)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output_rows[0]) if output_rows else ["group", "n"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    if args.markdown:
        markdown_path = Path(args.markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        headers = ["Group", "n"] + [f"{metric} (mean ± std)" for metric in metrics]
        lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        for row in output_rows:
            values = [str(row["group"]), str(row["n"])]
            values.extend(
                f"{float(row[f'{metric}_mean']):.6f} ± {float(row[f'{metric}_std']):.6f}"
                for metric in metrics
            )
            lines.append("| " + " | ".join(values) + " |")
        markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
