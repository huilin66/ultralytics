# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Record what a training run was made of, and refuse to resume one that has changed since.

A run directory is keyed by name and reused, so a config edited between two invocations would
silently mix two graphs in one set of weights. ``pin`` writes the record for a new run and blocks
the second invocation when any part of the contract has moved. Runs trained before this file
existed cannot be stamped retroactively: ``pin`` warns instead of inventing a record for them.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import torch

from ultralytics import __version__

REGISTRY = "provenance.json"
CONTRACT = ("template_sha", "data_sha", "recipe")


def digest(*paths: Path) -> str:
    """Return a short sha256 over the given files, in order."""
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(Path(path).read_bytes())
    return hasher.hexdigest()[:16]


def describe(
    template: Path | list[Path],
    data: Path,
    image_dir: Path,
    checkpoint: Path,
    recipe: dict,
) -> dict:
    """Describe one cell: the graph it builds, the data it reads and the recipe that shapes it."""
    files = template if isinstance(template, list) else [template]
    return {
        "template": "+".join(path.name for path in files),
        "template_sha": digest(*files),
        "data_yaml": str(data),
        "data_sha": digest(data),
        "images": {split: len(list((image_dir / split).iterdir())) for split in ("train", "val")},
        "pretrained": {"file": checkpoint.name, "bytes": checkpoint.stat().st_size if checkpoint.is_file() else None},
        "recipe": dict(recipe),
        "env": {"python": platform.python_version(), "torch": torch.__version__, "ultralytics": __version__},
    }


def pin(run_dir: Path, record: dict, overwrite: bool = False) -> dict:
    """Record a new run, verify a known one, and never back-date a run trained before this existed.

    ``overwrite`` rebuilds the directory from scratch, so a legacy directory may be stamped then.
    A directory whose recorded contract differs from the current one is refused either way: the
    old numbers were produced by another graph and must not be replaced in place.
    """
    path = run_dir / REGISTRY
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        for key in CONTRACT:
            if previous.get(key) != record[key]:
                raise SystemExit(
                    f"[conflict] {run_dir.name}: {key} changed since this run was trained "
                    f"({previous.get(key)} -> {record[key]}). Train into a new name or delete the stale run; "
                    "never mix two recipes in one directory."
                )
    elif (run_dir / "results.csv").is_file() and not overwrite:
        print(f"[unverified] {run_dir.name}: trained before provenance existed, cannot be checked", flush=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
