# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Leakage-safe, reproducible split assignment."""

from __future__ import annotations

import hashlib
from collections import defaultdict

SPLIT_ORDER = ("train", "val", "test")


def _rank(seed: str, group: str) -> str:
    """Stable per-group pseudo-random key, independent of iteration order."""
    return hashlib.sha1(f"{seed}|{group}".encode()).hexdigest()


def assign_grouped_splits(
    group_sizes: dict[str, int], ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: str = "mdet"
) -> dict[str, str]:
    """Assign whole groups to splits so no scene/sequence straddles two splits.

    Args:
        group_sizes: Number of samples per group (scene, sequence or collection date).
        ratios: Target ``(train, val, test)`` fractions by sample count.
        seed: Change this to reshuffle deterministically.

    Returns:
        Mapping ``group -> split``.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {ratios}")
    total = sum(group_sizes.values())
    if total == 0:
        return {}

    ordered = sorted(group_sizes, key=lambda g: _rank(seed, g))
    targets = {split: ratios[i] * total for i, split in enumerate(SPLIT_ORDER)}
    assigned: dict[str, str] = {}
    counts = dict.fromkeys(SPLIT_ORDER, 0)

    for group in ordered:
        # Put the group into the split furthest below its target, so small groups
        # still land where they are needed most.
        split = max(SPLIT_ORDER, key=lambda s: targets[s] - counts[s])
        assigned[group] = split
        counts[split] += group_sizes[group]

    # A split with no group at all is unusable downstream; move the smallest group.
    for split in SPLIT_ORDER:
        if counts[split] == 0 and len(ordered) > len(SPLIT_ORDER):
            donor = max((s for s in SPLIT_ORDER if counts[s] > 0), key=lambda s: counts[s])
            candidate = min((g for g in assigned if assigned[g] == donor), key=lambda g: group_sizes[g])
            assigned[candidate] = split
            counts[donor] -= group_sizes[candidate]
            counts[split] += group_sizes[candidate]
    return assigned


def summarize_splits(assignments: dict[str, str], group_sizes: dict[str, int]) -> dict[str, dict]:
    """Return per-split group and sample counts for the manifest."""
    summary: dict[str, dict] = {split: {"groups": 0, "samples": 0} for split in SPLIT_ORDER}
    for group, split in assignments.items():
        summary[split]["groups"] += 1
        summary[split]["samples"] += group_sizes[group]
    return summary


def groups_from_samples(samples) -> dict[str, int]:
    """Count samples per group id."""
    sizes: defaultdict[str, int] = defaultdict(int)
    for sample in samples:
        sizes[sample.group] += 1
    return dict(sizes)
