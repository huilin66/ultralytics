# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Fixture tests for ``tools/build_cls_dataset.py`` (no real dataset required)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import build_cls_dataset as build  # noqa: E402


def make_crop(root: Path, class_name: str, filename: str) -> None:
    """Write a placeholder image file with unique bytes (the builder never decodes it)."""
    target = root / class_name / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(f"{class_name}/{filename}".encode())


def test_source_stem_strips_instance_suffix() -> None:
    """The trailing underscore and instance index are removed."""
    assert build.source_stem("Nanjing_B_ladybug_panoramic_005207_1") == "Nanjing_B_ladybug_panoramic_005207"
    assert build.source_stem("Shanghai_B_00000000000000273_12") == "Shanghai_B_00000000000000273"
    with pytest.raises(ValueError):
        build.source_stem("no_instance_suffix")


def test_split_lists_reject_the_same_source_in_two_splits(tmp_path: Path) -> None:
    """A frame may not appear in both train and val."""
    (tmp_path / "train.txt").write_text("/data/images/frame1.jpg\n", encoding="utf-8")
    (tmp_path / "val.txt").write_text("/data/images/frame1.jpg\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="more than one split"):
        build.read_split_lists(tmp_path)


def test_build_keeps_source_image_crops_together(tmp_path: Path, monkeypatch) -> None:
    """All crops from one frame land in the split of that frame."""
    src = tmp_path / "crop"
    make_crop(src, "a", "frame1_1.jpg")
    make_crop(src, "a", "frame1_2.jpg")
    make_crop(src, "a", "frame2_2.jpg")
    make_crop(src, "b", "frame2_1.jpg")
    split_dir = tmp_path / "lists"
    split_dir.mkdir()
    (split_dir / "train.txt").write_text("/data/images/frame1.jpg\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("/data/images/frame2.jpg\n", encoding="utf-8")
    out = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        ["build_cls_dataset.py", "--src", str(src), "--out", str(out), "--split-dir", str(split_dir)],
    )
    assert build.main() == 0

    assert (out / "train" / "a" / "frame1_1.jpg").is_file()
    assert (out / "train" / "a" / "frame1_2.jpg").is_file()
    assert (out / "val" / "a" / "frame2_2.jpg").is_file()
    assert (out / "val" / "b" / "frame2_1.jpg").is_file()
    assert not (out / "val" / "a" / "frame1_1.jpg").exists()

    metadata = json.loads((out / "classification_metadata.json").read_text(encoding="utf-8"))
    assert metadata["counts"] == {"train": {"a": 2}, "val": {"a": 1, "b": 1}}
    assert metadata["cross_split_duplicates"] == 0
