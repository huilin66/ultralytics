# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Fixture tests for the conversion package (no real dataset required)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

from mdet_convert.adapters.base import Adapter  # noqa: E402
from mdet_convert.encode import encode_image  # noqa: E402
from mdet_convert.labels import dedupe, xyxy_to_yolo, yolo_to_xyxy  # noqa: E402
from mdet_convert.pipeline import convert  # noqa: E402
from mdet_convert.schema import Box, DatasetPlan, EncodeSpec, ModalitySpec, RawSample  # noqa: E402
from mdet_convert.splits import assign_grouped_splits  # noqa: E402


def test_yolo_roundtrip_is_stable():
    """Pixel -> normalised -> pixel must survive with sub-pixel error."""
    xyxy = (10.0, 20.0, 110.0, 220.0)
    cx, cy, bw, bh = xyxy_to_yolo(xyxy, 640, 512)
    back = yolo_to_xyxy(cx, cy, bw, bh, 640, 512)
    assert back == pytest.approx(xyxy, abs=1e-6)


def test_yolo_clips_to_frame():
    """Boxes hanging outside the image are clipped, not dropped."""
    cx, cy, bw, bh = xyxy_to_yolo((-20.0, -10.0, 320.0, 256.0), 640, 512)
    assert cx == pytest.approx(0.25)
    assert bw == pytest.approx(0.5)


def test_dedupe_drops_identical_boxes():
    """Duplicate boxes of the same class collapse, different classes survive."""
    boxes = [
        Box("person", (0, 0, 10, 10)),
        Box("person", (0, 0, 10, 10)),
        Box("car", (0, 0, 10, 10)),
    ]
    kept, dropped = dedupe(boxes)
    assert len(kept) == 2 and dropped == 1


def test_grouped_splits_are_deterministic_and_disjoint():
    """Same seed reproduces the split and no group appears in two splits."""
    sizes = {f"scene{i}": 10 for i in range(20)}
    first = assign_grouped_splits(sizes, seed="abc")
    second = assign_grouped_splits(sizes, seed="abc")
    assert first == second
    assert set(first) == set(sizes)
    assert set(first.values()) <= {"train", "val", "test"}


def test_gray_encoding_reduces_replicated_channels(tmp_path):
    """A three-channel replicated grayscale file becomes a single channel."""
    image = np.full((8, 8, 3), 77, np.uint8)
    path = tmp_path / "t.png"
    cv2.imwrite(str(path), image)
    encoded, stats = encode_image(path, EncodeSpec("gray"))
    assert encoded.shape == (8, 8)
    assert stats.replicated_channels is True


def test_scaled_encoding_maps_sentinel_to_zero(tmp_path):
    """Out-of-range and sentinel values are mapped to 0 / 255 explicitly."""
    image = np.array([[0, 64536, 10000, 20000]], np.uint16)
    path = tmp_path / "d.png"
    cv2.imwrite(str(path), image)
    encoded, stats = encode_image(
        path, EncodeSpec("scaled", units="mm", lo=0.0, hi=20000.0, invalid=(64536,))
    )
    assert encoded.dtype == np.uint8
    assert encoded[0, 0] == 0 and encoded[0, 1] == 0 and encoded[0, 3] == 255
    assert stats.invalid_fraction == pytest.approx(0.25)


class _FakeAdapter(Adapter):
    """Adapter over a synthetic three-modality fixture."""

    key = "fake"
    display_name = "Fake"

    def plan(self) -> DatasetPlan:
        root = self.raw_root
        samples = []
        for index in range(10):
            stem = f"{index:03d}"
            files = {
                "rgb": root / "rgb" / f"{stem}.png",
                "thermal": root / "thermal" / f"{stem}.png",
                "depth": root / "depth" / f"{stem}.png",
            }
            samples.append(
                RawSample(
                    sample_id=stem,
                    group=f"scene{index // 2}",
                    files=files,
                    boxes=[Box("person", (10.0, 10.0, 60.0, 80.0))],
                )
            )
        return DatasetPlan(
            key=self.key,
            display_name=self.display_name,
            reference_modality="rgb",
            modalities=(
                ModalitySpec("rgb", 3, EncodeSpec("rgb"), color="bgr"),
                ModalitySpec("thermal", 1, EncodeSpec("gray")),
                ModalitySpec("depth", 1, EncodeSpec("scaled", units="mm", lo=0.0, hi=2000.0)),
            ),
            classes={"person": 0},
            samples=samples,
        )


@pytest.fixture()
def fake_raw(tmp_path):
    """Create ten aligned RGB/thermal/depth samples."""
    for index in range(10):
        stem = f"{index:03d}"
        for name, array in (
            ("rgb", np.full((32, 48, 3), 90, np.uint8)),
            ("thermal", np.full((32, 48), 120, np.uint8)),
            ("depth", np.full((32, 48), 1000, np.uint16)),
        ):
            directory = tmp_path / "raw" / name
            directory.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(directory / f"{stem}.png"), array)
    return tmp_path / "raw"


def test_pipeline_writes_loadable_layout(fake_raw, tmp_path):
    """The pipeline emits the exact tree the multi-modal loader expects."""
    dataset = convert(
        _FakeAdapter(fake_raw).plan(),
        tmp_path / "out",
        repo=tmp_path,
        previews=0,
    )
    assert dataset.name == "Fake_mdet"
    yaml_text = (dataset / "data.yaml").read_text(encoding="utf-8")
    assert "channels: 5" in yaml_text

    images = list((dataset / "images" / "rgb").rglob("*.png"))
    labels = list((dataset / "labels" / "rgb").rglob("*.txt"))
    assert len(images) == len(labels) == 10
    # label dir mirrors the image sub-path, which is what img2label_paths requires
    assert {p.parent.name for p in labels} == {p.parent.name for p in images}

    for label in labels:
        row = label.read_text(encoding="utf-8").strip().split()
        assert len(row) == 5
        values = [float(v) for v in row[1:]]
        assert all(0.0 <= v <= 1.0 for v in values)

    assert (dataset / "metadata" / "samples.jsonl").is_file()
    assert (dataset / "metadata" / "split_manifest.csv").is_file()


def test_pipeline_refuses_uncalibrated_mismatch(fake_raw, tmp_path):
    """A modality at a different resolution is excluded instead of resized."""
    bad = fake_raw / "thermal" / "000.png"
    cv2.imwrite(str(bad), np.full((16, 24), 120, np.uint8))
    dataset = convert(_FakeAdapter(fake_raw).plan(), tmp_path / "out2", repo=tmp_path, previews=0)
    exclusions = (dataset / "metadata" / "exclusions.csv").read_text(encoding="utf-8")
    assert "refusing to resize" in exclusions
