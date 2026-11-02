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
from mdet_convert.projection import (  # noqa: E402
    LidarCalibration,
    depth_to_uint16_mm,
    load_calibration,
    load_sequence_mapping,
    match_lidar_frame,
    project_lidar_to_depth,
)
from mdet_convert.schema import Box, DatasetPlan, EncodeSpec, ModalitySpec, RawSample  # noqa: E402
from mdet_convert.splits import assign_grouped_splits, assign_sample_splits  # noqa: E402


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


def test_sample_level_splits_are_deterministic_and_bounded():
    """The fallback split picks a reproducible fraction of the samples."""
    ids = [f"{i:04d}" for i in range(100)]
    first = assign_sample_splits(ids, ratio=0.1, seed="abc")
    second = assign_sample_splits(ids, ratio=0.1, seed="abc")
    assert first == second
    assert len(first) == 10
    assert first <= set(ids)
    assert assign_sample_splits(ids, ratio=0.0) == set()


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

    import json

    first = json.loads((dataset / "metadata" / "samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["reference_modality"] == "rgb"
    assert first["reference_camera"] == "rgb"
    depth = first["modalities"]["depth"]
    assert depth["alignment_status"] == "verified_size_match"
    assert depth["aligned_to"] == "rgb"
    assert depth["units"] == "mm"
    assert depth["invalid_value"] == []
    assert depth["scale"] == pytest.approx(2000.0 / 255.0)


def test_parallel_workers_produce_identical_manifests(fake_raw, tmp_path):
    """Threaded encoding must not change manifest order or contents."""
    serial = convert(_FakeAdapter(fake_raw).plan(), tmp_path / "serial", repo=tmp_path, previews=0, workers=1)
    parallel = convert(_FakeAdapter(fake_raw).plan(), tmp_path / "parallel", repo=tmp_path, previews=0, workers=4)
    assert (serial / "metadata" / "samples.jsonl").read_text(encoding="utf-8") == (
        parallel / "metadata" / "samples.jsonl"
    ).read_text(encoding="utf-8")
    assert (serial / "metadata" / "split_manifest.csv").read_text(encoding="utf-8") == (
        parallel / "metadata" / "split_manifest.csv"
    ).read_text(encoding="utf-8")


def test_pipeline_refuses_uncalibrated_mismatch(fake_raw, tmp_path):
    """A modality at a different resolution is excluded instead of resized."""
    bad = fake_raw / "thermal" / "000.png"
    cv2.imwrite(str(bad), np.full((16, 24), 120, np.uint8))
    dataset = convert(_FakeAdapter(fake_raw).plan(), tmp_path / "out2", repo=tmp_path, previews=0)
    exclusions = (dataset / "metadata" / "exclusions.csv").read_text(encoding="utf-8")
    assert "refusing to resize" in exclusions


def test_projection_keeps_nearest_return_per_pixel():
    """Two returns landing on one pixel must keep the nearer depth."""
    calibration = LidarCalibration(
        intrinsic=np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]]),
        lidar_to_rgb=np.eye(4),
        width=10,
        height=10,
    )
    points = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 2.0]], np.float32)
    depth, stats = project_lidar_to_depth(points, calibration)
    assert depth.shape == (10, 10)
    assert depth[5, 5] == pytest.approx(2.0)
    assert stats["points_in_image"] == 2
    assert stats["coverage"] == pytest.approx(0.01)
    assert stats["depth_min_m"] == pytest.approx(2.0)


def test_projection_drops_points_outside_window_or_image():
    """Behind-camera and out-of-frame returns leave the depth map empty."""
    calibration = LidarCalibration(np.eye(3), np.eye(4), width=8, height=8)
    points = np.array([[1.0, 1.0, -2.0], [50.0, 50.0, 3.0]], np.float32)
    depth, stats = project_lidar_to_depth(points, calibration, min_depth_m=0.5, max_depth_m=10.0)
    assert not depth.any()
    assert stats["points_in_front"] == 1
    assert stats["points_in_image"] == 0


def test_depth_to_uint16_mm_keeps_invalid_zero_and_saturates():
    """0 m stays "no return"; values above the cut-off saturate instead of wrapping."""
    depth = np.array([[0.0, 1.234, 60.0, 50.0]], np.float32)
    encoded, stats = depth_to_uint16_mm(depth, max_mm=50000.0)
    assert encoded.dtype == np.uint16
    assert encoded.tolist() == [[0, 1234, 50000, 50000]]
    assert stats["coverage"] == pytest.approx(0.75)
    assert stats["saturated_fraction_of_valid"] == pytest.approx(1 / 3)


def test_calibration_and_sequence_mapping_parsing(tmp_path):
    """The shipped calibration/mapping formats are parsed without JSON hacks."""
    calibration_path = tmp_path / "calibration_params.json"
    calibration_path.write_text(
        """
        {
          "rgb_camera": {
            "width": 640, "height": 480,
            "intrinsic_matrix": [[100, 0, 320], [0, 100, 240], [0, 0, 1]],
            "distortion_coefficients": [0.1, -0.2, 0.0, 0.0, 0.0]
          },
          "lidar_to_rgb": {"extrinsic_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
        }
        """,
        encoding="utf-8",
    )
    calibration = load_calibration(calibration_path)
    assert (calibration.width, calibration.height) == (640, 480)
    assert calibration.distortion == pytest.approx((0.1, -0.2, 0.0, 0.0, 0.0))
    assert len(calibration.fingerprint()) == 64

    mapping_path = tmp_path / "rgb_to_lidar_sequence_mapping.csv"
    mapping_path.write_text(
        'rgbt_seq_id,lidar_seq_id,rgb_imgs\n02,000,"[02,06,10]"\n02,001,"[14]"\n',
        encoding="utf-8",
    )
    mapping = load_sequence_mapping(mapping_path)
    assert mapping["02"][2] == ("000",)
    assert mapping["02"][6] == ("000",)
    assert mapping["02"][14] == ("001",)


def test_timestamp_matching_respects_tolerance():
    """The nearest scan inside the tolerance wins; too-far scans do not match."""
    mapping = {"02": {2: ("000", "001")}}
    timestamps = {
        "000": np.array([[0, 100.00], [1, 100.10]]),
        "001": np.array([[0, 100.02]]),
    }
    match = match_lidar_frame("02", 2, 100.015, mapping, timestamps, tolerance_s=0.1)
    assert match is not None
    assert match.lidar_sequence == "001"
    assert match.delta_s == pytest.approx(0.005)
    assert match_lidar_frame("02", 2, 100.015, mapping, timestamps, tolerance_s=0.001) is None
    assert match_lidar_frame("02", 99, 100.015, mapping, timestamps) is None


def _materialized_plan(raw: Path):
    """Plan whose depth modality is produced by the materialize hook."""
    samples = []
    for index in range(6):
        stem = f"{index:03d}"
        samples.append(
            RawSample(
                sample_id=stem,
                group=f"scene{index // 3}",
                files={"rgb": raw / "rgb" / f"{stem}.png"},
                boxes=[Box("person", (2.0, 2.0, 30.0, 30.0))],
            )
        )
    return DatasetPlan(
        key="derived",
        display_name="Derived",
        reference_modality="rgb",
        modalities=(
            ModalitySpec("rgb", 3, EncodeSpec("rgb"), color="bgr"),
            ModalitySpec("depth", 1, EncodeSpec("scaled", units="mm", lo=0.0, hi=50000.0, invalid=(0,))),
        ),
        classes={"person": 0},
        samples=samples,
    )


def test_materialize_hook_only_runs_for_real_conversion(fake_raw, tmp_path):
    """Derived depth is rendered once and merged into samples.jsonl; --dry-run skips it."""
    calls = []

    def materialize(selected, derived_dir, workers):
        calls.append(len(selected))
        for sample, split in selected:
            target = derived_dir / "depth" / split / f"{sample.sample_id}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), np.full((32, 48), 1234, np.uint16))
            sample.files["depth"] = target
            sample.modality_meta["depth"] = {
                "source_path": "scan.bin",
                "alignment_status": "calibrated_lidar_to_rgb_projection",
                "projection_status": "ok",
            }

    plan = _materialized_plan(fake_raw)
    plan.materialize = materialize
    dry = convert(plan, tmp_path / "dry", repo=tmp_path, dry_run=True)
    assert calls == []
    assert not dry.exists()

    plan = _materialized_plan(fake_raw)
    plan.materialize = materialize
    dataset = convert(plan, tmp_path / "out", repo=tmp_path, previews=0)
    assert calls == [6]
    assert len(list((dataset / "images" / "depth").rglob("*.png"))) == 6

    import json

    first = json.loads((dataset / "metadata" / "samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
    depth = first["modalities"]["depth"]
    # The true sensor file wins over the intermediate PNG, which is kept in encoded_from.
    assert depth["source_path"] == "scan.bin"
    assert depth["encoded_from"].endswith(".png")
    assert depth["alignment_status"] == "calibrated_lidar_to_rgb_projection"
    assert depth["projection_status"] == "ok"
