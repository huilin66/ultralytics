# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Extract the zipped raw downloads into ``<raw-root>/_extracted``.

Examples:
    python tools/extract_raw.py --dataset mid3k
    python tools/extract_raw.py --dataset infraparis --only BBox Depth Infra
    python tools/extract_raw.py --dataset r_livit --with-lidar
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

SOURCES: dict[str, list[str]] = {
    "mid3k": ["MID-3K/MID-3K-rgb-main.zip", "MID-3K/MID-3K-thermal-main.zip",
              "MID-3K/MID-3K-depth-main.zip", "MID-3K/MID-3K-intensity-main.zip"],
    # Right (60 GB) and the segmentation archives are not needed for detection.
    "infraparis": ["InfraParis/Left.zip", "InfraParis/Depth.zip", "InfraParis/Infra.zip", "InfraParis/BBox.zip"],
    "r_livit": ["R-LiViT/R-LiViT_RGB-T.zip"],
    "mm5": ["MM5/MM5_ALIGNED.zip"],
    "vdt2048": ["VDT-2048/VDT-2048 dataset.zip"],
}

OPTIONAL: dict[str, list[str]] = {
    "r_livit": ["R-LiViT/R-LiViT_LiDAR.zip"],
    "infraparis": ["InfraParis/semantic_segmentation_truth_reprojected_IR.zip"],
}


def main(argv: list[str] | None = None) -> int:
    """Extract the archives for one dataset."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(SOURCES))
    parser.add_argument("--raw-root", default=r"Y:\dataset")
    parser.add_argument("--dest", default=None, help="extraction root (defaults to <raw-root>/_extracted)")
    parser.add_argument("--only", nargs="*", default=None, help="restrict to archives whose name contains one of these")
    parser.add_argument("--with-lidar", action="store_true", help="also extract optional archives for this dataset")
    parser.add_argument("--force", action="store_true", help="re-extract even when the target folder exists")
    args = parser.parse_args(argv)

    raw_root = Path(args.raw_root)
    dest = Path(args.dest) if args.dest else raw_root / "_extracted"
    dest.mkdir(parents=True, exist_ok=True)

    archives = list(SOURCES[args.dataset])
    if args.with_lidar:
        archives += OPTIONAL.get(args.dataset, [])
    if args.only:
        needles = [needle.lower() for needle in args.only]
        archives = [a for a in archives if any(n in a.lower() for n in needles)]

    for relative in archives:
        archive = raw_root / relative
        if not archive.is_file():
            print(f"[skip] missing {archive}")
            continue
        marker = dest / f".{Path(relative).stem}.complete"
        if marker.is_file() and not args.force:
            print(f"[skip] already extracted {Path(relative).stem}")
            continue
        started = time.time()
        print(f"[extract] {archive.name} -> {dest}", flush=True)
        result = subprocess.run(["tar", "-xf", str(archive), "-C", str(dest)], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[error] {archive.name}: {result.stderr.strip()}")
            continue
        marker.write_text("ok", encoding="utf-8")
        print(f"[ok] {archive.name} in {time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
