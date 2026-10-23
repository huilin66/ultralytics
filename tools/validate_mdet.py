# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Validate a generated ``<name>_mdet`` dataset with the real multi-modal loader.

Examples:
    python tools/validate_mdet.py --data Y:/dataset/MID-3K_mdet/data.yaml
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    """Run structural checks plus a real ``MultiModalDataset`` load."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="path to the generated data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--checks", type=int, default=4, help="samples to materialise per split")
    args = parser.parse_args(argv)

    from ultralytics.data.multimodal import build_multimodal_dataset
    from ultralytics.utils import DEFAULT_CFG, YAML

    yaml_path = Path(args.data)
    if not yaml_path.is_file():
        print(f"[FAIL] missing yaml: {yaml_path}")
        return 1
    data = YAML.load(str(yaml_path))
    root = Path(data["path"]) if data.get("path") else yaml_path.parent

    problems: list[str] = []
    print(f"[dataset] {root}")
    print(f"[channels] declared {data.get('channels')} / modalities {[m['name'] for m in data.get('modalities', [])]}")
    print(f"[classes] {data.get('names')}")

    cfg = deepcopy(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.task = data.get("task", "detect")
    cfg.classes = None
    cfg.rect = False
    cfg.cache = None
    cfg.single_cls = False
    cfg.fraction = 1.0
    cfg.mask_ratio = 4

    for split in ("train", "val", "test"):
        relative = data.get(split)
        if not relative:
            continue
        image_dir = root / relative
        if not image_dir.is_dir():
            problems.append(f"{split}: image dir missing: {image_dir}")
            continue
        try:
            dataset = build_multimodal_dataset(
                cfg, str(image_dir), 4, data, mode="val", rect=False, stride=32
            )
        except Exception as error:  # noqa: BLE001 - surface the loader's own message
            problems.append(f"{split}: loader failed: {error}")
            continue

        missing_labels = [p for p in dataset.label_files if not Path(p).is_file()]
        if missing_labels:
            problems.append(f"{split}: {len(missing_labels)} images have no label file, e.g. {missing_labels[0]}")

        for index in range(min(args.checks, len(dataset))):
            sample = dataset[index]
            channels, height, width = sample["img"].shape
            if channels != data.get("channels"):
                problems.append(f"{split}[{index}]: fused tensor has {channels} channels, yaml says {data.get('channels')}")
                break
            if str(sample["img"].dtype) != "torch.uint8":
                problems.append(f"{split}[{index}]: fused tensor dtype {sample['img'].dtype}, expected uint8")
                break
            boxes = sample["bboxes"]
            if boxes.numel() and (boxes.min() < 0 or boxes.max() > 1):
                problems.append(f"{split}[{index}]: normalised boxes out of [0, 1]")
                break

        modality_files = dataset.modalities.paths(dataset.im_files[0])
        print(f"[{split}] {len(dataset)} samples, {len(modality_files)} modalities, "
              f"label={Path(dataset.label_files[0]).name}, first sample={tuple(sample['img'].shape)}")

    metadata = root / "metadata" / "quality_report.json"
    if metadata.is_file():
        import json

        report = json.loads(metadata.read_text(encoding="utf-8"))
        print(f"[report] excluded={report.get('exclusions')} warnings={len(report.get('warnings', []))}")

    if problems:
        print("\n[FAIL]")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\n[OK] dataset loads through the Ultralytics multi-modal dataloader")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
