"""Inspect decoded labels before long multi-label training runs."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.data.utils import check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

from multilabel_yolo.dataset import MultiLabelYOLODataset


def main():
    """Print physical boxes, transport IDs, decoded labels, and n-hot vectors."""
    parser = argparse.ArgumentParser(description="Inspect a multi-label YOLO batch")
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    data = check_det_dataset(args.data)
    split = data.get(args.split) or data.get("val") or data.get("test")
    cfg = get_cfg(
        DEFAULT_CFG,
        overrides={
            "imgsz": args.imgsz,
            "task": "detect",
            "rect": True,
            "cache": False,
            "single_cls": False,
            "classes": None,
            "fraction": 1.0,
        },
    )
    dataset = MultiLabelYOLODataset(
        img_path=split,
        imgsz=args.imgsz,
        batch_size=1,
        augment=False,
        hyp=cfg,
        rect=True,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.5,
        prefix="inspect: ",
        task="detect",
        classes=None,
        data=data,
    )

    index = max(0, min(args.index, len(dataset) - 1))
    raw = dataset.labels[index]
    sample = dataset[index]
    print(f"image={Path(sample['im_file']).name}")
    for row, (transport_id, bbox, nhot) in enumerate(
        zip(sample["cls"].reshape(-1).tolist(), sample["bboxes"].tolist(), sample["cls_nhot"].tolist())
    ):
        print(
            f"  object={row} bbox={bbox} transport_id={int(transport_id)} "
            f"labels={dataset.codec.decode(int(transport_id))} nhot={nhot}"
        )
    print(f"physical_boxes={len(sample['bboxes'])} raw_cache_rows={len(raw['bboxes'])} nc={dataset.nc}")


if __name__ == "__main__":
    main()
