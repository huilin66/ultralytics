# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Evaluate decision-level (late) fusion of two independently trained detectors.

Late fusion is the control that the feature-level graph has to beat: each detector sees one modality, produces its
own boxes, and the boxes are combined after the fact. Nothing is shared between the two networks, so the comparison
isolates the value of fusing at P3/P4/P5 rather than at the detection output.

Each seed pairs the RGB-only checkpoint with the thermal-only checkpoint of the same seed, exactly as the paper's
cells do. The evaluation runs inside the repository's own validator (``MultiModalDetectionValidator``), so the ground
truth handling, letterbox round-trip and AP computation are the ones behind every number in the paper. The RGB-only
row is reported first as a sanity check: it must reproduce the published single-modality score.

The two boxes are combined two ways: a score-ordered NMS over the union, and weighted box fusion (WBF), which
averages the coordinates of matched boxes in proportion to their confidence and divides the fused confidence by the
number of detectors. Both are untuned.

Usage::

    python tools/late_fusion_eval.py --project runs/e2_rlivit --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.multimodal.val import MultiModalDetectionValidator
from ultralytics.models.yolo.detect import DetectionValidator

VARIANTS = ("rgb", "thermal", "nms", "wbf")


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=str(REPO / "runs" / "e2_rlivit"), help="directory holding the cells")
    parser.add_argument("--rgb", default="b0-rgb", help="RGB-only cell name")
    parser.add_argument("--thermal", default="b1-thermal", help="thermal-only cell name")
    parser.add_argument("--data-root", default="/srv/nas/home/dataset/R-LiViT_mdet")
    parser.add_argument("--data-rgb", default="data-rgb.yaml", help="three-channel view the RGB detector was trained on")
    parser.add_argument("--data-thermal", default="data-thermal.yaml", help="one-channel view the thermal detector was trained on")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001, help="per-detector confidence floor before fusion")
    parser.add_argument("--iou", type=float, default=0.7, help="per-detector NMS IoU")
    parser.add_argument("--fusion-iou", type=float, default=0.55, help="IoU above which two boxes are one object")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--out", default=None, help="result JSON; defaults to <project>/late_fusion.json")
    return parser.parse_args(argv)


def build_validator(data_yaml: Path, args: argparse.Namespace, model, multimodal: bool):
    """Build a validator without running it, so predictions can be fed in from outside."""
    cfg = get_cfg(
        overrides={
            "data": str(data_yaml),
            "split": "test",
            "imgsz": args.imgsz,
            "batch": args.batch,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "task": "detect",
            "mode": "val",
            "device": args.device,
            "workers": args.workers,
            "plots": False,
            "save_json": False,
            "save_txt": False,
            "single_cls": False,
            "agnostic_nms": False,
            "augment": False,
            "rect": True,
            "dnn": False,
            "compile": False,
        }
    )
    validator_class = MultiModalDetectionValidator if multimodal else DetectionValidator
    validator = validator_class(args=cfg)
    validator.args.val = True
    validator.args.training = False
    validator.data = check_det_dataset(str(data_yaml))
    validator.device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    validator.stride = 32
    validator.training = False
    validator.save_dir = Path(".")
    validator.on_plot = lambda *_, **__: None
    validator.init_metrics(model)
    return validator


def load_detector(path: Path, device: torch.device) -> torch.nn.Module:
    """Load the raw detection module of a checkpoint, bypassing the Model facade."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint["model"].float().to(device).eval()


def collect_predictions(val_rgb, val_thermal, model_rgb, model_thermal, args) -> list[tuple[dict, list, list]]:
    """Run each detector through its own dataset view and keep the per-image boxes plus the ground-truth metadata.

    Each detector is evaluated exactly as it was trained: the RGB model through the three-channel view and the
    thermal model through the one-channel view. Both views letterbox the same frames to the same geometry, which is
    checked per batch, so their boxes live in one coordinate frame and can be merged directly.
    """
    loader_rgb = val_rgb.get_dataloader(val_rgb.data["test"], args.batch)
    loader_thermal = val_thermal.get_dataloader(val_thermal.data["test"], args.batch)
    records = []
    with torch.no_grad():
        for batch_rgb, batch_thermal in zip(loader_rgb, loader_thermal):
            names_rgb = [Path(p).name for p in batch_rgb["im_file"]]
            names_thermal = [Path(p).name for p in batch_thermal["im_file"]]
            if names_rgb != names_thermal or batch_rgb["ratio_pad"] != batch_thermal["ratio_pad"]:
                raise SystemExit("the two modality views disagree on the test frames or their letterbox geometry")
            img_rgb = batch_rgb["img"].to(val_rgb.device, non_blocking=True).float() / 255
            img_thermal = batch_thermal["img"].to(val_thermal.device, non_blocking=True).float() / 255
            rgb_pred = _to_cpu(val_rgb.postprocess(model_rgb(img_rgb)))
            thermal_pred = _to_cpu(val_thermal.postprocess(model_thermal(img_thermal)))
            meta = {k: v for k, v in batch_rgb.items() if k != "img"}
            meta["img"] = torch.empty((1, 1, *img_rgb.shape[2:]))
            records.append((meta, rgb_pred, thermal_pred))
    return records


def _to_cpu(preds: list[dict]) -> list[dict]:
    """Move one batch of predictions off the GPU; scoring runs on the CPU side of the validator."""
    return [{key: value.cpu() for key, value in pred.items()} for pred in preds]


def _as_numpy(pred: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split one post-NMS prediction into boxes, scores and classes as numpy arrays."""
    return (
        pred["bboxes"].float().cpu().numpy(),
        pred["conf"].float().cpu().numpy(),
        pred["cls"].float().cpu().numpy(),
    )


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box of ``a`` and every box of ``b``."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)[:, None]
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)[None, :]
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def merge_nms(preds: list[dict], iou: float) -> dict:
    """Concatenate the two detectors' boxes and keep the highest-scoring box of every overlapping pair."""
    boxes = np.concatenate([_as_numpy(p)[0] for p in preds])
    scores = np.concatenate([_as_numpy(p)[1] for p in preds])
    classes = np.concatenate([_as_numpy(p)[2] for p in preds])
    keep: list[int] = []
    for cls in np.unique(classes):
        index = np.flatnonzero(classes == cls)
        order = index[np.argsort(-scores[index])]
        suppressed = np.zeros(len(boxes), dtype=bool)
        for i in order:
            if suppressed[i]:
                continue
            keep.append(int(i))
            overlap = _iou_matrix(boxes[[i]], boxes[index])[0] > iou
            suppressed[index[overlap]] = True
    return _to_pred(boxes[keep], scores[keep], classes[keep])


def merge_wbf(preds: list[dict], iou: float) -> dict:
    """Weighted box fusion: cluster boxes of one class by IoU, average coordinates by confidence, average the score."""
    boxes = np.concatenate([_as_numpy(p)[0] for p in preds]).astype(np.float64)
    scores = np.concatenate([_as_numpy(p)[1] for p in preds]).astype(np.float64)
    classes = np.concatenate([_as_numpy(p)[2] for p in preds])
    clusters: list[dict] = []
    for i in np.argsort(-scores):
        best, best_iou = None, iou
        for cluster in clusters:
            if cluster["cls"] != classes[i]:
                continue
            overlap = _iou_matrix(boxes[[i]], cluster["box"][None, :])[0, 0]
            if overlap > best_iou:
                best, best_iou = cluster, overlap
        if best is None:
            clusters.append({"box": boxes[i].copy(), "conf": scores[i], "cls": classes[i], "n": 1})
        else:
            total = best["conf"] + scores[i]
            best["box"] = (best["box"] * best["conf"] + boxes[i] * scores[i]) / total
            best["conf"], best["n"] = total, best["n"] + 1
    if not clusters:
        return _to_pred(np.zeros((0, 4)), np.zeros(0), np.zeros(0))
    fused = np.stack([c["box"] for c in clusters])
    # The fused score is the mean over the two detectors, so a box found by one detector alone is halved.
    confs = np.array([c["conf"] / len(preds) for c in clusters]).clip(0, 1)
    fused_cls = np.array([c["cls"] for c in clusters])
    return _to_pred(fused.astype(np.float32), confs, fused_cls)


def _to_pred(boxes: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> dict:
    """Wrap numpy boxes back into the validator's prediction dict."""
    return {
        "bboxes": torch.as_tensor(boxes, dtype=torch.float32),
        "conf": torch.as_tensor(conf, dtype=torch.float32),
        "cls": torch.as_tensor(cls, dtype=torch.float32),
        "extra": torch.zeros((len(boxes), 0)),
    }


def merge_all(records, variant: str, fusion_iou: float, nms_iou: float) -> list[list[dict]]:
    """Produce the per-image predictions of one batch for the requested fusion variant."""
    merged = []
    for _, rgb_preds, thermal_preds in records:
        batch = []
        for rgb, thermal in zip(rgb_preds, thermal_preds):
            if variant == "rgb":
                batch.append(rgb)
            elif variant == "thermal":
                batch.append(thermal)
            else:
                pair = [rgb, thermal]
                batch.append(merge_nms(pair, nms_iou) if variant == "nms" else merge_wbf(pair, fusion_iou))
        merged.append(batch)
    return merged


def score(validator, model, records, variant: str, args) -> dict:
    """Run one variant through the repository's metric accumulation and return its results."""
    validator.device = torch.device("cpu")
    validator.init_metrics(model)
    for (meta, _, _), preds in zip(records, merge_all(records, variant, args.fusion_iou, args.iou)):
        validator.update_metrics(preds, meta)
    return validator.get_stats()


def main(argv=None) -> int:
    """Evaluate the late-fusion variants for every requested seed."""
    args = parse_args(argv)
    data_rgb = Path(args.data_root) / args.data_rgb
    data_thermal = Path(args.data_root) / args.data_thermal
    project = Path(args.project)
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)

    rows = {}
    for seed in args.seeds:
        rgb_ckpt = project / f"{args.rgb}-seed{seed}" / "weights" / "best.pt"
        thermal_ckpt = project / f"{args.thermal}-seed{seed}" / "weights" / "best.pt"
        missing = [str(p) for p in (rgb_ckpt, thermal_ckpt) if not p.is_file()]
        if missing:
            raise SystemExit(f"missing checkpoints: {missing}")
        model_rgb = load_detector(rgb_ckpt, device)
        model_thermal = load_detector(thermal_ckpt, device)
        validator = build_validator(data_rgb, args, model_rgb, multimodal=False)
        val_thermal = build_validator(data_thermal, args, model_thermal, multimodal=True)
        records = collect_predictions(validator, val_thermal, model_rgb, model_thermal, args)

        seed_rows = {}
        for variant in VARIANTS:
            results = score(validator, model_rgb, records, variant, args)
            seed_rows[variant] = {
                "mAP50-95": round(float(results["metrics/mAP50-95(B)"]), 4),
                "mAP50": round(float(results["metrics/mAP50(B)"]), 4),
                "precision": round(float(results["metrics/precision(B)"]), 4),
                "recall": round(float(results["metrics/recall(B)"]), 4),
            }
            print(f"[seed{seed}] {variant:<8} mAP50-95={seed_rows[variant]['mAP50-95']:.4f}", flush=True)
        rows[seed] = seed_rows
        del model_rgb, model_thermal, records

    summary = {}
    for variant in VARIANTS:
        values = np.array([rows[seed][variant]["mAP50-95"] for seed in args.seeds])
        summary[variant] = {"mean": round(float(values.mean()), 4), "std": round(float(values.std(ddof=1)), 4)}

    out = Path(args.out) if args.out else project / "late_fusion.json"
    out.write_text(json.dumps({"rows": rows, "summary": summary, "fusion_iou": args.fusion_iou}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
