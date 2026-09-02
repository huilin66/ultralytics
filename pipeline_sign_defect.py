# -*- coding: utf-8 -*-
"""
Two-stage traffic-sign defect pipeline.

Stage 1: tf_det_1-[yolov9e] detects sign boards on large highway images
         (232 TT100K traffic-sign classes).
Stage 2: each sign board is cropped together with a PAD_RATIO margin around it,
         then tf_defect_3-[yolov8x] detects the 9 defect classes on the crops.

Outputs (under OUT_DIR):
  crops/       cropped sign regions  (original aspect ratio, original resolution)
  crops_viz/   same crops with defect boxes drawn (in crop coordinate frame)
  labels/      yolo-format defect labels per crop (normalized to the crop, conf appended)
  full_viz/    original image with sign box (green) + remapped defect boxes (red)
  summary.csv  one row per crop: sign info + defect list (boxes in both crop & full-image frame)

Usage:
    python pipeline_sign_defect.py                     # all images, default config
    python pipeline_sign_defect.py --limit 10          # first 10 images (smoke test)
    python pipeline_sign_defect.py --img_dir <path> --out_dir <path> --device 0
"""

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ---------------------------------------------------------------- config
DETECT_WEIGHT = "runs/detect/tf_det_1-[yolov9e]/weights/best.pt"
DEFECT_WEIGHT = "runs/detect/tf_defect_3-[yolov8x]/weights/best.pt"

IMG_DIR = r"/scrinvme/huilin/traffic_sign/highway/road sign defect/road sign defect"
OUT_DIR = "runs/two_stage_sign_defect"

DETECT_IMGSZ = 640
DETECT_CONF = 0.25
DEFECT_IMGSZ = 640
DEFECT_CONF = 0.25
PAD_RATIO = 0.20            # margin added to every side of the sign box (of its own size)
MAX_DET = 300
IMG_SUFFIX = (".jpg", ".jpeg", ".png", ".bmp")


def letterbox(img, size=640, color=(114, 114, 114)):
    """Resize keeping aspect ratio, center-pad to `size`x`size`.

    Returns (padded, scale, top, left) such that a point p in the original
    image maps to ((p * scale) + (left, top)) in the padded frame.
    """
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), color, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out, scale, top, left


def expand_box(x1, y1, x2, y2, W, H, ratio):
    """Grow the box by `ratio` of its own width/height on every side, clamp to image."""
    dx, dy = (x2 - x1) * ratio, (y2 - y1) * ratio
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(W, x2 + dx)
    y2 = min(H, y2 + dy)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return int(x1), int(y1), int(x2), int(y2)


def draw_boxes(img, boxes, color, labels=None):
    """Mutate a copy of `img` with (x1, y1, x2, y2) boxes (int) drawn."""
    out = img.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)  # OpenCV requires integer coords
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if labels is not None:
            cv2.putText(out, labels[i], (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def parse_args():
    """Command-line overrides (defaults come from the config block above)."""
    global IMG_DIR, OUT_DIR, DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img_dir", default=IMG_DIR, help="input highway image dir")
    parser.add_argument("--out_dir", default=OUT_DIR, help="output root dir")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="only process first N images (0 = all)")
    parser.add_argument("--detect_conf", type=float, default=DETECT_CONF)
    parser.add_argument("--defect_conf", type=float, default=DEFECT_CONF)
    parser.add_argument("--pad", type=float, default=PAD_RATIO, help="sign-box expansion ratio")
    args = parser.parse_args()
    IMG_DIR, OUT_DIR, DEVICE = args.img_dir, args.out_dir, args.device
    return args


def main():
    args = parse_args()
    img_dir, out_dir = Path(IMG_DIR), Path(OUT_DIR)
    if not img_dir.is_dir():
        raise SystemExit(f"[ERROR] input dir not found: {img_dir}")

    for sub in ("crops", "crops_viz", "labels", "full_viz"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    detect_model = YOLO(DETECT_WEIGHT)
    defect_model = YOLO(DEFECT_WEIGHT)
    detect_names, defect_names = detect_model.names, defect_model.names

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_SUFFIX)
    if args.limit > 0:
        images = images[: args.limit]
    print(f"[INFO] {len(images)} images | device={DEVICE} | "
          f"detect_conf={args.detect_conf} defect_conf={args.defect_conf} pad={args.pad}")

    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "src_image", "sign_idx", "sign_class", "sign_conf",
            "sign_box", "crop_path", "crop_w", "crop_h",
            "n_defects", "defects_max_conf", "defects",
        ])
        writer.writeheader()

        n_sign_total, n_defect_total, n_img_sign = 0, 0, 0
        for it, img_path in enumerate(images):
            img = cv2.imread(str(img_path))  # BGR
            if img is None:
                print(f"[WARN] cannot read {img_path.name}")
                continue
            H, W = img.shape[:2]

            det = detect_model.predict(img, imgsz=DETECT_IMGSZ, conf=args.detect_conf,
                                       device=DEVICE, max_det=MAX_DET, verbose=False)[0]
            boxes = det.boxes
            if boxes is None or len(boxes) == 0:
                print(f"[    ] {img_path.name}: no sign")
                continue
            n_img_sign += 1

            # ---- stage 1 -> crops (letterboxed, aspect preserved) ----
            crops, metas = [], []
            coords = boxes.xyxy.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for i, (x1, y1, x2, y2) in enumerate(coords):
                crop_rect = expand_box(int(x1), int(y1), int(x2), int(y2), W, H, args.pad)
                if crop_rect is None:
                    continue
                cx1, cy1, cx2, cy2 = crop_rect
                crop = img[cy1:cy2, cx1:cx2]
                padded, scale, top, left = letterbox(crop, DEFECT_IMGSZ)
                crops.append(padded)
                metas.append({
                    "cx1": cx1, "cy1": cy1, "cx2": cx2, "cy2": cy2,
                    "sign_cls": int(clss[i]), "sign_name": detect_names[int(clss[i])],
                    "sign_conf": float(confs[i]), "scale": scale, "top": top, "left": left,
                })

            # ---- stage 2: defect inference on all crops of this image ----
            dres = defect_model.predict(crops, imgsz=DEFECT_IMGSZ, conf=args.defect_conf,
                                        device=DEVICE, verbose=False)

            for idx, (meta, res) in enumerate(zip(metas, dres)):
                cx1, cy1, cx2, cy2 = meta["cx1"], meta["cy1"], meta["cx2"], meta["cy2"]
                scale, top, left = meta["scale"], meta["top"], meta["left"]
                crop = img[cy1:cy2, cx1:cx2]  # original (BGR)
                cw, ch = cx2 - cx1, cy2 - cy1

                defects = []
                db = res.boxes
                if db is not None and len(db):
                    for cls, conf, (xr1, yr1, xr2, yr2) in zip(
                            db.cls.cpu().numpy(), db.conf.cpu().numpy(), db.xyxy.cpu().numpy()):
                        # letterboxed-640 frame -> original crop frame -> full-image frame
                        ox1, oy1 = (xr1 - left) / scale, (yr1 - top) / scale
                        ox2, oy2 = (xr2 - left) / scale, (yr2 - top) / scale
                        defects.append({
                            "class": defect_names[int(cls)],
                            "cls_id": int(cls),
                            "conf": round(float(conf), 3),
                            "box_crop": [round(float(ox1), 1), round(float(oy1), 1),
                                         round(float(ox2), 1), round(float(oy2), 1)],
                            "box_full": [int(cx1 + ox1), int(cy1 + oy1), int(cx1 + ox2), int(cy1 + oy2)],
                        })

                # ---- save outputs per crop ----
                crop_name = f"{img_path.stem}_s{idx:02d}_{meta['sign_name']}.jpg"
                txt_name = os.path.splitext(crop_name)[0] + ".txt"
                cv2.imwrite(str(out_dir / "crops" / crop_name), crop)

                with open(out_dir / "labels" / txt_name, "w") as f:
                    for d in defects:
                        bx1, by1, bx2, by2 = d["box_crop"]
                        f.write(f"{d['cls_id']} {(bx1 + bx2) / 2 / cw:.6f} {(by1 + by2) / 2 / ch:.6f} "
                                f"{(bx2 - bx1) / cw:.6f} {(by2 - by1) / ch:.6f} {d['conf']:.6f}\n")

                draw_boxes(crop, [d["box_crop"] for d in defects], (0, 0, 255),
                           labels=[f"{d['class']} {d['conf']:.2f}" for d in defects])
                cv2.imwrite(str(out_dir / "crops_viz" / crop_name), crop)

                max_conf = max((d["conf"] for d in defects), default=0.0)
                writer.writerow({
                    "src_image": img_path.name,
                    "sign_idx": idx,
                    "sign_class": meta["sign_name"],
                    "sign_conf": meta["sign_conf"],
                    "sign_box": [int(cx1), int(cy1), int(cx2), int(cy2)],
                    "crop_path": f"crops/{crop_name}",
                    "crop_w": cw, "crop_h": ch,
                    "n_defects": len(defects),
                    "defects_max_conf": max_conf,
                    "defects": json.dumps(defects, ensure_ascii=False),
                })
                n_sign_total += 1
                n_defect_total += len(defects)

            # ---- full-image visualization with remapped defect boxes ----
            full = img.copy()
            sign_boxes, sign_labels = [], []
            for meta in metas:
                x1, y1, x2, y2 = meta["cx1"], meta["cy1"], meta["cx2"], meta["cy2"]
                sign_boxes.append((x1, y1, x2, y2))
                sign_labels.append(f"{meta['sign_name']} {meta['sign_conf']:.2f}")
            full = draw_boxes(full, sign_boxes, (0, 255, 0), sign_labels)
            defect_boxes = []
            for meta, res in zip(metas, dres):
                db = res.boxes
                if db is None:
                    continue
                cx1, cy1 = meta["cx1"], meta["cy1"]
                scale, top, left = meta["scale"], meta["top"], meta["left"]
                for (xr1, yr1, xr2, yr2) in db.xyxy.cpu().numpy():
                    fx1 = int(cx1 + (xr1 - left) / scale)
                    fy1 = int(cy1 + (yr1 - top) / scale)
                    fx2 = int(cx1 + (xr2 - left) / scale)
                    fy2 = int(cy1 + (yr2 - top) / scale)
                    defect_boxes.append((fx1, fy1, fx2, fy2))
            full = draw_boxes(full, defect_boxes, (0, 0, 255))
            cv2.imwrite(str(out_dir / "full_viz" / img_path.name), full)

            print(f"[{it + 1:>3}/{len(images)}] {img_path.name}: {len(metas)} signs, "
                  f"{n_defect_total} defects so far")

    print(f"[DONE] images_with_signs={n_img_sign} signs={n_sign_total} defects={n_defect_total}")
    print(f"[DONE] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
