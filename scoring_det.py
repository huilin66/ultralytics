import json
import os
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


IOU_THRESHOLDS = np.arange(0.40, 0.96, 0.05)

MATCH_PER_CATEGORY = True
SCORE_AWARE_MATCH  = True

# -----------------------
# Metric helpers
# -----------------------
def _f_beta(p, r, beta):
    denom = beta * beta * p + r
    return (1 + beta * beta) * p * r / denom if denom > 0 else 0.0


def _prf_from_counts(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return p, r, _f_beta(p, r, 1.0), _f_beta(p, r, 2.0)


# -----------------------
# IoU matching
# -----------------------
def iou_match(ious, thresholds):
    """Greedy IoU matching for multiple thresholds.
    
    Args:
        ious: (N_pred, N_gt) matrix of IoU values
        thresholds: list of IoU thresholds
    Returns:
        List of (tp, fp, fn) tuples, one per threshold
    """
    n_pred, n_gt = ious.shape
    
    if n_pred == 0:
        return [(0, 0, n_gt)] * len(thresholds)
    if n_gt == 0:
        return [(0, n_pred, 0)] * len(thresholds)

    results = []
    for thr in thresholds:
        matched_gt = np.zeros(n_gt, dtype=bool)
        tp = fp = 0
        
        for i in range(n_pred):
            row = ious[i]
            eligible_mask = (row >= thr) & (~matched_gt)
            
            if eligible_mask.any():
                best_idx = np.argmax(np.where(eligible_mask, row, -1.0))
                tp += 1
                matched_gt[best_idx] = True
            else:
                fp += 1
                
        fn = int((~matched_gt).sum())
        results.append((tp, fp, fn))
        
    return results


# -----------------------
# BBOX evaluation
# -----------------------
def _bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)

    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _eval_bbox(gt, pr, thresholds):
    n_gt = len(gt)
    n_pr = len(pr)
    
    if n_gt == 0 and n_pr == 0:
        return [(0,0,0)] * len(thresholds)
    if n_gt == 0:
        return [(0, n_pr, 0)] * len(thresholds)
    if n_pr == 0:
        return [(0, 0, n_gt)] * len(thresholds)

    ious = np.zeros((n_pr, n_gt), dtype=np.float32)
    for i, p in enumerate(pr):
        for j, g in enumerate(gt):
            ious[i, j] = _bbox_iou(p["bbox"], g["bbox"])

    return iou_match(ious, thresholds)


def evaluate_bbox(gt_coco, pred_coco, thresholds):
    # Initialize separate counters for each threshold
    # counts = [ [TP, FP, FN], [TP, FP, FN], ... ]
    
    # Ensure thresholds is a list
    if isinstance(thresholds, (float, int)):
        thresholds = [thresholds]
        
    counts = [[0, 0, 0] for _ in thresholds]

    for img_id in gt_coco.getImgIds():
        gt_anns = [a for a in gt_coco.loadAnns(gt_coco.getAnnIds(imgIds=img_id)) if "bbox" in a]
        pr_anns = [a for a in pred_coco.loadAnns(pred_coco.getAnnIds(imgIds=img_id)) if "bbox" in a]

        if SCORE_AWARE_MATCH:
            pr_anns.sort(key=lambda a: a.get("score", 0), reverse=True)

        if MATCH_PER_CATEGORY:
            gt_by_cat = defaultdict(list)
            pr_by_cat = defaultdict(list)
            for a in gt_anns: gt_by_cat[a["category_id"]].append(a)
            for a in pr_anns: pr_by_cat[a["category_id"]].append(a)

            for c in set(gt_by_cat) | set(pr_by_cat):
                # Returns list of (tp, fp, fn) per threshold
                res_list = _eval_bbox(gt_by_cat[c], pr_by_cat[c], thresholds)
                for idx, (tp, fp, fn) in enumerate(res_list):
                    counts[idx][0] += tp
                    counts[idx][1] += fp
                    counts[idx][2] += fn
        else:
            res_list = _eval_bbox(gt_anns, pr_anns, thresholds)
            for idx, (tp, fp, fn) in enumerate(res_list):
                counts[idx][0] += tp
                counts[idx][1] += fp
                counts[idx][2] += fn

    # Calculate PRF for each threshold
    out_metrics = []
    for (TP, FP, FN) in counts:
        out_metrics.append(_prf_from_counts(TP, FP, FN) + ((TP, FP, FN),))
        
    return out_metrics


# -----------------------
# SEG evaluation
# -----------------------
def _eval_seg(gt_coco, gt, pr, thresholds):
    n_gt = len(gt)
    n_pr = len(pr)
    
    if n_gt == 0 and n_pr == 0:
        return [(0,0,0)] * len(thresholds)
    if n_gt == 0:
        return [(0, n_pr, 0)] * len(thresholds)
    if n_pr == 0:
        return [(0, 0, n_gt)] * len(thresholds)

    gt_rle = [gt_coco.annToRLE(a) for a in gt]
    pr_rle = [gt_coco.annToRLE(a) for a in pr]

    # Compute IoU matrix ONCE
    ious = maskUtils.iou(pr_rle, gt_rle, np.zeros(len(gt_rle)))
    
    return iou_match(ious, thresholds)


def evaluate_seg(gt_coco, pred_coco, thresholds):
    # Ensure thresholds is a list
    if isinstance(thresholds, (float, int)):
        thresholds = [thresholds]

    counts = [[0, 0, 0] for _ in thresholds]

    for img_id in gt_coco.getImgIds():
        gt_anns = [a for a in gt_coco.loadAnns(gt_coco.getAnnIds(imgIds=img_id)) if "segmentation" in a]
        pr_anns = [a for a in pred_coco.loadAnns(pred_coco.getAnnIds(imgIds=img_id)) if "segmentation" in a]

        if SCORE_AWARE_MATCH:
            pr_anns.sort(key=lambda a: a.get("score", 0), reverse=True)

        if MATCH_PER_CATEGORY:
            gt_by_cat = defaultdict(list)
            pr_by_cat = defaultdict(list)
            for a in gt_anns: gt_by_cat[a["category_id"]].append(a)
            for a in pr_anns: pr_by_cat[a["category_id"]].append(a)

            for c in set(gt_by_cat) | set(pr_by_cat):
                # Returns list of (tp, fp, fn) per threshold
                res_list = _eval_seg(gt_coco, gt_by_cat[c], pr_by_cat[c], thresholds)
                for idx, (tp, fp, fn) in enumerate(res_list):
                    counts[idx][0] += tp
                    counts[idx][1] += fp
                    counts[idx][2] += fn
        else:
            res_list = _eval_seg(gt_coco, gt_anns, pr_anns, thresholds)
            for idx, (tp, fp, fn) in enumerate(res_list):
                counts[idx][0] += tp
                counts[idx][1] += fp
                counts[idx][2] += fn

    # Calculate PRF for each threshold
    out_metrics = []
    for (TP, FP, FN) in counts:
        out_metrics.append(_prf_from_counts(TP, FP, FN) + ((TP, FP, FN),))
        
    return out_metrics


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    pred_name = 'a100_yolov8x-seg'
    # Codabench-specific paths
    ### MODIFY THESE FILES TO MAKE IT WORK LOCALLY ###
    ground_truth_json = r"data/huilin/scrinvme/huilin/bdd/cp_data/rip_seg/val_no_annotations.json"
    input_path = rf"data/huilin/localnvme/project/ultralytics_main/submission/{pred_name}"
    output_dir = input_path
    
    predictions_both = os.path.join(input_path, "predictions_both.json")
    predictions_det = os.path.join(input_path, "predictions_det.json")
    PREDICTIONS_JSON = predictions_both if os.path.exists(predictions_both) else predictions_det
    ### UP UNTIL HERE ####
    # Handle missing predictions file
    if not os.path.exists(PREDICTIONS_JSON):
        print(f"ERROR: No predictions file found. Expected: {predictions_both} or {predictions_det}")
        results = {
            "F1[50]": 0.0,
            "F2[50]": 0.0,
            "F1[40:95]": 0.0,
            "F2[40:95]": 0.0,
            "Score": 0.0
        }
        print(results)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "scores.json"), "w") as f:
            json.dump(results, f, indent=2)
        exit(0)

    gt_coco = COCO(ground_truth_json)
    with open(PREDICTIONS_JSON) as f:
        preds = json.load(f)
    pred_coco = gt_coco.loadRes(preds)

    # Evaluate once with all thresholds (0.50 is at index 2 in IOU_THRESHOLDS)
    # IOU_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, ...]
    all_thresholds = list(IOU_THRESHOLDS)
    idx_50 = 2  # 0.50 is at index 2 (0.40, 0.45, 0.50...)
    
    all_results = evaluate_bbox(gt_coco, pred_coco, all_thresholds)
    
    # Extract IoU=0.50 metrics
    p50, r50, f1_50, f2_50, counts50 = all_results[idx_50]
    
    # Compute averages over all thresholds
    f1s = [res[2] for res in all_results]
    f2s = [res[3] for res in all_results]
    f1_4095 = float(np.mean(f1s))
    f2_4095 = float(np.mean(f2s))

    # Composite score
    score = 0.25 * f1_50 + 0.25 * f2_50 + 0.25 * f1_4095 + 0.25 * f2_4095

    results = {
        "F1[50]": float(f1_50),
        "F2[50]": float(f2_50),
        "F1[40:95]": f1_4095,
        "F2[40:95]": f2_4095,
        "Score": float(score)
    }

    print(results)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "scores.json"), "w") as f:
        json.dump(results, f, indent=2)
