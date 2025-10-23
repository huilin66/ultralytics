import os
import cv2
import math
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from confusion_matrix_analysis import risk_analysis
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')


def get_attributes(attribute_path):
    if attribute_path is None:
        return None
    with open(attribute_path, 'r') as file:
        attribute_dict = yaml.safe_load(file)['attributes']
    attribute_keys = list(attribute_dict.keys())
    return attribute_keys

def poly2xywh(mask):
    mask = np.array([mask[::2], mask[1::2]])
    x_min,y_min = np.min(mask, axis=1)
    x_max,y_max = np.max(mask, axis=1)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    width = x_max - x_min
    height = y_max - y_min
    return [x_center, y_center, width, height]

def df_xywh_to_xyxy(df):
    df = df.copy()
    df["x1"] = df["x"] - df["w"] / 2
    df["y1"] = df["y"] - df["h"] / 2
    df["x2"] = df["x"] + df["w"] / 2
    df["y2"] = df["y"] + df["h"] / 2
    return df

def get_yolo_label_df(gt_path, mdet=False, attributes=None):
    if mdet:
        assert attributes is not None, 'attribute_path must be provided, which is "%s"' % attributes
        if isinstance(attributes, str):
            attribute_keys = get_attributes(attributes)
        elif isinstance(attributes, list):
            attribute_keys = attributes
        names = ['id', 'category'] + ['attribute_len'] + attribute_keys + ['x', 'y', 'w', 'h']
    else:
        names = ['id', 'category', 'x', 'y', 'w', 'h']

    df = pd.DataFrame(None, columns=names + ['image'])
    with open(gt_path, 'r') as f:
        data = f.readlines()
        for id_line, line in enumerate(data):
            parts = line.strip().split(' ')
            category = int(parts[0])
            image_name = Path(gt_path).stem
            if mdet:
                att_len = int(parts[1])
                atts = list(map(float, parts[2:2 + att_len]))
                polygons = list(map(float, parts[2 + att_len:]))
                if len(polygons) == 0:
                    continue
                xywh = poly2xywh(polygons)
                info = [id_line, category, att_len] + atts + xywh + [image_name]
            else:
                polygons = list(map(float, parts[1:]))
                if len(polygons) == 0:
                    continue
                xywh = poly2xywh(polygons)
                info = [id_line, category] + xywh + [image_name]
            df.loc[len(df)] = info
    df = df_xywh_to_xyxy(df)
    return df

def box_iou(box1, box2):
    # box1, box2: [x1,y1,x2,y2]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_w = max(inter_x2 - inter_x1, 0)
    inter_h = max(inter_y2 - inter_y1, 0)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area

    return inter_area / union if union > 0 else 0

def match_and_merge(df_pred, df_gt, iou_thr=0.5, att_list=None):
    matched_gt_idx = set()
    merged_rows = []

    show_columns = ['id', "category", "x", "y", "w", "h"]
    if att_list is not None:
       show_columns += att_list
    for i, pred_row in df_pred.iterrows():
        pred_box = [pred_row.x1, pred_row.y1, pred_row.x2, pred_row.y2]
        ious = [box_iou(pred_box, [gt.x1, gt.y1, gt.x2, gt.y2]) for _, gt in df_gt.iterrows()]
        if ious:
            max_iou = max(ious)
            max_idx = np.argmax(ious)
        else:
            max_iou = 0
            max_idx = None

        if max_iou > iou_thr and max_idx not in matched_gt_idx:
            gt_row = df_gt.iloc[max_idx]
            matched_gt_idx.add(max_idx)
            merged_rows.append({
                **{f"pred_{col}": pred_row[col] for col in show_columns},
                **{f"gt_{col}": gt_row[col] for col in show_columns},
                "iou": max_iou
            })
        else:
            merged_rows.append({
                **{f"pred_{col}": pred_row[col] for col in show_columns},
                **{f"gt_{col}": None for col in show_columns},
                "iou": None
            })

    # 把没被匹配的 GT 也补上
    for j, gt_row in df_gt.iterrows():
        if j not in matched_gt_idx:
            merged_rows.append({
                **{f"pred_{col}": None for col in show_columns},
                **{f"gt_{col}": gt_row[col] for col in show_columns},
                "iou": None
            })

    return pd.DataFrame(merged_rows)

def compute_macro_metrics(df, att_list, class_labels=[0, 1]):
    results = {}
    for att in att_list:
        y_true = list(map(int, df[f"gt_{att}"].to_list()))
        y_pred = list(map(int, df[f"pred_{att}"].to_list()))
        labels = class_labels
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        tp = np.diag(cm)  # 对角线
        fp = cm.sum(axis=0) - tp  # 每列总和减去 TP
        fn = cm.sum(axis=1) - tp  # 每行总和减去 TP

        # per-class precision, recall
        precision_per_class = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp+fp)>0)
        recall_per_class = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp+fn)>0)
        f1_per_class = np.divide(2 * precision_per_class * recall_per_class,
                                 precision_per_class + recall_per_class,
                                 out=np.zeros_like(precision_per_class, dtype=float),
                                 where=(precision_per_class + recall_per_class) > 0)

        macro_precision = precision_per_class.mean()
        macro_recall = recall_per_class.mean()
        macro_f1 = f1_per_class.mean()

        results.update({
            f"{att}_precision": macro_precision,
            f"{att}_recall": macro_recall,
            f"{att}_f1": macro_f1,
            f"{att}_confusion_matrix": pd.DataFrame(cm, index=[f"True_{l}" for l in labels],
                                             columns=[f"Pred_{l}" for l in labels])
        })

    return results

def compute_precision_recall(df, att_list=None):
    gt_len = len(df["pred_x"].notna())
    pred_len = len(df["gt_x"].notna())
    # 条件：有预测框、有 GT 框、类别一致
    tp = ((df["pred_x"].notna()) & (df["gt_x"].notna()) & (df["pred_category"] == df["gt_category"])).sum()
    # FP：有预测框，但 GT 不存在或类别不一致
    fp = ((df["pred_x"].notna()) & ((df["gt_x"].isna()) | (df["pred_category"] != df["gt_category"]))).sum()
    # FN：有 GT，但预测不存在或类别不一致
    fn = ((df["gt_x"].notna()) & ((df["pred_x"].isna()) | (df["pred_category"] != df["gt_category"]))).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    result = {"gt_len": gt_len, "pred_len": pred_len, "TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall}

    if att_list is not None:
        df_box_true = df[(df["pred_x"].notna()) & (df["gt_x"].notna()) & (df["pred_category"] == df["gt_category"]) & (df['iou'] >=0.5)]
        result_att = compute_macro_metrics(df_box_true, att_list)
        result.update(result_att)
    return result

def model_pred_compare(label_dir, pred_dir, save_dir=None, att_file=None, threshold=0.5):
    att_list = get_attributes(att_file)
    mdet = True if att_list is not None else False

    if save_dir is None:
        save_dir = label_dir + f'_compare_{os.path.basename(os.path.dirname(pred_dir))}'
    os.makedirs(save_dir, exist_ok=True)

    merged_columns = ["gt_len", "pred_len", "TP", "FP", "FN", "precision", "recall"]
    if att_list is not None:
        for att in att_list:
            merged_columns += [
               f"{att}_precision",
               f"{att}_recall",
               f"{att}_f1",
               f"{att}_confusion_matrix"
            ]

    df = pd.DataFrame(None, columns=merged_columns)

    txt_list = os.listdir(pred_dir)
    for txt_name in tqdm(txt_list):
        pred_path = os.path.join(pred_dir, txt_name)
        label_path = os.path.join(label_dir, txt_name)
        save_path = os.path.join(save_dir, txt_name)

        df_pred = get_yolo_label_df(pred_path, mdet=mdet, attributes=att_list)
        df_label = get_yolo_label_df(label_path, mdet=mdet, attributes=att_list)

        df_match = match_and_merge(df_pred, df_label, iou_thr=threshold, att_list=att_list)

        df_stats = compute_precision_recall(df_match, att_list=att_list)
        df.loc[txt_name] = df_stats
        df_match.to_csv(save_path.replace('.txt', '.csv'))
    df.to_csv(save_dir+'.csv')

def filter_pre_match_defect(df, idx, att_list, ):
    row = df[df['pred_id'] == idx]

    # not exists
    if row.empty:
        return False
    row = row.iloc[0]

    # match
    if pd.isna(row['gt_id']):
        return False
    if row['pred_category'] != row['gt_category']:
        return False

    for att in att_list:
        if row[f'pred_{att}'] != row[f'gt_{att}'] and row[f'pred_{att}'] > 0:
            return True
    return False

def filter_gt_match_defect(df, idx, att_list, ):
    row = df[df['gt_id'] == idx]

    # not exists
    if row.empty:
        return False
    row = row.iloc[0]

    # match
    if pd.isna(row['pred_id']):
        return False
    if row['pred_category'] != row['gt_category']:
        return False

    for att in att_list:
        if row[f'pred_{att}'] != row[f'gt_{att}'] and row[f'gt_{att}'] > 0:
            return True
    return False

def filter_txt(input_dir, output_dir, ref_dir, att_file, filter_func):
    att_list = get_attributes(att_file)
    os.makedirs(output_dir, exist_ok=True)
    txt_list = os.listdir(ref_dir)
    for txt_name in tqdm(txt_list):
        input_path = os.path.join(input_dir, txt_name.replace('.csv', '.txt'))
        output_path = os.path.join(output_dir, txt_name.replace('.csv', '.txt'))
        ref_path = os.path.join(ref_dir, txt_name)
        df_merge = pd.read_csv(ref_path, header=0, index_col=0)
        with open(input_path, 'r', encoding='utf-8') as f1:
            lines = f1.readlines()
            new_lines = []
            for idx, line in enumerate(lines):
                keep = filter_func(df_merge, idx, att_list)
                if keep:
                    new_lines.append(line)
            if len(new_lines) > 0:
                with open(output_path, 'w', encoding='utf-8') as f2:
                    f2.writelines(new_lines)


if __name__ == "__main__":
    pass
    # root_dir = r'/localnvme/data/billboard/fused_data/data7436_mseg_c5_l2_0917'
    # analysis_dir = os.path.join(root_dir, 'result_analysis')
    # gt_dir = os.path.join(root_dir, 'labels')
    # pred_dir = os.path.join(root_dir, 'images_infer', 'labels')
    # att_file = os.path.join(root_dir, 'attribute_l2.yaml')
    # compare_dir = os.path.join(analysis_dir, f'labels_vs_{os.path.basename(os.path.dirname(pred_dir))}')
    # # model_pred_compare(gt_dir, pred_dir, att_file=att_file, save_dir=compare_dir)
    #
    # filter_pre_match_defect_dir = os.path.join(analysis_dir, 'filter_pre_match_defect')
    # filter_txt(pred_dir, filter_pre_match_defect_dir, compare_dir, att_file, filter_func=filter_pre_match_defect)
    # filter_gt_match_defect_dir = os.path.join(analysis_dir, 'filter_gt_match_defect')
    # filter_txt(gt_dir, filter_gt_match_defect_dir, compare_dir, att_file, filter_func=filter_gt_match_defect)

