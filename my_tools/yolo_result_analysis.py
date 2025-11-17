import os

import cv2
import yaml
import shutil
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from seg_yolo2coco_analysis import seg_analysis, select_val, seg_cfm_analysis
from risk_confusion_matrix_analysis import risk_analysis

warnings.filterwarnings('ignore')


def get_attributes(attribute_path):
    if attribute_path is None:
        return None
    with open(attribute_path, 'r') as file:
        attribute_dict = yaml.safe_load(file)['attributes']
    attribute_keys = list(attribute_dict.keys())
    return attribute_keys
def get_classes(class_path):
    if class_path is None:
        return None
    df_class = pd.read_csv(class_path, header=None, index_col=None, names=['classes'])
    classes = df_class['classes'].to_list()
    if classes[0] == 'background':
        classes[0] = 'none0'
    return classes, ['background']+classes

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

def box_iou(box1, box2, eps=1e-7):
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

    return inter_area / (union + eps)

def match_and_merge(df_pred, df_gt, iou_thr=0.5, att_list=None):
    df_gt = df_gt.reset_index(drop=True)
    matched_gt_idx = set()
    merged_rows = []

    show_columns = ['id', "category", "x", "y", "w", "h", "x1", "y1", "x2", "y2"]
    if att_list is not None:
       show_columns += att_list + ['defect']
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

        df_pred = get_yolo_label_df(pred_path, mdet=mdet, attributes=att_list, with_conf=True)
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

def get_yolo_label_df(gt_path, mdet=False, attributes=None, with_track_id=False, with_object_id=False, with_conf=False,
                      conf_threshold=0.001, defect_conf_threshold=None):
    if mdet:
        assert attributes is not None, 'attribute_path must be provided, which is "%s"' % attributes
        if isinstance(attributes, str):
            attribute_keys = get_attributes(attributes)
        elif isinstance(attributes, list):
            attribute_keys = attributes
        names = ['category'] + ['attribute_len'] + attribute_keys + ['x', 'y', 'w', 'h', 'image']
    else:
        names = ['category', 'x', 'y', 'w', 'h', 'image_name']
    if with_track_id:
        names = names + ['track_id']
    if with_object_id:
        names = names + ['id']
    if with_conf:
        names = names + ['conf']


    df = pd.DataFrame(None, columns=names)
    with open(gt_path, 'r') as f:
        data = f.readlines()
        for id_line, line in enumerate(data):
            parts = line.strip().split(' ')
            if with_conf:
                conf = float(parts[-1])
                parts = parts[:-1]
            category = int(parts[0])
            image_name = Path(gt_path).name
            if mdet:
                att_len = int(parts[1])
                atts = list(map(float, parts[2:2 + att_len]))
                if with_track_id:
                    track_id = int(parts[-1])
                    polygons = list(map(float, parts[2 + att_len:-1]))
                    if len(polygons) < 4:
                        continue
                    xywh = poly2xywh(polygons)
                    info = [category, att_len] + atts + xywh + [image_name, track_id]
                else:
                    polygons = list(map(float, parts[2 + att_len:]))
                    if len(polygons) < 4:
                        continue
                    xywh = poly2xywh(polygons)
                    info = [category, att_len] + atts + xywh + [image_name]
            else:
                polygons = list(map(float, parts[1:]))
                xywh = poly2xywh(polygons)
                info = [category] + xywh + [image_name]
            if with_object_id:
                info += [id_line]
            if with_conf:
                info += [conf]
            df.loc[len(df)] = info

    if mdet:
        df['defect'] = (df[attribute_keys] > 0).any(axis=1)
        attribute_keys_noc = [a for a in attribute_keys if a != 'corrosion']
        df['defect_no_c'] = (df[attribute_keys_noc] > 0).any(axis=1)
    if with_conf:
        if defect_conf_threshold is None:
            df = df[(df['conf'] >= conf_threshold)]
        else:
            # df = df[~(
            #     ((df['defect_no_c'] == True) & (df['conf'] < defect_conf_threshold)) |
            #     ((df['defect_no_c'] == False) & (df['conf'] < conf_threshold))
            # )]
            df = df[~(
                ((df['defect'] == True) & (df['conf'] < defect_conf_threshold)) |
                ((df['defect'] == False) & (df['conf'] < conf_threshold))
            )]
    df = df_xywh_to_xyxy(df)
    return df


def predictions_compare_show(input_dir_list, save_dir, nums=[2, 3], size=[12, 8]):
    os.makedirs(save_dir)
    file_list = os.listdir(input_dir_list[0])
    for file_name in tqdm(file_list):
        input_path_list = [os.path.join(input_dir, file_name) for input_dir in input_dir_list if os.path.isfile(os.path.join(input_dir, file_name))]
        save_path = os.path.join(save_dir, file_name)
        prediction_compare_show(input_path_list, save_path, nums=nums, size=size)

def prediction_compare_show(input_path_list, save_path, nums=[2, 3], size=[12, 8]):
    pass
    fig, axes = plt.subplots(nums[0], nums[1], figsize=size)
    [ax.imshow(plt.imread(p)) for ax,p in zip(axes.flat, input_path_list)]
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def cfm_match_2(keep, df_a, df_b, df_c, df_d, row, df_seg_all, df_seg_with_defect, df_seg_without_defect):
    if keep == 'all':
        if pd.isna(row['gt_category']):
            if int(row['pred_abandonment']) > 0:
                df_a.loc['pred_high', 'label_no'] += 1
            else:
                df_a.loc['pred_no', 'label_no'] += 1
            if int(row['pred_broken']) > 0:
                df_b.loc['pred_high', 'label_no'] += 1
            else:
                df_b.loc['pred_no', 'label_no'] += 1
            if int(row['pred_corrosion']) > 0:
                df_c.loc['pred_high', 'label_no'] += 1
            else:
                df_c.loc['pred_no', 'label_no'] += 1
            if int(row['pred_deformation']) > 0:
                df_d.loc['pred_high', 'label_no'] += 1
            else:
                df_d.loc['pred_no', 'label_no'] += 1
        elif pd.isna(row['pred_category']):
            if int(row['gt_abandonment']) > 0:
                df_a.loc['pred_no', 'label_high'] += 1
            else:
                df_a.loc['pred_no', 'label_no'] += 1
            if int(row['gt_broken']) > 0:
                df_b.loc['pred_no', 'label_high'] += 1
            else:
                df_b.loc['pred_no', 'label_no'] += 1
            if int(row['gt_corrosion']) > 0:
                df_c.loc['pred_no', 'label_high'] += 1
            else:
                df_c.loc['pred_no', 'label_no'] += 1
            if int(row['gt_deformation']) > 0:
                df_d.loc['pred_no', 'label_high'] += 1
            else:
                df_d.loc['pred_no', 'label_no'] += 1
        else:
            if int(row['gt_abandonment']) > 0 and int(row['pred_abandonment']) > 0:
                df_a.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_abandonment']) > 0 and int(row['pred_abandonment']) == 0:
                df_a.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_abandonment']) == 0 and int(row['pred_abandonment']) > 0:
                df_a.loc['pred_high', 'label_no'] += 1
            else:
                df_a.loc['pred_no', 'label_no'] += 1
            if int(row['gt_broken']) > 0 and int(row['pred_broken']) > 0:
                df_b.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_broken']) > 0 and int(row['pred_broken']) == 0:
                df_b.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_broken']) == 0 and int(row['pred_broken']) > 0:
                df_b.loc['pred_high', 'label_no'] += 1
            else:
                df_b.loc['pred_no', 'label_no'] += 1
            if int(row['gt_corrosion']) > 0 and int(row['pred_corrosion']) > 0:
                df_c.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_corrosion']) > 0 and int(row['pred_corrosion']) == 0:
                df_c.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_corrosion']) == 0 and int(row['pred_corrosion']) > 0:
                df_c.loc['pred_high', 'label_no'] += 1
            else:
                df_c.loc['pred_no', 'label_no'] += 1
            if int(row['gt_deformation']) > 0 and int(row['pred_deformation']) > 0:
                df_d.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_deformation']) > 0 and int(row['pred_deformation']) == 0:
                df_d.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_deformation']) == 0 and int(row['pred_deformation']) > 0:
                df_d.loc['pred_high', 'label_no'] += 1
            else:
                df_d.loc['pred_no', 'label_no'] += 1
    elif keep == 'ignore_other':
        if pd.isna(row['gt_category']):
            if not int(row['pred_category']) == 6:
                if int(row['pred_abandonment']) > 0:
                    df_a.loc['pred_high', 'background'] += 1
                else:
                    df_a.loc['pred_no', 'background'] += 1
                if int(row['pred_broken']) > 0:
                    df_b.loc['pred_high', 'background'] += 1
                else:
                    df_b.loc['pred_no', 'background'] += 1
                if int(row['pred_corrosion']) > 0:
                    df_c.loc['pred_high', 'background'] += 1
                else:
                    df_c.loc['pred_no', 'background'] += 1
                if int(row['pred_deformation']) > 0:
                    df_d.loc['pred_high', 'background'] += 1
                else:
                    df_d.loc['pred_no', 'background'] += 1
        elif pd.isna(row['pred_category']):
            if not int(row['gt_category']) == 6:
                if int(row['gt_abandonment']) > 0:
                    df_a.loc['background', 'label_high'] += 1
                else:
                    df_a.loc['background', 'label_no'] += 1
                if int(row['gt_broken']) > 0:
                    df_b.loc['background', 'label_high'] += 1
                else:
                    df_b.loc['background', 'label_no'] += 1
                if int(row['gt_corrosion']) > 0:
                    df_c.loc['background', 'label_high'] += 1
                else:
                    df_c.loc['background', 'label_no'] += 1
                if int(row['gt_deformation']) > 0:
                    df_d.loc['background', 'label_high'] += 1
                else:
                    df_d.loc['background', 'label_no'] += 1
        else:
            if not int(row['gt_category']) == 6:
                if int(row['gt_abandonment']) > 0 and int(row['pred_abandonment']) > 0:
                    df_a.loc['pred_high', 'label_high'] += 1
                elif int(row['gt_abandonment']) > 0 and int(row['pred_abandonment']) == 0:
                    df_a.loc['pred_no', 'label_high'] += 1
                elif int(row['gt_abandonment']) == 0 and int(row['pred_abandonment']) > 0:
                    df_a.loc['pred_high', 'label_no'] += 1
                else:
                    df_a.loc['pred_no', 'label_no'] += 1
                if int(row['gt_broken']) > 0 and int(row['pred_broken']) > 0:
                    df_b.loc['pred_high', 'label_high'] += 1
                elif int(row['gt_broken']) > 0 and int(row['pred_broken']) == 0:
                    df_b.loc['pred_no', 'label_high'] += 1
                elif int(row['gt_broken']) == 0 and int(row['pred_broken']) > 0:
                    df_b.loc['pred_high', 'label_no'] += 1
                else:
                    df_b.loc['pred_no', 'label_no'] += 1
                if int(row['gt_corrosion']) > 0 and int(row['pred_corrosion']) > 0:
                    df_c.loc['pred_high', 'label_high'] += 1
                elif int(row['gt_corrosion']) > 0 and int(row['pred_corrosion']) == 0:
                    df_c.loc['pred_no', 'label_high'] += 1
                elif int(row['gt_corrosion']) == 0 and int(row['pred_corrosion']) > 0:
                    df_c.loc['pred_high', 'label_no'] += 1
                else:
                    df_c.loc['pred_no', 'label_no'] += 1
                if int(row['gt_deformation']) > 0 and int(row['pred_deformation']) > 0:
                    df_d.loc['pred_high', 'label_high'] += 1
                elif int(row['gt_deformation']) > 0 and int(row['pred_deformation']) == 0:
                    df_d.loc['pred_no', 'label_high'] += 1
                elif int(row['gt_deformation']) == 0 and int(row['pred_deformation']) > 0:
                    df_d.loc['pred_high', 'label_no'] += 1
                else:
                    df_d.loc['pred_no', 'label_no'] += 1
    elif keep == 'iou':
        if pd.isna(row['gt_category']) or pd.isna(row['pred_category']):
            pass
        else:
            if int(row['gt_abandonment']) > 0 and int(row['pred_abandonment']) > 0:
                df_a.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_abandonment']) > 0 and int(row['pred_abandonment']) == 0:
                df_a.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_abandonment']) == 0 and int(row['pred_abandonment']) > 0:
                df_a.loc['pred_high', 'label_no'] += 1
            else:
                df_a.loc['pred_no', 'label_no'] += 1
            if int(row['gt_broken']) > 0 and int(row['pred_broken']) > 0:
                df_b.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_broken']) > 0 and int(row['pred_broken']) == 0:
                df_b.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_broken']) == 0 and int(row['pred_broken']) > 0:
                df_b.loc['pred_high', 'label_no'] += 1
            else:
                df_b.loc['pred_no', 'label_no'] += 1
            if int(row['gt_corrosion']) > 0 and int(row['pred_corrosion']) > 0:
                df_c.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_corrosion']) > 0 and int(row['pred_corrosion']) == 0:
                df_c.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_corrosion']) == 0 and int(row['pred_corrosion']) > 0:
                df_c.loc['pred_high', 'label_no'] += 1
            else:
                df_c.loc['pred_no', 'label_no'] += 1
            if int(row['gt_deformation']) > 0 and int(row['pred_deformation']) > 0:
                df_d.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_deformation']) > 0 and int(row['pred_deformation']) == 0:
                df_d.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_deformation']) == 0 and int(row['pred_deformation']) > 0:
                df_d.loc['pred_high', 'label_no'] += 1
            else:
                df_d.loc['pred_no', 'label_no'] += 1
    elif keep == 'correct':
        if pd.isna(row['gt_category']) or pd.isna(row['pred_category']) and row['gt_category'] != row['pred_category']:
            pass
        else:
            if int(row['gt_abandonment']) > 0 and int(row['pred_abandonment']) > 0:
                df_a.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_abandonment']) > 0 and int(row['pred_abandonment']) == 0:
                df_a.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_abandonment']) == 0 and int(row['pred_abandonment']) > 0:
                df_a.loc['pred_high', 'label_no'] += 1
            else:
                df_a.loc['pred_no', 'label_no'] += 1
            if int(row['gt_broken']) > 0 and int(row['pred_broken']) > 0:
                df_b.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_broken']) > 0 and int(row['pred_broken']) == 0:
                df_b.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_broken']) == 0 and int(row['pred_broken']) > 0:
                df_b.loc['pred_high', 'label_no'] += 1
            else:
                df_b.loc['pred_no', 'label_no'] += 1
            if int(row['gt_corrosion']) > 0 and int(row['pred_corrosion']) > 0:
                df_c.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_corrosion']) > 0 and int(row['pred_corrosion']) == 0:
                df_c.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_corrosion']) == 0 and int(row['pred_corrosion']) > 0:
                df_c.loc['pred_high', 'label_no'] += 1
            else:
                df_c.loc['pred_no', 'label_no'] += 1
            if int(row['gt_deformation']) > 0 and int(row['pred_deformation']) > 0:
                df_d.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_deformation']) > 0 and int(row['pred_deformation']) == 0:
                df_d.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_deformation']) == 0 and int(row['pred_deformation']) > 0:
                df_d.loc['pred_high', 'label_no'] += 1
            else:
                df_d.loc['pred_no', 'label_no'] += 1

def cfm_match_3(keep, df_a, df_b, df_c, df_d, row, df_seg_all, df_seg_with_defect, df_seg_without_defect):
    pass
    if keep == 'all':
        if pd.isna(row['gt_category']):
            df_seg_all.loc[int(row['pred_category'])+1, 0] += 1
            if row['pred_defect']:
                df_seg_with_defect.loc[int(row['pred_category']) + 1, 0] += 1
            else:
                df_seg_without_defect.loc[int(row['pred_category']) + 1, 0] += 1
            if int(row['pred_abandonment']) > 0:
                df_a.loc['pred_high', 'background'] += 1
            else:
                df_a.loc['pred_no', 'background'] += 1
            if int(row['pred_broken']) > 0:
                df_b.loc['pred_high', 'background'] += 1
            else:
                df_b.loc['pred_no', 'background'] += 1
            if int(row['pred_corrosion']) > 0:
                df_c.loc['pred_high', 'background'] += 1
            else:
                df_c.loc['pred_no', 'background'] += 1
            if int(row['pred_deformation']) > 0:
                df_d.loc['pred_high', 'background'] += 1
            else:
                df_d.loc['pred_no', 'background'] += 1
        elif pd.isna(row['pred_category']):
            df_seg_all.loc[0, int(row['gt_category'])+1] += 1
            if row['gt_defect']:
                df_seg_with_defect.loc[0, int(row['gt_category']) + 1] += 1
            else:
                df_seg_without_defect.loc[0, int(row['gt_category']) + 1] += 1
            if int(row['gt_abandonment']) > 0:
                df_a.loc['background', 'label_high'] += 1
            else:
                df_a.loc['background', 'label_no'] += 1
            if int(row['gt_broken']) > 0:
                df_b.loc['background', 'label_high'] += 1
            else:
                df_b.loc['background', 'label_no'] += 1
            if int(row['gt_corrosion']) > 0:
                df_c.loc['background', 'label_high'] += 1
            else:
                df_c.loc['background', 'label_no'] += 1
            if int(row['gt_deformation']) > 0:
                df_d.loc['background', 'label_high'] += 1
            else:
                df_d.loc['background', 'label_no'] += 1
        else:
            df_seg_all.loc[int(row['pred_category'])+1, int(row['gt_category'])+1] += 1
            if row['gt_defect']:
                df_seg_with_defect.loc[int(row['pred_category'])+1, int(row['gt_category']) + 1] += 1
            else:
                df_seg_without_defect.loc[int(row['pred_category'])+1, int(row['gt_category']) + 1] += 1
            if int(row['gt_abandonment']) > 0 and int(row['pred_abandonment']) > 0:
                df_a.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_abandonment']) > 0 and int(row['pred_abandonment']) == 0:
                df_a.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_abandonment']) == 0 and int(row['pred_abandonment']) > 0:
                df_a.loc['pred_high', 'label_no'] += 1
            else:
                df_a.loc['pred_no', 'label_no'] += 1
            if int(row['gt_broken']) > 0 and int(row['pred_broken']) > 0:
                df_b.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_broken']) > 0 and int(row['pred_broken']) == 0:
                df_b.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_broken']) == 0 and int(row['pred_broken']) > 0:
                df_b.loc['pred_high', 'label_no'] += 1
            else:
                df_b.loc['pred_no', 'label_no'] += 1
            if int(row['gt_corrosion']) > 0 and int(row['pred_corrosion']) > 0:
                df_c.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_corrosion']) > 0 and int(row['pred_corrosion']) == 0:
                df_c.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_corrosion']) == 0 and int(row['pred_corrosion']) > 0:
                df_c.loc['pred_high', 'label_no'] += 1
            else:
                df_c.loc['pred_no', 'label_no'] += 1
            if int(row['gt_deformation']) > 0 and int(row['pred_deformation']) > 0:
                df_d.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_deformation']) > 0 and int(row['pred_deformation']) == 0:
                df_d.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_deformation']) == 0 and int(row['pred_deformation']) > 0:
                df_d.loc['pred_high', 'label_no'] += 1
            else:
                df_d.loc['pred_no', 'label_no'] += 1
    elif keep == 'ignore_other':
        ValueError('seg confusion matrix not imply!')
        if pd.isna(row['gt_category']):
            if int(row['pred_category']) not in [6]:
                if int(row['pred_abandonment']) > 0:
                    df_a.loc['pred_high', 'background'] += 1
                else:
                    df_a.loc['pred_no', 'background'] += 1
                if int(row['pred_broken']) > 0:
                    df_b.loc['pred_high', 'background'] += 1
                else:
                    df_b.loc['pred_no', 'background'] += 1
                if int(row['pred_corrosion']) > 0:
                    df_c.loc['pred_high', 'background'] += 1
                else:
                    df_c.loc['pred_no', 'background'] += 1
                if int(row['pred_deformation']) > 0:
                    df_d.loc['pred_high', 'background'] += 1
                else:
                    df_d.loc['pred_no', 'background'] += 1
        elif pd.isna(row['pred_category']):
            if int(row['gt_category']) not in [6]:
                if int(row['gt_abandonment']) > 0:
                    df_a.loc['background', 'label_high'] += 1
                else:
                    df_a.loc['background', 'label_no'] += 1
                if int(row['gt_broken']) > 0:
                    df_b.loc['background', 'label_high'] += 1
                else:
                    df_b.loc['background', 'label_no'] += 1
                if int(row['gt_corrosion']) > 0:
                    df_c.loc['background', 'label_high'] += 1
                else:
                    df_c.loc['background', 'label_no'] += 1
                if int(row['gt_deformation']) > 0:
                    df_d.loc['background', 'label_high'] += 1
                else:
                    df_d.loc['background', 'label_no'] += 1
        else:
            if int(row['gt_abandonment']) > 0 and int(row['pred_abandonment']) > 0:
                df_a.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_abandonment']) > 0 and int(row['pred_abandonment']) == 0:
                df_a.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_abandonment']) == 0 and int(row['pred_abandonment']) > 0:
                df_a.loc['pred_high', 'label_no'] += 1
            else:
                df_a.loc['pred_no', 'label_no'] += 1
            if int(row['gt_broken']) > 0 and int(row['pred_broken']) > 0:
                df_b.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_broken']) > 0 and int(row['pred_broken']) == 0:
                df_b.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_broken']) == 0 and int(row['pred_broken']) > 0:
                df_b.loc['pred_high', 'label_no'] += 1
            else:
                df_b.loc['pred_no', 'label_no'] += 1
            if int(row['gt_corrosion']) > 0 and int(row['pred_corrosion']) > 0:
                df_c.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_corrosion']) > 0 and int(row['pred_corrosion']) == 0:
                df_c.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_corrosion']) == 0 and int(row['pred_corrosion']) > 0:
                df_c.loc['pred_high', 'label_no'] += 1
            else:
                df_c.loc['pred_no', 'label_no'] += 1
            if int(row['gt_deformation']) > 0 and int(row['pred_deformation']) > 0:
                df_d.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_deformation']) > 0 and int(row['pred_deformation']) == 0:
                df_d.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_deformation']) == 0 and int(row['pred_deformation']) > 0:
                df_d.loc['pred_high', 'label_no'] += 1
            else:
                df_d.loc['pred_no', 'label_no'] += 1
    elif keep == 'ignore_other_pred_frame':
        if pd.isna(row['gt_category']):
            if int(row['pred_category']) not in [2, 6]:
                df_seg_all.loc[int(row['pred_category']) + 1, 0] += 1
                if row['pred_defect']:
                    df_seg_with_defect.loc[int(row['pred_category']) + 1, 0] += 1
                else:
                    df_seg_without_defect.loc[int(row['pred_category']) + 1, 0] += 1
                if int(row['pred_abandonment']) > 0:
                    df_a.loc['pred_high', 'background'] += 1
                else:
                    df_a.loc['pred_no', 'background'] += 1
                if int(row['pred_broken']) > 0:
                    df_b.loc['pred_high', 'background'] += 1
                else:
                    df_b.loc['pred_no', 'background'] += 1
                if int(row['pred_corrosion']) > 0:
                    df_c.loc['pred_high', 'background'] += 1
                else:
                    df_c.loc['pred_no', 'background'] += 1
                if int(row['pred_deformation']) > 0:
                    df_d.loc['pred_high', 'background'] += 1
                else:
                    df_d.loc['pred_no', 'background'] += 1
        elif pd.isna(row['pred_category']):
            if int(row['gt_category']) not in [6]:
                df_seg_all.loc[0, int(row['gt_category']) + 1] += 1
                if row['gt_defect']:
                    df_seg_with_defect.loc[0, int(row['gt_category']) + 1] += 1
                else:
                    df_seg_without_defect.loc[0, int(row['gt_category']) + 1] += 1
                if int(row['gt_abandonment']) > 0:
                    df_a.loc['background', 'label_high'] += 1
                else:
                    df_a.loc['background', 'label_no'] += 1
                if int(row['gt_broken']) > 0:
                    df_b.loc['background', 'label_high'] += 1
                else:
                    df_b.loc['background', 'label_no'] += 1
                if int(row['gt_corrosion']) > 0:
                    df_c.loc['background', 'label_high'] += 1
                else:
                    df_c.loc['background', 'label_no'] += 1
                if int(row['gt_deformation']) > 0:
                    df_d.loc['background', 'label_high'] += 1
                else:
                    df_d.loc['background', 'label_no'] += 1
        else:
            df_seg_all.loc[int(row['pred_category'])+1, int(row['gt_category'])+1] += 1
            if row['gt_defect']:
                df_seg_with_defect.loc[int(row['pred_category'])+1, int(row['gt_category']) + 1] += 1
            else:
                df_seg_without_defect.loc[int(row['pred_category'])+1, int(row['gt_category']) + 1] += 1
            if int(row['gt_abandonment']) > 0 and int(row['pred_abandonment']) > 0:
                df_a.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_abandonment']) > 0 and int(row['pred_abandonment']) == 0:
                df_a.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_abandonment']) == 0 and int(row['pred_abandonment']) > 0:
                df_a.loc['pred_high', 'label_no'] += 1
            else:
                df_a.loc['pred_no', 'label_no'] += 1
            if int(row['gt_broken']) > 0 and int(row['pred_broken']) > 0:
                df_b.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_broken']) > 0 and int(row['pred_broken']) == 0:
                df_b.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_broken']) == 0 and int(row['pred_broken']) > 0:
                df_b.loc['pred_high', 'label_no'] += 1
            else:
                df_b.loc['pred_no', 'label_no'] += 1
            if int(row['gt_corrosion']) > 0 and int(row['pred_corrosion']) > 0:
                df_c.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_corrosion']) > 0 and int(row['pred_corrosion']) == 0:
                df_c.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_corrosion']) == 0 and int(row['pred_corrosion']) > 0:
                df_c.loc['pred_high', 'label_no'] += 1
            else:
                df_c.loc['pred_no', 'label_no'] += 1
            if int(row['gt_deformation']) > 0 and int(row['pred_deformation']) > 0:
                df_d.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_deformation']) > 0 and int(row['pred_deformation']) == 0:
                df_d.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_deformation']) == 0 and int(row['pred_deformation']) > 0:
                df_d.loc['pred_high', 'label_no'] += 1
            else:
                df_d.loc['pred_no', 'label_no'] += 1
    elif keep == 'iou':
        ValueError('seg confusion matrix not imply!')
        if pd.isna(row['gt_category']) or pd.isna(row['pred_category']):
            pass
        else:
            if int(row['gt_abandonment']) > 0 and int(row['pred_abandonment']) > 0:
                df_a.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_abandonment']) > 0 and int(row['pred_abandonment']) == 0:
                df_a.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_abandonment']) == 0 and int(row['pred_abandonment']) > 0:
                df_a.loc['pred_high', 'label_no'] += 1
            else:
                df_a.loc['pred_no', 'label_no'] += 1
            if int(row['gt_broken']) > 0 and int(row['pred_broken']) > 0:
                df_b.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_broken']) > 0 and int(row['pred_broken']) == 0:
                df_b.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_broken']) == 0 and int(row['pred_broken']) > 0:
                df_b.loc['pred_high', 'label_no'] += 1
            else:
                df_b.loc['pred_no', 'label_no'] += 1
            if int(row['gt_corrosion']) > 0 and int(row['pred_corrosion']) > 0:
                df_c.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_corrosion']) > 0 and int(row['pred_corrosion']) == 0:
                df_c.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_corrosion']) == 0 and int(row['pred_corrosion']) > 0:
                df_c.loc['pred_high', 'label_no'] += 1
            else:
                df_c.loc['pred_no', 'label_no'] += 1
            if int(row['gt_deformation']) > 0 and int(row['pred_deformation']) > 0:
                df_d.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_deformation']) > 0 and int(row['pred_deformation']) == 0:
                df_d.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_deformation']) == 0 and int(row['pred_deformation']) > 0:
                df_d.loc['pred_high', 'label_no'] += 1
            else:
                df_d.loc['pred_no', 'label_no'] += 1
    elif keep == 'correct':
        ValueError('seg confusion matrix not imply!')
        if pd.isna(row['gt_category']) or pd.isna(row['pred_category']) and row['gt_category'] != row['pred_category']:
            pass
        else:
            if int(row['gt_abandonment']) > 0 and int(row['pred_abandonment']) > 0:
                df_a.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_abandonment']) > 0 and int(row['pred_abandonment']) == 0:
                df_a.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_abandonment']) == 0 and int(row['pred_abandonment']) > 0:
                df_a.loc['pred_high', 'label_no'] += 1
            else:
                df_a.loc['pred_no', 'label_no'] += 1
            if int(row['gt_broken']) > 0 and int(row['pred_broken']) > 0:
                df_b.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_broken']) > 0 and int(row['pred_broken']) == 0:
                df_b.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_broken']) == 0 and int(row['pred_broken']) > 0:
                df_b.loc['pred_high', 'label_no'] += 1
            else:
                df_b.loc['pred_no', 'label_no'] += 1
            if int(row['gt_corrosion']) > 0 and int(row['pred_corrosion']) > 0:
                df_c.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_corrosion']) > 0 and int(row['pred_corrosion']) == 0:
                df_c.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_corrosion']) == 0 and int(row['pred_corrosion']) > 0:
                df_c.loc['pred_high', 'label_no'] += 1
            else:
                df_c.loc['pred_no', 'label_no'] += 1
            if int(row['gt_deformation']) > 0 and int(row['pred_deformation']) > 0:
                df_d.loc['pred_high', 'label_high'] += 1
            elif int(row['gt_deformation']) > 0 and int(row['pred_deformation']) == 0:
                df_d.loc['pred_no', 'label_high'] += 1
            elif int(row['gt_deformation']) == 0 and int(row['pred_deformation']) > 0:
                df_d.loc['pred_high', 'label_no'] += 1
            else:
                df_d.loc['pred_no', 'label_no'] += 1


def seg2bi(df):
    df_new = pd.DataFrame(index=['background', 'segmented'], columns=['background', 'segmented'])
    df_new.loc['background', 'background'] = df.loc['background', 'background']
    df_new.loc['background', 'segmented'] = df.loc['background'].drop('background').sum()
    df_new.loc['segmented', 'background'] = df.drop('background').loc[:, 'background'].sum()
    df_new.loc['segmented', 'segmented'] = df.drop(index='background',columns='background').values.sum()
    return df_new


def pred2cfm_risk_single(label_dir, pred_dir_a, pred_dir_b, pred_dir_c, pred_dir_d, pred_dir_all, save_dir,
                         attributes=None, classes=None, with_conf=True, conf_threshold=0.001, defect_conf_threshold=None,
                         iou_thr=0.5, keep='all', filter_small=None, show_list=[], cfm_num=3, only_save=False):
    label_list = os.listdir(label_dir)
    os.makedirs(pred_dir_all, exist_ok=True)
    for label_name in tqdm(label_list, desc='label merge'):
        pred_a_path = os.path.join(pred_dir_a, label_name)
        pred_b_path = os.path.join(pred_dir_b, label_name)
        pred_c_path = os.path.join(pred_dir_c, label_name)
        pred_d_path = os.path.join(pred_dir_d, label_name)
        pred_all_path = os.path.join(pred_dir_all, label_name)
        with open(pred_a_path, 'r') as fa, open(pred_b_path, 'r') as fb, open(pred_c_path, 'r') as fc, open(pred_d_path, 'r') as fd:
            lines_a, lines_b, lines_c, lines_d = fa.readlines(), fb.readlines(), fc.readlines(), fd.readlines()
            assert len(lines_a) == len(lines_b) == len(lines_c) == len(lines_d), f"{label_name} error"
            lines_all = []
            for idx in range(len(lines_a)):
                line_a, line_b, line_c, line_d = lines_a[idx], lines_b[idx], lines_c[idx], lines_d[idx]
                parts_a, parts_b, parts_c, parts_d = line_a.strip().split(' '), line_b.strip().split(' '), line_c.strip().split(' '), line_d.strip().split(' ')
                assert (len(parts_a) == len(parts_b) == len(parts_c) == len(parts_d)) and (parts_a[-1] == parts_b[-1] == parts_c[-1] == parts_d[-1]), f"{label_name} error"
                parts_all = [parts_a[0], '4', parts_d[2], parts_b[2], parts_a[2], parts_c[2]] + parts_a[3:]
                line_all = ' '.join(parts_all)+'\n'
                lines_all.append(line_all)
            with open(pred_all_path, 'w') as fw:
                fw.writelines(lines_all)

    if not only_save:
        pred2cfm_risk(label_dir, pred_dir_all, save_dir, attributes=attributes, classes=classes, with_conf=with_conf, conf_threshold=conf_threshold,
                      defect_conf_threshold=defect_conf_threshold, iou_thr=iou_thr, keep=keep, filter_small=filter_small, show_list=show_list, cfm_num=cfm_num)


def pred2cfm_risk(label_dir, pred_dir, save_dir, attributes=None, classes=None, with_conf=True, conf_threshold=0.001,
                  defect_conf_threshold=None, iou_thr=0.5, keep='all', filter_small=None, show_list=[], cfm_num=3):
    save_dir_risk3 = os.path.join(save_dir, 'risk3')
    save_dir_risk2 = os.path.join(save_dir, 'risk2')
    save_dir_seg = os.path.join(save_dir, 'seg')
    save_dir_seg2 = os.path.join(save_dir, 'seg2')
    os.makedirs(save_dir_risk3, exist_ok=True)
    os.makedirs(save_dir_risk2, exist_ok=True)
    os.makedirs(save_dir_seg, exist_ok=True)
    os.makedirs(save_dir_seg2, exist_ok=True)

    attributes = get_attributes(attributes)
    classes, classes_bg = get_classes(classes)
    txt_list = os.listdir(label_dir)

    df_risk_3 = pd.DataFrame(np.zeros([3,3]), columns=['background', 'label_no', 'label_high'], index=['background', 'pred_no', 'pred_high'])
    df_risk_2 = pd.DataFrame(np.zeros([2,2]), columns=['label_no', 'label_high'], index=['pred_no', 'pred_high'])
    final_columns_2 = ['False', 'high']
    final_columns_3 = ['background', 'False', 'high']
    if cfm_num==2:
        df_risk = df_risk_2
        cfm_match = cfm_match_2
        final_columns = final_columns_2
    elif cfm_num==3:
        df_risk = df_risk_3
        cfm_match = cfm_match_3
        final_columns = final_columns_3
    else:
        raise ValueError('cfm_num must be 2 or 3')


    df_seg_all = pd.DataFrame(np.zeros([len(classes_bg), len(classes_bg)]),columns=list(range(len(classes_bg))), index=list(range(len(classes_bg))))
    df_seg_with_defect = df_seg_all.copy()
    df_seg_without_defect = df_seg_all.copy()
    df_a, df_b, df_c, df_d = df_risk.copy(), df_risk.copy(), df_risk.copy(), df_risk.copy()


    for txt_name in tqdm(txt_list, desc='read and process label'):
        label_path = os.path.join(label_dir, txt_name)
        pred_path = os.path.join(pred_dir, txt_name)

        df_label = get_yolo_label_df(label_path, mdet=True, attributes=attributes, with_object_id=True)
        if not os.path.exists(pred_path):
            df_pred = pd.DataFrame(columns=df_label.columns)
        else:
            df_pred = get_yolo_label_df(pred_path, mdet=True, attributes=attributes, with_object_id=True, with_conf=with_conf, conf_threshold=conf_threshold, defect_conf_threshold=defect_conf_threshold)

        if filter_small is not None:
            df_label = df_label.loc[(df_label['w']>filter_small) | (df_label['h']>filter_small)]
            df_pred = df_pred.loc[(df_pred['w']>filter_small) | (df_pred['h']>filter_small)]

        df_match = match_and_merge(df_pred, df_label, iou_thr=iou_thr, att_list=attributes)
        for idx, row in df_match.iterrows():
            cfm_match(keep, df_a, df_b, df_c, df_d, row, df_seg_all, df_seg_with_defect, df_seg_without_defect)

    df_seg_all.columns, df_seg_with_defect.columns, df_seg_without_defect.columns = classes_bg, classes_bg, classes_bg
    df_seg_all.index, df_seg_with_defect.index, df_seg_without_defect.index = classes_bg, classes_bg, classes_bg
    df_a.columns, df_b.columns, df_c.columns, df_d.columns = final_columns, final_columns, final_columns, final_columns
    df_a.index, df_b.index, df_c.index, df_d.index = final_columns, final_columns, final_columns, final_columns
    df_a_2 = df_a.drop(index='background', columns='background')
    df_b_2 = df_b.drop(index='background', columns='background')
    df_c_2 = df_c.drop(index='background', columns='background')
    df_d_2 = df_d.drop(index='background', columns='background')
    df_seg2_all = seg2bi(df_seg_all)
    df_seg2_with_defect = seg2bi(df_seg_with_defect)
    df_seg2_without_defect = seg2bi(df_seg_without_defect)
    df_a.to_csv(os.path.join(save_dir_risk3, "confusion_matrix_for_attribute_abandonment.csv"), header=True, index=True)
    df_b.to_csv(os.path.join(save_dir_risk3, "confusion_matrix_for_attribute_broken.csv"), header=True, index=True)
    df_c.to_csv(os.path.join(save_dir_risk3, "confusion_matrix_for_attribute_corrosion.csv"), header=True, index=True)
    df_d.to_csv(os.path.join(save_dir_risk3, "confusion_matrix_for_attribute_deformation.csv"), header=True, index=True)
    df_a_2.to_csv(os.path.join(save_dir_risk2, "confusion_matrix_for_attribute_abandonment.csv"), header=True, index=True)
    df_b_2.to_csv(os.path.join(save_dir_risk2, "confusion_matrix_for_attribute_broken.csv"), header=True, index=True)
    df_c_2.to_csv(os.path.join(save_dir_risk2, "confusion_matrix_for_attribute_corrosion.csv"), header=True, index=True)
    df_d_2.to_csv(os.path.join(save_dir_risk2, "confusion_matrix_for_attribute_deformation.csv"), header=True, index=True)
    df_seg_all.to_csv(os.path.join(save_dir_seg, "confusion_matrix_for_seg_all.csv"), header=True, index=True)
    df_seg_with_defect.to_csv(os.path.join(save_dir_seg, "confusion_matrix_for_seg_with_defect.csv"), header=True, index=True)
    df_seg_without_defect.to_csv(os.path.join(save_dir_seg, "confusion_matrix_for_seg_without_defect.csv"), header=True, index=True)
    df_seg2_all.to_csv(os.path.join(save_dir_seg2, "confusion_matrix_for_seg_all.csv"), header=True, index=True)
    df_seg2_with_defect.to_csv(os.path.join(save_dir_seg2, "confusion_matrix_for_seg_with_defect.csv"), header=True, index=True)
    df_seg2_without_defect.to_csv(os.path.join(save_dir_seg2, "confusion_matrix_for_seg_without_defect.csv"), header=True, index=True)

    if 'risk3' in show_list:
        risk_analysis(save_dir_risk3, rm_bg=True)
    if 'risk2' in show_list:
        risk_analysis(save_dir_risk2)
    if 'seg' in show_list:
        seg_cfm_analysis(save_dir_seg)
    if 'seg2' in show_list:
        seg_cfm_analysis(save_dir_seg2)

def get_stem2img_dict(img_dir):
    img_list = [img_name for img_name in os.listdir(img_dir)]
    stem_list = [Path(img).stem for img in img_list]
    stem2img_dict = dict(zip(stem_list, img_list))
    return stem2img_dict

def row_save(row, all_dir, image_path, save_dir, key):
    # x1, y1, x2, y2 = row[f'{key}_x1'], row[f'{key}_y1'], row[f'{key}_x2'], row[f'{key}_y2']
    pred_id = row[f'{key}_id']
    save_name = Path(image_path).stem + f'_{int(pred_id)}' + Path(image_path).suffix
    save_path = os.path.join(save_dir, save_name)
    if not os.path.exists(save_path):
        input_all_path = os.path.join(all_dir, save_name)
        if os.path.exists(input_all_path):
            shutil.copy(input_all_path, save_path)
        else:
            print(f'{input_all_path} miss')

def find_files(pred_conf, label_conf, risk, image_dir, label_dir, predict_dir, root_dir, attributes, with_conf=True,
               conf_threshold=0.001, iou_thr=0.5, filter_small=0.05, defect_conf_threshold=None):
    save_name = f'risk_{risk}_pred_{pred_conf}_label_{label_conf}'
    save_dir = os.path.join(root_dir, save_name)
    all_gt_dir = os.path.join(os.path.dirname(root_dir), 'all_gt')
    all_pred_dir = os.path.join(root_dir, 'all_pred')
    os.makedirs(save_dir, exist_ok=True)
    attributes = get_attributes(attributes)
    stem2image_list = get_stem2img_dict(image_dir)
    label_list = os.listdir(label_dir)

    for label_name in tqdm(label_list, desc=save_name):
        label_path = os.path.join(label_dir, label_name)
        pred_path = os.path.join(predict_dir, label_name)
        image_path = os.path.join(image_dir, stem2image_list[Path(label_name).stem])
        df_label = get_yolo_label_df(label_path, mdet=True, attributes=attributes, with_object_id=True)
        if not os.path.exists(pred_path):
            df_pred = pd.DataFrame(columns=df_label.columns)
        else:
            df_pred = get_yolo_label_df(pred_path, mdet=True, attributes=attributes, with_object_id=True,
                                        with_conf=with_conf, conf_threshold=conf_threshold, defect_conf_threshold=defect_conf_threshold)
        if filter_small is not None:
            df_label = df_label.loc[(df_label['w']>filter_small) | (df_label['h']>filter_small)]
            df_pred = df_pred.loc[(df_pred['w']>filter_small) | (df_pred['h']>filter_small)]
        df_match = match_and_merge(df_pred, df_label, iou_thr=iou_thr, att_list=attributes)
        if df_match.empty:
            continue
        if pred_conf == 'background' and label_conf == 'no':
            df = df_match[(df_match[f'pred_{risk}'].isna()) & (df_match[f'gt_{risk}'] == 0)]
            all_dir = all_gt_dir
            key = 'gt'
        elif pred_conf == 'background' and label_conf == 'high':
            df = df_match[(df_match[f'pred_{risk}'].isna()) & (df_match[f'gt_{risk}'] > 0)]
            all_dir = all_gt_dir
            key = 'gt'
        elif pred_conf == 'no' and label_conf == 'background':
            df = df_match[(df_match[f'pred_{risk}'] == 0) & (df_match[f'gt_{risk}'].isna())]
            all_dir = all_pred_dir
            key = 'pred'
        elif pred_conf == 'high' and label_conf == 'background':
            df = df_match[(df_match[f'pred_{risk}'] > 0) & (df_match[f'gt_{risk}'].isna())]
            all_dir = all_pred_dir
            key = 'pred'
        elif pred_conf == 'no' and label_conf == 'high':
            df = df_match[(df_match[f'pred_{risk}'] == 0) & (df_match[f'gt_{risk}'] > 0)]
            all_dir = all_gt_dir
            key = 'gt'
        elif pred_conf == 'high' and label_conf == 'no':
            df = df_match[(df_match[f'pred_{risk}'] > 0) & (df_match[f'gt_{risk}'] == 0)]
            all_dir = all_gt_dir
            key = 'gt'
        else:
            raise ValueError
        for idx, row in df.iterrows():
            row_save(row, all_dir, image_path, save_dir, key=key)


def eval_for_emsd(label_dir, predict_dir, attributes, output_path, with_conf=True, conf_threshold=0.001, iou_thr=0.5, filter_small=0.05):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    attributes = get_attributes(attributes)
    label_list = os.listdir(label_dir)
    df_all = pd.DataFrame(None, columns=['file_name', 'object_order', 'T/F', 'small object',  'seg error', 'abandonment error', 'broken error', 'corrosion error', 'deformation error'])
    for label_name in tqdm(label_list):
        label_path = os.path.join(label_dir, label_name)
        pred_path = os.path.join(predict_dir, label_name)
        df_label = get_yolo_label_df(label_path, mdet=True, attributes=attributes, with_object_id=True)
        if not os.path.exists(pred_path):
            df_pred = pd.DataFrame(columns=df_label.columns)
        else:
            df_pred = get_yolo_label_df(pred_path, mdet=True, attributes=attributes, with_object_id=True,
                                        with_conf=with_conf, conf_threshold=conf_threshold)

        df_match = match_and_merge(df_pred, df_label, iou_thr=iou_thr, att_list=attributes)

        for idx, row in df_match.iterrows():
            if pd.isna(row['gt_id']):
                continue
            file_name = Path(label_name).stem

            if row['gt_w']>filter_small or row['gt_h']>filter_small:
                small_object = False
            else:
                small_object = True

            object_order = row['gt_id']

            seg_error = not pd.isna(row['pred_category'])
            a_error = row['gt_abandonment'] == row['pred_abandonment']
            b_error = row['gt_broken'] == row['pred_broken']
            c_error = row['gt_corrosion'] == row['pred_corrosion']
            d_error = row['gt_deformation'] == row['pred_deformation']

            TF = seg_error & a_error& b_error & c_error & d_error

            info = [file_name, object_order, TF, small_object, seg_error, a_error, b_error, c_error, d_error]

            df_all.loc[len(df_all)] = info
    df_all.to_csv(output_path, index=True, header=True, encoding='utf-8')
    print(output_path)


def eval_for_emsd_v2(label_dir, predict_dir, attributes, output_path, with_conf=True, conf_threshold=0.001, iou_thr=0.5, filter_small=0.05, defect_conf_threshold=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    attributes = get_attributes(attributes)
    label_list = os.listdir(label_dir)
    risk_list = ['abandonment', 'broken', 'corrosion', 'deformation']
    df_all = pd.DataFrame(None, columns=['object_name_anno', 'seg_pred', 'risk_pred',  'object_name_pred', 'annotated risk', 'pred risk', 'small_object'])
    for label_name in tqdm(label_list):
        label_path = os.path.join(label_dir, label_name)
        pred_path = os.path.join(predict_dir, label_name)
        df_label = get_yolo_label_df(label_path, mdet=True, attributes=attributes, with_object_id=True)
        if not os.path.exists(pred_path):
            df_pred = pd.DataFrame(columns=df_label.columns)
        else:
            df_pred = get_yolo_label_df(pred_path, mdet=True, attributes=attributes, with_object_id=True,
                                        with_conf=with_conf, conf_threshold=conf_threshold, defect_conf_threshold=defect_conf_threshold)

        df_match = match_and_merge(df_pred, df_label, iou_thr=iou_thr, att_list=attributes)

        df_filter = df_match[(df_match['gt_with_defect']) | (df_match['pred_with_defect'])]

        for idx, row in df_filter.iterrows():
            pred_id = row['pred_id']
            gt_id = row['gt_id']
            anno_name = Path(label_name).stem + f'_{gt_id}'
            pred_name = Path(label_name).stem + f'_{pred_id}'
            if row['pred_category'] == row['gt_category']:
                seg_pred = 'TP'
            else:
                seg_pred = 'FN'
            if row['gt_w']>filter_small or row['gt_h']>filter_small:
                small_object = False
            else:
                small_object = True

            for risk in risk_list:
                if not pd.isna(row[f'gt_{risk}']) and row[f'gt_{risk}']>0:
                    gt_risk = risk
                    if row[f'gt_{risk}']==row[f'pred_{risk}']:
                        risk_pred = 'TP'
                        pred_risk = risk
                    else:
                        risk_pred = 'FN'
                        pred_risk = row[f'pred_{risk}']
                elif not pd.isna(row[f'pred_{risk}']) and row[f'pred_{risk}']>0:
                    pred_risk = risk
                    if row[f'gt_{risk}']==row[f'pred_{risk}']:
                        risk_pred = 'TP'
                        gt_risk = risk
                    else:
                        risk_pred = 'FP'
                        gt_risk = row[f'gt_{risk}']
                else:
                    continue
                df_all.loc[len(df_all)] = [anno_name, seg_pred, risk_pred, pred_name, gt_risk, pred_risk, small_object]

    df_all.to_csv(output_path, index=True, header=True, encoding='utf-8')
    print(output_path)


def get_all_high(input_dir, ref_txt=None, attributes=None, with_conf=False, conf_threshold=0.4, filter_small=None):
    attributes = get_attributes(attributes)
    file_list = os.listdir(input_dir)
    if ref_txt is not None:
        ref_df = pd.read_csv(ref_txt, header=None, index_col=None, names=['file_name'])
        ref_list = [Path(file_name).stem for file_name in ref_df['file_name'].to_list()]
        file_list = [file_name for file_name in file_list if Path(file_name).stem in ref_list]
    counts = [0, 0, 0, 0]
    cat_counts = [0, 0, 0, 0]
    for file_name in tqdm(file_list):
        file_path = os.path.join(input_dir, file_name)
        df = get_yolo_label_df(file_path, mdet=True, attributes=attributes, with_conf=with_conf, conf_threshold=conf_threshold)
        if filter_small is not None:
            df = df.loc[(df['w']>filter_small) | (df['h']>filter_small)]
        for idx, row in df.iterrows():
            if int(row['abandonment']) > 0:
                counts[0] += 1
            if int(row['broken']) > 0:
                counts[1] += 1
            if int(row['corrosion']) > 0:
                counts[2] += 1
            if int(row['deformation']) > 0:
                counts[3] += 1
            if int(row['category']) not in [6]:
                if int(row['abandonment']) >0:
                    cat_counts[0] += 1
                if int(row['broken']) >0:
                    cat_counts[1] += 1
                if int(row['corrosion']) >0:
                    cat_counts[2] += 1
                if int(row['deformation']) >0:
                    cat_counts[3] += 1

    print(counts)
    print(cat_counts)


if __name__ == "__main__":
    pass

    #
    val_dir = r'/localnvme/project/ultralytics/runs/msegment/val699'
    data_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine'
    pred_dir = os.path.join(val_dir, 'labels')
    label_dir = os.path.join(data_dir, 'val_test', 'labels')
    save_dir = os.path.join(data_dir, 'val_test','result_analysis', os.path.basename(val_dir))
    att_file = os.path.join(data_dir, 'attribute.yaml')
    class_file = os.path.join(data_dir, 'class.txt')


    # pred_dir = r'/localnvme/project/ultralytics/runs/msegment/val653/labels'
    # data_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1023_src/val_80p_ref'

    # label_dir = os.path.join(data_dir, 'labels')
    # save_dir = os.path.join(data_dir, 'result_analysis', 'val653')
    # save_path = save_dir + '.csv'
    # att_file = os.path.join(data_dir, 'attribute.yaml')


    # get_all_high(label_dir, with_conf=False)

    # pred2cfm_risk(label_dir, pred_dir, save_dir, att_file, class_file, with_conf=True, iou_thr=0.3, conf_threshold=0.4,
    #               filter_small=None, keep='all', )
    # pred2cfm_risk(label_dir, pred_dir, save_dir, att_file, class_file, with_conf=True, iou_thr=0.3, conf_threshold=0.4,
    #               defect_conf_threshold=0.001, filter_small=0.05, keep='ignore_other_pred_frame', )
    # save_path = save_dir + '.csv'
    # eval_for_emsd_v2(label_dir, pred_dir, att_file, save_path, with_conf=True, conf_threshold=0.4, iou_thr=0.3, filter_small=0.05)



    # pred_dir = r'/localnvme/project/ultralytics/runs/msegment/val629/labels'
    # data_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1023_src/val_80p_ref'
    # pred_dir = r'/localnvme/project/ultralytics/runs/msegment/val628/labels'
    # data_dir = r'/localnvme/data/added_data/test_data/test_data_mseg_c5_l2_1021_broken_refine'

    # pred_dir = r'/localnvme/project/ultralytics/runs/msegment/val629/labels'
    # data_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1023_src/val_80p_ref'
    #
    # label_dir = os.path.join(data_dir, 'labels')
    # save_dir = os.path.join(data_dir, 'result_analysis', os.path.basename(os.path.dirname(pred_dir)))
    # save_path = save_dir + '.csv'
    # att_file = os.path.join(data_dir, 'attribute.yaml')
    # eval_for_emsd_v2(label_dir, pred_dir, att_file, save_path, with_conf=True, conf_threshold=0.001, iou_thr=0.5,
    #               filter_small=0.05)


    # val_dir = r'/localnvme/project/ultralytics/runs/msegment/val694'
    # pred_dir = os.path.join(val_dir, 'labels')
    # data_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine'
    # image_dir= os.path.join(data_dir, 'images')
    # label_dir= os.path.join(data_dir, 'labels')
    # save_dir = os.path.join(data_dir, 'result_analysis', os.path.basename(os.path.dirname(pred_dir)))
    # att_file = os.path.join(data_dir, 'attribute.yaml')
    # risks = ['abandonment', 'broken', 'corrosion', 'deformation']
    # confs = [
    #     # ['background', 'no'],
    #     # ['background', 'high'],
    #     ['no', 'background'],
    #     ['high', 'background'],
    #     # ['no', 'high'],
    #     # ['high', 'no'],
    # ]
    # print(save_dir)
    # for risk in risks:
    #     for pred_conf, label_conf in confs:
    #         find_files(pred_conf, label_conf, risk, image_dir, label_dir, pred_dir, save_dir, att_file, with_conf=True,
    #                    conf_threshold=0.4, iou_thr=0.3, filter_small=0.05)