import os
import yaml
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from seg_yolo2coco_analysis import seg_analysis, select_val
from risk_confusion_matrix_analysis import risk_analysis

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

def get_yolo_label_df(gt_path, mdet=False, attributes=None, with_track_id=False, with_object_id=False, with_conf=False, conf_threshold=0.001):
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
    if with_conf:
        df = df[df['conf'] >= conf_threshold]
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



def pred2cfm_risk(label_dir, pred_dir, save_dir, mdet=True, attributes=None, with_conf=True, conf_threshold=0.001, iou_thr=0.5, keep='all', filter_small=None):
    os.makedirs(save_dir, exist_ok=True)
    attributes = get_attributes(attributes)
    txt_list = os.listdir(label_dir)
    df_a = pd.DataFrame([[0, 0], [0, 0]], columns=['label_no', 'label_high'], index=['pred_no', 'pred_high'])
    df_b = pd.DataFrame([[0, 0], [0, 0]], columns=['label_no', 'label_high'], index=['pred_no', 'pred_high'])
    df_c = pd.DataFrame([[0, 0], [0, 0]], columns=['label_no', 'label_high'], index=['pred_no', 'pred_high'])
    df_d = pd.DataFrame([[0, 0], [0, 0]], columns=['label_no', 'label_high'], index=['pred_no', 'pred_high'])
    for txt_name in tqdm(txt_list):
        label_path = os.path.join(label_dir, txt_name)
        pred_path = os.path.join(pred_dir, txt_name)

        df_label = get_yolo_label_df(label_path, mdet=mdet, attributes=attributes, with_object_id=True)
        if not os.path.exists(pred_path):
            df_pred = pd.DataFrame(columns=df_label.columns)
        else:
            df_pred = get_yolo_label_df(pred_path, mdet=mdet, attributes=attributes, with_object_id=True, with_conf=with_conf, conf_threshold=conf_threshold)

        if filter_small is not None:
            df_label = df_label.loc[(df_label['w']>filter_small) | (df_label['h']>filter_small)]
            df_pred = df_pred.loc[(df_pred['w']>filter_small) | (df_pred['h']>filter_small)]

        df_match = match_and_merge(df_pred, df_label, iou_thr=iou_thr, att_list=attributes)
        
        for idx, row in df_match.iterrows():
            pass
            if keep=='all':
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
            elif keep=='iou':
                if pd.isna(row['gt_category']) or pd.isna(row['pred_category']):
                    continue
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
            elif keep=='correct':
                if pd.isna(row['gt_category']) or pd.isna(row['pred_category']) and row['gt_category']!=row['pred_category']:
                    continue
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
    df_a.columns = ['False', 'high']
    df_a.index = ['False', 'high']
    df_b.columns = ['False', 'high']
    df_b.index = ['False', 'high']
    df_c.columns = ['False', 'high']
    df_c.index = ['False', 'high']
    df_d.columns = ['False', 'high']
    df_d.index = ['False', 'high']
    df_a.to_csv(os.path.join(save_dir, "confusion_matrix_for_attribute_abandonment.csv"), header=True, index=True)
    df_b.to_csv(os.path.join(save_dir, "confusion_matrix_for_attribute_broken.csv"), header=True, index=True)
    df_c.to_csv(os.path.join(save_dir, "confusion_matrix_for_attribute_corrosion.csv"), header=True, index=True)
    df_d.to_csv(os.path.join(save_dir, "confusion_matrix_for_attribute_deformation.csv"), header=True, index=True)
            
if __name__ == "__main__":
    pass
    # pred_dir = r'/localnvme/project/ultralytics/runs/msegment/val607/labels'
    # data_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1023_src'
    # save_dir = os.path.join(data_dir, 'result_analysis', 'val607')
    # label_dir = os.path.join(data_dir, 'val_80p_ref', 'labels')

    # pred_dir = r'/localnvme/project/ultralytics/runs/msegment/val614/labels'
    # data_dir = r'/localnvme/data/added_data/test_data/test_data_mseg_c5_l2_1021_broken_refine'
    # save_dir = os.path.join(data_dir, 'result_analysis', 'val614')
    # label_dir = os.path.join(data_dir, 'labels')
    # att_file = os.path.join(data_dir, 'attribute.yaml')


    # pred2cfm_risk(label_dir, pred_dir, save_dir=save_dir, mdet=True, attributes=att_file, with_conf=True, conf_threshold=0.001, iou_thr=0.5, keep='all')
    # risk_analysis(save_dir)
    # pred2cfm_risk(label_dir, pred_dir, save_dir=save_dir, mdet=True, attributes=att_file, with_conf=True, conf_threshold=0.001, iou_thr=0.5, keep='correct')
    # risk_analysis(save_dir)


    # pred_dir = r'/localnvme/project/ultralytics/runs/msegment/val615/labels'
    # data_dir = r'/localnvme/data/added_data/test_data/test_data_mseg_c5_l2_1021_broken_refine'
    # save_dir = os.path.join(data_dir, 'result_analysis', 'val615')
    # label_dir = os.path.join(data_dir, 'labels')
    # att_file = os.path.join(data_dir, 'attribute.yaml')

    # pred2cfm_risk(label_dir, pred_dir, save_dir=save_dir, mdet=True, attributes=att_file, with_conf=True, conf_threshold=0.001, iou_thr=0.5, keep='all')
    # risk_analysis(save_dir)

    pred_dir = r'/localnvme/project/ultralytics/runs/msegment/val610/labels'
    data_dir = r'/localnvme/data/added_data/test_data/test_data_mseg_c5_l2_1021_broken_refine'
    save_dir = os.path.join(data_dir, 'result_analysis', 'val610')
    label_dir = os.path.join(data_dir, 'labels')
    att_file = os.path.join(data_dir, 'attribute.yaml')

    # select_val(data_dir, val_txt='val_80p_ref.txt')
    pred2cfm_risk(label_dir, pred_dir, save_dir=save_dir, mdet=True, attributes=att_file, with_conf=True, conf_threshold=0.001, iou_thr=0.5, keep='all')
    risk_analysis(save_dir)

