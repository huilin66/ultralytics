import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_iou(box1, box2):
    box1 = [box1[0]-box1[2]/2, box1[1]-box1[3]/2, box1[0]+box1[2]/2, box1[1]+box1[3]/2]
    box2 = [box2[0]-box2[2]/2, box2[1]-box2[3]/2, box2[0]+box2[2]/2, box2[1]+box2[3]/2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2] - box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def parse_yolo_txt(txt_path, seg=True, with_conf=False, with_att=True):
    data = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            class_id = int(parts[0])
            if with_conf:
                conf = parts.pop(-1)
            else:
                conf = None
            if with_att:
                att_len = int(parts[1])
                att = list(map(int, parts[2:2+att_len]))

                polygon = list(map(float, parts[2+att_len:]))
                data.append((class_id, att, polygon))
            else:
                polygon = list(map(float, parts[1:]))
                data.append((class_id, polygon))
    return data

def results_analysis(labels_dir, pred_dir, seg=True, with_conf=False, with_att=True):
    pass
    file_list = os.listdir(pred_dir)
    columns = ['file_name', 'label_class', 'label_att', 'label_conf', 'label_polygon', 'pred_class', 'pred_att', 'pred_conf', 'pred_polygon', 'iou']
    df = pd.DataFrame(None, columns=columns)
    for file_name in tqdm(file_list):
        label_path = os.path.join(labels_dir, file_name)
        pred_path = os.path.join(pred_dir, file_name)
        label_info = parse_yolo_txt(label_path, seg=seg, with_conf=with_conf, with_att=with_att)
        pred_info = parse_yolo_txt(pred_path, seg=seg, with_conf=with_conf, with_att=with_att)



if __name__ == '__main__':
    pass
    labels_dir = r'/nfsv4/23039356r/data/billboard/bd_data/data127/labels'
    pred_dir = r'/nfsv4/23039356r/repository/ultralytics/runs/msegment/val17/labels'
    results_analysis(labels_dir, pred_dir)