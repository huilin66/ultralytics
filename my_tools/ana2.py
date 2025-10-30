import os
import sys
sys.path.append(r'./')
from my_tools.yolo_result_analysis import find_files


if __name__ == '__main__':
    pass
    
    val_dir = r'/localnvme/project/ultralytics/runs/msegment/val694'
    pred_dir = os.path.join(val_dir, 'labels')
    data_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine'
    image_dir= os.path.join(data_dir, 'images')
    label_dir= os.path.join(data_dir, 'labels')
    save_dir = os.path.join(data_dir, 'result_analysis', os.path.basename(os.path.dirname(pred_dir)))
    att_file = os.path.join(data_dir, 'attribute.yaml')
    risks = [
        # 'abandonment',
        'broken', 'corrosion', 'deformation']
    confs = [
        # ['background', 'no'],
        # ['background', 'high'],
        # ['no', 'background'],
        # ['high', 'background'],
        # ['no', 'high'],
        ['high', 'no'],
    ]
    print(save_dir)
    for risk in risks:
        for pred_conf, label_conf in confs:
            find_files(pred_conf, label_conf, risk, image_dir, label_dir, pred_dir, save_dir, att_file, with_conf=True,
                       conf_threshold=0.4, iou_thr=0.3, filter_small=0.05)