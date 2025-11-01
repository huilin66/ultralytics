import os
import sys
sys.path.append(r'./')
from my_tools.yolo_result_analysis import pred2cfm_risk, risk_analysis, select_val


if __name__ == '__main__':

    val_dir = r'/localnvme/project/ultralytics/runs/msegment/val694'
    data_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine'
    val_name = ''

    pred_dir = os.path.join(val_dir, 'labels')
    label_dir = os.path.join(data_dir, val_name, 'labels')
    save_dir = os.path.join(data_dir, val_name, 'result_analysis', val_dir)
    att_file = os.path.join(data_dir, 'attribute.yaml')
    class_file = os.path.join(data_dir, 'class.txt')

    pred2cfm_risk(label_dir, pred_dir, save_dir, att_file, class_file, with_conf=True, iou_thr=0.3, conf_threshold=0.4,
                  filter_small=None, keep='all', )

