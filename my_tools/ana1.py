import os
import sys
sys.path.append(r'./')
from my_tools.yolo_result_analysis import get_all_high, pred2cfm_risk, risk_analysis, select_val


if __name__ == '__main__':
    pass
    # val_dir = r'/localnvme/project/ultralytics/runs/msegment/val717'
    # val_name = 'val_80p_ref'
    val_dir = r'/localnvme/project/ultralytics/runs/msegment/val724'
    val_name = 'val_80p'
    data_dir = r'/localnvme/data/billboard/all_data/mseg_c5_l2/data7961_mseg_c5_l2_1113_v17'
    pred_dir = os.path.join(val_dir, 'labels')
    label_dir = os.path.join(data_dir, val_name, 'labels')
    save_dir = os.path.join(data_dir, val_name, 'result_analysis', val_dir)
    att_file = os.path.join(data_dir, 'attribute.yaml')
    class_file = os.path.join(data_dir, 'class.txt')

    if not os.path.exists(label_dir) or len(os.listdir(label_dir)) == 0:
        select_val(data_dir, val_txt=f'{val_name}.txt')

    # get_all_high(label_dir, attributes=att_file)
    # get_all_high(label_dir, attributes=att_file, filter_small=0.05)
    # risk_analysis(val_dir)

    params = [
        # [0.5, 0.001, 0.001, None],
        # [0.5, 0.001, 0.001, 0.05],
        # [0.3, 0.001, 0.001, 0.05],
        # [0.3, 0.1, 0.001, 0.05],
        # [0.3, 0.2, 0.001, 0.05],
        # [0.3, 0.3, 0.001, 0.05],
        # [0.3, 0.4, 0.001, 0.05],
        # [0.3, 0.5, 0.001, 0.05],
        # [0.3, 0.6, 0.001, 0.05],
        # [0.3, 0.7, 0.001, 0.05],

        # [0.3, 0.4, None, 0.05],
        [0.3, 0.4, 0.001, 0.05],
    ]
    # for param in params:
    #     pred2cfm_risk(label_dir, pred_dir, save_dir, att_file, class_file, with_conf=True, iou_thr=param[0],
    #                   conf_threshold=param[1], defect_conf_threshold=param[2], filter_small=param[3],
    #                   keep='ignore_other_pred_frame', show_list=['risk3'])

