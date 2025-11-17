import os
import sys
sys.path.append(r'./')
from my_tools.yolo_result_analysis import pred2cfm_risk, risk_analysis, select_val, pred2cfm_risk_single


if __name__ == '__main__':
    pass

    val_dir = r'/localnvme/project/ultralytics/runs/msegment/val844'
    data_dir = r'/localnvme/data/billboard/all_data/mseg_c5_l2/data7961_mseg_c5_l2_1113_v17'
    val_name = 'val_test'

    pred_dir = os.path.join(val_dir, 'labels')
    label_dir = os.path.join(data_dir, val_name, 'labels')
    save_dir = os.path.join(data_dir, val_name, 'result_analysis', val_dir)
    att_file = os.path.join(data_dir, 'attribute.yaml')
    class_file = os.path.join(data_dir, 'class.txt')

    pred2cfm_risk(label_dir, pred_dir, save_dir, att_file, class_file, with_conf=True, iou_thr=0.3, conf_threshold=0.4,
                  defect_conf_threshold=0.1, filter_small=0.05, keep='all', show_list=['risk2'])

    pred2cfm_risk(label_dir, pred_dir, save_dir, att_file, class_file, with_conf=True, iou_thr=0.3, conf_threshold=0.4,
                  defect_conf_threshold=0.1, filter_small=None, keep='all', show_list=['risk2'])


    # val_dir = r'/localnvme/project/ultralytics/runs/msegment/val778'
    # data_dir = r'/localnvme/data/billboard/all_data/mseg_c5_l2/data7961_mseg_c5_l2_1113_v17'
    # val_name = 'val_test'
    #
    # pred_dir = os.path.join(val_dir, 'labels')
    # label_dir = os.path.join(data_dir, val_name, 'labels')
    # save_dir = os.path.join(data_dir, val_name, 'result_analysis', val_dir)
    # att_file = os.path.join(data_dir, 'attribute.yaml')
    # class_file = os.path.join(data_dir, 'class.txt')
    #
    # pred2cfm_risk(label_dir, pred_dir, save_dir, att_file, class_file, with_conf=True, iou_thr=0.3, conf_threshold=0.4,
    #               defect_conf_threshold=0.1, filter_small=0.05, keep='ignore_other_pred_frame', show_list=['risk3'])


    # val_dir_d = r'/localnvme/project/ultralytics/runs/msegment/val768'
    # val_dir_b = r'/localnvme/project/ultralytics/runs/msegment/val769'
    # val_dir_a = r'/localnvme/project/ultralytics/runs/msegment/val770'
    # val_dir_c = r'/localnvme/project/ultralytics/runs/msegment/val771'
    # pred_dir_a = os.path.join(val_dir_a, 'labels')
    # pred_dir_b = os.path.join(val_dir_b, 'labels')
    # pred_dir_c = os.path.join(val_dir_c, 'labels')
    # pred_dir_d = os.path.join(val_dir_d, 'labels')
    # pred_dir_all = os.path.join(data_dir, val_name, 'result_analysis', f'{os.path.basename(val_dir_d)}_{os.path.basename(val_dir_c)}')
    # save_dir = os.path.join(data_dir, val_name, 'result_analysis', f'{os.path.basename(val_dir_d)}_{os.path.basename(val_dir_c)}_ana')
    # pred2cfm_risk_single(label_dir, pred_dir_a, pred_dir_b, pred_dir_c, pred_dir_d, pred_dir_all,
    #                      save_dir, att_file, class_file, with_conf=True, iou_thr=0.3, conf_threshold=0.4, only_save=False,
    #                      defect_conf_threshold=0.1, filter_small=0.05, keep='ignore_other_pred_frame', show_list=['risk2'])
