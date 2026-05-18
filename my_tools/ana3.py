import os
import sys
sys.path.append(r'./')
from my_tools.yolo_result_analysis import pred2cfm_risk, risk_analysis, select_val


if __name__ == '__main__':
    pass
    
    # val_dir = r'/localnvme/project/ultralytics/runs/msegment/val709'
    # val_name = 'val_80p_ref'

    # val_dir = r'/localnvme/project/ultralytics/runs/msegment/val715'
    # val_name = 'val_test'

    # data_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1030_v4'


    # val_dir = r'/localnvme/project/ultralytics/runs/msegment/val925'
    # data_dir = r'/localnvme/data/billboard/all_data/mseg_c5_l2/data7961_mseg_c5_l2_1113_v17'
    # val_name = 'val_test'

    # # val_dir = r'/localnvme/project/ultralytics/runs/msegment/val653'
    # # data_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1023_src'
    # # val_name = 'val_80p_ref'

    # pred_dir = os.path.join(val_dir, 'labels')
    # label_dir = os.path.join(data_dir, val_name, 'labels')
    # save_dir = os.path.join(data_dir, val_name, 'result_analysis', val_dir)
    # att_file = os.path.join(data_dir, 'attribute.yaml')
    # class_file = os.path.join(data_dir, 'class.txt')

    # if not os.path.exists(label_dir) or len(os.listdir(label_dir)) == 0:
    #     select_val(data_dir, val_txt=f'{val_name}.txt')
    # pred2cfm_risk(label_dir, pred_dir, save_dir, att_file, class_file, with_conf=True, iou_thr=0.3, conf_threshold=0.4,
    #               filter_small=None, keep='all', show_list=['risk3', 'risk2', 'seg2'])


    # pred2cfm_risk(label_dir, pred_dir, save_dir, att_file, class_file, with_conf=True, iou_thr=0.3, conf_threshold=0.4,
    #               filter_small=0.05, keep='ignore_other_pred_frame', show_list=['risk2'])
    # pred2cfm_risk(label_dir, pred_dir, save_dir, att_file, class_file, with_conf=True, iou_thr=0.3, conf_threshold=0.3,
    #               filter_small=0.05, keep='ignore_other_pred_frame', show_list=['risk2'])
    # pred2cfm_risk(label_dir, pred_dir, save_dir, att_file, class_file, with_conf=True, iou_thr=0.3, conf_threshold=0.4,
    #               defect_conf_threshold = 0.1,
    #               filter_small=0.05, keep='ignore_other_pred_frame', show_list=['risk3'])
    # pred2cfm_risk(label_dir, pred_dir, save_dir, att_file, class_file, with_conf=True, iou_thr=0.3, conf_threshold=0.3,
    #               defect_conf_threshold = 0.1,
    #               filter_small=0.05, keep='ignore_other_pred_frame', show_list=['risk3'])
    # pred2cfm_risk(label_dir, pred_dir, save_dir, att_file, class_file, with_conf=True, iou_thr=0.3, conf_threshold=0.4,
    #               defect_conf_threshold = 0.01,
    #               filter_small=0.05, keep='ignore_other_pred_frame', show_list=['risk2'])
    # pred2cfm_risk(label_dir, pred_dir, save_dir, att_file, class_file, with_conf=True, iou_thr=0.3, conf_threshold=0.4,
    #               defect_conf_threshold = 0.001,
    #               filter_small=0.05, keep='ignore_other_pred_frame', show_list=['risk2'])
    # pred2cfm_risk(label_dir, pred_dir, save_dir, att_file, class_file, with_conf=True, iou_thr=0.3, conf_threshold=0.4,
    #               filter_small=0.05, keep='ignore_other_pred_frame', show_list=['risk3', 'risk2', 'seg2'])


    data_dir = r'/localnvme/data/billboard/all_data/mseg_c5_l2/data7961_mseg_c5_l2_1113_v17'
    att_file = os.path.join(data_dir, 'attribute.yaml')
    class_file = os.path.join(data_dir, 'class.txt')

    # val_dir = r'/localnvme/project/ultralytics/runs/msegment/val929'
    # val_name = 'val_test'

    val_dir = r'/localnvme/project/ultralytics/runs/msegment/val653'
    val_name = 'val_80p_ref'

    pred_dir = os.path.join(val_dir, 'labels')
    label_dir = os.path.join(data_dir, val_name, 'labels')
    save_dir = os.path.join(data_dir, val_name, 'result_analysis', val_dir)


    if not os.path.exists(label_dir) or len(os.listdir(label_dir)) == 0:
        select_val(data_dir, val_txt=f'{val_name}.txt')
    pred2cfm_risk(label_dir, pred_dir, save_dir, att_file, class_file, with_conf=True, iou_thr=0.3, conf_threshold=0.4,
                defect_conf_threshold = 0.1, filter_small=0.05, keep='ignore_other_pred_frame', show_list=['risk3'])