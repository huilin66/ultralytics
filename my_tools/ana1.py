import os
import sys
sys.path.append(r'./')
from my_tools.yolo_result_analysis import pred2cfm_risk, risk_analysis, select_val


if __name__ == '__main__':
    pass
    
    val_dir = r'/localnvme/project/ultralytics/runs/msegment/val700'
    data_dir = r'/localnvme/data/billboard/fused_data/fusedata7961_mseg_c5_l2_1030_abandonment_refine_test'
    val_name = 'val_test'
    pred_dir = os.path.join(val_dir, 'labels')
    label_dir = os.path.join(data_dir, val_name, 'labels')
    save_dir = os.path.join(data_dir, val_name, 'result_analysis', val_dir)
    att_file = os.path.join(data_dir, 'attribute.yaml')

    if not os.path.exists(label_dir) or len(os.listdir(label_dir)) == 0:
        select_val(data_dir, val_txt=f'{val_name}.txt')
    risk_analysis(val_dir)
    pred2cfm_risk(label_dir, pred_dir, cfm_num=3, save_dir=save_dir, attributes=att_file, with_conf=True, conf_threshold=0.001, iou_thr=0.5, filter_small=None, keep='ignore_other_pred_frame', )
    pred2cfm_risk(label_dir, pred_dir, cfm_num=3, save_dir=save_dir, attributes=att_file, with_conf=True, conf_threshold=0.001, iou_thr=0.5, filter_small=0.05, keep='ignore_other_pred_frame', )
    pred2cfm_risk(label_dir, pred_dir, cfm_num=3, save_dir=save_dir, attributes=att_file, with_conf=True, conf_threshold=0.001, iou_thr=0.3, filter_small=0.05, keep='ignore_other_pred_frame', )
    pred2cfm_risk(label_dir, pred_dir, cfm_num=3, save_dir=save_dir, attributes=att_file, with_conf=True, conf_threshold=0.01, iou_thr=0.3, filter_small=0.05, keep='ignore_other_pred_frame', )
    pred2cfm_risk(label_dir, pred_dir, cfm_num=3, save_dir=save_dir, attributes=att_file, with_conf=True, conf_threshold=0.1, iou_thr=0.3, filter_small=0.05, keep='ignore_other_pred_frame', )
    pred2cfm_risk(label_dir, pred_dir, cfm_num=3, save_dir=save_dir, attributes=att_file, with_conf=True, conf_threshold=0.2, iou_thr=0.3, filter_small=0.05, keep='ignore_other_pred_frame', )
    pred2cfm_risk(label_dir, pred_dir, cfm_num=3, save_dir=save_dir, attributes=att_file, with_conf=True, conf_threshold=0.3, iou_thr=0.3, filter_small=0.05, keep='ignore_other_pred_frame', )
    pred2cfm_risk(label_dir, pred_dir, cfm_num=3, save_dir=save_dir, attributes=att_file, with_conf=True, conf_threshold=0.4, iou_thr=0.3, filter_small=0.05, keep='ignore_other_pred_frame', )
    pred2cfm_risk(label_dir, pred_dir, cfm_num=3, save_dir=save_dir, attributes=att_file, with_conf=True, conf_threshold=0.5, iou_thr=0.3, filter_small=0.05, keep='ignore_other_pred_frame', )
    pred2cfm_risk(label_dir, pred_dir, cfm_num=3, save_dir=save_dir, attributes=att_file, with_conf=True, conf_threshold=0.6, iou_thr=0.3, filter_small=0.05, keep='ignore_other_pred_frame', )
    pred2cfm_risk(label_dir, pred_dir, cfm_num=3, save_dir=save_dir, attributes=att_file, with_conf=True, conf_threshold=0.7, iou_thr=0.5, filter_small=0.05, keep='ignore_other_pred_frame', )
