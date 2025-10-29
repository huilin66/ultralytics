import os
import sys
sys.path.append(r'/localnvme/project/ultralytics/my_tools')
from my_tools.yolo_result_analysis import pred2cfm_risk, risk_analysis, find_files

if __name__ == '__main__':
    pass
    # # pred_dir = r'/localnvme/project/ultralytics/runs/msegment/val624/labels'
    # # data_dir = r'/localnvme/data/added_data/test_data/test_data_mseg_c5_l2_1021_broken_refine'
    # pred_dir = r'/localnvme/project/ultralytics/runs/msegment/val623/labels'
    # data_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1023_src/val_80p_ref'
    # label_dir = os.path.join(data_dir, 'labels')
    # save_dir = os.path.join(data_dir, 'result_analysis', os.path.dirname(os.path.dirname(pred_dir)))
    # att_file = os.path.join(data_dir, 'attribute.yaml')
    #
    #
    #
    # pred2cfm_risk(label_dir, pred_dir, cfm_num=3, save_dir=save_dir, attributes=att_file, with_conf=True, conf_threshold=0.3, iou_thr=0.3, filter_small=0.05, keep='all', )
    # risk_analysis(save_dir, rm_bg=True)
    #
    # pred2cfm_risk(label_dir, pred_dir, cfm_num=3, save_dir=save_dir, attributes=att_file, with_conf=True, conf_threshold=0.3, iou_thr=0.3, filter_small=0.05, keep='ignore_other', )
    # risk_analysis(save_dir, rm_bg=True)


    pred_dir = r'/localnvme/project/ultralytics/runs/msegment/val627/labels'
    data_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1023_src'
    # pred_dir = r'/localnvme/project/ultralytics/runs/msegment/val629/labels'
    # data_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1023_src/val_80p_ref'
    image_dir= os.path.join(data_dir, 'images')
    label_dir= os.path.join(data_dir, 'labels')
    save_dir = os.path.join(data_dir, 'result_analysis', os.path.basename(os.path.dirname(pred_dir)))
    att_file = os.path.join(data_dir, 'attribute.yaml')
    risks = ['abandonment', 'broken', 'corrosion', 'deformation']
    confs = [
        ['background', 'no'],
        ['background', 'high'],
        ['no', 'background'],
        ['high', 'background'],
        ['no', 'high'],
        ['high', 'no'],
    ]
    print(save_dir)
    for risk in risks:
        for pred_conf, label_conf in confs:
            find_files(pred_conf, label_conf, risk, image_dir, label_dir, pred_dir, save_dir, att_file, with_conf=True,
                       conf_threshold=0.4, iou_thr=0.3, filter_small=0.05)