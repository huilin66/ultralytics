# import os
# import sys

# from ultralytics import YOLO

# sys.path.append(r'/localnvme/project/ultralytics/my_tools')
# from my_tools.yolo_result_analysis import pred2cfm_risk, risk_analysis, find_files
# import pandas as pd

if __name__ == '__main__':
    pass
    # for i in range(628, 699):
    #     input_dir = r'/localnvme/project/ultralytics/runs/msegment/val'+str(i)
    #     labels_dir = os.path.join(input_dir, 'labels')
    #     a_path = os.path.join(input_dir, 'confusion_matrix_for_attribute_abandonment.csv')
    #     if os.path.exists(labels_dir) and os.path.exists(a_path) and len(os.listdir(labels_dir)) > 80:
    #         df = pd.read_csv(a_path)
    #         if 60<df.loc[1, 'high'] <100:
    #             print(i, df.loc[1, 'high'])
    precision = 0.597
    recall = 0.537
    f1 = 2* precision * recall / (precision + recall) 
    print(precision, recall, f1)
    # import numpy as np
    # import onnxruntime as ort

    # import torch
    # data = r'/scrinvme/huilin/isds/other_data/upload1014/Val_set/test/cam1/DA5148683_20250812150605700.jpg'
    # torch_model_path = r'/localnvme/project/ultralytics/runs/msegment/fusedata7961_mseg_c5_l2_1111_v15_test-[yolov10x-mseg-dlka3res-7-unet-sep]3/weights/best.pt'
    # onnx_model_path = r'/localnvme/project/ultralytics/runs/msegment/fusedata7961_mseg_c5_l2_1111_v15_test-[yolov10x-mseg-dlka3res-7-unet-sep]3/weights/best.onnx'
    # torch_model = YOLO(torch_model_path)
    # result = torch_model(data)
    # print(result[0].boxes)


    # onnx_model = YOLO(onnx_model_path)
    # result = onnx_model(data)
    # print(result[0].boxes)

    # sess = ort.InferenceSession(onnx_model_path)
    # onnx_input_name = sess.get_inputs()[0].name
    # y_onnx = sess.run(None, {onnx_input_name: data.numpy()})[0]
    #
    # print('max diff', np.max(np.abs(y_onnx - result_torch)))

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


    # pred_dir = r'/localnvme/project/ultralytics/runs/msegment/val627/labels'
    # data_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1023_src'
    # # pred_dir = r'/localnvme/project/ultralytics/runs/msegment/val629/labels'
    # # data_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1023_src/val_80p_ref'
    # image_dir= os.path.join(data_dir, 'images')
    # label_dir= os.path.join(data_dir, 'labels')
    # save_dir = os.path.join(data_dir, 'result_analysis', os.path.basename(os.path.dirname(pred_dir)))
    # att_file = os.path.join(data_dir, 'attribute.yaml')
    # risks = ['abandonment', 'broken', 'corrosion', 'deformation']
    # confs = [
    #     ['background', 'no'],
    #     ['background', 'high'],
    #     ['no', 'background'],
    #     ['high', 'background'],
    #     ['no', 'high'],
    #     ['high', 'no'],
    # ]
    # print(save_dir)
    # for risk in risks:
    #     for pred_conf, label_conf in confs:
    #         find_files(pred_conf, label_conf, risk, image_dir, label_dir, pred_dir, save_dir, att_file, with_conf=True,
    #                    conf_threshold=0.4, iou_thr=0.3, filter_small=0.05)