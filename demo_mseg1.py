import demo_base
import torch

demo_base.TASK = 'msegment'
demo_base.EPOCHS = 100
demo_base.IMGSZ = 960
demo_base.BATCH_SIZE = 32
demo_base.DEVICE = torch.device('cuda:0')
demo_base.DATA = "fusedata2419_mseg_c5_0730.yaml"
# demo_base.CONF_VAL = 0.5
# demo_base.CONF_PREDICT = 0.5

MODEL1 = 'yolov8x-mseg-7.yaml'
MODEL2 = 'yolov8x-mseg-dlka3res-7.yaml'
MODEL3 = 'yolov10x-mseg-dlka3res-7.yaml'
MODEL4 = 'yolov10x-mseg-dlka3res-7-unet.yaml'
MODEL6 = 'yolov10x-mseg-dlka3res-7-unet-sep.yaml'
MODEL6A = 'yolov10x-mseg-dlka3res-7-unet-sep-sa.yaml'
SEG_WEIGHT8 = "runs/segment/fusedata5894_seg_c5_0822_80p-[yolov8x-seg-dlka3res]2/weights/best.pt"
SEG_WEIGHT10 = "/localnvme/project/ultralytics/runs/segment/fusedata7961_seg_c5_l2_1022_re_80p_ref-[yolov10x-seg-dlka3res]8/weights/best.pt"
SEG_WEIGHT10_v12 = r'runs/segment/fusedata7961_seg_c5_1106_v12_src-[yolov10x-seg-dlka3res]/weights/best.pt'
DATA0 = "fusedata7961_mseg_c5_l2_1112_v16_test.yaml"
DATA1 = "fusedata7961_mseg_c5_l2_1113_v17_test_rs.yaml"

DATA_TEST = "data80_v21_b_test.yaml"
MODEL4_S = 'yolov10x-mseg-dlka3res-7-unet-single-ContrastLoss.yaml'
MODEL4_ST = 'yolov10x-mseg-dlka3res-7-unet-single-texture-ContrastLoss.yaml'
DATA_SB1 = "fusedata7961_mseg_c5_l2_1123_v22_sb_test_broken_syn_v4.yaml"
DATA_SB2 = "fusedata7961_mseg_c5_l2_1117_v20_sb_test_broken_syn_v1.yaml"

if __name__ == '__main__':
    pass
    # demo_base.yolo10(
    #     MODEL4_ST, weight_path=SEG_WEIGHT10, data=DATA_SB1, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=10, save_period=1, contrastive_loss=True, contrastive_loss_weight=0.5
    # )
    # demo_base.yolo10(
    #     MODEL4_ST, weight_path=SEG_WEIGHT10, data=DATA_SB1, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=10, save_period=1, contrastive_loss=True, contrastive_loss_weight=0.1
    # )
    # demo_base.yolo10(
    #     MODEL4_S, weight_path=SEG_WEIGHT10, data=DATA_SB2, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=20,
    # )

    # demo_base.yolo10(
    #     MODEL6, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=5,
    # )
    # demo_base.yolo10(
    #     MODEL6, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=10,
    # )
    # demo_base.yolo10(
    #     MODEL6, weight_path=SEG_WEIGHT10_v12, data=DATA0, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=20,
    # )
    # demo_base.yolo10(
    #     MODEL6, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=20,
    # )

    # demo_base.yolo10(
    #     MODEL6, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=50,
    # )
    # demo_base.yolo10(
    #     MODEL6, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=100,
    # )
    # demo_base.yolo10(
    #     MODEL4, weight_path=SEG_WEIGHT10, data=DATA_SB, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=5,
    # )
    # demo_base.yolo10(
    #     MODEL4, weight_path=SEG_WEIGHT10, data=DATA_SC, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=5,
    # )
    # demo_base.yolo10(
    #     MODEL4, weight_path=SEG_WEIGHT10, data=DATA_SD, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=5,
    # )
    # demo_base.yolo10(
    #     MODEL6A, weight_path=SEG_WEIGHT10, data=DATA_SA, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=5, name='debug', freeze_att_nums=[0, 1, 3]
    # )
    # demo_base.yolo10(
    #     MODEL6, weight_path=SEG_WEIGHT10, data=DATA_SB, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=5,
    # )
    # demo_base.yolo10(
    #     MODEL6, weight_path=SEG_WEIGHT10, data=DATA_SC, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=5,
    # )
    # demo_base.yolo10(
    #     MODEL6, weight_path=SEG_WEIGHT10, data=DATA_SD, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=5,
    # )
    # demo_base.yolo10(
    #     MODEL3, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=5,
    # )

    # demo_base.yolo10(
    #     MODEL4, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=5,
    # )
    # demo_base.yolo10(
    #     MODEL4, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=10,
    # )
    # demo_base.yolo10(
    #     MODEL3, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
    #     mloss_mask=True, mloss_weight=False, mloss_enlarge=0,
    # )

    # val_name = 'runs/msegment/fusedata7961_mseg_c5_l2_1029_abandonment_refine_test-[yolov10x-mseg-dlka3res-7-unet]4/weights/last.pt'
    # data = 'fusedata7961_mseg_c5_l2_1029_abandonment_refine_80p_ref_src.yaml'
    # demo_base.model_val(val_name, save_txt=True, save_conf=True, data=data, weight_name=False)
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1015_80p_ref-[yolov10x-mseg-dlka3res-7]')
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1015_80p_ref-[yolov10x-mseg-dlka3res-7]2')
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1015_80p_ref-[yolov10x-mseg-dlka3res-7]3')
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1015_80p_ref-[yolov10x-mseg-dlka3res-7]4')
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1021_80p_ref-[yolov10x-mseg-dlka3res-7]')
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1021_80p_ref-[yolov10x-mseg-dlka3res-7]2')
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1021_80p_ref-[yolov10x-mseg-dlka3res-7]3')
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1021_80p_ref-[yolov10x-mseg-dlka3res-7]4')
    #
    #
    # val_name = 'fusedata7961_mseg_c5_l2_1015_80p_ref-[yolov10x-mseg-dlka3res-7]2'
    # demo_base.model_val(val_name, )
    # demo_base.model_val(val_name, filter_small=0.05)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.1)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.2)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.5)
    #
    # val_name = 'fusedata7961_mseg_c5_l2_1021_80p_ref-[yolov10x-mseg-dlka3res-7]2'
    # demo_base.model_val(val_name, )
    # demo_base.model_val(val_name, filter_small=0.05)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.1)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.2)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.5)
    #
    #
    # val_name = 'fusedata7961_mseg_c5_l2_1015_80p_ref-[yolov10x-mseg-dlka3res-7]2'
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.1)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.2)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.3)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.4)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.5)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.6)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.7)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.8)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.9)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=2.0)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=[1.0, 1.4, 1.6, 1.3])
    #
    # val_name = 'fusedata7961_mseg_c5_l2_1021_80p_ref-[yolov10x-mseg-dlka3res-7]2'
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.1)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.2)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.3)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.4)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.5)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.6)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.7)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.8)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.9)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=2.0)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=[1.0, 1.6, 1.4, 1.8])
    #
    #
    # val_name = 'fusedata7961_mseg_c5_l2_1015_80p_ref-[yolov10x-mseg-dlka3res-7]2'
    # data = 'defect_test_1021.yaml'
    # demo_base.model_val(val_name, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.1, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.2, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.5, data=data)
    #
    # val_name = 'fusedata7961_mseg_c5_l2_1021_80p_ref-[yolov10x-mseg-dlka3res-7]2'
    # data = 'defect_test_1023.yaml'
    # demo_base.model_val(val_name, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.1, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.2, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.5, data=data)
    #
    # val_name = 'fusedata7961_mseg_c5_l2_1021_80p_ref-[yolov10x-mseg-dlka3res-7]2'
    # data = 'defect_test_1023_defect.yaml'
    # demo_base.model_val(val_name, filter_small=0.05, data=data)
    #
    #
    # val_name = 'fusedata7961_mseg_c5_l2_1015_80p_ref-[yolov10x-mseg-dlka3res-7]2'
    # data = 'defect_test_1021.yaml'
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.1, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.2, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.3, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.4, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.5, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.6, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.7, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.8, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.9, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=2.0, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=[1.0, 1.4, 1.6, 1.3], data=data)
    #
    # val_name = 'fusedata7961_mseg_c5_l2_1021_80p_ref-[yolov10x-mseg-dlka3res-7]2'
    # data = 'defect_test_1023.yaml'
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.1, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.2, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.3, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.4, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.5, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.6, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.7, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.8, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=1.9, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=2.0, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=[1.0, 1.4, 1.6, 1.3], data=data)
    #
    # data = 'defect_test_1021.yaml'
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1022_80p_ref-[yolov10x-mseg-dlka3res-7]2', data=data)
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1022_80p_ref-[yolov10x-mseg-dlka3res-7]3', data=data)
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1022_80p_ref-[yolov10x-mseg-dlka3res-7]4', data=data)
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1022_80p_ref-[yolov10x-mseg-dlka3res-7]2', filter_small=0.05, data=data)
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1022_80p_ref-[yolov10x-mseg-dlka3res-7]3', filter_small=0.05, data=data)
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1022_80p_ref-[yolov10x-mseg-dlka3res-7]4', filter_small=0.05, data=data)
    # data = 'defect_test_1023.yaml'
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7]2', data=data)
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7]3', data=data)
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7]4', data=data)
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7]5', data=data)
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7]2', filter_small=0.05, data=data)
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7]3', filter_small=0.05, data=data)
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7]4', filter_small=0.05, data=data)
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7]5', filter_small=0.05, data=data)
    #
    #
    #
    # val_name = 'fusedata7961_mseg_c5_l2_1022_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # data = 'defect_test_1021.yaml'
    # demo_base.model_val(val_name, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.1, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.2, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.5, data=data)
    #
    # val_name = 'fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # data = 'defect_test_1023.yaml'
    # demo_base.model_val(val_name, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.1, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.2, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.5, data=data)
    #
    #
    # val_name = 'fusedata7961_mseg_c5_l2_1022_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # data = 'defect_test_1021.yaml'
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, risk_enlarge=1.1, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, risk_enlarge=1.2, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, risk_enlarge=1.3, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, risk_enlarge=1.4, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, risk_enlarge=1.5, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, risk_enlarge=1.6, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, risk_enlarge=1.7, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, risk_enlarge=1.8, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, risk_enlarge=1.9, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, risk_enlarge=2.0, data=data)
    #
    #
    # val_name = 'fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # data = 'defect_test_1023.yaml'
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, risk_enlarge=1.1, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, risk_enlarge=1.2, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, risk_enlarge=1.3, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, risk_enlarge=1.4, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, risk_enlarge=1.5, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, risk_enlarge=1.6, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, risk_enlarge=1.7, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, risk_enlarge=1.8, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, risk_enlarge=1.9, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, risk_enlarge=2.0, data=data)

    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]3')
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]6')
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]8')
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]10')

    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]5')
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]7')
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]9')
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1111_v15_test-[yolov10x-mseg-dlka3res-7-unet-sep]3', save_txt=True, save_conf=True)
    # data000 = 'data80_v17_test.yaml'
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1111_v15_test-[yolov10x-mseg-dlka3res-7-unet-sep]3',
    #                     data= data000, save_txt=True, save_conf=True, iou=0.5, conf=0.1)

    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1111_v15_test-[yolov10x-mseg-dlka3res-7-unet-sep]3',
    #                     save_txt=True, save_conf=True, iou=0.5, conf=0.1)

    # demo_base.model_predict(r'fusedata7961_mseg_c5_l2_1111_v15_test-[yolov10x-mseg-dlka3res-7-unet-sep]3',
    #                         r'/localnvme/data/billboard/all_data/mseg_c5_l2/data80_v17/images')



    # img_dir = r'/localnvme/data/billboard/all_data/mseg_c5_l2/data80_v17_rs/images/DA4930148_20250930165152399.jpg'
    # # img_dir = r'/localnvme/data/billboard/all_data/mseg_c5_l2/data80_v17_rs/images'
    # # # img_dir = r'/localnvme/data/billboard/all_data/mseg_c5_l2/data7961_mseg_c5_l2_1104_v5/images/cam_DA5324655_cam_image_20250704155539099.jpg'
    # # save_dir = r'/localnvme/data/billboard/infer'
    # save_dir = r'/localnvme/data/billboard/all_data/mseg_c5_l2/data80_v17_rs/images_pred'
    # demo_base.model_predict(r'fusedata7961_mseg_c5_l2_1111_v15_test-[yolov10x-mseg-dlka3res-7-unet-sep]3', img_dir = img_dir, name=save_dir,conf=0.1, save_conf=True)
    # demo_base.model_predict(r'fusedata7961_mseg_c5_l2_1111_v15_sd_test-[yolov10x-mseg-dlka3res-7-unet-single]', img_dir = img_dir, name=save_dir,conf=0.001)
    # demo_base.model_predict(r'fusedata7961_mseg_c5_l2_1111_v15_sb_test-[yolov10x-mseg-dlka3res-7-unet-single]', img_dir = img_dir, name=save_dir,conf=0.001)
    # demo_base.model_predict(r'fusedata7961_mseg_c5_l2_1111_v15_sa_test-[yolov10x-mseg-dlka3res-7-unet-single]', img_dir = img_dir, name=save_dir,conf=0.001)
    # demo_base.model_predict(r'fusedata7961_mseg_c5_l2_1111_v15_sc_test-[yolov10x-mseg-dlka3res-7-unet-single]', img_dir = img_dir, name=save_dir,conf=0.001)


    # val_name = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # # img_dir = r'/localnvme/data/added_data/test_data/test_data_mseg_c6_1021_broken_refine/images'
    # # save_dir = r'/localnvme/data/added_data/test_data/test_data_mseg_c6_1021_broken_refine/predicts/infer'
    # img_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1023_src/val_80p_ref/images'
    # save_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1023_src/val_80p_ref/images_infer'
    # demo_base.model_predict(val_name, img_dir = img_dir, conf=0.001, name=save_dir,)


    # demo_base.model_val('fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7-unet]',
    #                     data='fusedata7961_mseg_c5_l2_1023_src_all.yaml', save_txt=True, save_conf=True)
    # demo_base.model_val('fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7-unet]', save_txt=True, save_conf=True, eval_att_by_class=False)

    # demo_base.model_export(r'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]3',
    #                         imgsz=(608,960),batch=6)
    # demo_base.model_export(r'fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7]3',
    #                         imgsz=(608,960),batch=6)


