import demo_base
import torch

demo_base.TASK = 'msegment'
demo_base.EPOCHS = 100
demo_base.IMGSZ = 960
demo_base.BATCH_SIZE = 16
demo_base.DEVICE = torch.device('cuda:0')
demo_base.DATA = "fusedata2419_mseg_c5_0730.yaml"
# demo_base.CONF_VAL = 0.5
# demo_base.CONF_PREDICT = 0.5

MODEL1 = 'yolov8x-mseg-7.yaml'
MODEL2 = 'yolov8x-mseg-dlka3res-7.yaml'
MODEL3 = 'yolov10x-mseg-dlka3res-7.yaml'
MODEL4 = 'yolov10x-mseg-dlka3res-7-unet.yaml'
SEG_WEIGHT8 = "runs/segment/fusedata5894_seg_c5_0822_80p-[yolov8x-seg-dlka3res]2/weights/best.pt"
SEG_WEIGHT10 = "/localnvme/project/ultralytics/runs/segment/fusedata7961_seg_c5_l2_1022_re_80p_ref-[yolov10x-seg-dlka3res]8/weights/best.pt"
DATA1 = "fusedata7720_mseg_c5_l2_1002_70p_ref.yaml"
DATA2 = "fusedata7720_mseg_c5_l2_1002_60p_ref.yaml"
DATA3 = "fusedata7720_mseg_c5_l2_1002_70p.yaml"
DATA4 = "fusedata7720_mseg_c5_l2_1002_60p.yaml"
DATA0 = 'fusedata7436_mseg_c5_l2_0922_80p_ref.yaml'

DATA5 = "testdata80_mseg_c5_l2_1021.yaml"
DATA6 = "testdata80_mseg_c5_l2_1021_broke_refine.yaml"

DATA00 = 'defect_test_1023.yaml'
DATA0 = "fusedata7961_mseg_c5_l2_1029_abandonment_refine_test.yaml"
DATA1 = "fusedata7961_mseg_c5_1015_80p_ref.yaml"

if __name__ == '__main__':
    pass
    demo_base.yolo10(
        MODEL4, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
        mloss_mask=False, mloss_weight=0, mloss_enlarge=5,
    )
    demo_base.yolo10(
        MODEL4, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
        mloss_mask=False, mloss_weight=0, mloss_enlarge=10,
    )
    # demo_base.yolo10(
    #     MODEL3, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
    #     mloss_mask=True, mloss_weight=False, mloss_enlarge=0,
    # )

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
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]11')


    # val_name = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # data = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref.yaml'
    # demo_base.model_val(val_name, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.1, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.2, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.5, data=data)
    #
    # val_name = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # data = 'defect_test_1023.yaml'
    # demo_base.model_val(val_name, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.1, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.2, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.3, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, data=data)
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.5, data=data)

    # val_name = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # data = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref.yaml'
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
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=[1.0, 1.0, 1.5, 1.0], data=data)

    # val_name = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]3'
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
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=[1.0, 1.0, 1.5, 1.0], data=data)



    # val_name = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # data = 'defect_test_1023.yaml'
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=[1.0, 1.3, 2.0, 1.0], data=data, save_conf=True, save_txt=True)
    # val_name = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # data = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref.yaml'
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=[1.0, 1.3, 2.0, 1.0], data=data, save_conf=True, save_txt=True)


    val_name = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    data = 'defect_test_1023.yaml'
    demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=[1.0, 1.3, 2.0, 1.0], data=data, save_conf=True, save_txt=True)
    # val_name = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # data = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref.yaml'
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=[1.0, 1.3, 2.0, 1.0], data=data, save_conf=True, save_txt=True)

    # val_name = 'fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # data = 'defect_test_1023.yaml'
    # demo_base.model_val(val_name, filter_small=0.05, data=data)
    # val_name = 'fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7-unet]'
    # data = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref.yaml'
    # demo_base.model_val(val_name, data=data, save_conf=True, save_txt=True)
    # val_name = 'fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7-unet]'
    # data = 'defect_test_1023.yaml'
    # demo_base.model_val(val_name, data=data, save_conf=True, save_txt=True)

    # val_name = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # data = 'fusedata7961_mseg_c5_l2_1023_src_80p_ref.yaml'
    # demo_base.model_val(val_name, filter_small=0.05, conf=0.4, risk_enlarge=[1.0, 1.3, 2.0, 1.0], data=data, save_conf=True, save_txt=True)
    # val_name = 'fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # demo_base.model_val(val_name, save_conf=True, save_txt=True)

    # val_name = 'fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7]3'
    # img_dir = r'/localnvme/data/added_data/test_data/test_data_mseg_c5_l2_1021_broken_refine/images'
    # save_dir = r'/localnvme/data/added_data/test_data/test_data_mseg_c5_l2_1021_broken_refine/predicts/infer'
    # demo_base.model_predict(val_name, img_dir = img_dir,name=save_dir, conf=0.001)
    # demo_base.model_predict(val_name, img_dir = img_dir,name=save_dir, conf=0.1)
    # demo_base.model_predict(val_name, img_dir = img_dir,name=save_dir, conf=0.2)
    # demo_base.model_predict(val_name, img_dir = img_dir,name=save_dir, conf=0.3)
    # demo_base.model_predict(val_name, img_dir = img_dir,name=save_dir, conf=0.4)
    # demo_base.model_predict(val_name, img_dir = img_dir,name=save_dir, conf=0.5)


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


