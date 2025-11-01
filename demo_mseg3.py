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
MODEL5 = 'yolov10x-mseg-dlka3res-7-c3str.yaml'
MODEL6 = 'yolov10x-mseg-dlka3res-7-unet-sep.yaml'

MSEG_WEIGHT = "fusedata7436_mseg_c5_l2_0917_80p_ref-[yolov8x-mseg-dlka3res-7]2"
SEG_WEIGHT10 = "/localnvme/project/ultralytics/runs/segment/fusedata7961_seg_c5_l2_1022_re_80p_ref-[yolov10x-seg-dlka3res]8/weights/best.pt"
DATA0 = "fusedata7961_mseg_c5_l2_1029_abandonment_refine_test.yaml"
DATA1 = "fusedata7436_mseg_c5_0914_80p.yaml"
DATA2 = "fusedata7436_mseg_c5_l2_0914_80p.yaml"
DATA3 = "fusedata4197_mseg_c5_0914_80p.yaml"
DATA4 = "fusedata4197_mseg_c5_l2_0914_80p.yaml"
DATA5 = "fusedata3617_mseg_c5_0915_80p.yaml"
DATA6 = "fusedata3617_mseg_c5_l2_0915_80p.yaml"
DATA7 = "fusedata4197_mseg_c5_l2_0914_80p_ref.yaml"
DATA8 = "fusedata7436_mseg_c5_l2_0914_80p_ref.yaml"
DATA9 = "fusedata4197_mseg_c5_0914_80p_ref.yaml"
DATA10 = "fusedata7436_mseg_c5_0914_80p_ref.yaml"
DATA_test = "defect_test_1023.yaml"

if __name__ == '__main__':
    pass
    # demo_base.yolo10(
    #     MODEL6, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=5,
    # )

    # demo_base.yolo10(
    #     MODEL3, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=2,
    # )
    # demo_base.yolo10(
    #     MODEL3, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=5,
    # )
    # demo_base.yolo10(
    #     MODEL3, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=10,
    # )
    # demo_base.yolo10(
    #     MODEL3, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
    #     mloss_mask=True, mloss_weight=False, mloss_enlarge=0,
    # )

    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1030_v4_src-[yolov10x-mseg-dlka3res-7-unet]', save_txt=True, save_conf=True)
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1030_v4_src-[yolov10x-mseg-dlka3res-7-unet-sep]', save_txt=True, save_conf=True)
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1030_v4_test-[yolov10x-mseg-dlka3res-7-unet-sep]', save_txt=True, save_conf=True)

    # demo_base.model_predict(
    #     MSEG_WEIGHT,
    #     data = r'fusedata7436_mseg_c5_l2_0917_80p_ref_valall.yaml',
    #     img_dir = r'/localnvme/data/billboard/fused_data/data7436_mseg_c5_l2_0917/images',
    #     name= r'/localnvme/data/billboard/fused_data/data7436_mseg_c5_l2_0917/images_infer',
    #     save_txt=True, plots=False,
    # )

    # demo_base.model_export(r'fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7]2',
    #                         imgsz=(608,960),
    #                         # dynamic=True,
    #                         batch=6,
    #                        )

