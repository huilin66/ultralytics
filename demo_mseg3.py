import demo_base
import torch

demo_base.TASK = 'msegment'
demo_base.EPOCHS = 100
demo_base.IMGSZ = 960
demo_base.BATCH_SIZE = 16
demo_base.DEVICE = torch.device('cuda:1')
demo_base.DATA = "fusedata2419_mseg_c5_0730.yaml"
demo_base.CONF_VAL = 0.5
demo_base.CONF_PREDICT = 0.5

MODEL1 = 'yolov8x-mseg-7.yaml'
MODEL2 = 'yolov8x-mseg-dlka3res-7.yaml'
MSEG_WEIGHT = "fusedata7436_mseg_c5_l2_0917_80p_ref-[yolov8x-mseg-dlka3res-7]2"
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

if __name__ == '__main__':
    pass
    # demo_base.yolo8(
    #     MODEL2, weight_path=SEG_WEIGHT, data=DATA9, auto_optim=False, retrain=True,
    #     mloss_mask=True, mloss_weight=False, mloss_enlarge=0,
    # )
    # demo_base.yolo8(
    #     MODEL2, weight_path=SEG_WEIGHT, data=DATA1, auto_optim=False, retrain=True,
    #     mloss_mask=True, mloss_weight=False, mloss_enlarge=0,
    # )

    # demo_base.model_val(r'fusedata4197_mseg_c5_0914_80p-[yolov8x-mseg-dlka3res-7]',
    #                     filter_small=0.05,
    #                     )

    demo_base.model_predict(
        MSEG_WEIGHT,
        data = r'fusedata7436_mseg_c5_l2_0917_80p_ref_valall.yaml',
        img_dir = r'/localnvme/data/billboard/fused_data/data7436_mseg_c5_l2_0917/images',
        name= r'/localnvme/data/billboard/fused_data/data7436_mseg_c5_l2_0917/images_infer',
        save_txt=True, plots=False,
    )

    # demo_base.model_export(r'runs/msegment/fusedata5894_mseg_c5_0822_80p-[yolov8x-mseg-dlka3res-7]/weights/best.pt',
    #                         imgsz=(608,960),
    #                         # dynamic=True,
    #                         batch=6,
    #                        )

