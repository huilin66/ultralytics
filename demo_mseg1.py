import demo_base
import torch

demo_base.TASK = 'msegment'
demo_base.EPOCHS = 100
demo_base.IMGSZ = 960
demo_base.BATCH_SIZE = 16
demo_base.DEVICE = torch.device('cuda:1')
demo_base.DATA = "fusedata2419_mseg_c5_0730.yaml"

MODEL1 = 'yolov8x-mseg-7.yaml'
MODEL2 = 'yolov8x-mseg-dlka3res-7.yaml'
SEG_WEIGHT = "runs/segment/fusedata5894_seg_c5_0822_80p-[yolov8x-seg-dlka3res]2/weights/best.pt"
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
    demo_base.yolo8(
        MODEL2, weight_path=SEG_WEIGHT, data=DATA9, auto_optim=False, retrain=True,
        mloss_mask=True, mloss_weight=False, mloss_enlarge=0,
    )
    # demo_base.yolo8(
    #     MODEL2, weight_path=SEG_WEIGHT, data=DATA1, auto_optim=False, retrain=True,
    #     mloss_mask=True, mloss_weight=False, mloss_enlarge=0,
    # )

    # demo_base.model_val(r'runs/msegment/fusedata5894_mseg_c5_0822_80p-[yolov8x-mseg-dlka3res-7]/weights/best.pt',)

    # demo_base.model_predict(
    #     r'runs/msegment/fusedata3044_mseg_c5_0731-[yolov8x-mseg-dlka3res-7]4/weights/best.pt',
    #     img_dir = r'/localnvme/data/billboard/bd_data/data626_mseg_c6_check0624/demo_images',
    #     name=r'/localnvme/data/billboard/bd_data/data626_mseg_c6_check0624/demo_images_infer',
    # )

    # demo_base.model_export(r'runs/msegment/fusedata5894_mseg_c5_0822_80p-[yolov8x-mseg-dlka3res-7]/weights/best.pt',
    #                         imgsz=(608,960),
    #                         # dynamic=True,
    #                         batch=6,
    #                        )

