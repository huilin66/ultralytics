import demo_base
import torch

demo_base.TASK = 'msegment'
demo_base.EPOCHS = 100
demo_base.IMGSZ = 960
demo_base.BATCH_SIZE = 16
demo_base.DEVICE = torch.device('cuda:1')
demo_base.DATA = "fusedata2419_mseg_c5_0730.yaml"
# demo_base.CONF_VAL = 0.5
# demo_base.CONF_PREDICT = 0.5

MODEL1 = 'yolov8x-mseg-7.yaml'
MODEL2 = 'yolov8x-mseg-dlka3res-7.yaml'
MODEL3 = 'yolov10x-mseg-dlka3res-7.yaml'
SEG_WEIGHT = "runs/segment/fusedata5894_seg_c5_0822_80p-[yolov8x-seg-dlka3res]2/weights/best.pt"
SEG_WEIGHT10 = "runs/segment/fusedata7436_seg_c5_0922_80p_ref-[yolov10x-seg-dlka3res]_a100/weights/best.pt"
DATA0 = "fusedata7961_mseg_c5_l2_1021_80p_ref.yaml"
DATA1 = "fusedata7961_mseg_c5_1015_80p_ref.yaml"

if __name__ == '__main__':
    pass
    demo_base.yolo10(
        MODEL3, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
        mloss_mask=True, mloss_weight=False, mloss_enlarge=0,
    )
    demo_base.yolo10(
        MODEL3, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
        mloss_mask=False, mloss_weight=0, mloss_enlarge=2,
    )
    demo_base.yolo10(
        MODEL3, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
        mloss_mask=False, mloss_weight=0, mloss_enlarge=5,
    )
    demo_base.yolo10(
        MODEL3, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
        mloss_mask=False, mloss_weight=0, mloss_enlarge=10,
    )


    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0922_80p_ref-[yolov10x-mseg-dlka3res-7]3',)
    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0922_80p_ref-[yolov10x-mseg-dlka3res-7]3', filter_small=0.05)
    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0922_80p_ref-[yolov10x-mseg-dlka3res-7]3', augment=True)
    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0922_80p_ref-[yolov10x-mseg-dlka3res-7]3', augment=True)
    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0922_80p_ref-[yolov10x-mseg-dlka3res-7]3', augment=True)
    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0922_80p_ref-[yolov10x-mseg-dlka3res-7]3', augment=True)
    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0922_80p_ref-[yolov10x-mseg-dlka3res-7]3', augment=True)


    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0922_80p_ref-[yolov10x-mseg-dlka3res-7]3', risk_enlarge=1.1)
    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0922_80p_ref-[yolov10x-mseg-dlka3res-7]3', risk_enlarge=1.2)
    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0922_80p_ref-[yolov10x-mseg-dlka3res-7]3', risk_enlarge=1.3)
    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0922_80p_ref-[yolov10x-mseg-dlka3res-7]3', risk_enlarge=1.4)
    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0922_80p_ref-[yolov10x-mseg-dlka3res-7]3', risk_enlarge=1.5)
    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0922_80p_ref-[yolov10x-mseg-dlka3res-7]3', risk_enlarge=1.6)
    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0922_80p_ref-[yolov10x-mseg-dlka3res-7]3', risk_enlarge=1.7)
    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0922_80p_ref-[yolov10x-mseg-dlka3res-7]3', risk_enlarge=1.8)
    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0922_80p_ref-[yolov10x-mseg-dlka3res-7]3', risk_enlarge=1.9)

    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0922_80p_ref-[yolov10x-mseg-dlka3res-7]3', risk_enlarge=1.5)
    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0922_80p_ref-[yolov10x-mseg-dlka3res-7]3', risk_enlarge=2)
    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0922_80p_ref-[yolov10x-mseg-dlka3res-7]3', risk_enlarge=3)
    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0922_80p_ref-[yolov10x-mseg-dlka3res-7]3', risk_enlarge=4)
    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0922_80p_ref-[yolov10x-mseg-dlka3res-7]3', risk_enlarge=5)
    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0922_80p_ref-[yolov10x-mseg-dlka3res-7]3', risk_enlarge=10)


    # demo_base.model_predict(
    #     r'runs/msegment/fusedata3044_mseg_c5_0731-[yolov8x-mseg-dlka3res-7]4/weights/best.pt',
    #     img_dir = r'/localnvme/data/billboard/bd_data/data626_mseg_c6_check0624/demo_images',
    #     name=r'/localnvme/data/billboard/bd_data/data626_mseg_c6_check0624/demo_images_infer',
    # )

    # demo_base.model_export(r'fusedata7720_mseg_c5_l2_1002_80p_ref-[yolov10x-mseg-dlka3res-7]2',
    #                         imgsz=(608,960),
    #                         # dynamic=True,
    #                         batch=6,
    #                        )

    # demo_base.model_val(r'fusedata7720_mseg_c5_l2_1002_80p_ref-[yolov10x-mseg-dlka3res-7]2', risk_enlarge=1.8)