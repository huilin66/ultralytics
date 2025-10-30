import demo_base
import torch

demo_base.TASK = 'msegment'
demo_base.EPOCHS = 100
demo_base.IMGSZ = 960
demo_base.BATCH_SIZE = 64
demo_base.DEVICE = torch.device('cuda:1')
demo_base.DATA = "fusedata2419_mseg_c5_0730.yaml"
# demo_base.CONF_VAL = 0.5
# demo_base.CONF_PREDICT = 0.5

MODEL1 = 'yolov8x-mseg-7.yaml'
MODEL2 = 'yolov8x-mseg-dlka3res-7.yaml'
MODEL3 = 'yolov10x-mseg-dlka3res-7.yaml'
MODEL4 = 'yolov10x-mseg-dlka3res-7-unet.yaml'
MODEL5 = 'yolov10x-mseg-dlka3res-7-c3str.yaml'
MODEL6 = 'yolov10x-mseg-dlka3res-7-c3str-unet1.yaml'
MODEL7 = 'yolov10x-mseg-dlka3res-7-c3str-unet2.yaml'
MODEL06 = 'yolov10x-mseg-dlka3res-7-dfl.yaml'
MODEL07 = 'yolov10x-mseg-dlka3res-7-dlka.yaml'
MODEL08 = 'yolov10x-mseg-dlka3res-7-dlkaatt.yaml'
SEG_WEIGHT = "runs/segment/fusedata5894_seg_c5_0822_80p-[yolov8x-seg-dlka3res]2/weights/best.pt"
SEG_WEIGHT10 = "/localnvme/project/ultralytics/runs/segment/fusedata7961_seg_c5_l2_1022_re_80p_ref-[yolov10x-seg-dlka3res]8/weights/best.pt"
DATA0 = "fusedata7961_mseg_c5_l2_1030_v4_src.yaml"
DATA1 = "fusedata7961_mseg_c5_l2_1030_v4_test.yaml"

if __name__ == '__main__':
    pass
    demo_base.yolo10(
        MODEL4, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
        mloss_mask=False, mloss_weight=0, mloss_enlarge=5,
    )
    demo_base.yolo10(
        MODEL4, weight_path=SEG_WEIGHT10, data=DATA1, auto_optim=False, retrain=True,
        mloss_mask=False, mloss_weight=0, mloss_enlarge=5,
    )
    # demo_base.yolo10(
    #     MODEL4, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=10,
    # )
    # demo_base.yolo10(
    #     MODEL3, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=5, name='debug'
    # )
    # demo_base.yolo10(
    #     MODEL8,  weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=5, name='debug'
    # )

    # demo_base.yolo10(
    #     MODEL7, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=5,
    # )
    # demo_base.yolo10(
    #     MODEL6, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=5,
    # )
    # demo_base.yolo10(
    #     MODEL5, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=5,
    # )
    # demo_base.yolo10(
    #     MODEL4, weight_path=SEG_WEIGHT10, data=DATA0, auto_optim=False, retrain=True,
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


    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1029_abandonment_refine_80p_ref_src-[yolov10x-mseg-dlka3res-7-unet]3',)
    # demo_base.model_val(r'fusedata7961_mseg_c5_l2_1029_abandonment_refine_test-[yolov10x-mseg-dlka3res-7-unet-sep]2',)
    # demo_base.model_val('fusedata7961_mseg_c5_l2_1029_abandonment_refine_80p_ref_src-[yolov10x-mseg-dlka3res-7-unet]2')
    # demo_base.model_val('fusedata7961_mseg_c5_l2_1029_abandonment_refine_test-[yolov10x-mseg-dlka3res-7-unet]4')
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
