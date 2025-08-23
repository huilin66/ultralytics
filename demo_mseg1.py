import demo_base
import torch

demo_base.TASK = 'msegment'
demo_base.EPOCHS = 100
demo_base.IMGSZ = 960
demo_base.BATCH_SIZE = 16
demo_base.DEVICE = torch.device('cuda:1')
demo_base.DATA = "fusedata2419_mseg_c5_0730.yaml"

MODEL1 = 'yolov8x-mseg-dlka3res-7-dlka.yaml'
MODEL2 = 'yolov8x-mseg-dlka3res-7-dlkaatt.yaml'
MODEL3 = 'yolov8x-mseg-dlka3res-7-c3str.yaml'
MODEL4 = 'yolov8x-mseg-dlka3res-7-dfl.yaml'
SEG_WEIGHT = "runs/segment/fusedata3899_seg_c5_0818_80p-[yolov8x-seg-dlka3res]/weights/best.pt"
DATA1 = "fusedata3899_mseg_c5_0818_80p.yaml"
DATA2 = "fusedata3899_mseg_c5_l2_0818_80p.yaml"


if __name__ == '__main__':
    pass

    # 1 loss
    # demo_base.yolo8(
    #     MODEL1, weight_path=SEG_WEIGHT, data=DATA1, auto_optim=False, retrain=True,
    #     mloss_mask=True, mloss_weight=False, mloss_enlarge=0
    # )
    # demo_base.model_val(r'runs/msegment/fusedata3899_mseg_c5_0818_80p-[yolov8x-mseg-dlka3res-7-dlka]3/weights/best.pt',
    #                     # save_txt=True, save_npy=True
    #                     )
    # demo_base.yolo8(
    #     MODEL2, weight_path=SEG_WEIGHT, data=DATA1, auto_optim=False, retrain=True,
    #     mloss_mask=True, mloss_weight=False, mloss_enlarge=0
    # )
    # demo_base.yolo8(
    #     MODEL3, weight_path=SEG_WEIGHT, data=DATA1, auto_optim=False, retrain=True,
    #     mloss_mask=True, mloss_weight=False, mloss_enlarge=0
    # )
    # demo_base.yolo8(
    #     MODEL4, weight_path=SEG_WEIGHT, data=DATA1, auto_optim=False, retrain=True,
    #     mloss_mask=True, mloss_weight=False, mloss_enlarge=0
    # )
    # demo_base.yolo8(
    #     MODEL2, weight_path=SEG_WEIGHT, data=DATA1, auto_optim=False, retrain=True,
    #     mloss_mask=True, mloss_weight=True, mloss_enlarge=0
    # )
    # demo_base.yolo8(
    #     MODEL2, weight_path=SEG_WEIGHT, data=DATA1, auto_optim=False, retrain=True,
    #     mloss_mask=True, mloss_weight=True, mloss_enlarge=2
    # )
    # demo_base.yolo8(
    #     MODEL2, weight_path=SEG_WEIGHT, data=DATA1, auto_optim=False, retrain=True,
    #     mloss_mask=True, mloss_weight=True, mloss_enlarge=5
    # )

    # # 2 loss
    # # loss=loss1
    # demo_base.yolo8(
    #     MODEL2, weight_path=SEG_WEIGHT, data=DATA2, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=0
    # )
    # demo_base.yolo8(
    #     MODEL2, weight_path=SEG_WEIGHT, data=DATA2, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=1
    # )
    # demo_base.yolo8(
    #     MODEL2, weight_path=SEG_WEIGHT, data=DATA2, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=2
    # )
    # demo_base.yolo8(
    #     MODEL2, weight_path=SEG_WEIGHT, data=DATA2, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=5
    # )
    # demo_base.yolo8(
    #     MODEL2, weight_path=SEG_WEIGHT, data=DATA2, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=10
    # )
    # # loss=0.75loss1+0.25loss2
    # demo_base.yolo8(
    #     MODEL2, weight_path=SEG_WEIGHT, data=DATA2, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0.25, mloss_enlarge=1
    # )
    # # loss=0.5loss1+0.5loss2
    # demo_base.yolo8(
    #     MODEL2, weight_path=SEG_WEIGHT, data=DATA2, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0.5, mloss_enlarge=1
    # )
    # # loss=0.75loss1+0.25loss2
    # demo_base.yolo8(
    #     MODEL2, weight_path=SEG_WEIGHT, data=DATA2, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0.75, mloss_enlarge=1
    # )

    # demo_base.yolo8(
    #     MODEL2, weight_path=SEG_WEIGHT, data=DATA2, auto_optim=False, retrain=True,
    #     mloss_mask=True, mloss_weight=False, mloss_enlarge=0
    # )
    # demo_base.yolo8(
    #     MODEL2, weight_path=SEG_WEIGHT, data=DATA2, auto_optim=False, retrain=True,
    #     mloss_mask=True, mloss_weight=True, mloss_enlarge=0
    # )
    # demo_base.yolo8(
    #     MODEL2, weight_path=SEG_WEIGHT, data=DATA2, auto_optim=False, retrain=True,
    #     mloss_mask=True, mloss_weight=True, mloss_enlarge=2
    # )
    # demo_base.yolo8(
    #     MODEL2, weight_path=SEG_WEIGHT, data=DATA2, auto_optim=False, retrain=True,
    #     mloss_mask=True, mloss_weight=True, mloss_enlarge=5
    # )
    # demo_base.yolo8(
    #     MODEL2, weight_path=SEG_WEIGHT, data=DATA2, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=1
    # )
    # demo_base.yolo8(
    #     MODEL2, weight_path=SEG_WEIGHT, data=DATA2, auto_optim=False, retrain=True,
    #     mloss_mask=False, mloss_weight=0, mloss_enlarge=2
    # )

    # demo_base.model_val(r'runs/msegment/fusedata3899_mseg_c5_0818_80p-[yolov8x-mseg-dlka3res-7-dlka]3/weights/best.pt',
    #                     # save_txt=True, save_npy=True
    #                     )
    # demo_base.model_val(r'runs/msegment/fusedata2177_mseg_c5_0718-[yolov8x-mseg-dlka3res-7]19/weights/best.pt',
    #                     data='fusedata2177_mseg_c5_large_0718.yaml',
    #                     # save_txt=True, save_npy=True
    #                     )

    # demo_base.model_predict(
    #     r'runs/msegment/fusedata3044_mseg_c5_0731-[yolov8x-mseg-dlka3res-7]4/weights/best.pt',
    #     img_dir = r'/localnvme/data/billboard/bd_data/data626_mseg_c6_check0624/demo_images',
    #     name=r'/localnvme/data/billboard/bd_data/data626_mseg_c6_check0624/demo_images_infer',
    # )

    # demo_base.model_export(r'runs/msegment/fusedata3044_mseg_c5_0731-[yolov8x-mseg-dlka3res-7]4/weights/best.pt',
    #                         imgsz=(608,960),
    #                         # dynamic=True,
    #                         batch=6,
    #                        )


    demo_base.model_track(
        r'runs/msegment/fusedata3044_mseg_c5_0731-[yolov8x-mseg-dlka3res-7]4/weights/best.pt',
        img_dir=r'/localnvme/data/billboard/demo_data/data118_mseg/images',
        name=r'/localnvme/data/billboard/demo_data/data118_mseg/images_infer',
    )