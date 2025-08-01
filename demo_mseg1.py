import demo_base
import torch

demo_base.TASK = 'msegment'
demo_base.EPOCHS = 100
demo_base.IMGSZ = 960
demo_base.BATCH_SIZE = 32
demo_base.DEVICE = torch.device('cuda:0')
demo_base.DATA = "fusedata2419_mseg_c5_0730.yaml"

if __name__ == '__main__':
    pass
    demo_base.yolo8('yolov8x-mseg-dlka3res-7.yaml',
                    weight_path=r'runs/msegment/fusedata2419_seg_c5_0730-[yolov8x-seg-dlka3res]4/weights/best.pt',
                    auto_optim=False,
                    retrain=True,
                    mloss_mask=False,
                    mloss_weight=False,
                    mloss_enlarge=2,
                    # name='debug'
                    data = "fusedata2419_mseg_c5_0730.yaml",
                    )
    demo_base.yolo8('yolov8x-mseg-dlka3res-7.yaml',
                    weight_path=r'runs/msegment/fusedata2419_seg_c5_0730-[yolov8x-seg-dlka3res]4/weights/best.pt',
                    auto_optim=False,
                    retrain=True,
                    mloss_mask=False,
                    mloss_weight=False,
                    mloss_enlarge=2,
                    # name='debug'
                    data = "fusedata2419_mseg_c5_l2_0730.yaml",
                    )
    demo_base.yolo8('yolov8x-mseg-dlka3res-7.yaml',
                    weight_path=r'runs/msegment/fusedata2419_seg_c5_0730-[yolov8x-seg-dlka3res]4/weights/best.pt',
                    auto_optim=False,
                    retrain=True,
                    mloss_mask=False,
                    mloss_weight=True,
                    mloss_enlarge=0,
                    # name='debug'
                    data = "fusedata2419_mseg_c5_0730.yaml",
                    )
    demo_base.yolo8('yolov8x-mseg-dlka3res-7.yaml',
                    weight_path=r'runs/msegment/fusedata2419_seg_c5_0730-[yolov8x-seg-dlka3res]4/weights/best.pt',
                    auto_optim=False,
                    retrain=True,
                    mloss_mask=False,
                    mloss_weight=True,
                    mloss_enlarge=0,
                    # name='debug'
                    data = "fusedata2419_mseg_c5_l2_0730.yaml",
                    )
    demo_base.yolo8('yolov8x-mseg-dlka3res-7.yaml',
                    weight_path=r'runs/msegment/fusedata2419_seg_c5_0730-[yolov8x-seg-dlka3res]4/weights/best.pt',
                    auto_optim=False,
                    retrain=True,
                    mloss_mask=False,
                    mloss_weight=True,
                    mloss_enlarge=1,
                    # name='debug'
                    data = "fusedata2419_mseg_c5_0730.yaml",
                    )
    demo_base.yolo8('yolov8x-mseg-dlka3res-7.yaml',
                    weight_path=r'runs/msegment/fusedata2419_seg_c5_0730-[yolov8x-seg-dlka3res]4/weights/best.pt',
                    auto_optim=False,
                    retrain=True,
                    mloss_mask=False,
                    mloss_weight=True,
                    mloss_enlarge=1,
                    # name='debug'
                    data = "fusedata2419_mseg_c5_l2_0730.yaml",
                    )
    # demo_base.yolo8('yolov8x-mseg-dlka3res-7.yaml',
    #                 weight_path=r'runs/msegment/fusedata2177_mseg_c5_l2_0718-[yolov8x-mseg-dlka3res-7]10/weights/best.pt',
    #                 auto_optim=False,
    #                 retrain=True,
    #                 # mloss_mask=True,
    #                 mloss_weight=0,
    #                 mloss_enlarge=1,
    #                 # name='debug'
    #                 )
    # demo_base.yolo8('yolov8x-mseg-dlka3res-7.yaml',
    #                 weight_path=r'runs/msegment/fusedata2177_mseg_c5_l2_0718-[yolov8x-mseg-dlka3res-7]10/weights/best.pt',
    #                 auto_optim=False,
    #                 retrain=True,
    #                 # mloss_mask=True,
    #                 mloss_weight=0.25,
    #                 mloss_enlarge=1,
    #                 # name='debug'
    #                 )
    # demo_base.yolo8('yolov8x-mseg-dlka3res-7.yaml',
    #                 weight_path=r'runs/msegment/fusedata2177_mseg_c5_l2_0718-[yolov8x-mseg-dlka3res-7]10/weights/best.pt',
    #                 auto_optim=False,
    #                 retrain=True,
    #                 # mloss_mask=True,
    #                 mloss_weight=0.5,
    #                 mloss_enlarge=1,
    #                 # name='debug'
    #                 )
    # demo_base.yolo8('yolov8x-mseg-dlka3res-7.yaml',
    #                 weight_path=r'runs/msegment/fusedata2177_mseg_c5_l2_0718-[yolov8x-mseg-dlka3res-7]10/weights/best.pt',
    #                 auto_optim=False,
    #                 retrain=True,
    #                 # mloss_mask=True,
    #                 mloss_weight=0.75,
    #                 mloss_enlarge=1,
    #                 # name='debug'
    #                 )
    # demo_base.yolo8('yolov8x-mseg-dlka3res-7.yaml',
    #                 weight_path=r'runs/msegment/fusedata2177_mseg_c5_l2_0718-[yolov8x-mseg-dlka3res-7]10/weights/best.pt',
    #                 auto_optim=False,
    #                 retrain=True,
    #                 # mloss_mask=True,
    #                 mloss_weight=1,
    #                 mloss_enlarge=1,
    #                 # name='debug'
    #                 )
    # for mloss_enlarge in [2, 4, 6, 8, 10]:
    #     demo_base.yolo8('yolov8x-mseg-dlka3res-7.yaml',
    #                     weight_path=r'runs/segment/fusedata2177_seg_c5_0718-[yolov8x-seg]/weights/best.pt',
    #                     auto_optim=False,
    #                     retrain=True,
    #                     # mloss_mask=True,
    #                     mloss_weight=0,
    #                     mloss_enlarge=mloss_enlarge,
    #                     # name='debug'
    #                     )
    # for mloss_weight in [0.5, 1, 2]:
    #     demo_base.yolo8('yolov8x-mseg-dlka3res-7.yaml',
    #                     weight_path=r'runs/segment/fusedata2177_seg_c5_0718-[yolov8x-seg]/weights/best.pt',
    #                     auto_optim=False,
    #                     retrain=True,
    #                     # mloss_mask=True,
    #                     mloss_weight=mloss_weight,
    #                     mloss_enlarge=1,
    #                     # name='debug'
    #                     )


    # demo_base.yolo8('yolov8x-mseg-dlka3res-7.yaml',
    #                 weight_path=r'runs/segment/fusedata2177_seg_c5_0718-[yolov8x-seg-dlka3res]/weights/best.pt',
    #                 auto_optim=False,
    #                 retrain=True,
    #                 # mloss_mask=True,
    #                 mloss_weight=0,
    #                 mloss_enlarge=2,
    #                 # name='debug'
    #                 )
    # demo_base.yolo8('yolov8x-mseg-dlka3res-7.yaml',
    #                 weight_path=r'runs/segment/fusedata2177_seg_c5_0718-[yolov8x-seg-dlka3res]/weights/best.pt',
    #                 auto_optim=False,
    #                 retrain=True,
    #                 mloss_mask=True,
    #                 mloss_weight=0,
    #                 # mloss_enlarge=0,
    #                 # name='debug'
    #                 )

    demo_base.model_val(r'runs/msegment/fusedata2177_mseg_c5_0718-[yolov8x-mseg-dlka3res-7]19/weights/best.pt',
                        # save_txt=True, save_npy=True
                        data='fusedata2177_mseg_c5_0718.yaml',
                        )
    demo_base.model_val(r'runs/msegment/fusedata2177_mseg_c5_0718-[yolov8x-mseg-dlka3res-7]19/weights/best.pt',
                        data='fusedata2177_mseg_c5_large_0718.yaml',
                        # save_txt=True, save_npy=True
                        )

    # demo_base.model_predict(
    #     r'runs/msegment/fusedata1600_mseg_c6_0710-[yolov8x-mseg-7]/weights/best.pt',
    #     img_dir = r'/localnvme/data/billboard/ps_data/0702/images',
    #     name=r'/localnvme/data/billboard/ps_data/0702/images_infer',
    # )

    # demo_base.model_export(r'runs/msegment/fusedata1600_mseg_c6_0710-[yolov8x-mseg-7]/weights/best.pt',
    #                         imgsz=(608,960),
    #                         # dynamic=True,
    #                         batch=6,
    #                        )