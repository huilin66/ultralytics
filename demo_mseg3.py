import demo_base
import torch

demo_base.TASK = 'msegment'
demo_base.EPOCHS = 100
demo_base.IMGSZ = 960
demo_base.BATCH_SIZE = 64
demo_base.DEVICE = torch.device('cuda:1')
demo_base.DATA = "fusedata2177_mseg_c5_l2_0718.yaml"

if __name__ == '__main__':
    pass
    # demo_base.yolo8('yolov8x-mseg-dlka3res-7.yaml',
    #                 weight_path=r'runs/segment/fusedata2177_seg_c5_0718-[yolov8x-seg-dlka3res]/weights/best.pt',
    #                 auto_optim=False,
    #                 retrain=True,
    #                 mloss_mask=True,
    #                 mloss_weight=0,
    #                 # mloss_enlarge=0,
    #                 # name='debug'
    #                 )
    # demo_base.model_val(r'runs/msegment/fusedata3072_mseg_c5_l2_0809_80p-[yolov8x-mseg-dlka3res-7]6/weights/best.pt',
    #                     # save_txt=True, save_npy=True
    #                     )
    # demo_base.model_val(r'runs/msegment/fusedata2177_mseg_c5_0718-[yolov8x-mseg-dlka3res-7]23/weights/best.pt',
    #             # save_txt=True, save_npy=True
    #             )
    # demo_base.model_val(r'runs/msegment/fusedata2177_mseg_c5_l2_0718-[yolov8x-mseg-dlka3res-7]11/weights/best.pt',
    #                     # save_txt=True, save_npy=True
    #                     )
    # demo_base.model_val(r'runs/msegment/fusedata2177_mseg_c5_l2_0718-[yolov8x-mseg-dlka3res-7]8/weights/best.pt',
    #                     # save_txt=True, save_npy=True
    #                     )

    # demo_base.model_predict(
    #     r'runs/msegment/fusedata1600_mseg_c6_0710-[yolov8x-mseg-7]/weights/best.pt',
    #     img_dir = r'/localnvme/data/billboard/ps_data/0702/images',
    #     name=r'/localnvme/data/billboard/ps_data/0702/images_infer',
    # )

    # demo_base.model_export(r'runs/msegment/fusedata2177_mseg_c5_0718-[yolov8x-mseg-dlka3res-7]21/weights/best.pt',
    #                         imgsz=(608,960),
    #                         # dynamic=True,
    #                         batch=6,
    #                        )