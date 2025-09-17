import demo_base
import torch

demo_base.TASK = 'msegment'
demo_base.EPOCHS = 100
demo_base.IMGSZ = 960
demo_base.BATCH_SIZE = 16
demo_base.DEVICE = torch.device('cuda:1')
demo_base.DATA = "fusedata2419_mseg_c5_0730.yaml"

MODEL0 = 'yolov8x-mseg-dlka3res-7.yaml'
MODEL1 = 'yolov8x-mseg-dlka3res-7-dlka.yaml'
MODEL2 = 'yolov8x-mseg-dlka3res-7-dlkaatt.yaml'
MODEL3 = 'yolov8x-mseg-dlka3res-7-c3str.yaml'
MODEL4 = 'yolov8x-mseg-dlka3res-7-dfl.yaml'
SEG_WEIGHT = "runs/segment/fusedata5894_seg_c5_0822_80p-[yolov8x-seg-dlka3res]2/weights/best.pt"
MSEG_WEIGHT = "runs/msegment/fusedata6010_mseg_c5_0903_80p-[yolov8x-mseg-dlka3res-7]5/weights/best.pt"
DATA1 = "fusedata6010_mseg_c5_0911_80p.yaml"
DATA2 = "fusedata6010_mseg_c5_0911_75p.yaml"
DATA3 = "fusedata6010_mseg_c5_0911_70p.yaml"
DATA4 = "fusedata6010_mseg_c5_0911_65p.yaml"
DATA5 = "fusedata6010_mseg_c5_0911_60p.yaml"

if __name__ == '__main__':
    pass
    # demo_base.yolo8(
    #     MODEL2, weight_path=r'runs/msegment/fusedata6010_mseg_c5_0903_80p-[yolov8x-mseg-dlka3res-7]5/weights/best.pt', data=DATA1, auto_optim=False, retrain=True,
    #     mloss_mask=True, mloss_weight=False, mloss_enlarge=0,
    # )


    # demo_base.model_val(r'fusedata7436_mseg_c5_0912_80p-[yolov8x-mseg-dlka3res-7]')
    # demo_base.model_val(r'fusedata4197_mseg_c5_l2_0912_80p-[yolov8x-mseg-dlka3res-7]')
    # demo_base.model_val(r'fusedata7436_mseg_c5_0914_80p-[yolov8x-mseg-dlka3res-7]')
    # demo_base.model_val(r'fusedata7436_mseg_c5_l2_0914_80p-[yolov8x-mseg-dlka3res-7]')
    # demo_base.model_val(r'fusedata4197_mseg_c5_0914_80p-[yolov8x-mseg-dlka3res-7]')
    # demo_base.model_val(r'fusedata4197_mseg_c5_l2_0914_80p-[yolov8x-mseg-dlka3res-7]')


    # demo_base.model_predict(
    #     r'runs/msegment/fusedata6010_mseg_c5_0903_80p-[yolov8x-mseg-dlka3res-7]5/weights/best.pt',
    #     img_dir = r'/localnvme/data/billboard/fused_data/data5894_mseg_c5_0822/val/images',
    #     name=r'/localnvme/data/billboard/fused_data/data5894_mseg_c5_0822/val/images_infer_mseg',
    #     save_risk_score=True,
    # )
    demo_base.model_export(r'fusedata4197_mseg_c5_0914_80p-[yolov8x-mseg-dlka3res-7]',
                            imgsz=(608,960), batch=6,
                            # dynamic=True,
                           )
