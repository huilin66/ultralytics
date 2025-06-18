import demo_base
import torch

demo_base.TASK = 'msegment'
demo_base.EPOCHS = 100
demo_base.IMGSZ = 960
demo_base.BATCH_SIZE = 16
demo_base.DEVICE = torch.device('cuda:0')
demo_base.DATA = "psdata735_mseg_c6.yaml"

if __name__ == '__main__':
    pass
    # demo_base.yolo8('yolov8x-mseg-7.yaml', auto_optim=False, retrain=True, mloss_mask=True, mloss_weight=True,
    #        weight_path=r'runs/segment/psdata735_seg_c6-[yolov8x-seg]3/weights/best.pt', )

    # demo_base.yolo8('yolov8x-mseg-7.yaml', auto_optim=False, retrain=True, mloss_mask=True, mloss_weight=True,
    #        data="psdata735_mseg_c6_check0618.yaml",
    #        weight_path=r'runs/segment/psdata735_seg_c6-[yolov8x-seg]3/weights/best.pt', )

    # demo_base.yolo8('yolov8x-mseg-7.yaml', auto_optim=False, retrain=True, mloss_mask=True, mloss_weight=True,
    #        data="psdata735_mseg_c6_check0618_update.yaml",
    #        weight_path=r'runs/segment/psdata735_seg_c6-[yolov8x-seg]3/weights/best.pt', )


    # demo_base.yolo8('yolov8x-mseg-7.yaml', auto_optim=False, retrain=True, mloss_mask=True, mloss_weight=True,
    #        data="fusedata1361_mseg_c6_check0618.yaml",
    #        weight_path=r'runs/segment/fusedata1361_seg_c6-[yolov8x-seg]3/weights/best.pt', )
    demo_base.yolo8('yolov8x-mseg-7.yaml', auto_optim=False, retrain=True, mloss_mask=True, mloss_weight=True,
           data="fusedata1361_mseg_c6_check0618_update.yaml",
           weight_path=r'runs/segment/fusedata1361_seg_c6-[yolov8x-seg]3/weights/best.pt', )

    # demo_base.model_val(r'runs/msegment/psdata735_mseg_c6-[yolov8x-mseg-7]11/weights/best.pt')
    # demo_base.model_val(r'runs/msegment/psdata735_mseg_c6_check0618-[yolov8x-mseg-7]6/weights/best.pt')
    # demo_base.model_val(r'runs/msegment/psdata735_mseg_c6_check0618-[yolov8x-mseg-7]6/weights/best.pt')