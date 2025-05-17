import demo_base
import torch

demo_base.TASK = 'msegment'
demo_base.EPOCHS = 100
demo_base.IMGSZ = 960
demo_base.BATCH_SIZE = 16
demo_base.DATA = "billboard_mseg_389_c6.yaml"

if __name__ == '__main__':
    pass
    demo_base.yolo8x('yolov8x-mseg-dlka3res-7.yaml', auto_optim=False, retrain=True, mloss_mask=False,
           weight_path=r'runs/segment/billboard_seg_389_c618/weights/best.pt', )
    demo_base.yolo8x('yolov8x-mseg-dlka3res-7.yaml', auto_optim=False, retrain=True, mloss_mask=True, mloss_weight=False,
           weight_path=r'runs/segment/billboard_seg_389_c618/weights/best.pt', )
    demo_base.yolo8x('yolov8x-mseg-dlka3res-7.yaml', auto_optim=False, retrain=True,  mloss_mask=True, mloss_weight=True,
           weight_path=r'runs/segment/billboard_seg_389_c618/weights/best.pt',)
    # demo_base.model_predict('runs/msegment/billboard_mseg_389re1_c6-yolov8x-mseg-7/weights/best.pt',
    #                         '/nfsv4/23039356r/data/billboard/bd_data/data389re1_mseg_c6/images')