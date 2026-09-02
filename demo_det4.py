import demo_base
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
demo_base.TASK = 'detect'
demo_base.EPOCHS = 300
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device('cuda:2')
demo_base.BATCH_SIZE = 16
demo_base.DATA = "PVEL_AD.yaml"


if __name__ == '__main__':
    pass
    NAME = None
    demo_base.model_val('/root/huilin/projects/ultralytics/runs/detect/panel-[yolov8x]/weights/best.pt', weight_name=False)
    demo_base.model_val('/root/huilin/projects/ultralytics/runs/detect/panel-[yolov9e]/weights/best.pt', weight_name=False)
    demo_base.model_val('/root/huilin/projects/ultralytics/runs/detect/panel-[yolov10x]/weights/best.pt', weight_name=False)
    demo_base.model_val('/root/huilin/projects/ultralytics/runs/detect/panel-[yolo11x]/weights/best.pt', weight_name=False)
    demo_base.model_val('/root/huilin/projects/ultralytics/runs/detect/panel-[yolo12x]/weights/best.pt', weight_name=False)
    demo_base.model_val('/root/huilin/projects/ultralytics/runs/detect/panel-[yolo26x]/weights/best.pt', weight_name=False)

    demo_base.model_val('/root/huilin/projects/ultralytics/runs/detect/solarpanel-[yolov8x]2/weights/best.pt', weight_name=False)
    demo_base.model_val('/root/huilin/projects/ultralytics/runs/detect/solarpanel-[yolov9e]/weights/best.pt', weight_name=False)
