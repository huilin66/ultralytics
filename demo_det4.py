import os

import torch

import demo_base

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
demo_base.TASK = "detect"
demo_base.EPOCHS = 300
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device("cuda:2")
demo_base.BATCH_SIZE = 16
demo_base.DATA = "PVEL_AD.yaml"


if __name__ == "__main__":
    NAME = None
    demo_base.model_val("runs/detect/rmcc_re-[yolov8x]/weights/best.pt", weight_name=False)
    demo_base.model_val("runs/detect/rmcc-[yolov8x]/weights/best.pt", weight_name=False)
