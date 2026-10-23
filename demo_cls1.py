import os

import torch

import demo_base

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
demo_base.TASK = "classify"
demo_base.EPOCHS = 100
demo_base.IMGSZ = 224
demo_base.DEVICE = torch.device("cuda:0")
demo_base.BATCH_SIZE = 64
demo_base.DATA = "/root/huilin/data/sign_defect_v1/"


if __name__ == "__main__":
    demo_base.yolo26("yolo26x-cls.yaml", weight_path="yolo26x-cls.pt", name="sign_defect_v1-yolo26x-cls")
