import os

import torch

import demo_base

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
demo_base.TASK = "detect"
demo_base.EPOCHS = 200
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device("cuda:0")
demo_base.BATCH_SIZE = 16
# demo_base.DATA = ".yaml"
# demo_base.CONF = 0.5
demo_base.CONF_VAL = 0.001

if __name__ == "__main__":
    NAME = "debug"

    csv_path = r"summary_conf_0.01.csv"
    demo_base.yolo8("yolov8x.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml", cls_pw=0.5)
