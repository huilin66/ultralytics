import os

import torch

import demo_base

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
demo_base.TASK = "detect"
demo_base.EPOCHS = 300
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device("cuda:2")
demo_base.BATCH_SIZE = 32
demo_base.DATA = "gpr_v2.yaml"
# demo_base.DATA = "panel.yaml"  # [local] new panel dataset test
# demo_base.DATA = "BP_HMT_1216.yaml"
# demo_base.DATA = "hmt_t.yaml"  # origin/main_demo
# demo_base.CONF = 0.5


if __name__ == "__main__":
    NAME = None
    demo_base.yolo8("yolov8x.yaml", auto_optim=False, name=NAME)
