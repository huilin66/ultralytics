import os

import torch

import demo_base

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
demo_base.TASK = "detect"
demo_base.EPOCHS = 300
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device("cuda:0")
demo_base.BATCH_SIZE = 16
# demo_base.DATA = ".yaml"
# demo_base.CONF = 0.5


if __name__ == "__main__":
    NAME = "debug"
    demo_base.model_track("runs\\key_results\rmcc-[yolov8x]\\weights\best.pt", weight_name=False, name=NAME)
