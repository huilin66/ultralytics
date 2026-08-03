import os

import torch

import demo_base

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
demo_base.TASK = "detect"
demo_base.EPOCHS = 300
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device("cuda:1")
demo_base.BATCH_SIZE = 16
demo_base.CONF_VAL = 0.01

if __name__ == "__main__":
    NAME = None

    # demo_base.model_val("hmt_t-[yolov8x]-8")

    # demo_base.model_val("hmt_rgb_merge-[yolov8x]-4")

    # demo_base.model_val_summary("hmt_bp_cube-[yolov8x]4")
