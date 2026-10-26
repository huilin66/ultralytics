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

    # demo_base.model_val("debug-12")
    # demo_base.model_val("debug-13")
    # demo_base.model_val("debug-16")
    # demo_base.model_val("hmt_sua_t_v6-[yolov8x]")
    # demo_base.model_val("hmt_sua_t_v6-[yolov9e]")
    # demo_base.model_val("hmt_sua_t_v6-[yolov10x]")
    # demo_base.model_val("hmt_sua_t_v6-[yolo11x]")
    # demo_base.model_val("hmt_sua_t_v6-[yolo12x]")
    # demo_base.model_val("hmt_sua_t_v6-[yolo26x]")
    # demo_base.model_val("debug-23")
    # demo_base.model_val("debug-24")
    demo_base.model_val("debug-26")
    demo_base.model_val("debug-27")

    # demo_base.model_val("debug-18")
    # demo_base.model_val("debug-19")
    # demo_base.model_val("debug-20")
    # demo_base.model_val("debug-21")
    # demo_base.model_val("debug-22")

    # demo_base.model_val("debug-30")
    # demo_base.model_val("debug-31")
    # demo_base.model_val("debug-32")
    # demo_base.model_val("debug-33")
    # demo_base.model_val("debug-36")
    # demo_base.model_val("debug-37")
    # demo_base.model_val("debug-38")
    # img_dir=r'/data/huilin/data/bdd_hmt/hmt_rgb_update_v5/images'
    # demo_base.model_predict("debug-30", img_dir, save_conf=True)
    # demo_base.model_predict("debug-31", img_dir, save_conf=True)
