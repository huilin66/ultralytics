import os

import torch

import demo_base

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
demo_base.TASK = "detect"
demo_base.EPOCHS = 100
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device("cuda:0")
demo_base.BATCH_SIZE = 16
demo_base.CONF_VAL = 0.001


if __name__ == "__main__":
    pass
    # Primary HMT clean-v1 experiment (thermal, 3 classes).
    demo_base.yolo8(
        "yolov8x.yaml",
        weight_path="yolov8x.pt",
        load_as_model=False,
        auto_optim=False,
        name="debug",
        data="hmt_clean_v1_sua_t.yaml",
        cls_pw=0.5,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        mosaic=0.25,
    )
    # Variant with a fixed run name.
    # demo_base.yolo8(
    #     "yolov8x.yaml",
    #     weight_path="yolov8x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     name="hmt_clean_v1_sua_t-[yolov8x]",
    #     data="hmt_clean_v1_sua_t.yaml",
    #     cls_pw=0.5,
    #     hsv_h=0.0,
    #     hsv_s=0.0,
    #     hsv_v=0.0,
    #     mosaic=0.25,
    # )
