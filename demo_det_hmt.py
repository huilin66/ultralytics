import os

import torch

import demo_base

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
demo_base.TASK = "detect"
demo_base.EPOCHS = 100
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device("cuda:1")
demo_base.BATCH_SIZE = 16
demo_base.CONF_VAL = 0.001
DEFAULT_V2_WEIGHTS = (
    r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t_update_v2-[yolov8x]/weights/best.pt"
)
os.environ["LEAKAGE_ONLY_LIST"] = "/localnvme/data/bdd_hmt/hmt_t_update_v3/train_leakage_loss_mask.txt"


if __name__ == "__main__":
    demo_base.yolo8(
        "yolov8x.yaml",
        weight_path="yolov8x.pt",
        load_as_model=False,
        auto_optim=False,
        name="hmt_t_update_v6-[yolov8x]",
        data="hmt_t_update_v6.yaml",
        cls_pw=0.5,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        mosaic=0.25,
    )
    demo_base.yolo9(
        "yolov9e.yaml",
        weight_path="yolov9e.pt",
        load_as_model=False,
        auto_optim=False,
        name="hmt_t_update_v6-[yolov9e]",
        data="hmt_t_update_v6.yaml",
        cls_pw=0.5,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        mosaic=0.25,
    )
    demo_base.yolo10(
        "yolov10x.yaml",
        weight_path="yolov10x.pt",
        load_as_model=False,
        auto_optim=False,
        name="hmt_t_update_v6-[yolov10x]",
        data="hmt_t_update_v6.yaml",
        cls_pw=0.5,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        mosaic=0.25,
    )
    demo_base.yolo11(
        "yolo11x.yaml",
        weight_path="yolo11x.pt",
        load_as_model=False,
        auto_optim=False,
        name="hmt_t_update_v6-[yolo11x]",
        data="hmt_t_update_v6.yaml",
        cls_pw=0.5,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        mosaic=0.25,
    )
    demo_base.yolo12(
        "yolo12x.yaml",
        weight_path="yolo12x.pt",
        load_as_model=False,
        auto_optim=False,
        name="hmt_t_update_v6-[yolo12x]",
        data="hmt_t_update_v6.yaml",
        cls_pw=0.5,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        mosaic=0.25,
    )
    demo_base.yolo26(
        "yolo26x.yaml",
        weight_path="yolo26x.pt",
        load_as_model=False,
        auto_optim=False,
        name="hmt_t_update_v6-[yolo26x]",
        data="hmt_t_update_v6.yaml",
        cls_pw=0.5,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        mosaic=0.25,
    )
