import os

import torch

import demo_base

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
demo_base.TASK = "classify"
demo_base.EPOCHS = 200
demo_base.IMGSZ = 224
demo_base.DEVICE = torch.device("cuda:0")
demo_base.BATCH_SIZE = 64
demo_base.DATA = "/srv/nas/home/dataset/WHU_Infra2D_cube_yolo_shape_core/ydm_vis/crop_split"


if __name__ == "__main__":
    # demo_base.yolo8("yolov8x-cls.yaml", weight_path="yolov8x-cls.pt")
    # demo_base.yolo26("yolo26x-cls.yaml", weight_path="yolo26x-cls.pt")
    demo_base.model_val("WHU_Infra2D_cube_yolo_shape_core-[yolov8x-cls]")
    demo_base.model_val("WHU_Infra2D_cube_yolo_shape_core-[yolo26x-cls]")
