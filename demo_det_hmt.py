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
demo_base.DATA="hmt_sua_t_v6.yaml"
# os.environ["LEAKAGE_ONLY_LIST"] = r"/data/huilin/data/bdd_hmt/hmt_t_update_v6/train_leakage_loss_mask.txt"
img_dir=r'/data/huilin/data/bdd_hmt/hmt_t_update_v6/images'
if __name__ == "__main__":
    pass
    # demo_base.yolo8(
    #     "yolov8x.yaml",
    #     weight_path="yolov8x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     name="debug",
    #     data="hmt_sua_rgb_v2.yaml",
    # )
    # demo_base.yolo8(
    #     "yolov8x.yaml",
    #     weight_path="yolov8x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     name="debug",
    #     data="hmt_sua_rgb_v3.yaml",
    # )
    # demo_base.yolo8(
    #     "yolov8x.yaml",
    #     weight_path="yolov8x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     name="debug",
    #     data="hmt_sua_rgb_v2.yaml",
    #     cls_pw=0.5,
    # )
    # demo_base.yolo8(
    #     "yolov8x.yaml",
    #     weight_path="yolov8x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     name="debug",
    #     data="hmt_sua_rgb_v3.yaml",
    #     cls_pw=0.5,
    # )
    # # rgb leakage
    # demo_base.yolo8(
    #     "yolov8x.yaml",
    #     weight_path="yolov8x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     cls_pw=0.5,
    #     hsv_h=0.0,
    #     hsv_s=0.0,
    #     hsv_v=0.0,
    #     mosaic=0.25,
    # )
    # demo_base.yolo9(
    #     "yolov9e.yaml",
    #     weight_path="yolov9e.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     cls_pw=0.5,
    #     hsv_h=0.0,
    #     hsv_s=0.0,
    #     hsv_v=0.0,
    #     mosaic=0.25,
    # )
    # demo_base.yolo10(
    #     "yolov10x.yaml",
    #     weight_path="yolov10x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     cls_pw=0.5,
    #     hsv_h=0.0,
    #     hsv_s=0.0,
    #     hsv_v=0.0,
    #     mosaic=0.25,
    # )
    # demo_base.yolo11(
    #     "yolo11x.yaml",
    #     weight_path="yolo11x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     cls_pw=0.5,
    #     hsv_h=0.0,
    #     hsv_s=0.0,
    #     hsv_v=0.0,
    #     mosaic=0.25,
    # )
    # demo_base.yolo12(
    #     "yolo12x.yaml",
    #     weight_path="yolo12x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     cls_pw=0.5,
    #     hsv_h=0.0,
    #     hsv_s=0.0,
    #     hsv_v=0.0,
    #     mosaic=0.25,
    # )
    # demo_base.yolo26(
    #     "yolo26x.yaml",
    #     weight_path="yolo26x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     cls_pw=0.5,
    #     hsv_h=0.0,
    #     hsv_s=0.0,
    #     hsv_v=0.0,
    #     mosaic=0.25,
    # )
    # demo_base.yolo8(
    #     "yolov8x.yaml",
    #     weight_path="yolov8x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     name="debug",
    #     data="hmt_sua_t_v6.yaml",
    #     cls_pw=0.5,
    #     hsv_h=0.0,
    #     hsv_s=0.0,
    #     hsv_v=0.0,
    #     mosaic=0.25,
    #     imgsz=960,
    # )
    # demo_base.yolo8(
    #     "yolov8x.yaml",
    #     weight_path="yolov8x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     name="debug",
    #     data="hmt_sua_t_v6.yaml",
    #     cls_pw=0.5,
    #     hsv_h=0.0,
    #     hsv_s=0.0,
    #     hsv_v=0.0,
    #     mosaic=0.25,
    #     imgsz=1280,
    # )
    # rgb
    # demo_base.yolo8(
    #     "yolov8x.yaml",
    #     weight_path="yolov8x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     name="debug",
    #     data="hmt_sua_rgb_v1.yaml",
    # )
    # demo_base.yolo8(
    #     "yolov8x.yaml",
    #     weight_path="yolov8x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     name="debug",
    #     data="hmt_sua_rgb_v2.yaml",
    # )
    # demo_base.yolo8(
    #     "yolov8x.yaml",
    #     weight_path="yolov8x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     name="debug",
    #     data="hmt_sua_rgb_v3.yaml",
    # )
    # demo_base.yolo8(
    #     "yolov8x.yaml",
    #     weight_path="yolov8x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     name="debug",
    #     data="hmt_sua_rgb_v2.yaml",
    #     cls_pw=0.5,
    # )
    # demo_base.yolo8(
    #     "yolov8x.yaml",
    #     weight_path="yolov8x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     name="debug",
    #     data="hmt_sua_rgb_v3.yaml",
    #     cls_pw=0.5,
    # )
    # demo_base.yolo8(
    #     "yolov8x.yaml",
    #     weight_path="yolov8x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     name="debug",
    #     data="hmt_sua_rgb_v6.yaml",
    # )
    # demo_base.yolo8(
    #     "yolov8x.yaml",
    #     weight_path="yolov8x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     name="debug",
    #     data="hmt_sua_rgb_v6.yaml",
    #     cls_pw=0.5,
    #     imgsz=960,
    # )
    # demo_base.yolo8(
    #     "yolov8x.yaml",
    #     weight_path="yolov8x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     name="debug",
    #     data="hmt_sua_rgb_v6.yaml",
    #     cls_pw=0.5,
    #     imgsz=1280,
    # )
    demo_base.yolo8(
        "yolov8x.yaml",
        weight_path="yolov8x.pt",
        load_as_model=False,
        auto_optim=False,
        name="debug",
        data="hmt_bp_cube_v6.yaml",
    )
    demo_base.yolo8(
        "yolov8x.yaml",
        weight_path="yolov8x.pt",
        load_as_model=False,
        auto_optim=False,
        name="debug",
        data="hmt_bp_cube_v6.yaml",
        cls_pw=0.5,
    )
    demo_base.yolo8(
        "yolov8x.yaml",
        weight_path="yolov8x.pt",
        load_as_model=False,
        auto_optim=False,
        name="debug",
        data="hmt_bp_cube_v6.yaml",
        imgsz=960,
    )
    demo_base.yolo8(
        "yolov8x.yaml",
        weight_path="yolov8x.pt",
        load_as_model=False,
        auto_optim=False,
        name="debug",
        data="hmt_bp_cube_v6.yaml",
        cls_pw=0.5,
        imgsz=960,
    )
    demo_base.yolo8(
        "yolov8x.yaml",
        weight_path="yolov8x.pt",
        load_as_model=False,
        auto_optim=False,
        name="debug",
        data="hmt_bp_cube_v6.yaml",
        imgsz=1280,
    )
    demo_base.yolo8(
        "yolov8x.yaml",
        weight_path="yolov8x.pt",
        load_as_model=False,
        auto_optim=False,
        name="debug",
        data="hmt_bp_cube_v6.yaml",
        cls_pw=0.5,
        imgsz=1280,
    )