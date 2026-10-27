import os

import torch

import demo_base

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
demo_base.TASK = "detect"
demo_base.EPOCHS = 300
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device("cuda:0")
demo_base.BATCH_SIZE = 8
# demo_base.DATA = "road_forniture_v2.yaml"
# demo_base.DATA = "BP_HMT_1216.yaml"
# demo_base.DATA = "hmt_t.yaml"  # origin/main_demo
# demo_base.CONF = 0.5


if __name__ == "__main__":
    NAME = None
    # demo_base.yolo8(
    #     "yolov8x.yaml",
    #     weight_path="yolov8x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     # name="debug",
    #     data="hmt_sua_rgb_v6.yaml",
    #     cls_pw=0.5,
    #     imgsz=960,
    # )
    # demo_base.yolo9(
    #     "yolov9e.yaml",
    #     weight_path="yolov9e.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     # name="debug",
    #     data="hmt_sua_rgb_v6.yaml",
    #     cls_pw=0.5,
    #     imgsz=960,
    # )
    # demo_base.yolo10(
    #     "yolov10x.yaml",
    #     weight_path="yolov10x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     # name="debug",
    #     data="hmt_sua_rgb_v6.yaml",
    #     cls_pw=0.5,
    #     imgsz=960,
    # )
    # demo_base.yolo11(
    #     "yolo11x.yaml",
    #     weight_path="yolo11x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     # name="debug",
    #     data="hmt_sua_rgb_v6.yaml",
    #     cls_pw=0.5,
    #     imgsz=960,
    # )
    # demo_base.yolo26(
    #     "yolo26x.yaml",
    #     weight_path="yolo26x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     # name="debug",
    #     data="hmt_sua_rgb_v6.yaml",
    #     cls_pw=0.5,
    #     imgsz=960,
    # )
    # demo_base.yolo12(
    #     "yolo12x.yaml",
    #     weight_path="yolo12x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     # name="debug",
    #     data="hmt_sua_rgb_v6.yaml",
    #     cls_pw=0.5,
    #     imgsz=960,
    # )
    # demo_base.model_val("hmt_sua_rgb_v6-[yolov8x]")
    # demo_base.model_val("hmt_sua_rgb_v6-[yolov9e]")
    # demo_base.model_val("hmt_sua_rgb_v6-[yolov10x]")
    # demo_base.model_val("hmt_sua_rgb_v6-[yolo11x]")
    # demo_base.model_val("hmt_sua_rgb_v6-[yolo12x]")
    # demo_base.model_val("hmt_sua_rgb_v6-[yolo26x]")

    img_dir = r"/srv/nas/home/dataset/hmt_rgb_update_v6/images"
    demo_base.model_predict("hmt_sua_rgb_v6-[yolov8x]", img_dir=img_dir, save_conf=True)
    demo_base.model_predict("hmt_sua_rgb_v6-[yolov9e]", img_dir=img_dir, save_conf=True)
    demo_base.model_predict("hmt_sua_rgb_v6-[yolov10x]", img_dir=img_dir, save_conf=True)
    demo_base.model_predict("hmt_sua_rgb_v6-[yolo11x]", img_dir=img_dir, save_conf=True)
    demo_base.model_predict("hmt_sua_rgb_v6-[yolo12x]", img_dir=img_dir, save_conf=True)
    demo_base.model_predict("hmt_sua_rgb_v6-[yolo26x]", img_dir=img_dir, save_conf=True)
