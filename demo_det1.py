import os

import torch

import demo_base

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
demo_base.TASK = "detect"
demo_base.EPOCHS = 100
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device("cuda:0")
demo_base.BATCH_SIZE = 16
demo_base.DATA = "road_forniture_v2.yaml"
# demo_base.DATA = "BP_HMT_1216.yaml"
# demo_base.DATA = "hmt_t.yaml"  # origin/main_demo
# demo_base.CONF = 0.5


if __name__ == "__main__":
    NAME = None
    # demo_base.yolo8("yolov8x.yaml", auto_optim=False, name=NAME)
    # demo_base.yolo9("yolov9e.yaml", auto_optim=False, name=NAME)
    # demo_base.yolo11("yolo11x.yaml", auto_optim=False, name=NAME)
    demo_base.yolo26("yolo26x.yaml", auto_optim=False, name=NAME)

    # demo_base.model_val("road_forniture_v1-[yolov8x]-3")
    # demo_base.model_val("road_forniture_v1-[yolov9e]")
    # demo_base.model_val("road_forniture_v1-[yolo11x]")
    # demo_base.model_val("road_forniture_v1-[yolo26x]")

    # img_dir = r"D:\zhl\data\road_forniture_v1_rename\images"
    # demo_base.model_predict("road_forniture_v1-[yolov8x]-3", img_dir=img_dir, conf=0.1)
    # demo_base.model_predict("road_forniture_v1-[yolov9e]", img_dir=img_dir, conf=0.1)
    # demo_base.model_predict("road_forniture_v1-[yolo11x]", img_dir=img_dir, conf=0.1)
    # demo_base.model_predict("road_forniture_v1-[yolo26x]", img_dir=img_dir, conf=0.1)
