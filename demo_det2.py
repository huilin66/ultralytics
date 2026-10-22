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

    # demo_base.model_val("road_forniture_v2-[yolo26x]", data="road_forniture_add_v1_c2.yaml")
    # demo_base.model_val("road_forniture_v2-[yolo26x]", data="road_forniture_add_v2_c2.yaml")
    # demo_base.model_val("road_forniture_add_v1_c2-[yolo26x]-4", data="road_forniture_add_v1_c2.yaml")
    # demo_base.model_val("road_forniture_add_v2_c2-[yolo26x]", data="road_forniture_add_v2_c2.yaml")
    # demo_base.model_val("road_forniture_add_v1_c2-[yolo26x]-4", data="road_forniture_add_v2_c2.yaml")
    # demo_base.model_val("road_forniture_add_v2_c2-[yolo26x]", data="road_forniture_add_v1_c2.yaml")
    # demo_base.model_val("road_furniture_add_v3-[yolo26x]")
    # demo_base.model_val("road_furniture_add_v3-[yolo26x]-2")
    # demo_base.model_val("road_furniture_add_v3-[yolo26x]-3")
    # demo_base.model_val("road_furniture_add_v3_c3-[yolo26x]")
    # demo_base.model_val("road_furniture_add_v3_c3-[yolo26x]-2")
    # demo_base.model_val("road_furniture_add_v3_c3-[yolo26x]-3")
    demo_base.model_val("road_forniture_add_v2_c2-[yolo26x]-2")
    demo_base.model_val("road_furniture_add_v3_c2-[yolo26x]")
    demo_base.model_val("road_furniture_add_v3_c2_split-[yolo26x]-2")
