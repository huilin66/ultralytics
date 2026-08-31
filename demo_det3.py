import os

import torch

import demo_base

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
demo_base.TASK = "detect"
demo_base.EPOCHS = 200
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device("cuda:1")
demo_base.BATCH_SIZE = 16
# demo_base.DATA = ".yaml"
# demo_base.CONF = 0.5
demo_base.CONF_VAL = 0.001

if __name__ == "__main__":
    NAME = "debug"

    csv_path = r"summary_hmt_t_update_v6.csv"

    demo_base.model_val_summary(
        [
            r"hmt_t_update_v6-[yolov8x]-2",
            r"hmt_t_update_v6-[yolov9e]-2",
            r"hmt_t_update_v6-[yolov10x]-2",
            r"hmt_t_update_v6-[yolo11x]-2",
            r"hmt_t_update_v6-[yolo12x]-2",
            r"hmt_t_update_v6-[yolo26x]-2",
        ],
        csv_path=csv_path,
        # weight_name=False,
        # save_txt=True,
        # save_conf=True,
    )
