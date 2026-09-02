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
# demo_base.DATA = "PVEL_AD.yaml"  # [local] new PVEL_AD dataset test
# demo_base.CONF = 0.5
demo_base.CONF_VAL = 0.001

if __name__ == "__main__":
    NAME = "debug"

    csv_path = r"summary_conf_0.01.csv"
    # demo_base.model_val(
    #     [
    #         # "tf_det_1-[yolov8x]-2",
    #         # "tf_det_1-[yolov8x]-3",
    #         "tf_det_1-[yolov9e]",
    #         # "tf_det_1-[yolov9e]-2",
    #         "tf_defect_3-[yolov8x]",
    #         # "tf_defect_3-[yolov8x]-2",
    #         # "tf_defect_3-[yolo26x]",
    #         # "tf_defect_3-[yolo26x]-2",
    #         # "f_defect_1-[yolov8x]-3",
    #         "f_defect_1-[yolov8x]-4",
    #         # "f_defect_1-[yolov9e]",
    #         # "f_defect_1-[yolov9e]-2",
    #     ],
    #     save_txt=True,
    #     save_conf=True,
    # )
    # demo_base.model_val(
    #     [
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t_update_v2-[yolov8x]/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t_update_v3-[yolov8x]/weights/best.pt",
    #     ],
    #     weight_name=False,
    #     save_txt=True,
    #     save_conf=True,
    # )
    demo_base.model_predict(
        "tf_defect_3-[yolov8x]",
        img_dir=r"/scrinvme/huilin/traffic_sign/highway/road sign defect/road sign defect",
        save_txt=True,
        save_conf=True,
    )
    # [local] PVEL_AD dataset quick test (commit f0353944)
    # demo_base.yolo12('yolo12x.yaml', auto_optim=False, batch=8)
    # demo_base.yolo26('yolo26x.yaml', auto_optim=False)
