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
    # demo_base.model_predict(
    #     [
    #         # "tf_defect_3-[yolov8x]",
    #         "f_defect_1-[yolov8x]-4",
    #     ],
    #     r"/scrinvme/huilin/traffic_sign/defect/detection/data_seg_1_damaged-guardrails/images",
    #     save_txt=True,
    #     save_conf=True,
    # )

    # demo_base.model_track("tf_det_1-[yolov9e]", img_dir=r"/scrinvme/huilin/tp/front")

    demo_base.model_val(
        [
            # r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t_update_v2-[yolov8x]/weights/best.pt",
            # r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t_update_v3-[yolov8x]/weights/best.pt",
            # r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t_update_v3-[yolov8x]-from-v2-7/weights/best.pt",
            # r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t_update_v3-[yolov8x]-from-v2-8/weights/best.pt",
            # r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t_update_v3-[yolov8x]-3/weights/best.pt",
            # r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t_update_v3-[yolov8x]-4/weights/best.pt",
            # r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t_update_v3-[yolov8x]-8/weights/best.pt",
            # r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t_update_v4-[yolov8x]-2/weights/best.pt",
            # r"hmt_t_update_v6-[yolov8x]-5",
            # r"hmt_t_update_v6-[yolov8x]-6",
            # r"hmt_t_update_v6-[yolov8x]-7",
            # r"hmt_t_update_v6-[yolov8x]-8",
            r"hmt_t_update_v6-[yolov8x]-10",
        ],
        # weight_name=False,
        # save_txt=True,
        # save_conf=True,
    )
    # demo_base.model_predict(
    #     [
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t_update_v3-[yolov8x]-7/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t_update_v3-[yolov8x]-8/weights/best.pt",
    #     ],
    #     r"/localnvme/data/bdd_hmt/hmt_t_update_v3/images",
    #     weight_name=False,
    #     save_txt=True,
    #     save_conf=True,
    # )
    # demo_base.model_predict(
    #     [
    #         # r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t_update_v4-[yolov8x]-2/weights/best.pt",
    #         # r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t_update_v4-[yolov8x]-3/weights/best.pt",
    #         # r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t_update_v5-[yolov8x]/weights/best.pt",
    #         # r"hmt_t_update_v6-[yolov8x]-5",
    #         r"hmt_t_update_v6-[yolov8x]-10",
    #     ],
    #     r"/localnvme/data/bdd_hmt/hmt_t_update_v6/images",
    #     # weight_name=False,
    #     save_txt=True,
    #     save_conf=True,
    # )
