import os

import torch

import demo_base

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
demo_base.TASK = "detect"
demo_base.EPOCHS = 300
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device("cuda:0")
demo_base.BATCH_SIZE = 16
# demo_base.DATA = ".yaml"
# demo_base.CONF = 0.5
demo_base.CONF_VAL = 0.01

if __name__ == "__main__":
    NAME = None

    csv_path = r"summary_conf_0.01.csv"

    # demo_base.model_val(
    #     "/localnvme/project/aic_mdet/models/ultralytics/runs/detect/tf_det_1-[yolov8x]-2/weights/best.pt",
    #     weight_name=False,
    # )

    demo_base.model_val_summary(
        "/localnvme/project/aic_mdet/models/ultralytics/runs/detect/f_defect_1-[yolov8x]-3/weights/best.pt",
        weight_name=False,
    )
    demo_base.model_val_summary(
        "/localnvme/project/aic_mdet/models/ultralytics/runs/detect/f_defect_1-[yolov8x]-4/weights/best.pt",
        weight_name=False,
    )

    demo_base.model_val(
        "/localnvme/project/aic_mdet/models/ultralytics/runs/detect/f_defect_1-[yolov9e]/weights/best.pt",
        weight_name=False,
    )
    demo_base.model_val(
        "/localnvme/project/aic_mdet/models/ultralytics/runs/detect/f_defect_1-[yolov9e]-2/weights/best.pt",
        weight_name=False,
    )

    # demo_base.model_val("tf_defect_3-[yolov8x]")
    # demo_base.model_val("tf_defect_3-[yolov8x]-2")
    # demo_base.model_val("tf_defect_3-[yolo26x]")
    # demo_base.model_val("tf_defect_3-[yolo26x]-2")

    # demo_base.model_val("hmt_t-[yolov8x]-6")
    # demo_base.model_val("hmt_rgb_merge_v2-[yolov8x]-5")
    # demo_base.model_val("hmt_bp_cube-[yolov8x]2")

    # demo_base.model_val('hmt_t-[yolov8x]-8', conf=0.001, iou=0.5)
    # demo_base.model_val('hmt_rgb_merge_f01-[yolov8x]')
    # demo_base.model_val('hmt_rgb_merge-[yolov8x]-2')

    # demo_base.model_val("tf_defect_3-[yolov8x]", save_conf=True, save_txt=True)
    # demo_base.model_val("tf_defect_3-[yolov8x]-2")
    # demo_base.model_val("tf_defect_3-[yolo26x]")
    # demo_base.model_val("tf_defect_3-[yolo26x]-2")

    # demo_base.model_val('hmt_rgb_merge-[yolov8x]-4', save_conf=True, save_txt=True)
    # demo_base.model_val('hmt_t-[yolov8x]-6', save_conf=True, save_txt=True)
    # demo_base.model_val('hmt_bp_cube_f_p01-[yolov8x]', save_conf=True, save_txt=True)

    # demo_base.model_val('hmt_t_f_p01_merge-[yolov8x]', save_conf=True, save_txt=True)
    # demo_base.model_val('hmt_t_f_p01_merge-[yolov8x]', save_conf=True, save_txt=True)

    # demo_base.model_val_summary('hmt_bp_cube-[yolov8x]2', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_bp_cube-[yolov8x]3', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_bp_cube-[yolov8x]4', csv_path=csv_path)

    # demo_base.model_val_summary('hmt_bp_cube_f_p01-[yolov8x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_bp_cube_f_p01-[yolov8x]2', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_bp_cube_f_p01-[yolov8x]3', csv_path=csv_path)

    # demo_base.model_val_summary('hmt_rgb-[yolov8x]2', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_rgb-[yolov8x]3', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_rgb-[yolov8x]4', csv_path=csv_path)

    # demo_base.model_val_summary('hmt_rgb_f_p01-[yolov8x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_rgb_f_p01-[yolov8x]2', csv_path=csv_path)

    # demo_base.model_val_summary('hmt_t-[yolov8x]2', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_t-[yolov8x]3', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_t-[yolov8x]4', csv_path=csv_path)

    # demo_base.model_val_summary('hmt_t_f_p01-[yolov8x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_t_f_p01-[yolov8x]2', csv_path=csv_path)

    # demo_base.model_val_summary('hmt_bp_cube-[yolov8x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_bp_cube-[yolov9e]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_bp_cube-[yolov10x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_bp_cube-[yolo11x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_bp_cube-[yolo12x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_bp_cube-[yolo26x]', csv_path=csv_path)

    # demo_base.model_val_summary('hmt_t-[yolov8x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_t-[yolov9e]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_t-[yolov10x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_t-[yolo11x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_t-[yolo12x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_t-[yolo26x]', csv_path=csv_path)

    # demo_base.model_val_summary('hmt_t_rgbt-[yolov8x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_t_rgbt-[yolov9e]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_t_rgbt-[yolov10x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_t_rgbt-[yolo11x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_t_rgbt-[yolo12x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_t_rgbt-[yolo26x]', csv_path=csv_path)

    # demo_base.model_val_summary('hmt_rgb-[yolov8x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_rgb-[yolov9e]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_rgb-[yolov10x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_rgb-[yolo11x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_rgb-[yolo12x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_rgb-[yolo26x]', csv_path=csv_path)

    # demo_base.model_val_summary('hmt_rgb_rgbt-[yolov8x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_rgb_rgbt-[yolov9e]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_rgb_rgbt-[yolov10x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_rgb_rgbt-[yolo11x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_rgb_rgbt-[yolo12x]', csv_path=csv_path)
    # demo_base.model_val_summary('hmt_rgb_rgbt-[yolo26x]', csv_path=csv_path)
