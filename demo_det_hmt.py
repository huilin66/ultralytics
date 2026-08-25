import os

import torch

import demo_base

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
demo_base.TASK = "detect"
demo_base.EPOCHS = 300
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device("cuda:1")
demo_base.BATCH_SIZE = 16
demo_base.CONF_VAL = 0.001
DEFAULT_V2_WEIGHTS = (
    r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t_update_v2-[yolov8x]/weights/best.pt"
)
os.environ["LEAKAGE_ONLY_LIST"] = "/localnvme/data/bdd_hmt/hmt_t_update_v3/train_leakage_loss_mask.txt"


if __name__ == "__main__":
    # demo_base.model_val("hmt_t-[yolov8x]-8")
    # demo_base.model_val("hmt_rgb_merge-[yolov8x]-4")
    # demo_base.model_val("hmt_bp_cube-[yolov8x]4")

    # demo_base.model_val("hmt_t-[yolov8x]-8", save_txt=True, save_conf=True)
    # demo_base.model_predict("hmt_t-[yolov8x]-8", r"/localnvme/data/bdd_hmt/sua_t/images", save_txt=True, save_conf=True)

    # demo_base.yolo8("yolov8x.yaml", auto_optim=False, name=NAME, data="hmt_t.yaml")
    # demo_base.yolo9("yolov9e.yaml", auto_optim=False, name=NAME, data="hmt_t.yaml")
    # demo_base.yolo10("yolov10x.yaml", auto_optim=False, name=NAME, data="hmt_t.yaml")
    # demo_base.yolo11("yolo11x.yaml", auto_optim=False, name=NAME, data="hmt_t.yaml")
    # demo_base.yolo12("yolo12x.yaml", auto_optim=False, name=NAME, data="hmt_t.yaml")
    # demo_base.yolo26("yolo26x.yaml", auto_optim=False, name=NAME, data="hmt_t.yaml")

    # demo_base.yolo8("yolov8x.yaml", auto_optim=False, name=NAME, data="hmt_t.yaml", cls_pw=0.5)
    # demo_base.yolo9("yolov9e.yaml", auto_optim=False, name=NAME, data="hmt_t.yaml", cls_pw=0.5)
    # demo_base.yolo10("yolov10x.yaml", auto_optim=False, name=NAME, data="hmt_t.yaml", cls_pw=0.5)
    # demo_base.yolo11("yolo11x.yaml", auto_optim=False, name=NAME, data="hmt_t.yaml", cls_pw=0.5)
    # demo_base.yolo12("yolo12x.yaml", auto_optim=False, name=NAME, data="hmt_t.yaml", cls_pw=0.5)
    # demo_base.yolo26("yolo26x.yaml", auto_optim=False, name=NAME, data="hmt_t.yaml", cls_pw=0.5)

    # demo_base.yolo8("yolov8x.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml")
    # demo_base.yolo9("yolov9e.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml")
    # demo_base.yolo10("yolov10x.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml")
    # demo_base.yolo11("yolo11x.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml")
    # demo_base.yolo12("yolo12x.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml")
    # demo_base.yolo26("yolo26x.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml")

    # demo_base.yolo8("yolov8x.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml", cls_pw=0.5)
    # demo_base.yolo9("yolov9e.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml", cls_pw=0.5)
    # demo_base.yolo10("yolov10x.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml", cls_pw=0.5)
    # demo_base.yolo11("yolo11x.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml", cls_pw=0.5)
    # demo_base.yolo12("yolo12x.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml", cls_pw=0.5)
    # demo_base.yolo26("yolo26x.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml", cls_pw=0.5)

    # csv_path = r"runs/summary_hmt_t.csv"
    # demo_base.model_val_summary(
    #     [
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t-[yolov8x]/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t-[yolov9e]/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t-[yolov10x]/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t-[yolo11x]/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t-[yolo12x]-2/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t-[yolo26x]/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t-[yolov8x]-3/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t-[yolov9e]-2/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t-[yolov10x]-2/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t-[yolo11x]-2/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t-[yolo12x]-3/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_t-[yolo26x]-2/weights/best.pt",
    #     ],
    #     csv_path=csv_path,
    #     weight_name=False,
    # )

    # csv_path = r"runs/summary_hmt_rgb.csv"
    # demo_base.model_val_summary(
    #     [
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/debug-5/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/debug-6/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/debug-7/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/debug-8/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/debug-9/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/debug-10/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/debug-11/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/debug-12/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/debug-13/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/debug-14/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/debug-15/weights/best.pt",
    #         r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/debug-16/weights/best.pt",
    #     ],
    #     csv_path=csv_path,
    #     weight_name=False,
    # )

    # demo_base.model_val(
    #     r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/debug-11/weights/best.pt",
    #     save_txt=True,
    #     save_conf=True,
    #     weight_name=False,
    # )
    # demo_base.model_predict(
    #     r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/debug-11/weights/best.pt",
    #     r"/localnvme/data/bdd_hmt/sua_rgb/images",
    #     save_txt=True,
    #     save_conf=True,
    #     weight_name=False,
    #     conf=0.001,
    # )

    # demo_base.yolo8("yolov8x.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml")
    # demo_base.yolo8("yolov8x.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml", cls_pw=0.5)
    # demo_base.model_val(
    #     "/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_rgb-[yolov8x]/weights/best.pt",
    #     weight_name=False,
    #     save_txt=True,
    #     save_conf=True,
    # )
    # demo_base.model_val(
    #     "/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_rgb-[yolov8x]-2/weights/best.pt",
    #     weight_name=False,
    #     save_txt=True,
    #     save_conf=True,
    # )

    # demo_base.model_predict(
    #     r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_rgb-[yolov8x]/weights/best.pt",
    #     r"/localnvme/data/bdd_hmt/sua_rgb/images",
    #     save_txt=True,
    #     save_conf=True,
    #     weight_name=False,
    #     conf=0.001,
    # )
    # demo_base.yolo8("yolov8x.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml")
    # demo_base.yolo8("yolov8x.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml", cls_pw=0.5)
    # demo_base.model_val(
    #     "/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_rgb-[yolov8x]-7/weights/best.pt",
    #     weight_name=False,
    #     save_txt=True,
    #     save_conf=True,
    # )
    # demo_base.model_val(
    #     "/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_rgb-[yolov8x]-8/weights/best.pt",
    #     weight_name=False,
    #     save_txt=True,
    #     save_conf=True,
    # )
    # demo_base.model_predict(
    #     r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_rgb-[yolov8x]-8/weights/best.pt",
    #     r"/localnvme/data/bdd_hmt/sua_rgb/images",
    #     save_txt=True,
    #     save_conf=True,
    #     weight_name=False,
    #     conf=0.001,
    # )
    # demo_base.yolo8("yolov8x.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml")
    # demo_base.yolo8("yolov8x.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml", cls_pw=0.5)
    # demo_base.yolo8("yolov8x.yaml", auto_optim=False, name=NAME, data="hmt_rgb_update.yaml")
    # demo_base.yolo8("yolov8x.yaml", auto_optim=False, name=NAME, data="hmt_rgb_update.yaml", cls_pw=0.5)
    # demo_base.model_val(
    #     "/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_rgb-[yolov8x]-9/weights/best.pt",
    #     weight_name=False,
    #     save_txt=True,
    #     save_conf=True,
    # )
    # demo_base.model_val(
    #     "/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_rgb-[yolov8x]-11/weights/best.pt",
    #     weight_name=False,
    #     save_txt=True,
    #     save_conf=True,
    # )

    # demo_base.model_val("hmt_bp_cube-[yolov8x]4", save_txt=True, save_conf=True)
    # demo_base.model_predict(
    #     r"hmt_bp_cube-[yolov8x]4",
    #     r"/localnvme/data/bdd_hmt/bp_cube/images",
    #     save_txt=True,
    #     save_conf=True,
    #     conf=0.001,
    # )
    # demo_base.yolo8("yolov8x.yaml", auto_optim=False, name=NAME, data="hmt_bp_cube.yaml")
    # demo_base.yolo8("yolov8x.yaml", auto_optim=False, name=NAME, data="hmt_bp_cube.yaml", cls_pw=0.5)

    # demo_base.model_val(
    #     "/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_bp_cube-[yolov8x]/weights/best.pt",
    #     weight_name=False,
    #     save_txt=True,
    #     save_conf=True,
    # )
    # demo_base.model_val(
    #     "/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_bp_cube-[yolov8x]-2/weights/best.pt",
    #     weight_name=False,
    #     save_txt=True,
    #     save_conf=True,
    # )

    # demo_base.model_predict(
    #     r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/hmt_bp_cube-[yolov8x]/weights/best.pt",
    #     r"/localnvme/data/bdd_hmt/bp_cube/images",
    #     weight_name=False,
    #     save_txt=True,
    #     save_conf=True,
    #     conf=0.001,
    # )

    # demo_base.yolo8("yolov8x.yaml", auto_optim=False, name=NAME, data="hmt_t.yaml")
    # demo_base.yolo9("yolov9e.yaml", auto_optim=False, name=NAME, data="hmt_t.yaml")
    # demo_base.yolo10("yolov10x.yaml", auto_optim=False, name=NAME, data="hmt_t.yaml")
    # demo_base.yolo11("yolo11x.yaml", auto_optim=False, name=NAME, data="hmt_t.yaml")
    # demo_base.yolo12("yolo12x.yaml", auto_optim=False, name=NAME, data="hmt_t.yaml")
    # demo_base.yolo26("yolo26x.yaml", auto_optim=False, name=NAME, data="hmt_t.yaml")

    # demo_base.yolo8("yolov8x.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml")
    # demo_base.yolo9("yolov9e.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml")
    # demo_base.yolo10("yolov10x.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml")
    # demo_base.yolo11("yolo11x.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml")
    # demo_base.yolo12("yolo12x.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml")
    # demo_base.yolo26("yolo26x.yaml", auto_optim=False, name=NAME, data="hmt_rgb.yaml")

    # demo_base.yolo8("yolov8x.yaml", auto_optim=False, name=NAME, data="hmt_bp_cube.yaml")
    # demo_base.yolo9("yolov9e.yaml", auto_optim=False, name=NAME, data="hmthmt_bp_cube_rgb.yaml")
    # demo_base.yolo10("yolov10x.yaml", auto_optim=False, name=NAME, data="hmt_bp_cube.yaml")
    # demo_base.yolo11("yolo11x.yaml", auto_optim=False, name=NAME, data="hmt_bp_cube.yaml")
    # demo_base.yolo12("yolo12x.yaml", auto_optim=False, name=NAME, data="hmt_bp_cube.yaml")
    # demo_base.yolo26("yolo26x.yaml", auto_optim=False, name=NAME, data="hmt_bp_cube.yaml")

    # demo_base.model_val(
    #     [
    #         "hmt_t-[yolov8x]",
    #         "hmt_t-[yolov9e]",
    #         "hmt_t-[yolov10x]",
    #         "hmt_t-[yolo11x]",
    #         "hmt_t-[yolo12x]",
    #         "hmt_t-[yolo26x]",
    #         "hmt_rgb-[yolov8x]",
    #         "hmt_rgb-[yolov9e]",
    #         "hmt_rgb-[yolov10x]",
    #         "hmt_rgb-[yolo11x]",
    #         "hmt_rgb-[yolo12x]",
    #         "hmt_rgb-[yolo26x]",
    #         "hmt_bp_cube-[yolov8x]",
    #         "hmt_bp_cube-[yolov9e]",
    #         "hmt_bp_cube-[yolov10x]",
    #         "hmt_bp_cube-[yolo11x]",
    #         "hmt_bp_cube-[yolo12x]",
    #         "hmt_bp_cube-[yolo26x]",
    #     ],
    #     save_txt=True,
    #     save_conf=True,
    # )

    # demo_base.model_val("hmt_t-[yolov8x]", data="hmt_t.yaml", save_txt=True, save_conf=True)
    # demo_base.model_val("hmt_t-[yolov9e]", data="hmt_t.yaml", save_txt=True, save_conf=True)
    # demo_base.model_val("hmt_t-[yolov10x]", data="hmt_t.yaml", save_txt=True, save_conf=True)
    # demo_base.model_val("hmt_t-[yolo11x]", data="hmt_t.yaml", save_txt=True, save_conf=True)
    # demo_base.model_val("hmt_t-[yolo12x]", data="hmt_t.yaml", save_txt=True, save_conf=True)
    # demo_base.model_val("hmt_t-[yolo26x]", data="hmt_t.yaml", save_txt=True, save_conf=True)

    # demo_base.model_val("hmt_rgb-[yolov8x]", data="hmt_rgb.yaml", save_txt=True, save_conf=True)
    # demo_base.model_val("hmt_rgb-[yolov9e]", data="hmt_rgb.yaml", save_txt=True, save_conf=True)
    # demo_base.model_val("hmt_rgb-[yolov10x]", data="hmt_rgb.yaml", save_txt=True, save_conf=True)
    # demo_base.model_val("hmt_rgb-[yolo11x]", data="hmt_rgb.yaml", save_txt=True, save_conf=True)
    # demo_base.model_val("hmt_rgb-[yolo12x]", data="hmt_rgb.yaml", save_txt=True, save_conf=True)
    # demo_base.model_val("hmt_rgb-[yolo26x]", data="hmt_rgb.yaml", save_txt=True, save_conf=True)

    # demo_base.model_val("hmt_bp_cube-[yolov8x]", data="hmt_bp_cube.yaml", save_txt=True, save_conf=True)
    # demo_base.model_val("hmt_bp_cube-[yolov9e]", data="hmt_bp_cube.yaml", save_txt=True, save_conf=True)
    # demo_base.model_val("hmt_bp_cube-[yolov10x]", data="hmt_bp_cube.yaml", save_txt=True, save_conf=True)
    # demo_base.model_val("hmt_bp_cube-[yolo11x]", data="hmt_bp_cube.yaml", save_txt=True, save_conf=True)
    # demo_base.model_val("hmt_bp_cube-[yolo12x]", data="hmt_bp_cube.yaml", save_txt=True, save_conf=True)
    # demo_base.model_val("hmt_bp_cube-[yolo26x]", data="hmt_bp_cube.yaml", save_txt=True, save_conf=True)

    # demo_base.yolo8(
    #     "yolov8x.yaml",
    #     weight_path="yolov8x.pt",
    #     load_as_model=False,
    #     auto_optim=False,
    #     name="hmt_t_update_v3-[yolov8x]",
    #     data="hmt_t_update_v3.yaml",
    # )
    # demo_base.yolo8(
    #     "yolov8x.yaml",
    #     weight_path=DEFAULT_V2_WEIGHTS,
    #     load_as_model=True,
    #     auto_optim=False,
    #     name="hmt_t_update_v3-[yolov8x]-from-v2",
    #     data="hmt_t_update_v3.yaml",
    # )
    demo_base.yolo8(
        "yolov8x.yaml",
        weight_path="yolov8x.pt",
        load_as_model=False,
        auto_optim=False,
        name="hmt_t_update_v4-[yolov8x]",
        data="hmt_t_update_v4.yaml",
        cls_pw=0.5,
    )
    demo_base.yolo8(
        "yolov8x.yaml",
        weight_path="yolov8x.pt",
        load_as_model=False,
        auto_optim=False,
        name="hmt_t_update_v5-[yolov8x]",
        data="hmt_t_update_v5.yaml",
        cls_pw=0.5,
    )
