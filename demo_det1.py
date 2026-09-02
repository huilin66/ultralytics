import os

import torch

import demo_base

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
demo_base.TASK = "detect"
demo_base.EPOCHS = 300
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device("cuda:1")
demo_base.BATCH_SIZE = 16
demo_base.DATA = "tf_det_1.yaml"
# demo_base.DATA = "panel.yaml"  # [local] new panel dataset test
# demo_base.DATA = "BP_HMT_1216.yaml"
# demo_base.DATA = "hmt_t.yaml"  # origin/main_demo
# demo_base.CONF = 0.5


if __name__ == "__main__":
    NAME = None
    demo_base.model_train_resume(
        r"/localnvme/project/aic_mdet/models/ultralytics/runs/detect/tf_det_1-[yolov8x]-2/weights/last.pt"
    )
    # demo_base.yolo8("yolov8x.yaml", auto_optim=False, name=NAME)
    demo_base.yolo8("yolov8x.yaml", auto_optim=False, name=NAME, cls_pw=0.5)

    demo_base.yolo9("yolov9e.yaml", auto_optim=False, name=NAME)
    demo_base.yolo9("yolov9e.yaml", auto_optim=False, name=NAME, cls_pw=0.5)

    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_rgb.yaml")
    # demo_base.yolo9('yolov9e.yaml', auto_optim=False, name=NAME, data="hmt_rgb.yaml")
    # demo_base.yolo10('yolov10x.yaml', auto_optim=False, name=NAME, data="hmt_rgb.yaml")
    # demo_base.yolo11('yolo11x.yaml', auto_optim=False, name=NAME, data="hmt_rgb.yaml")
    # demo_base.yolo12('yolo12x.yaml', auto_optim=False, name=NAME, data="hmt_rgb.yaml")
    # demo_base.yolo26('yolo26x.yaml', auto_optim=False, name=NAME, data="hmt_rgb.yaml")
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_rgbt.yaml")
    # demo_base.yolo9('yolov9e.yaml', auto_optim=False, name=NAME, data="hmt_rgb_rgbt.yaml")
    # demo_base.yolo10('yolov10x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_rgbt.yaml")
    # demo_base.yolo11('yolo11x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_rgbt.yaml")
    # demo_base.yolo12('yolo12x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_rgbt.yaml")
    # demo_base.yolo26('yolo26x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_rgbt.yaml")

#     # demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_bp_cube_f_p01.yaml", imgsz=1280)
#     # demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_f_p01.yaml", imgsz=1280)
#     # demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_t_f_p01.yaml", imgsz=1280)

#     # demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_bp_cube.yaml")
#     # demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_rgb.yaml")
#     # demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_t.yaml")
#     # demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_bp_cube_f_p01.yaml")
#     # demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_f_p01.yaml")
#     # demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_t_f_p01.yaml")

# demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_bp_cube.yaml", imgsz=960)
# demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_rgb.yaml", imgsz=960)
# demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_t.yaml", imgsz=960)
# demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_bp_cube_f_p01.yaml", imgsz=960)
# demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_f_p01.yaml", imgsz=960)
# demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_t_f_p01.yaml", imgsz=960)

# demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_bp_cube.yaml", imgsz=1280)
# demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_rgb.yaml", imgsz=1280)
# demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_t.yaml", imgsz=1280)
# demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_bp_cube_f_p01.yaml", imgsz=1280)
# demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_f_p01.yaml", imgsz=1280)
# demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_t_f_p01.yaml", imgsz=1280)

#     # demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_bp_cube.yaml")
#     # demo_base.yolo9('yolov9e.yaml', auto_optim=False, name=NAME, data="hmt_bp_cube.yaml")
#     # demo_base.yolo10('yolov10x.yaml', auto_optim=False, name=NAME, data="hmt_bp_cube.yaml")
#     # demo_base.yolo11('yolo11x.yaml', auto_optim=False, name=NAME, data="hmt_bp_cube.yaml")
#     # demo_base.yolo12('yolo12x.yaml', auto_optim=False, name=NAME, data="hmt_bp_cube.yaml")
#     # demo_base.yolo26('yolo26x.yaml', auto_optim=False, name=NAME, data="hmt_bp_cube.yaml")
#     # demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_rgb.yaml")
#     # demo_base.yolo9('yolov9e.yaml', auto_optim=False, name=NAME, data="hmt_rgb.yaml")
#     # demo_base.yolo10('yolov10x.yaml', auto_optim=False, name=NAME, data="hmt_rgb.yaml")
#     # demo_base.yolo11('yolo11x.yaml', auto_optim=False, name=NAME, data="hmt_rgb.yaml")
#     # demo_base.yolo12('yolo12x.yaml', auto_optim=False, name=NAME, data="hmt_rgb.yaml")
#     # demo_base.yolo26('yolo26x.yaml', auto_optim=False, name=NAME, data="hmt_rgb.yaml")
#     # demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_t.yaml")
#     # demo_base.yolo9('yolov9e.yaml', auto_optim=False, name=NAME, data="hmt_t.yaml")
#     # demo_base.yolo10('yolov10x.yaml', auto_optim=False, name=NAME, data="hmt_t.yaml")
#     # demo_base.yolo11('yolo11x.yaml', auto_optim=False, name=NAME, data="hmt_t.yaml")
#     # demo_base.yolo12('yolo12x.yaml', auto_optim=False, name=NAME, data="hmt_t.yaml")
#     # demo_base.yolo26('yolo26x.yaml', auto_optim=False, name=NAME, data="hmt_t.yaml")

#     # demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_t_f_p01_merge.yaml")
#     # demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_f_p01_merge.yaml")
#     # demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_t.yaml", cls_pw=0.5)

# demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_merge_f01.yaml", cls_pw=0.5)
# demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_merge.yaml", cls_pw=0.5)

# demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_merge_v2.yaml", cls_pw=0.5)
# demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_merge.yaml", cls_pw=0.5)
# demo_base.yolo26('yolo26x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_merge_v2.yaml", cls_pw=0.5)
# demo_base.yolo26('yolo26x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_merge.yaml", cls_pw=0.5)

# demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_merge_v2.yaml", cls_pw=0.5, imgsz=960)
# demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_merge.yaml", cls_pw=0.5, imgsz=960)
# demo_base.yolo26('yolo26x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_merge_v2.yaml", cls_pw=0.5, imgsz=960)
# demo_base.yolo26('yolo26x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_merge.yaml", cls_pw=0.5, imgsz=960)

# demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_merge_v2.yaml", cls_pw=0.5, imgsz=1280)
# demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_merge.yaml", cls_pw=0.5, imgsz=1280)
# demo_base.yolo26('yolo26x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_merge_v2.yaml", cls_pw=0.5, imgsz=1280)
# demo_base.yolo26('yolo26x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_merge.yaml", cls_pw=0.5, imgsz=1280)

    # [local] quick panel-dataset test (commit f0353944)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False)
    # demo_base.yolo9('yolov9e.yaml', auto_optim=False)
    # demo_base.yolo10('yolov10x.yaml', auto_optim=False)
    # demo_base.yolo11('yolo11x.yaml', auto_optim=False)
    # demo_base.yolo12('yolo12x.yaml', auto_optim=False)
    # demo_base.yolo26('yolo26x.yaml', auto_optim=False)
