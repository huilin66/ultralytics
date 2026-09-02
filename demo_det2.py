import demo_base
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
demo_base.TASK = 'detect'
demo_base.EPOCHS = 300
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device('cuda:1')
demo_base.BATCH_SIZE = 16
# demo_base.DATA = ".yaml"
# demo_base.DATA = "solarpanel.yaml"  # [local] new solar panel dataset test
# demo_base.CONF = 0.5


if __name__ == '__main__':
    pass
    NAME = None
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_t.yaml")
    # demo_base.yolo9('yolov9e.yaml', auto_optim=False, name=NAME, data="hmt_t.yaml")
    # demo_base.yolo10('yolov10x.yaml', auto_optim=False, name=NAME, data="hmt_t.yaml")
    # demo_base.yolo11('yolo11x.yaml', auto_optim=False, name=NAME, data="hmt_t.yaml")
    # demo_base.yolo12('yolo12x.yaml', auto_optim=False, name=NAME, data="hmt_t.yaml")
    # demo_base.yolo26('yolo26x.yaml', auto_optim=False, name=NAME, data="hmt_t.yaml")
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_t_rgbt.yaml")
    # demo_base.yolo9('yolov9e.yaml', auto_optim=False, name=NAME, data="hmt_t_rgbt.yaml")
    # demo_base.yolo10('yolov10x.yaml', auto_optim=False, name=NAME, data="hmt_t_rgbt.yaml")
    # demo_base.yolo11('yolo11x.yaml', auto_optim=False, name=NAME, data="hmt_t_rgbt.yaml")
    # demo_base.yolo12('yolo12x.yaml', auto_optim=False, name=NAME, data="hmt_t_rgbt.yaml")
    # demo_base.yolo26('yolo26x.yaml', auto_optim=False, name=NAME, data="hmt_t_rgbt.yaml")

    # demo_base.model_val('hmt_rgb_0211_slice640_v1-[yolov8x]')
    # demo_base.model_val('hmt_rgb_0211_slice640_v1-[yolov9e]')
    # demo_base.model_val('hmt_rgb_0211_v1-[yolov8x]')
    # demo_base.model_val('hmt_rgb_0211_v1-[yolov8x]2')
    # demo_base.model_val('hmt_rgb_0211_v1-[yolov8x]3')

    # demo_base.model_val('debug')
    # demo_base.model_val('debug5')
    # demo_base.model_val('debug6')
    # demo_base.model_val('debug7')
    # demo_base.model_val('debug8')

    demo_base.model_val('BP_HMT_1216-[yolov8x]2')
    demo_base.model_val('BP_HMT_1216-[yolov9e]')
    demo_base.model_val('BP_HMT_1216-[yolov10x]')
    demo_base.model_val('BP_HMT_1216-[yolo11x]')
    demo_base.model_val('BP_HMT_1216-[yolo12x]')
#     demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_rgbt.yaml")
#     demo_base.yolo9('yolov9e.yaml', auto_optim=False, name=NAME, data="hmt_rgb_rgbt.yaml")
#     demo_base.yolo10('yolov10x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_rgbt.yaml")
#     demo_base.yolo11('yolo11x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_rgbt.yaml")
#     demo_base.yolo12('yolo12x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_rgbt.yaml")
#     demo_base.yolo26('yolo26x.yaml', auto_optim=False, name=NAME, data="hmt_rgb_rgbt.yaml")
#     demo_base.yolo8('yolov8x.yaml', auto_optim=False, name=NAME, data="hmt_t_rgbt.yaml")
#     demo_base.yolo9('yolov9e.yaml', auto_optim=False, name=NAME, data="hmt_t_rgbt.yaml")
#     demo_base.yolo10('yolov10x.yaml', auto_optim=False, name=NAME, data="hmt_t_rgbt.yaml")
#     demo_base.yolo11('yolo11x.yaml', auto_optim=False, name=NAME, data="hmt_t_rgbt.yaml")
#     demo_base.yolo12('yolo12x.yaml', auto_optim=False, name=NAME, data="hmt_t_rgbt.yaml")
#     demo_base.yolo26('yolo26x.yaml', auto_optim=False, name=NAME, data="hmt_t_rgbt.yaml")

    # [local] solar panel dataset quick test (commit f0353944)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False)
    # demo_base.yolo9('yolov9e.yaml', auto_optim=False)
    # demo_base.yolo10('yolov10x.yaml', auto_optim=False)
    # demo_base.yolo11('yolo11x.yaml', auto_optim=False)
    # demo_base.yolo12('yolo12x.yaml', auto_optim=False)
    # demo_base.yolo26('yolo26x.yaml', auto_optim=False)
