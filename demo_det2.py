import demo_base
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
demo_base.TASK = 'detect'
demo_base.EPOCHS = 300
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device('cuda:0')
demo_base.BATCH_SIZE = 2
# demo_base.DATA = ".yaml"
# demo_base.CONF = 0.5


if __name__ == '__main__':
    pass
    # demo_base.model_val('hmt_t_0211_extendv1-[yolo11x]', conf=0.5)
    # demo_base.model_val('hmt_rgb_0211_v1-[yolov8x]', conf=0.5)
    # demo_base.model_val('hmt_rgb_0211_slice640_v1-[yolov8x]', conf=0.5)
    # demo_base.model_val('hmt_rgb_0211_slice640_v1-[yolov9e]', conf=0.5)

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

    demo_base.model_val('hmt_rgb_0211_v1-[yolov8x]3')
    demo_base.model_val('debug6')
