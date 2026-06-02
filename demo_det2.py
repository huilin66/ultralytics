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
    # demo_base.model_val('debug14')
    # demo_base.model_val('debug15')
    # demo_base.model_val('debug16')
    # demo_base.model_val('debug17')
    # demo_base.model_val('debug18')
    # demo_base.model_val('debug19')
    # demo_base.model_val('debug26')
    # demo_base.model_val('debug27')

    demo_base.model_val('cubit-[yolov8x]')
    demo_base.model_val('cubit-[yolov9e]')
    demo_base.model_val('cubit-[yolov10x]')
    demo_base.model_val('cubit-[yolo11x]')
    demo_base.model_val('cubit-[yolo12x]')
    demo_base.model_val('cubit-[yolo26x]')
    demo_base.model_val('cubit-[yolo26x_dlka]')
    demo_base.model_val('cubit-[yolo26x_dlkaatt]')