import demo_base
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
demo_base.TASK = 'detect'
demo_base.EPOCHS = 300
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device('cuda:0')
demo_base.BATCH_SIZE = 8
demo_base.DATA = "BP_HMT_1216.yaml"
# demo_base.CONF = 0.5


if __name__ == '__main__':
    pass
    NAME = None #'debug'
    demo_base.yolo8('yolov8x.yaml', weight_path=None, auto_optim=False, name=NAME)
    demo_base.yolo9('yolov9e.yaml', weight_path=None, auto_optim=False, name=NAME)
    demo_base.yolo10('yolov10x.yaml', weight_path=None, auto_optim=False, name=NAME)
    demo_base.yolo11('yolo11x.yaml', weight_path=None, auto_optim=False, name=NAME)
    demo_base.yolo12('yolo12x.yaml', weight_path=None, auto_optim=False, name=NAME)
    demo_base.yolo26('yolo26x.yaml', weight_path=None, auto_optim=False, name=NAME)
    # demo_base.yolo26('yolo26x_dlka.yaml', weight_path=None, auto_optim=False, name=NAME)
    # demo_base.yolo26('yolo26x_dlkaatt.yaml', weight_path=None, auto_optim=False, name=NAME)

    # demo_base.yolo26('yolo26x_dlka_tf.yaml', weight_path=None, auto_optim=False)