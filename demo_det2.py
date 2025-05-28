import demo_base
import torch

demo_base.TASK = 'detect'
demo_base.EPOCHS = 100
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device('cuda:1')
demo_base.DATA = ".yaml"

if __name__ == '__main__':
    pass
    # demo_base.yolo8x('yolov8x.yaml', auto_optim=False, data="obj_rgb.yaml")
    demo_base.yolo8('yolov8x.yaml', auto_optim=False, data="obj_t.yaml")
    # demo_base.yolo8x('yolov8x.yaml', auto_optim=False, data="obj_t.yaml", epochs=500)