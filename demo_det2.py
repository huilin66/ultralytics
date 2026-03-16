import demo_base
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
demo_base.TASK = 'detect'
demo_base.EPOCHS = 300
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device('cuda:0')
demo_base.BATCH_SIZE = 16
# demo_base.DATA = ".yaml"
# demo_base.CONF = 0.5


if __name__ == '__main__':
    pass
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_t_0211_extendv1.yaml', imgsz=640,  workers=0)
    # demo_base.yolo9('yolov9e.yaml', auto_optim=False, data='hmt_t_0211_extendv1.yaml', imgsz=640,  workers=0)
    # demo_base.yolo10('yolov10x.yaml', auto_optim=False, data='hmt_t_0211_extendv1.yaml', imgsz=640,  workers=0)
    # demo_base.yolo11('yolo11x.yaml', auto_optim=False, data='hmt_t_0211_extendv1.yaml', imgsz=640,  workers=0)
    # demo_base.yolo12('yolo12x.yaml', auto_optim=False, data='hmt_t_0211_extendv1.yaml', imgsz=640,  workers=0)
    # demo_base.yolo26('yolo26x.yaml', auto_optim=False, data='hmt_t_0211_extendv1.yaml', imgsz=640,  workers=0)

    demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_0211_v1.yaml', imgsz=640, workers=0)
    demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_0211_slice640_v1.yaml', imgsz=640, workers=0)
    demo_base.yolo9('yolov9e.yaml', auto_optim=False, data='hmt_rgb_0211_slice640_v1.yaml', imgsz=640,  workers=0)
    demo_base.yolo10('yolov10x.yaml', auto_optim=False, data='hmt_rgb_0211_slice640_v1.yaml', imgsz=640,  workers=0)
    demo_base.yolo11('yolo11x.yaml', auto_optim=False, data='hmt_rgb_0211_slice640_v1.yaml', imgsz=640,  workers=0)
    demo_base.yolo12('yolo12x.yaml', auto_optim=False, data='hmt_rgb_0211_slice640_v1.yaml', imgsz=640,  workers=0)
    demo_base.yolo26('yolo26x.yaml', auto_optim=False, data='hmt_rgb_0211_slice640_v1.yaml', imgsz=640,  workers=0)

    demo_base.yolo9('yolov9e.yaml', auto_optim=False, data='hmt_rgb_0211_v1.yaml', imgsz=640,  workers=0)
    demo_base.yolo10('yolov10x.yaml', auto_optim=False, data='hmt_rgb_0211_v1.yaml', imgsz=640,  workers=0)
    demo_base.yolo11('yolo11x.yaml', auto_optim=False, data='hmt_rgb_0211_v1.yaml', imgsz=640,  workers=0)
    demo_base.yolo12('yolo12x.yaml', auto_optim=False, data='hmt_rgb_0211_v1.yaml', imgsz=640,  workers=0)
    demo_base.yolo26('yolo26x.yaml', auto_optim=False, data='hmt_rgb_0211_v1.yaml', imgsz=640,  workers=0)