import demo_base
import torch

demo_base.TASK = 'segment'
demo_base.EPOCHS = 100
demo_base.IMGSZ = 640
demo_base.BATCH_SIZE = 16
demo_base.DEVICE = torch.device('cuda:1')
demo_base.DATA = "rip_seg.yaml"

if __name__ == '__main__':
    pass
    demo_base.yolo8('yolov8x-seg.yaml', auto_optim=False)     # 6.22m/18s
    demo_base.yolo9('yolov9e-seg.yaml', auto_optim=False)     # 9.07m/20.7s
    demo_base.yolo10('yolov10x-seg.yaml', auto_optim=False)   # 6.30m/21.7s
    demo_base.yolo11('yolo11x-seg.yaml', auto_optim=False)    # 7.12m/21.2s
    demo_base.yolo12('yolo12x-seg.yaml', auto_optim=False)    # 11.05m/25.5s
    demo_base.yolo26('yolo26x-seg.yaml', auto_optim=False)    # 9.12m/23.5s