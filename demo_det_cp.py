import demo_det
import torch
demo_det.DEVICE = torch.device('cuda:1')
if __name__ == '__main__':
    demo_det.yolo8x('yolov8x.yaml', auto_optim=False)
    demo_det.yolo9e('yolov9e.yaml', auto_optim=False)
    demo_det.yolo10x('yolov10x.yaml', auto_optim=False)
    demo_det.rtdetrx('rtdetr-x.yaml', auto_optim=False)