import demo_seg
import torch
demo_seg.DEVICE = torch.device('cuda:1')
demo_seg.DATA = "billboard_seg_611.yaml"
# demo_seg.EPOCHS = 500
if __name__ == '__main__':
    pass
    # demo_seg.yolo8('yolov8x-seg.yaml', auto_optim=False, name=demo_seg.DATA.replace('.yaml', ''))
    # demo_seg.yolo8('yolov8x-seg-p2.yaml', auto_optim=False, name=demo_seg.DATA.replace('.yaml', ''))

    # demo_seg.model_val(r'runs/segment/billboard_seg_3895/weights/best.pt')
    demo_seg.yolo8('yolov8x-seg.yaml', auto_optim=False, name=demo_seg.DATA.replace('.yaml', ''))