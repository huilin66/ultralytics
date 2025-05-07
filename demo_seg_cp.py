import demo_seg
import torch
demo_seg.DEVICE = torch.device('cuda:0')
demo_seg.IMGSZ = 960
demo_seg.BATCH_SIZE = 8
if __name__ == '__main__':
    pass
    # demo_seg.yolo8('yolov8x-seg.yaml', auto_optim=False, name=demo_seg.DATA.replace('.yaml', ''))
    # demo_seg.yolo8('yolov8x-seg-p2.yaml', auto_optim=False, name=demo_seg.DATA.replace('.yaml', ''))

    # demo_seg.model_val(r'runs/segment/billboard_seg_3895/weights/best.pt')
    # demo_seg.yolo8('yolov8x-seg.yaml', auto_optim=False, name='debug')
    # demo_seg.yolo8('yolov8x-seg-p2.yaml', auto_optim=False, name='debug')

    # demo_seg.yolo8('yolov8x-seg-p2.yaml', auto_optim=True, name=demo_seg.DATA.replace('.yaml', ''))

    demo_seg.yolo8('yolov8x-seg-p2.yaml', auto_optim=False, name=demo_seg.DATA.replace('.yaml', ''))
    demo_seg.yolo8('yolov8x-seg.yaml', auto_optim=False, name=demo_seg.DATA.replace('.yaml', ''))