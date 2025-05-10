import demo_seg
import torch
demo_seg.DEVICE = torch.device('cuda:1')
demo_seg.IMGSZ = 960
# demo_seg.EPOCHS = 1
demo_seg.BATCH_SIZE = 16
if __name__ == '__main__':
    pass
    # demo_seg.yolo8('yolov8x-seg.yaml', auto_optim=False, name=demo_seg.DATA.replace('.yaml', ''))
    # demo_seg.yolo8('yolov8x-seg-p2.yaml', auto_optim=False, name=demo_seg.DATA.replace('.yaml', ''))

    # demo_seg.model_val(r'runs/segment/billboard_seg_3895/weights/best.pt')
    # demo_seg.yolo8('yolov8x-seg.yaml', auto_optim=False, name='debug')
    # demo_seg.yolo8('yolov8x-seg-p2.yaml', auto_optim=False, name='debug')

    # demo_seg.yolo8('yolov8x-seg-p2.yaml', auto_optim=True, name=demo_seg.DATA.replace('.yaml', ''))

    # demo_seg.yolo8('yolov8x-seg-p2.yaml', auto_optim=False, name=demo_seg.DATA.replace('.yaml', ''))
    # demo_seg.yolo8('yolov8x-seg.yaml', auto_optim=False, name=demo_seg.DATA.replace('.yaml', ''))

    # demo_seg.yolo8('yolov8x-seg-dlkaatt3res.yaml', auto_optim=False, name=demo_seg.DATA.replace('.yaml', ''))
    # demo_seg.yolo8('yolov8x-seg-dlkaatt3nores.yaml', auto_optim=False, name=demo_seg.DATA.replace('.yaml', ''))
    # demo_seg.yolo8('yolov8x-seg-dlka3res.yaml', auto_optim=False, name=demo_seg.DATA.replace('.yaml', ''))
    # demo_seg.yolo8('yolov8x-seg-dlka3nores.yaml', auto_optim=False, name=demo_seg.DATA.replace('.yaml', ''))
    # demo_seg.yolo8('yolov8x-seg.yaml', auto_optim=False, name=demo_seg.DATA.replace('.yaml', ''))
    demo_seg.model_val('runs/segment/billboard_seg_389_c616/weights/best.pt')
    demo_seg.model_val('runs/segment/billboard_seg_389_c617/weights/best.pt')
    demo_seg.model_val('runs/segment/billboard_seg_389_c618/weights/best.pt')
    demo_seg.model_val('runs/segment/billboard_seg_389_c619/weights/best.pt')
    demo_seg.model_val('runs/segment/billboard_seg_389_c620/weights/best.pt')
