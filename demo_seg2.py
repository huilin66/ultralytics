import demo_base
import torch

demo_base.TASK = 'segment'
demo_base.EPOCHS = 500
demo_base.IMGSZ = 960
demo_base.BATCH_SIZE = 16
demo_base.DEVICE = torch.device('cuda:1')
demo_base.DATA = "billboard_seg_626_f010_c6.yaml"

if __name__ == '__main__':
    pass
    demo_base.yolo8('yolov8x-seg.yaml', auto_optim=False, data="fusedata1037_seg_c6.yaml")
    demo_base.yolo8('yolov8x-seg-dlka3res.yaml', auto_optim=False, data="fusedata1037_seg_c6.yaml")

    # demo_base.yolo8('yolov8x-seg.yaml', auto_optim=False, data="psdata244_seg_c6.yaml")
    # demo_base.yolo8('yolov8x-seg.yaml', auto_optim=False, data="psdata244_seg_f001_c6.yaml")
    # demo_base.yolo8('yolov8x-seg-dlka3res.yaml', auto_optim=False, data="psdata244_seg_c6.yaml")
    # demo_base.yolo8('yolov8x-seg-dlka3res.yaml', auto_optim=False, data="psdata244_seg_f001_c6.yaml")
    # demo_base.yolo8x('yolov8x-seg.yaml', auto_optim=False, data="billboard_seg_626_f010_c6.yaml")
    # demo_base.yolo8x('yolov8x-seg.yaml', auto_optim=False, data="billboard_seg_626_f010_c6_ref.yaml")
    # demo_base.yolo8x('yolov8x-seg-p2.yaml', auto_optim=False)
    # demo_base.yolo8x('yolov8x-seg.yaml', auto_optim=False)
    # demo_base.yolo8x('yolov8x-seg.yaml', auto_optim=False, data="billboard_seg_667_c6_ref.yaml")
    # demo_base.yolo10x('yolov10x-seg.yaml', auto_optim=False)
    # demo_base.yolo11x('yolov11x-seg.yaml', auto_optim=False)
    # demo_base.yolo9e('yolov9e-seg.yaml', auto_optim=False)
    # demo_base.yolo12x('yolov12x-seg.yaml', auto_optim=False)


    # demo_base.model_val(r'runs/segment/billboard_seg_389_c618/weights/best.pt', data='billboard_seg_389_c6.yaml')
    # demo_base.model_val(r'runs/segment/billboard_seg_389re1_c6_ref-[yolov8x-seg]/weights/best.pt',
    #                     data="billboard_seg_389re1_c6.yaml")
    # demo_base.model_predict(r'runs/segment/billboard_seg_389re1_c6_ref-[yolov8x-seg]/weights/best.pt',
    #                         r'/nfsv4/23039356r/data/billboard/bd_data/data389re1_seg_c6/images',
    #                         )/

    # demo_base.model_predict(r'runs/segment/billboard_seg_626_c6-[yolov8x-seg]/weights/best.pt',
    #                         r'/localnvme/data/billboard/bd_data/data626_seg_c6/images',
    #                         )