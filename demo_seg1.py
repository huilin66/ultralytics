import demo_base
import torch

demo_base.TASK = 'segment'
demo_base.EPOCHS = 500
demo_base.IMGSZ = 960
demo_base.BATCH_SIZE = 16
demo_base.DEVICE = torch.device('cuda:0')
demo_base.DATA = "billboard_seg_389re2_c6.yaml"

if __name__ == '__main__':
    pass
    # demo_base.yolo8x('yolov8x-seg.yaml', auto_optim=False)
    # demo_base.yolo8x('yolov8x-seg.yaml', auto_optim=False, data="billboard_seg_389re1_c6.yaml")
    # demo_base.yolo8x('yolov8x-seg.yaml', auto_optim=False, data="billboard_seg_618_c6.yaml")
    # demo_base.yolo8x('yolov8x-seg.yaml', auto_optim=False, data="billboard_seg_618_c6_ref.yaml")

    demo_base.yolo8x('yolov8x-seg-dlka3res.yaml', auto_optim=False)
    demo_base.yolo8x('yolov8x-seg-dlka3nores.yaml', auto_optim=False)
    demo_base.yolo8x('yolov8x-seg-dlkaatt3res.yaml', auto_optim=False)
    demo_base.yolo8x('yolov8x-seg-dlkaatt3nores.yaml', auto_optim=False)
    # demo_base.yolo8x('yolov8x-seg-bot33nores.yaml', auto_optim=False)
    # demo_base.yolo8x('yolov8x-seg-bot33res.yaml', auto_optim=False)

    # demo_base.model_val(r'runs/segment/billboard_seg_618_c6_ref-[yolov8x-seg]2/weights/best.pt')
    # demo_base.model_val(r'runs/segment/billboard_seg_618_c6_ref-yolov8x-seg/weights/best.pt')
    # demo_base.model_val(r'runs/segment/billboard_seg_618_c6-yolov8x-seg/weights/best.pt',)

    # demo_base.model_val(r'runs/segment/billboard_seg_618_c6_ref-[yolov8x-seg]2/weights/best.pt', data="billboard_seg_618_c6_ref.yaml", imgsz=640)
    # demo_base.model_val(r'runs/segment/billboard_seg_618_c6_ref-yolov8x-seg/weights/best.pt', data="billboard_seg_618_c6_ref.yaml",)
    # demo_base.model_val(r'runs/segment/billboard_seg_389_c618/weights/best.pt', data='billboard_mseg_389_c6.yaml')