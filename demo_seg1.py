import demo_base
import torch

demo_base.TASK = 'segment'
demo_base.EPOCHS = 500
demo_base.IMGSZ = 960
demo_base.BATCH_SIZE = 16
demo_base.DEVICE = torch.device('cuda:1')
demo_base.DATA = "fusedata870_seg_c6.yaml"

if __name__ == '__main__':
    pass
    demo_base.yolo8('yolov8x-seg.yaml', auto_optim=False, data="fusedata1422_seg_c6_check0708.yaml")
    demo_base.yolo8('yolov8x-seg-dlka3res.yaml', auto_optim=False, data="fusedata1422_seg_c6_check0708.yaml")

    # demo_base.model_predict('psdata411_seg_c6-[yolov8x-seg]', auto_optim=False, data="psdata411_seg_c6.yaml")
    

    # demo_base.yolo8('yolov8x-seg.yaml', auto_optim=False, data="psdata244_seg_c6.yaml", name='debug')
    # demo_base.yolo8x('yolov8x-seg.yaml', auto_optim=False, data="psdata244_seg_f001_c6.yaml")
    # demo_base.yolo8x('yolov8x-seg-dlka3res.yaml', auto_optim=False, data="psdata244_seg_c6.yaml")
    # demo_base.yolo8x('yolov8x-seg-dlka3res.yaml', auto_optim=False, data="psdata244_seg_f001_c6.yaml")
    # demo_base.yolo8x('yolov8x-seg.yaml', auto_optim=False, data="billboard_seg_389re1_c6.yaml")
    # demo_base.yolo8x('yolov8x-seg.yaml', auto_optim=False, data="billboard_seg_618_c6.yaml")
    # demo_base.yolo8x('yolov8x-seg.yaml', auto_optim=False, data="billboard_seg_618_c6_ref.yaml")

    # demo_base.yolo8x('yolov8x-seg-dlka3res.yaml', auto_optim=False)
    # demo_base.yolo8x('yolov8x-seg-dlka3nores.yaml', auto_optim=False)
    # demo_base.yolo8x('yolov8x-seg-dlkaatt3res.yaml', auto_optim=False)
    # demo_base.yolo8x('yolov8x-seg-dlkaatt3nores.yaml', auto_optim=False)
    # demo_base.yolo8x('yolov8x-seg-bot33nores.yaml', auto_optim=False)
    # demo_base.yolo8x('yolov8x-seg-bot33res.yaml', auto_optim=False)

    # psdata122_seg_f001_c6 - [yolov8x - seg]
    # demo_base.model_val(r'runs/segment/psdata122_seg_f001_c6-[yolov8x-seg]/weights/best.pt',
    #                         data="psdata122_seg_c6.yaml"
    #                         )
    # demo_base.model_val(r'runs/segment/psdata122_seg_f001_c6-[yolov8x-seg]/weights/best.pt',
    #                         data="psdata122_seg_f001_c6.yaml"
    #                         )

    # demo_base.model_predict(r'runs/segment/psdata122_seg_f001_c6-[yolov8x-seg]/weights/best.pt',
    #                         r'/localnvme/data/billboard/ps_data/0516/images_split/left',
    #                         name='/localnvme/data/billboard/ps_data/0516/images_split_pred/left',
    #                         )
    # demo_base.model_predict(r'runs/segment/psdata122_seg_f001_c6-[yolov8x-seg]/weights/best.pt',
    #                         r'/localnvme/data/billboard/ps_data/0516/images_split/right',
    #                         name='/localnvme/data/billboard/ps_data/0516/images_split_pred/right',
    #                         )

    # demo_base.model_val(r'runs/segment/psdata244_seg_c6-[yolov8x-seg]3/weights/best.pt', data="psdata244_seg_c6.yaml")
    # demo_base.model_val(r'runs/segment/psdata244_seg_c6-[yolov8x-seg]3/weights/best.pt', data="psdata244_seg_f001_c6.yaml")
    # demo_base.model_val(r'runs/segment/psdata244_seg_f001_c6-[yolov8x-seg]/weights/best.pt', data="psdata244_seg_f001_c6.yaml")
    # demo_base.model_val(r'runs/segment/billboard_seg_389_c618/weights/best.pt', data="billboard_seg_389_c6.yaml", imgsz=960)
    # demo_base.model_val(r'runs/segment/billboard_seg_389_c618/weights/best.pt', data="billboard_seg_389_c6.yaml", imgsz=1600)
    # demo_base.model_val(r'runs/segment/billboard_seg_389_c618/weights/best.pt', data="billboard_seg_389_c6.yaml", imgsz=1920)

    # demo_base.model_val(r'runs/segment/billboard_seg_626_c6-[yolov8x-seg]/weights/best.pt', data="billboard_seg_626_f001_c6.yaml")
    # demo_base.model_val(r'runs/segment/billboard_seg_626_c6-[yolov8x-seg]/weights/best.pt', data="billboard_seg_626_f010_c6.yaml")



    # demo_base.model_val(r'runs/segment/billboard_seg_626_c6-[yolov8x-seg]2/weights/best.pt', data="billboard_seg_626_c6.yaml")
    # demo_base.model_val(r'runs/segment/billboard_seg_626_c6-[yolov8x-seg]2/weights/best.pt', data="billboard_seg_626_f010_c6.yaml")
    # demo_base.model_val(r'runs/segment/billboard_seg_626_f010_c6-[yolov8x-seg]2/weights/best.pt', data="billboard_seg_626_f010_c6.yaml")
    # demo_base.model_val(r'runs/segment/fusedata1037_seg_c6-[yolov8x-seg]/weights/best.pt',)
    # demo_base.model_val(r'runs/segment/fusedata1361_seg_c6-[yolov8x-seg]3/weights/best.pt', )