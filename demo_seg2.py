import demo_base
import torch

demo_base.TASK = 'segment'
demo_base.EPOCHS = 500
demo_base.IMGSZ = 960
demo_base.BATCH_SIZE = 16
demo_base.DEVICE = torch.device('cuda:0')
demo_base.DATA = "fusedata7436_seg_c5_0922_80p_ref.yaml"

if __name__ == '__main__':
    pass

    # demo_base.yolo10('yolov10x-seg-dlka3res.yaml', auto_optim=False, name='debug')
    # demo_base.yolo8('yolov8x-seg.yaml', auto_optim=False)
    # demo_base.yolo9('yolov9e-seg.yaml', auto_optim=False)
    # demo_base.yolo10('yolov10x-seg.yaml', auto_optim=False)
    # demo_base.yolo11('yolov11x-seg.yaml', auto_optim=False)
    # demo_base.yolo12('yolov12x-seg.yaml', auto_optim=False)
    #
    # demo_base.yolo9('yolov9e-seg-dlka3res.yaml', auto_optim=False)
    # demo_base.yolo10('yolov10x-seg-dlka3res.yaml', auto_optim=False)


    # demo_base.yolo11('yolov11x-seg-dlka3res.yaml', auto_optim=False)
    # demo_base.yolo12('yolov12x-seg-dlka3res.yaml', auto_optim=False)

    # demo_base.yolo8('yolov8x-seg-dlka3res.yaml', auto_optim=False)

    # demo_base.model_val(
    #     "runs/segment/fusedata3072_seg_c5_0809_80p-[yolov8x-seg-dlka3res]2/weights/best.pt",
    #     save_json=True
    # )
    # demo_base.model_val(
    #     r'runs/segment/fusedata2419_seg_c5_0730-[yolov8x-seg-dlka3res]4/weights/best.pt',
    #     save_json=True
    # )
    # demo_base.model_predict(
    #     r'runs/segment/fusedata1422_seg_c6_check0708-[yolov8x-seg-dlka3res]/weights/best.pt',
    #     r'/localnvme/data/billboard/fused_data/data1422_seg_c6_check0708/val/images',
    # )


    # demo_base.model_val(r'fusedata7436_seg_c5_0922_80p_ref-[yolov10x-seg-dlka3res]_a100',
    #                     data='fusedata7436_seg_c5_0922_80p_ref.yaml',
    #                     # filter_small=0.05,
    #                     save_json=True)
    demo_base.model_val(r'fusedata7436_seg_c5_0922_80p_ref-[yolov10x-seg-dlka3res]2')
    demo_base.model_val(r'fusedata7436_seg_c5_0922_80p_ref-[yolov8x-seg-dlka3res]')
    # demo_base.model_val(r'runs/segment/fusedata3899_seg_c5_0818_80p-[yolov11x-seg]/weights/best.pt')
    # demo_base.model_val(r'runs/segment/fusedata3899_seg_c5_0818_80p-[yolov12x-seg]6/weights/best.pt')
    # demo_base.model_val(r'runs/segment/fusedata3899_seg_c5_0818_80p-[yolov9e-seg-dlka3res]2/weights/best.pt')
    # demo_base.model_val(r'runs/segment/fusedata3899_seg_c5_0818_80p-[yolov10x-seg-dlka3res]2/weights/best.pt')
    # demo_base.model_val(r'runs/segment/fusedata5894_seg_c5_0822_80p-[yolov8x-seg-dlka3res]2/weights/best.pt')