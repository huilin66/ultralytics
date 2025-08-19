import demo_base
import torch

demo_base.TASK = 'segment'
demo_base.EPOCHS = 500
demo_base.IMGSZ = 960
demo_base.BATCH_SIZE = 16
demo_base.DEVICE = torch.device('cuda:1')
demo_base.DATA = "fusedata3899_seg_c5_0818.yaml"

if __name__ == '__main__':
    pass

    # demo_base.yolo8('yolov8x-seg.yaml', auto_optim=False)
    demo_base.yolo8('yolov8x-seg-dlka3res.yaml', auto_optim=False)

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

