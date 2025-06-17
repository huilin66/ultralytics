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
    # demo_base.model_val(r'runs/segment/billboard_seg_626_c6-[yolov8x-seg]/weights/best.pt', data='billboard_seg_626_c6.yaml')
    # demo_base.model_val(r'runs/segment/psdata411_seg_c6-[yolov8x-seg]/weights/best.pt')
    demo_base.model_val(r'runs/segment/fusedata1037_seg_c6-[yolov8x-seg-dlka3res]/weights/best.pt')