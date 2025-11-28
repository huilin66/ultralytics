import demo_base
import torch

demo_base.TASK = 'detect'
demo_base.EPOCHS = 300
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device('cuda:1')
demo_base.BATCH_SIZE = 8
# demo_base.DATA = ".yaml"
# demo_base.CONF = 0.5

if __name__ == '__main__':
    pass
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data="dreality_1c_fv2_v3.yaml")
    demo_base.yolo12('yolo12x.yaml', auto_optim=False, data="dreality_1c_fv2_v2.yaml")
    demo_base.yolo11('yolov11x.yaml', auto_optim=False, data="dreality_1c_fv2_v2.yaml")
    demo_base.yolo10('yolov10x.yaml', auto_optim=False, data="dreality_1c_fv2_v2.yaml")
    demo_base.yolo9('yolov9e.yaml', auto_optim=False, data="dreality_1c_fv2_v2.yaml")
    # demo_base.model_val('dreality_1c_fv2-[yolov8x]2')
    # demo_base.model_predict('/localnvme/project/ultralytics/runs/detect/obj_t-[yolov8x]5/weights/best.pt',
    #                         r'/localnvme/data/added_data/rgbt/Testset/TIR',
    #                         name=r'/localnvme/data/added_data/rgbt/Testset/TIR_infer_obj_rgb-[yolov8x]-%.2f'%demo_base.CONF,
    #                         batch=32, save_conf=True)