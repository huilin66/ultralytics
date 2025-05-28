import demo_base
import torch

demo_base.TASK = 'detect'
demo_base.EPOCHS = 500
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device('cuda:1')
demo_base.DATA = ".yaml"
demo_base.CONF = 0.5

if __name__ == '__main__':
    pass
    # demo_base.yolo8x('yolov8x.yaml', auto_optim=False, data=".yaml")
    demo_base.model_val('/localnvme/project/ultralytics/runs/detect/obj_t-[yolov8x]5/weights/best.pt', batch=32)
    demo_base.model_predict('/localnvme/project/ultralytics/runs/detect/obj_t-[yolov8x]5/weights/best.pt',
                            r'/localnvme/data/added_data/rgbt/Testset/TIR',
                            name=r'/localnvme/data/added_data/rgbt/Testset/TIR_infer_obj_rgb-[yolov8x]-%.2f'%demo_base.CONF,
                            batch=32, save_conf=True)