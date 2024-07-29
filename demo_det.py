import torch
from ultralytics import YOLO, RTDETR
BATCH_SIZE = 2
EPOCHS = 10
IMGSZ = 640
DEVICE = torch.device('cuda:0')
DATA = "mmdet.yaml"


def yolo8_n():
    model = YOLO("yolov8n.yaml", task='detect')
    model.load('yolov8n.pt')

    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)

def yolo8_x():
    model = YOLO("yolov8x.yaml", task='detect')
    model.load('yolov8x.pt')

    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)

def yolo8_modifiy(model_path):
    model = YOLO(model_path, task='detect')
    model.load('yolov8x.pt')

    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)


def yolo9_e():
    model = YOLO("yolov9e.yaml", task='detect')
    model.load('yolov9e.pt')

    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)


def yolo10_x():
    model = YOLO("yolov10x.yaml", task='detect')
    model.load('yolov10x.pt')
    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)

def rtdetr_x():
    model = RTDETR("rtdetr-x.yaml")
    model.load('rtdetr-x.pt')
    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)


def yolo_val(weight_path):
    model = YOLO(weight_path, task='detect')
    model.val(data=DATA, device=DEVICE, imgsz=IMGSZ*2, batch=1)

def yolo_predict(weight_path, img_dir, conf=0.5):
    model = YOLO(weight_path, task='detect')
    model.predict(
        img_dir,
        save=True,
        conf=conf,
        device=DEVICE,
    )

def yolo_export_onnx(weight_path):
    model = YOLO(weight_path, task='detect')
    model.export(format="onnx", dynamic=True, imgsz=IMGSZ)


if __name__ == '__main__':
    pass
    # yolo8_x()
    # yolo9_e()
    # yolo10_x()
    # rtdetr_x()
    # yolo8_modifiy(r'yolov8x-n1.yaml')
    # yolo8_modifiy(r'yolov8x-n2.yaml')
    # yolo8_modifiy(r'yolov8x-n3.yaml')

    # yolo8_modifiy(r'yolov8x-n1.yaml')
    # yolo8_n()
    # yolo_export_onnx(r'runs/detect/train59/weights/best.pt')
    # yolo_val(r'E:\data\tp\multi_modal_airplane_train\best.pt')
    # yolo_predict(
    #     r'E:\data\tp\multi_modal_airplane_train\best.pt',
    #     r'E:\data\tp\multi_modal_airplane_train\infer_img'
    # )
    # yolo_export_onnx(r'E:\data\tp\multi_modal_airplane_train\best.pt')
    yolo_predict(
        r'E:\data\tp\multi_modal_airplane_train\best.onnx',
        r'E:\data\tp\multi_modal_airplane_train\infer_img'
    )

