import torch
from ultralytics import YOLO, RTDETR
BATCH_SIZE = 32
EPOCHS = 500
IMGSZ = 640
DEVICE = torch.device('cuda:1')
DATA = "billboard_mdet5_10_det_0806m.yaml"


def yolo8_x():
    model = YOLO("yolov8x.yaml", task='detect')
    # model.load('yolov8x.pt')

    model.train(data=DATA, device=DEVICE,
                optimizer='AdamW', lr0=0.0001,
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)

def yolo8_modifiy(model_path):
    model = YOLO(model_path, task='detect')
    model.load('yolov8x.pt')

    model.train(data=DATA, device=DEVICE,
                optimizer='AdamW', lr0=0.0001,
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)


def yolo9_e():
    model = YOLO("yolov9e.yaml", task='detect')
    # model.load('yolov9e.pt')

    model.train(data=DATA, device=DEVICE,
                optimizer='AdamW', lr0=0.0001,
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)


def yolo10_x():
    model = YOLO("yolov10x.yaml", task='detect')
    # model.load('yolov10x.pt')
    model.train(data=DATA, device=DEVICE,
                optimizer='AdamW', lr0=0.0001,
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)

def rtdetr_x():
    model = RTDETR("rtdetr-x.yaml")
    # model.load('rtdetr-x.pt')
    model.train(data=DATA, device=DEVICE,
                optimizer='AdamW', lr0=0.0001,
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)

def predict(weight_path, img_dir, conf=0.5):
    model = YOLO(weight_path, task='detect')
    model.predict(
        img_dir,
        save=True,
        conf=conf,
        device=DEVICE,
        imgsz=IMGSZ,
    )

def export_onnx(weight_path):
    model = YOLO(weight_path, task='detect')
    model.save('yolov8.pt')


if __name__ == '__main__':
    pass
    # yolo8_x()
    # yolo9_e()
    # yolo10_x()
    # rtdetr_x()
