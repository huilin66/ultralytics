import torch
from ultralytics import YOLO, YOLOv10, RTDETR
BATCH_SIZE = 2
EPOCHS = 2
DEVICE = torch.device('cuda:0')
DATA = "billboard_det5.yaml"


def yolo8_x():
    model = YOLO("yolov8x.yaml", task='detect')
    model.load('yolov8x.pt')

    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=640, val=True, batch=BATCH_SIZE, patience=EPOCHS)


def yolo9_e():
    model = YOLO("yolov9e.yaml", task='detect')
    model.load('yolov9e.pt')

    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=640, val=True, batch=BATCH_SIZE, patience=EPOCHS)


def yolo10_x():
    model = YOLOv10('yolov10x.pt')
    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=640, val=True, batch=BATCH_SIZE, patience=EPOCHS)

def rtdetr_x():
    model = RTDETR("rtdetr-x.yaml")
    model.load('rtdetr-x.pt')
    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=640, val=True, batch=BATCH_SIZE, patience=EPOCHS)

if __name__ == '__main__':
    pass
    # yolo8_x()
    # yolo9_e()
    # yolo10_x()
    rtdetr_x()