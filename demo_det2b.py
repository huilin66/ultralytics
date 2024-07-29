import torch
from ultralytics import YOLO, RTDETR
BATCH_SIZE = 16
EPOCHS = 500
IMGSZ = 640
DEVICE = torch.device('cuda:0')
DATA = "billboard_det5.yaml"
DATA = "billboard_det5_2b.yaml"

def yolo8_x():
    model = YOLO("yolov8x_2b.yaml", task='detect')
    model.load('yolov8x.pt')

    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=640, val=True, batch=BATCH_SIZE, patience=EPOCHS)

if __name__ == '__main__':
    pass
