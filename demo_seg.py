import torch
from ultralytics import YOLO
BATCH_SIZE = 32
EPOCHS = 500
IMGSZ = 640
DEVICE = torch.device('cuda:0')
DATA = "mm.yaml"


def myolo8_x():
    model = YOLO("yolov8x-seg.yaml", task = 'segment')
    model.load('yolov8x.pt')

    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS,
                )
    model.val()


def myolo9_e():
    model = YOLO("yolov9e-seg.yaml", task = 'segment')
    model.load('yolov9e.pt')

    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)
    model.val()


def model_predict(weight_path):
    model = YOLO(weight_path, task='mdetect')
    model.val()  # evaluate model performance on the validation set

    model.predict(
        r"E:\data\0417_signboard\data0521_m\yolo_rgb_detection3\images",
        save=True,
        conf=0.25,
        # iou=0.2
        device=DEVICE,
    )


if __name__ == '__main__':
    pass
    myolo8_x()
    myolo9_e()

    # model_predict(weight_path)
