import torch
from ultralytics import YOLO
BATCH_SIZE = 2
EPOCHS = 20
DEVICE = torch.device('cuda:0')
DATA = "billboard_mdet5.yaml"


def myolo8_x():
    model = YOLO("yolov8x-mdetect.yaml", task = 'mdetect')
    model.load('yolov8x.pt')

    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=640, val=True, batch=BATCH_SIZE, patience=EPOCHS,
                freeze=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, ],
                freeze_head=['.cv2', '.cv3'],
                )

    model.val()


def myolo9_e():
    model = YOLO("yolov9e-mdetect.yaml", task = 'mdetect')
    model.load('yolov9e.pt')

    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=640, val=True, batch=BATCH_SIZE, patience=EPOCHS)
    model.val()


def model_predict(weight_path):
    model = YOLO(weight_path, task='mdetect')

    model.predict(
        r"E:\data\0417_signboard\data0521_m\yolo_rgb_detection3\images",
        save=True,
        # conf=0.5,
        # iou=0.2
        device=DEVICE,
    )
def model_val(weight_path):
    model = YOLO(weight_path, task='mdetect')
    model.val(data=DATA)

if __name__ == '__main__':
    pass
    # myolo8_x()
    # myolo9_e()
    model_val(r'best.pt')


