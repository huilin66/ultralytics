import os
import warnings
from ultralytics import YOLO
BATCH_SIZE = 2
EPOCHS = 5
DEVICE = 0

def myolo_n():
    # Load a model
    model = YOLO("yolov8n-mdetect.yaml", task = 'mdetect')  # build a new model from scratch
    model.load('yolov8n.pt')

    # Use the model
    model.train(data="billboard_mdet4.yaml", device=DEVICE,
                epochs=EPOCHS, imgsz=640, val=True, batch=BATCH_SIZE, patience=EPOCHS)  # train the model
    model.val()  # evaluate model performance on the validation set
    # model.predict(
    #     r"E:\data\0417_signboard\data0521_m\yolo_rgb_detection3\images",
    #     save=True,
    #     conf=0.25,
    #     # iou=0.2
    # )


def myolo_x():
    # Load a model
    model = YOLO("yolov8x-mdetect.yaml", task = 'mdetect')  # build a new model from scratch
    model.load('yolov8x.pt')

    # Use the model
    model.train(data="billboard_mdet4.yaml", device=DEVICE,
                epochs=EPOCHS, imgsz=640, val=True, batch=2, patience=EPOCHS)  # train the model
    model.val()  # evaluate model performance on the validation set

    # model.predict(
    #     r"E:\data\0417_signboard\data0521_m\yolo_rgb_detection3\images",
    #     save=True,
    #     conf=0.25,
    #     # iou=0.2
    # )


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
    myolo_n()
    # myolo_x()
    # model_predict(r'runs/mdetect/train326/weights/best.pt')
    # model_predict(r'runs/mdetect/train327/weights/best.pt')


