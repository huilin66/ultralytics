import os
import warnings
from ultralytics import YOLO
BATCH_SIZE = 2
EPOCHS = 5

def yolo8_n():
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model.load('yolov8n.pt')

    # Use the model
    model.train(data="billboard_det4.yaml",
                epochs=EPOCHS, imgsz=640, val=True, batch=BATCH_SIZE, patience=EPOCHS)  # train the model
    model.val()  # evaluate model performance on the validation set
    # model.predict(
    #     r"E:\data\0417_signboard\data0521_m\yolo_rgb_detection3\images",
    #     save=True,
    #     conf=0.25,
    #     # iou=0.2
    # )


def yolo8_x():
    # Load a model
    model = YOLO("yolov8x.yaml")  # build a new model from scratch
    model.load('yolov8x.pt')

    # Use the model
    model.train(data="billboard_det4.yaml",
                epochs=EPOCHS, imgsz=640, val=True, batch=2, patience=EPOCHS)  # train the model
    model.val()  # evaluate model performance on the validation set

    # model.predict(
    #     r"E:\data\0417_signboard\data0521_m\yolo_rgb_detection3\images",
    #     save=True,
    #     conf=0.25,
    #     # iou=0.2
    # )


def yolo9_e():
    # Load a model
    model = YOLO("yolov9e.yaml")  # build a new model from scratch
    model.load('yolov9e.pt')

    # Use the model
    model.train(data="billboard_det4.yaml",
                epochs=EPOCHS, imgsz=640, val=True, batch=BATCH_SIZE, patience=EPOCHS)  # train the model
    model.val()  # evaluate model performance on the validation set
    # model.predict(
    #     r"E:\data\0417_signboard\data0521_m\yolo_rgb_detection3\images",
    #     save=True,
    #     conf=0.25,
    #     # iou=0.2
    # )

def yolo9_e_tfd():
    # Load a model
    model = YOLO("yolov9e_tfd.yaml")  # build a new model from scratch
    model.load('yolov9e.pt')

    # Use the model
    model.train(data="billboard_det4.yaml",
                epochs=EPOCHS, imgsz=640, val=True, batch=BATCH_SIZE, patience=EPOCHS)  # train the model
    model.val()  # evaluate model performance on the validation set
    # model.predict(
    #     r"E:\data\0417_signboard\data0521_m\yolo_rgb_detection3\images",
    #     save=True,
    #     conf=0.25,
    #     # iou=0.2
    # )

if __name__ == '__main__':
    pass
    # yolo8_n()
    # yolo8_x()
    # yolo9_e()
    yolo9_e_tfd()