import torch
from ultralytics import YOLO
BATCH_SIZE = 32
EPOCHS = 500
IMGSZ = 640
DEVICE = torch.device('cuda:0')
DATA = "billboard_mdet5_10.yaml"

def myolo8_x():
    model = YOLO("yolov8x-mdetect.yaml", task = 'mdetect')
    model.load('yolov8x.pt')

    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)
    model.val()


def myolo8_modifiy(model_path):
    model = YOLO(model_path, task = 'mdetect')
    model.load('yolov8x.pt')

    model.train(data=DATA, device=DEVICE,
                optimizer='AdamW', lr0=0.0001,
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)


def myolo8_m(weight_path):
    model = YOLO(weight_path, task = 'mdetect')

    model.train(data=DATA, device=DEVICE,
                epochs=200, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS,
                freeze=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,],
                freeze_head=['.cv2', '.cv3'],
                freeze_bn=True,
                mdet=10.0,
                lrf=0.001,
                )


def model_val(weight_path):
    model = YOLO(weight_path, task='mdetect')
    model.val(data=DATA, device=DEVICE)


def model_predict(weight_path, img_dir, conf=0.5):
    model = YOLO(weight_path, task='mdetect')
    model.predict(
        img_dir,
        save=True,
        conf=conf,
        device=DEVICE,
    )


if __name__ == '__main__':
    pass
    # myolo8_modifiy(r'yolov8x-mdetect10-n0.yaml')
    # myolo8_modifiy(r'yolov8x-mdetect10-n1.yaml')
    # myolo8_modifiy(r'yolov8x-mdetect10-n2.yaml')
    # myolo8_modifiy(r'yolov8x-mdetect10-n3.yaml')

    # model_predict(r'runs/mdetect/train21/weights/best.pt')
    # myolo8_x_m(r'runs/mdetect/train22/weights/best.pt')
    # model_predict(r'runs/mdetect/train27/weights/best.pt')
    # myolo8_m(r'runs/mdetect/train21/weights/best.pt')
    # myolo8_m(r'runs/mdetect/train22/weights/best.pt')
    # myolo8_m(r'runs/mdetect/train25/weights/best.pt')

    # myolo8_modifiy('yolov8x-mdetect-gat21.yaml')
    # myolo8_modifiy('yolov8x-mdetect-gat22.yaml')
    # myolo8_modifiy('yolov8x-mdetect-gat23.yaml')

    myolo8_modifiy('yolov8x-mdetect-gat11-n2.yaml')
    myolo8_modifiy('yolov8x-mdetect-gat12-n3.yaml')