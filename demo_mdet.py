import torch
from ultralytics import YOLO
BATCH_SIZE = 2
EPOCHS = 20
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
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)

def myolo9_e():
    model = YOLO("yolov9e-mdetect.yaml", task = 'mdetect')
    model.load('yolov9e.pt')

    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)
    model.val()


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
    # myolo8_x()
    # myolo9_e()
    model_val(r'best.pt')


