import torch
from ultralytics import YOLO, RTDETR
BATCH_SIZE = 32
EPOCHS = 500
IMGSZ = 640
DEVICE = torch.device('cuda:0')
DATA = "billboard_mdet5_10_det_0806m.yaml"


def yolo8_x():
    model = YOLO("yolov8x.yaml", task='detect')
    # model.load('yolov8x.pt')

    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)

def yolo8_modifiy(model_path):
    model = YOLO(model_path, task='detect')
    model.load('yolov8x.pt')

    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)


def yolo9_e():
    model = YOLO("yolov9e.yaml", task='detect')
    # model.load('yolov9e.pt')

    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)


def yolo10_x():
    model = YOLO("yolov10x.yaml", task='detect')
    # model.load('yolov10x.pt')
    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)

def rtdetr_x():
    model = RTDETR("rtdetr-x.yaml")
    # model.load('rtdetr-x.pt')
    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)

def model_val(weight_path):
    model = YOLO(weight_path, task='detect')
    model.val(data=DATA, device=DEVICE)
def model_val_rtdetr(weight_path):
    model = RTDETR(weight_path)
    model.val(data=DATA, device=DEVICE)

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
    # model_val(r'runs/detect/train38/weights/best.pt')
    # model_val(r'runs/detect/train39/weights/best.pt')
    # model_val(r'runs/detect/train40/weights/best.pt')
    # model_val(r'runs/detect/train41/weights/best.pt')
    # model_val(r'runs/detect/train43/weights/best.pt')
    # model_val_rtdetr(r'runs/detect/train44/weights/best.pt')
    model_val(r'runs/detect/train51/weights/best.pt')