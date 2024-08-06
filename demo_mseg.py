import torch
from ultralytics import YOLO
BATCH_SIZE = 1
EPOCHS = 500
IMGSZ = 320
DEVICE = torch.device('cuda:0')
DATA = "billboard_mseg2.yaml"


def myolo8_x():
    model = YOLO("yolov8x-mseg.yaml", task = 'msegment')
    model.load('yolov8x.pt')

    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)
    model.val()



def model_val(weight_path):
    model = YOLO(weight_path, task='msegment')
    model.val(data=DATA, device=DEVICE)


def model_predict(weight_path, img_dir, conf=0.5):
    model = YOLO(weight_path, task='msegment')
    model.predict(
        img_dir,
        save=True,
        conf=conf,
        device=DEVICE,
    )

if __name__ == '__main__':
    pass
    myolo8_x()
    # myolo9_e()
    # model_val(r'best.pt')
    # myolo8_modifiy('yolov8x-mdetect10-n1-gat1.yaml')
    # myolo8_modifiy('yolov8x-mdetect10-n1-gat2.yaml')
    # myolo8_modifiy('yolov8x-mdetect-gat11.yaml')
    # myolo8_modifiy('yolov8x-mdetect-gat12.yaml')
    # myolo8_modifiy('yolov8x-mdetect-gat13.yaml')

    # myolo8_modifiy('yolov8x-mdetect-gat11-n0.yaml')
    # myolo8_modifiy('yolov8x-mdetect-gat11-n1.yaml')

    # myolo8_modifiy('yolov8x-mdetect-gat22-n0.yaml')
    # myolo8_modifiy('yolov8x-mdetect-gat22-n1.yaml')
    # model_val(r'runs/mdetect/train76/weights/best.pt')