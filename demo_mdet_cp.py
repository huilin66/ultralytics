import torch
from ultralytics import YOLO
BATCH_SIZE = 32
EPOCHS = 500
IMGSZ = 640
DEVICE = torch.device('cuda:0')
DATA = "billboard_mdet5_10_0806m.yaml"
# DATA = "billboard_mdet5_10.yaml"

def myolo_train(cfg_path, pretrain_path, auto_optim=True):
    model = YOLO(cfg_path, task='mdetect')
    model.load(pretrain_path)
    if auto_optim:
        model.train(data=DATA, device=DEVICE,
                    epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)
    else:
        model.train(data=DATA, device=DEVICE,
                    optimizer='AdamW', lr0=0.0001,
                    epochs=EPOCHS, imgsz=IMGSZ, val=True, batch=BATCH_SIZE, patience=EPOCHS)

def myolo8_x(cfg_path, auto_optim=True):
    myolo_train(cfg_path, 'yolov8x.pt', auto_optim=auto_optim)

def myolo9_e(cfg_path, auto_optim=True):
    myolo_train(cfg_path, 'yolov9e.pt', auto_optim=auto_optim)

def myolo10_x(cfg_path, auto_optim=True):
    myolo_train(cfg_path, 'yolov10x.pt', auto_optim=auto_optim)


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
    # myolo10_x('yolov10x-mdetect0.yaml')
    # myolo10_x('yolov10x-mdetect0.yaml', auto_optim=False)
    # myolo10_x('yolov10x-mdetect0-sephead.yaml', auto_optim=False)
    # myolo10_x('yolov10x-mdetect0-sppftf0.yaml', auto_optim=False)
    # myolo10_x('yolov10x-mdetect0-sppftf0-sephead.yaml', auto_optim=False)
    # myolo10_x('yolov10x-mdetect0-sppftf1.yaml', auto_optim=False)
    # myolo10_x('yolov10x-mdetect0-sppftf1-sephead.yaml', auto_optim=False)

    # model_val(r'runs/mdetect/train96/weights/best.pt')
    # model_val(r'runs/mdetect/train97/weights/best.pt')
    # model_val(r'runs/mdetect/train100/weights/best.pt')
    # model_val(r'runs/mdetect/train101/weights/best.pt')
    # model_val(r'runs/mdetect/train102/weights/best.pt')