import torch
from ultralytics import YOLO
BATCH_SIZE = 16
EPOCHS = 500
DEVICE = torch.device('cuda:0')
DATA = "billboard_mdet5_10.yaml"
CONF = 0.5

def myolo8m_x_modifiy(model_path):
    model = YOLO(model_path, task = 'mdetect')
    model.load('yolov8x.pt')

    model.train(data=DATA, device=DEVICE,
                epochs=EPOCHS, imgsz=640, val=True, batch=BATCH_SIZE, patience=EPOCHS)


def model_val(weight_path):
    model = YOLO(weight_path, task='mdetect')

    model.val(data=DATA, device=DEVICE,
              # conf=conf
              )

def model_predict(weight_path):
    model = YOLO(weight_path, task='mdetect')
     # evaluate model performance on the validation set

    model.predict(
        # r"/nfsv4/23039356r/data/billboard/data0521_m/yolo_rgb_detection6/infer_image",
        r"/nfsv4/23039356r/data/billboard/data0521_m/yolo_rgb_detection5/images",
        save=True,
        conf=0.5,
        device=DEVICE,
    )

def myolo8_x_m(weight_path):
    model = YOLO(weight_path, task = 'mdetect')

    model.train(data=DATA, device=DEVICE,
                epochs=200, imgsz=640, val=True, batch=BATCH_SIZE, patience=EPOCHS,
                freeze=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,],
                freeze_head=['.cv2', '.cv3'],
                mdet=10.0,
                lrf=0.001,
                )

if __name__ == '__main__':
    pass
    # myolo8m_x_modifiy(r'yolov8x-mdetect10-n0.yaml')
    # myolo8m_x_modifiy(r'yolov8x-mdetect10-n1.yaml')
    # myolo8m_x_modifiy(r'yolov8x-mdetect10-n2.yaml')
    # myolo8m_x_modifiy(r'yolov8x-mdetect10-n3.yaml')

    # model_predict(r'runs/mdetect/train21/weights/best.pt')
    # myolo8_x_m(r'runs/mdetect/train22/weights/best.pt')
    # model_predict(r'runs/mdetect/train27/weights/best.pt')
    model_val('best.pt')