import os
import sys
import torch
from ultralytics import YOLO
from tz2yolo import convert_yolo_to_tz, result_check
BATCH_SIZE = 16
EPOCHS = 500
IMGSZ = 1280
DEVICE = torch.device('cuda:0')
DATA = "mm.yaml"

mapping_class = {
    0: 'Boeing737',
    1: 'Boeing747',
    2: 'Boeing777',
    3: 'Boeing787',
    4: 'C919',
    5: 'A220',
    6: 'A321',
    7: 'A330',
    8: 'A350',
    9: 'ARJ21',
    10: 'other-airplane',
    11: 'A320/321',
    12: 'Boeing737-800',
    13: 'other',
}

def predict(weight_path, img_dir, save_dir, conf=0.5):
    model = YOLO(weight_path, task='detect')
    model.predict(
        img_dir,
        save=True,
        conf=conf,
        device=DEVICE,
        save_dir=save_dir,
        save_txt=True,
        imgsz=IMGSZ,
    )

def predict_app(input_dir, output_dir, weight_path, temp_dir, conf=0.5):
    predict(weight_path, input_dir, temp_dir, conf=conf)
    convert_yolo_to_tz(os.path.join(temp_dir, 'labels'), output_dir, temp_dir, mapping_class)
    result_check(input_dir, output_dir)



if __name__ == '__main__':
    pass
    # predict_app(
    #     # r'/model_path',
    #     # r'/input_path',
    #     # r'/output_path',
    #
    #     r'E:\data\tp\multi_modal_airplane_train\infer_img',
    #     r'E:\data\tp\multi_modal_airplane_train\infer_result',
    #     r'v2_0801_01.pt',
    #     r'E:\data\tp\multi_modal_airplane_train\temp'
    # )
    # predict_app(
    #     # r'/model_path',
    #     # r'/input_path',
    #     # r'/output_path',
    #
    #     r'E:\data\tp\car_det_train\car_det_train\infer_img',
    #     r'E:\data\tp\car_det_train\car_det_train\infer_result',
    #     r'v1_0802_02.pt',
    #     r'E:\data\tp\car_det_train\car_det_train\temp'
    # )
    predict_app(
        input_dir=sys.argv[1],
        output_dir=sys.argv[2],
        weight_path="v1_0802_02.pt",
        temp_dir='temp_result'
    )
