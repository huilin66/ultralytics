import os
import shutil
from pandas.compat import F
from ultralytics import YOLO

DATASET_NAME = "mars-seg.yaml"
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = 0
CONF_VAL = 0.5
VAL_IMAGE = r'/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase/yolo_data/val/images'
VAL_RESULT = r'/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase/yolo_data/val/results'

if __name__ == "__main__":
    MODEL_LIST = [
        "yolo26x-seg.yaml",
        "yolov8x-seg.yaml",
        "yolov9e-seg.yaml",
        "yolov10x-seg.yaml",
        "yolo11x-seg.yaml",
        "yolo12x-seg.yaml",
        ]
    for MODEL_NAME in MODEL_LIST:
        NAME = f'{DATASET_NAME.split(".")[0]}_{MODEL_NAME.split(".")[0]}2'
        MODEL_WEITHT_TRAIN = os.path.join('runs', 'segment', NAME, 'weights', 'best.pt')
        model = YOLO(MODEL_WEITHT_TRAIN)
        print(f'{"=="*20} [VAL] {NAME} {"=="*20}')
        VAL_RESULT_DIR = os.path.join(VAL_RESULT, f'{NAME}_{CONF_VAL}')
        if os.path.exists(VAL_RESULT_DIR):
            shutil.rmtree(VAL_RESULT_DIR)
        model.val(data=DATASET_NAME, batch=BATCH_SIZE, imgsz=IMG_SIZE, device=DEVICE, conf=CONF_VAL)
        model.predict(source=VAL_IMAGE, batch=BATCH_SIZE, imgsz=IMG_SIZE, device=DEVICE, conf=CONF_VAL, name=VAL_RESULT_DIR, verbose=False, save_txt=True)
        print(f'results save to {VAL_RESULT_DIR}')
        print(f'{"=="*60}\n')