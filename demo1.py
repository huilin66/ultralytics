import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

DATASET_NAME = "coco.yaml"
EPOCHS = 100
BATCH_SIZE = 64
IMG_SIZE = 640
DEVICE = 0

if __name__ == "__main__":
    MODEL_LIST = [
        "yolo26n.yaml",             # 15, 17GB
        "yolo26n-c3k2_mhc.yaml",    # 13, 
        "yolo26n-concat_mhc.yaml"   # 14,
        ]
    for MODEL_NAME in MODEL_LIST:
        NAME = f'{DATASET_NAME.split(".")[0]}_{MODEL_NAME.split(".")[0]}'
        model = YOLO(MODEL_NAME)
        model.train(data=DATASET_NAME, epochs=EPOCHS, batch=BATCH_SIZE, imgsz=IMG_SIZE, device=DEVICE, name=NAME, pretrained=False)
