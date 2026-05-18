from ultralytics import YOLO

DATASET_NAME = "mars-seg.yaml"
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = 0

if __name__ == "__main__":
    MODEL_LIST = [
        "yolo26x-seg.yaml",
        "yolov8x-seg.yaml",
        "yolov9e-seg.yaml",
        "yolov10x-seg.yaml",
        "yolo11x-seg.yaml",
        "yolo12x-seg.yaml",
        ]
    MODEL_WEIGHTS_LIST = [
        "yolo26x.pt",
        "yolov8x.pt",
        "yolov9e.pt",
        "yolov10x.pt",
        "yolo11x.pt",
        "yolo12x.pt",
    ]
    for MODEL_NAME, MODEL_WEIGHT in zip(MODEL_LIST, MODEL_WEIGHTS_LIST):
        NAME = f'debug_{DATASET_NAME.split(".")[0]}_{MODEL_NAME.split(".")[0]}'
        model = YOLO(MODEL_NAME)
        model.load(MODEL_WEIGHT)
        model.train(data=DATASET_NAME, epochs=EPOCHS, batch=BATCH_SIZE, imgsz=IMG_SIZE, device=DEVICE, name=NAME, pretrained=False, optimizer='MuSGD', lr0=0.01)
