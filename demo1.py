from pathlib import Path
from ultralytics import YOLO

model_name = "yolo11un_moe.yaml"
data_name = "coco.yaml"
train_name = f"{Path(data_name).stem}-{Path(model_name).stem}-train"

# Load a pretrained YOLO11n model
model = YOLO(model_name)

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data=data_name,  # Path to dataset configuration file
    epochs=300,  # Number of training epochs
    imgsz=640,  # Image size for training
    batch=128,
    device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    # name=train_name,
    name='debug',
    seed=42,
    close_mosaic=10,
    workers=8,
    optimizer='auto',
)