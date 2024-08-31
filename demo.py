import torch
# from ultralytics import YOLO
# BATCH_SIZE = 2
# EPOCHS = 2
# DEVICE = torch.device('cuda:0')
# DATA = "billboard_mdet.yaml"
# weight_path = r'runs/mdetect/train326/weights/best.pt'
# model = YOLO(weight_path, task='mdetect')
#
# model.predict(
#     r"E:\data\0417_signboard\data0521_m\yolo_rgb_detection4\images",
#     save=True,
#     conf=0.2,
#     # iou=0.2
#     device=DEVICE,
# )
model = torch.load(r'runs/mdetect/train151/weights/best.pt')