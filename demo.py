# from ultralytics.data.dataset import YOLODataset
# from ultralytics.data.augment import Mosaic
# from ultralytics.utils.plotting import Annotator
# import cv2
#
# # 1) 构建数据集（示例：只用训练图像路径和类名；按你的数据改）
# ds = YOLODataset(img_path="/localnvme/data/billboard/fused_data/data3072_mseg_c5_0809/images",
#                  data={"names": {
#                       0: 'background',
#                       1: 'wall display',
#                       2: 'frame',
#                       3: 'projecting display',
#                       4: 'none',
#                       5: 'hanging display',
#                       6: 'other',
#                 }},
#                  task="segment")
#
# # 2) 创建 Mosaic 变换（p=1.0 表示必用，n=4 为 2x2 mosaic）
# mosaic = Mosaic(ds, imgsz=640, p=1.0, n=4)  # 返回的labels里含合成后的图与框
#
# # 3) 取一个样本并做 mosaic（内部会再采样其它3张进行拼接）
# labels = ds[0]
# aug = mosaic(labels)              # aug["img"] 为 HWC BGR 图像
# img = aug["img"]
#
# # 4) 可选：画上标注后保存
# ann = Annotator(img, pil=False)
# for xyxy, cls in zip(aug["instances"].xyxy.astype(int), aug["cls"].astype(int)):
#     ann.box_label(xyxy, str(cls))
# cv2.imwrite("mosaic_preview.jpg", ann.result())



# from ultralytics import YOLO
# YOLO("yolov8n.pt").train(
#     data="fusedata3072_seg_c5_0809_80p.yaml",
#     epochs=1, imgsz=960, batch=8,
#     mixup=1.0,
#     plots=True, save=True,
#     task='segment'
# )
from ultralytics import YOLO

model = YOLO("runs/segment/fusedata3899_seg_c5_0818_80p-[yolov10x-seg-dlka3res]2/weights/last.pt")
model.train(resume=True)