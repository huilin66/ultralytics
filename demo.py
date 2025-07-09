

from ultralytics import RTDETR
# 加载模型
model = RTDETR("weights/best.pt")

# 开始验证
validation_results = model.val(
    data="others.yaml",
    imgsz=640,
    batch=16,
    save_json=True,
    conf=0.25,
    iou=0.6,
    device="0"
)
print(validation_results.results_dict)


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# 加载标注和预测
annFile = 'coco_annotations.json'
resFile = 'predictions.json'
cocoGt = COCO(annFile)
cocoDt = cocoGt.loadRes(resFile)

# 先评估整体性能 (area='all')
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
