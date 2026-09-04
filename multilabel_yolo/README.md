# True multi-label YOLO detection

Each line in a label file represents one physical box. The first field may
contain comma-separated, zero-based class IDs:

```text
1,4,7 0.50 0.50 0.20 0.30
```

Use the custom entry points from the repository root:

```powershell
conda run -n mdet python scripts/inspect_multilabel_batch.py --data data_multilabel.yaml
conda run -n mdet python scripts/train_multilabel.py --model yolo11n.pt --data data_multilabel.yaml --epochs 100
conda run -n mdet python scripts/val_multilabel.py --model runs/detect/train/weights/best.pt --data data_multilabel.yaml
```

YOLOv10 end-to-end YAMLs are supported as well, for example
`ultralytics/cfg/models/v10/yolov10n.yaml`. Both its one-to-many and
one-to-one branches use the n-hot loss during training. For PyTorch
validation/prediction, the raw one-to-one branch is decoded before custom
multi-label NMS so the native single-class postprocess does not discard
additional labels.

Training keeps `cls` as an internal transport ID and uses `cls_nhot` for all
classification supervision. Ground-truth expansion happens only inside the
validator's metric preparation.

## Image-level multi-label classification

The image classifier is a separate task for classifying a whole image or a
crop produced by a detector. It does not use `ImageFolder` because one image
can have several labels. Use a YAML like:

```yaml
path: /data/billboard_cls
train: images/train
val: images/val
labels: labels
names: [clean, damaged, occluded, text]
threshold: 0.5
```

For each image, create `labels/<split>/<stem>.txt` containing zero-based class
IDs, such as `1,3`. An empty file is an all-negative image; a missing file is
reported as an annotation error. The dataset returns `cls_nhot [B, nc]`, the
loss uses `BCEWithLogitsLoss`, and prediction uses independent sigmoid scores.

```powershell
conda run -n mdet python scripts/train_multilabel_classification.py `
    --model ultralytics/cfg/models/11/yolo11n-cls.yaml `
    --data path/to/multilabel_cls.yaml --epochs 100
conda run -n mdet python scripts/val_multilabel_classification.py `
    --model runs/classify/train/weights/best.pt `
    --data path/to/multilabel_cls.yaml --threshold 0.5
conda run -n mdet python scripts/predict_multilabel_classification.py `
    --model runs/classify/train/weights/best.pt --source path/to/crops --threshold 0.5
```

Each returned `Results` object exposes `multilabel_indices`,
`multilabel_names`, and `multilabel_scores`. Thus detector output can be
cropped and passed as a list of images to the same classifier without changing
the detector's physical-box/n-hot training representation.

For a direct two-stage call:

```python
from multilabel_yolo import classify_detection_crops

detections = detector.predict(source="image.jpg")
detections = classify_detection_crops(detections, classifier, threshold=0.5)
print(detections[0].detection_multilabel_names)  # one label list per box
```
