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
