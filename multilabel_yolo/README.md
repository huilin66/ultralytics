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

Training keeps `cls` as an internal transport ID and uses `cls_nhot` for all
classification supervision. Ground-truth expansion happens only inside the
validator's metric preparation.
