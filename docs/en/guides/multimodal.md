---
comments: true
description: Train, validate and run inference with pixel-aligned multi-modal YOLO detection and instance segmentation models.
---

# Multi-Modal YOLO

This extension keeps Ultralytics' normal model, loss, optimizer and export paths intact. It adds a paired-image dataset,
an arbitrary-channel splitter for YAML models, and task adapters for detection and instance segmentation.

## Dataset layout

Every sample is anchored by the first modality. Other modality files preserve the same relative path and filename below
their own roots. Set `suffix` only when a companion uses a different extension.

```text
dataset/
├── images/
│   ├── rgb/train/scene_001.jpg
│   ├── thermal/train/scene_001.png
│   └── depth/train/scene_001.png
└── labels/
    └── rgb/train/scene_001.txt
```

```yaml
# rgb-thermal-depth.yaml
path: /datasets/rgbtd
train: images/rgb/train
val: images/rgb/val
channels: 5 # 3 + 1 + 1
names:
    0: person

modalities:
    - name: rgb
      path: images/rgb
      channels: 3
      color: bgr # opt in to HSV augmentation for this modality
    - name: thermal
      path: images/thermal
      channels: 1
      suffix: .png
    - name: depth
      path: images/depth
      channels: 1
      suffix: .png
```

The images must be pixel-aligned, `uint8`, and have identical height and width before augmentation. Labels are read
from the primary-modality path using the standard YOLO `images/` → `labels/` convention.

## Train, validate and predict

```python
from ultralytics.models.multimodal import MultiModalYOLO

model = MultiModalYOLO("ultralytics/cfg/models/multimodal/yolo11-mm3-bf-seg.yaml")
model.train(data="rgb-thermal-depth.yaml", epochs=100, imgsz=640)
model.val(data="rgb-thermal-depth.yaml")
model.predict(source="/datasets/rgbtd/images/rgb/val/scene_001.jpg", data="rgb-thermal-depth.yaml")
```

For image-file prediction, `data=` supplies the companion-file mapping. A pre-fused HWC NumPy array with exactly the
model channel count can be passed directly and does not need a data YAML.

## Augmentation policy

Mosaic, affine/perspective, copy-paste, MixUp, CutMix, letterbox and flips operate on the fused HWC stack once, so all
modalities receive identical spatial geometry. `color: bgr` selects individual three-channel groups for one shared HSV
sample; thermal, depth and other sensor groups are left unchanged. Generic Albumentations color transforms are not
applied to a multi-channel stack, avoiding accidental per-modality desynchronization.

## Model topology

`yolo11-mm3-bf-seg.yaml` is a three-branch backbone-fusion example. Its `multimodal` block declares the channel layout,
fusion mode, operator and fusion points; `ModalSplit: [[3, 1, 1]]` returns RGB, thermal and depth tensors, while stock
`Index` selects each branch. `MultiModalFusion` validates and concatenates P3, P4 and P5 before the ordinary YOLO11 PAN
and `Segment` head. To add modalities, change both `input_sections` and `ModalSplit`, add the corresponding `Index` and
encoder branch, and include its P3/P4/P5 outputs in each fusion module.

`yolov8x-mm3-pre-sppf.yaml` is the corresponding three-modality YOLOv8x encoder-fusion example. It concatenates the
P3/P4/pre-nape-P5 branch outputs, projects them, then runs one shared `SPPF` on P5. This is the configuration to use
when fusion must happen before SPPF rather than after each branch has its own SPPF.

## Shared stages

Set `share_weight: true` only for a YAML that explicitly surrounds every shared stage with `ModalFold` and
`ModalUnfold`. Independent per-modality input adapters must first produce the same channel count; `ModalFold` then
converts `M` tensors of shape `(B, C, H, W)` into one `(B*M, C, H, W)` batch. Ordinary YOLO layers between the two
modules are therefore single shared module instances, including their BatchNorm statistics. `ModalUnfold: [M]` restores
the ordered modality features for later fusion.

The extension rejects `share_weight: true` configurations missing either module. It does not guess which layers in an
arbitrary YAML are semantically an encoder or nape; `shared_stages` records the intended scope, while `ModalFold` and
`ModalUnfold` make that scope unambiguous in the graph.

The implementation deliberately uses one standard segmentation head. Add a PGI/auxiliary branch only after establishing
this baseline; it then belongs in a custom loss/head experiment rather than in the paired-data infrastructure.
