---
comments: true
description: Reference for the two-branch YOLOv8x multimodal model configurations.
---

# YOLOv8x MM2 Model Configurations

The `ultralytics/cfg/mmodels/yolov8x-mm2-*.yaml` files are static, two-branch models for pixel-aligned image pairs.
They are separate from the `mm3` templates, which remain three-branch RGB + thermal + depth reference models.

Every MM2 configuration has this input contract:

```yaml
channels: 6
multimodal:
  input_sections: [3, 3]
```

The first three channels belong to the primary image and the final three to the auxiliary image. The data YAML must
declare the same ordered channel layout. `tools/train_yolo_mb.py` verifies that contract before constructing a model.

| Configuration | Fusion point | Description |
|---|---|---|
| `yolov8x-mm2-if.yaml` | Input | Stack both images before a single shared encoder. |
| `yolov8x-mm2-ef.yaml` | Encoder | Fuse P3, P4 and P5 after the two encoders. |
| `yolov8x-mm2-nif.yaml` | Nape-in | Fuse P3/P4 at encoder exits and P5 within the nape. |
| `yolov8x-mm2-bf.yaml` | Backbone | Fuse two completed backbone feature pyramids. |
| `yolov8x-mm2-nf.yaml` | Neck | Fuse two completed backbone-and-neck branches. |
| `yolov8x-mm2-hf.yaml` | Head feature | Fuse projected P3, P4 and P5 head features before `Detect`. |

For the project launcher, select one of the same names without the `.yaml` extension:

```bash
bash scripts/m2_double/train_yolo_mb.sh \
  --modalities ri \
  --models yolov8x-mm2-if,yolov8x-mm2-ef
```

Do not use an MM2 configuration with a three-modality data YAML, or an MM3 configuration with a paired data YAML.
Those graphs have different branch counts, `ModalSplit` sections and pretrained-layer mappings.
