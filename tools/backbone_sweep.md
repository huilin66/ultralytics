# Backbone sweep: single- against dual-modality

Purpose: show that the dual-modality gain is not an artifact of one detector family. Every
family contributes an RGB-only cell and an RGB+T cell with the identical recipe, so the only
difference inside a family is the second modality.

## Matrix

Largest scale each family ships (`yolov9` has no `x`; `e` is its widest). Parameters are for
the built model, not for the template's declared width.

| Family | Single: stock checkpoint + `data-rgb.yaml` | Dual: template + `data-rgb-t.yaml` | Params single / dual |
| --- | --- | --- | --- |
| YOLOv8x | `yolov8x.pt` | `yolov8x-mm2-31-bf.yaml` | 68.2 / 101.0 M |
| YOLOv9e | `yolov9e.pt` | `yolov9e-mm2-31-bf.yaml` | 58.2 / 91.6 M |
| YOLOv10x | `yolov10x.pt` | `yolov10x-mm2-31-bf.yaml` | 31.8 / 48.5 M |
| YOLO11x | `yolo11x.pt` | `yolo11x-mm2-31-bf.yaml` | 57.0 / 89.4 M |
| YOLO12x | `yolo12x.pt` | `yolo12x-mm2-31-bf.yaml` | 59.2 / 95.8 M |
| YOLO26x | `yolo26x.pt` | `yolo26x-mm2-31-bf.yaml` | 59.0 / 91.4 M |

The `mm2-31-bf` templates are generated from each family's own `mm3-bf` template by dropping
the depth branch and renumbering; they therefore inherit that family's stock neck depth. Data
YAMLs live in `/srv/nas/home/dataset/R-LiViT_mdet/`: `data-rgb.yaml` (3 channels, no modality
list) and `data-rgb-t.yaml` (4 channels, `[[3, 1]]`).

## Recipe

Identical to E1.5/E2: COCO transfer, AdamW at `lr0=1e-3`, cosine decay to `lrf=0.01`, 200
epochs, `imgsz=640`, physical batch 4 with `nbs=64`, `close_mosaic=10`, deterministic, three
seeds. A checkpoint is re-parsed through this repository's stock YAML before the transfer,
because a released `.pt` may declare the same modules with different channel caps.

## Commands

```bash
# one seed, one GPU; three processes give seeds 0/1/2 in parallel
CUDA_VISIBLE_DEVICES=0 python tools/run_backbone_sweep.py --seed 0
CUDA_VISIBLE_DEVICES=1 python tools/run_backbone_sweep.py --seed 1
CUDA_VISIBLE_DEVICES=2 python tools/run_backbone_sweep.py --seed 2

# one-epoch smoke of a subset
python tools/run_backbone_sweep.py --seed 0 --epochs 1 --items yolo11x-rgb yolo11x-rgb-t
```

Outputs land under `runs/backbone_sweep/<item>-seed<k>/` with a per-run `sweep.json`.
