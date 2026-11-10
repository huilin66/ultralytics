# Multimodal compatibility matrix (Phase 3 / E1)

Updated 2026-11-10. A capability is marked ✅ only when an automated test or a recorded
smoke run proves it. `—` means not implemented or not yet covered.

## Runtime contract

| Component | Implementation | Evidence |
|---|---|---|
| Paired RGB-T-D dataset | `ultralytics/data/multimodal.py` (`Modalities`, `MultiModalDataset`) | `tests/test_multimodal.py::test_multimodal_dataset_stacks_paired_modalities` |
| Channel-split input adapter | `ModalSplit` + `Index` in model YAML | every multimodal template |
| Feature fusion block | `MultiModalFusion` (concat / add, N inputs) | `test_multimodal_fusion_operators_validate_feature_geometry` |
| Shared-weight stage | `ModalFold` / `ModalUnfold` | `test_shared_weight_yaml_runs_one_stage_on_folded_modalities` |
| Native head / loss / NMS | stock `Detect`, `E2ELoss` / `v8DetectionLoss`, stock predictor | unchanged upstream code |
| Model factory | `MultiModalYOLO` facade | `tests/test_multimodal.py` |

## Architectures

| Architecture | Templates | Build | Forward | Backward | Train epoch | Predict | Status |
|---|---|---|---|---|---|---|---|
| YOLOv8x | `cfg/mmodels/yolov8x-mm{2,3}-{if,ef,nif,bf,nf,hf}.yaml` | ✅ | — | — | — | — | build + COCO transfer tested |
| YOLO11 (seg) | `cfg/models/multimodal/yolo11-mm3-bf-seg.yaml` | ✅ | — | — | — | — | template + task inference tested |
| YOLO26x (det) | `cfg/mmodels/yolo26x-mm3-bf.yaml` | ✅ | ✅ | ✅ | ✅ | ✅ | **development-verified 2026-11-10** |
| YOLOv9 / v10 / v12 / v13 | none | — | — | — | — | — | no templates yet |

YOLO26x evidence (`tests/test_multimodal.py::test_yolo26x_multimodal_model_keeps_the_native_end2end_head`
plus manual smoke runs):

- builds 121,815,102 parameters with `channels=5`, three backbone branches and P3/P4/P5 fusion;
- end-to-end inference output `(1, 84, 6)` for `nc=2` at 64 px (84 anchors, 4 box + 2 class);
- loss/backward reaches the first convolution of the RGB, thermal and depth branches;
- one epoch on a 4-image synthetic RGB-T-D dataset trains, validates, saves `best.pt`, and
  reloads it for a posed validation pass;
- `predict(..., data=<multimodal yaml>)` loads `rgb`, `thermal` and `depth` companions;
- a list of primary image paths is routed through `LoadImagesAndVideos`, so each sample keeps its
  real source path (`tests/test_multimodal.py::test_multimodal_predictor_keeps_primary_paths_for_a_list_source`).

## Real-dataset smoke run (2026-11-10)

YOLO26x RGB-T-D BF, `fraction=0.1`, 1 epoch, 640 px, batch 8, `pretrained=False`, RTX 5090. These runs
check that the data/training/inference pipeline works on the converted datasets; one 10%-epoch run is not a
baseline and its mAP is not a result.

| Dataset | Train box / cls / dfl | Val box / cls | Precision | Recall | mAP50-95 | Inference |
|---|---|---|---|---|---|---|
| MID-3K_mdet | 4.016 / 9.519 / 0.029 | 4.146 / 8.602 | 0.0005 | 0.0343 | 0.0000 | 5.49 ms/img |
| R-LiViT_mdet | 4.368 / 5.659 / 0.016 | 4.393 / 5.587 | 0.0000 | 0.0038 | 0.0000 | 3.97 ms/img |

Both runs wrote `weights/best.pt` (~245 MB). Reloading those checkpoints and predicting four val primary
images (as a path list) plus one as a single path succeeded on both datasets, with `data.yaml` companions
loaded from the `thermal/` and `depth/` trees. Confidence stayed near the `conf=0.001` floor
(0.0119 for MID-3K, 0.0016 for R-LiViT), as expected for a 10%-epoch model; detection quality on these
datasets still has to come from the full Phase 3 baselines.

## Not yet covered

- YOLO26x other fusion modes (`IF`, `EF`, `NIF`, `NF`, `HF`) and 2-modality (`mm2`) templates.
- Pretrained COCO transfer for YOLO26 (`pretrained.py` currently accepts only `source: yolov8`).
- Automated train/predict tests for YOLOv8x and YOLO11 templates.
- Formal (full-epoch, multi-seed) training on R-LiViT / MID-3K and the B0–B8 baseline matrix.
