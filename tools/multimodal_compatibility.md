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
| YOLO26x (det, all modes) | `cfg/mmodels/yolo26x-mm3-{if,ef,nif,bf,nf,hf}.yaml` | ✅ | ✅ | ✅ | ✅ | ✅ | **development-verified 2026-11-10** |
| YOLOv9 / v10 / v12 / v13 | none | — | — | — | — | — | no templates yet |

YOLO26x evidence (`tests/test_multimodal.py::test_yolo26x_multimodal_model_keeps_the_native_end2end_head`
and `::test_yolo26x_fusion_modes_build_and_backpropagate`, plus manual smoke runs):

- every fusion mode builds with `channels=5`, `end2end=True` and `reg_max=1`;
- parameters at `nc=2` (three RGB-T-D branches, `scale: x`):

  | Mode | Fusion point | Inputs per fusion node | Parameters |
  |---|---|---|---|
  | IF | early (input stack) | — | 58,814,916 |
  | EF | encoder P3/P4/P5 | 3 | 112,334,148 |
  | NIF | encoder P3/P4 + nape P5 | 3 | 115,287,876 |
  | BF | backbone P3/P4/P5 | 3 | 121,817,412 |
  | NF | neck P3/P4/P5 | 3 | 167,899,716 |
  | HF | head P3/P4/P5 (1×1 per-modality projections) | 3 | 171,892,548 |

- end-to-end inference output `(1, 84, 6)` for `nc=2` at 64 px (84 anchors, 4 box + 2 class) in all modes;
- loss/backward reaches the first convolution of every modality branch (IF: the single 5-channel stem);
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

### Fusion-mode sweep (R-LiViT_mdet)

Same protocol (1 epoch, `fraction=0.1`, 640 px, `pretrained=False`, RTX 5090) but batch 4; the BF row is
the batch-8 run above, so the speeds are not directly comparable. Every mode trained, validated, reloaded
its `best.pt` and predicted the same four val primary images.

| Mode | Fusion point | Parameters (nc=2) | Train box / cls / dfl | Val box / cls | P | R | mAP50-95 | Inference |
|---|---|---|---|---|---|---|---|---|
| IF | input stack | 58,814,916 | 4.483 / 5.620 / 0.016 | 4.411 / 5.581 | 0.0000 | 0.0240 | 0.0000 | 2.03 ms/img |
| EF | encoder P3/P4/P5 | 112,334,148 | 4.414 / 5.498 / 0.016 | 4.405 / 5.586 | 0.0001 | 0.0114 | 0.0000 | 4.11 ms/img |
| NIF | encoder P3/P4 + nape P5 | 115,287,876 | 4.311 / 5.397 / 0.015 | 4.393 / 5.588 | 0.0000 | 0.0000 | 0.0000 | 4.26 ms/img |
| BF | backbone P3/P4/P5 | 121,817,412 | 4.368 / 5.659 / 0.016 | 4.393 / 5.587 | 0.0000 | 0.0038 | 0.0000 | 3.97 ms/img |
| NF | neck P3/P4/P5 | 167,899,716 | 4.379 / 5.438 / 0.015 | 4.412 / 5.589 | 0.0000 | 0.0057 | 0.0000 | 5.44 ms/img |
| HF | head P3/P4/P5 | 171,892,548 | 4.303 / 5.387 / 0.015 | 4.391 / 5.566 | 0.0000 | 0.0000 | 0.0000 | 5.50 ms/img |

At `conf=0.001` each prediction returned `max_det` boxes (300) with a top confidence of 0.0016–0.0017 —
again a 10%-epoch artefact, not a quality signal. Parameter count rises monotonically from IF to HF, and
latency follows it except for the BF row (different batch).

## Not yet covered

- YOLO26x 2-modality (`mm2`) templates.
- Pretrained COCO transfer for YOLO26 (`pretrained.py` currently accepts only `source: yolov8`).
- Automated train/predict tests for YOLOv8x and YOLO11 templates.
- Formal (full-epoch, multi-seed) training on R-LiViT / MID-3K and the B0–B8 baseline matrix.
