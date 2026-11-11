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

## Shared template configuration

Ultralytics YAML has no cross-file inheritance (`yaml_model_load` only loads, guesses the scale and
records `yaml_file`), so the shared contract lives in `cfg/mmodels/base.yaml` and is merged by our own
loader: `MultiModalModelMixin.__init__` calls `load_multimodal_config` → `deep_merge`, and a key declared
by the model file always wins. `base.yaml` declares `channels: 5`, `multimodal.input_sections: [3, 1, 1]`,
`multimodal.operator: concat`, `multimodal.share_weight: false` and `multimodal.pretrained: {source: yolov8}`.
Every template therefore keeps only what makes it different:

- the `mm2` files override `channels`/`input_sections` (`[1, 1]` → 2, `[3, 1]` → 4, `[3, 3]` → 6);
- the 36 `yolov8*` templates inherit `pretrained: {source: yolov8}` and no longer spell out `copies`;
  `load_coco_pretrained` derives the copy plan from the donor and target graphs via
  `derive_copy_plan` (the mapping is not written into the YAML);
- the other 144 templates (`yolov9/10/11/12`, `yolo26x`) set `pretrained: false`, because a yolov8 donor
  has no matching graph for them; `load_coco_pretrained` raises a clear error if called anyway.

`tests/test_multimodal.py::test_every_template_inherits_the_shared_contract` checks this for all 180
files, including that re-merging an already-merged config is idempotent. Overriding any shared default is
still possible per file, for example `multimodal: {fusion: BF, pretrained: false}`.

## Architectures

| Architecture | Templates | Build | Forward | Backward | Train epoch | Predict | Status |
|---|---|---|---|---|---|---|---|
| YOLOv8x | `cfg/mmodels/yolov8x-mm{2,3}-{if,ef,nif,bf,nf,hf}.yaml` + `cfg/models/multimodal/yolov8x-mm3-pre-sppf.yaml` | ✅ | ✅ | ✅ | — | — | build/forward/backward + COCO transfer tested |
| YOLOv8s | `cfg/mmodels/yolov8{nsmlx}-mm3-{mode}.yaml` | ✅ | ✅ | ✅ | ✅ | ✅ | R-LiViT smoke 2026-11-10 |
| YOLOv9s | `cfg/mmodels/yolov9{tsmce}-mm3-{mode}.yaml` | ✅ | ✅ | ✅ | ✅ | ✅ | R-LiViT smoke 2026-11-10 |
| YOLOv10s | `cfg/mmodels/yolov10{nsmblx}-mm3-{mode}.yaml` | ✅ | ✅ | ✅ | ✅ | ✅ | end2end `v10Detect`; R-LiViT smoke 2026-11-10 |
| YOLO11 (det) | `cfg/mmodels/yolo11{nsmlx}-mm3-{mode}.yaml` | ✅ | ✅ | ✅ | ✅ | ✅ | R-LiViT smoke 2026-11-10 |
| YOLO11 (seg) | `cfg/models/multimodal/yolo11-mm3-bf-seg.yaml` | ✅ | ✅ | ✅ | — | — | build/forward/backward tested, task inference tested |
| YOLO12s | `cfg/mmodels/yolo12{nsmlx}-mm3-{mode}.yaml` | ✅ | ✅ | ✅ | ✅ | ✅ | R-LiViT smoke 2026-11-10 |
| YOLO26x (det, all modes) | `cfg/mmodels/yolo26x-mm3-{if,ef,nif,bf,nf,hf}.yaml` | ✅ | ✅ | ✅ | ✅ | ✅ | **development-verified 2026-11-10** |
| YOLO26x (det, 2 modalities) | `cfg/mmodels/yolo26x-mm2-{31,11}-{if,ef,nif,bf,nf,hf}.yaml` | ✅ | ✅ | ✅ | — | — | build/forward/backward tested 2026-11-10 |
| YOLOv13 | none | — | — | — | — | — | not shipped by Ultralytics 8.4.91 |

YOLOv8x and YOLO11 evidence: `::test_yolov8x_templates_build_and_backpropagate` covers the 13 YOLOv8
templates (6 mm3, 6 mm2 with 3+3 channels, 1 pre-SPPF) and
`::test_yolo11_segmentation_template_builds_and_backpropagates` covers the YOLO11 segmentation template.
Each case checks the native output shape, three Detect inputs, the fusion-node input count and nonzero
gradients in every modality stem.

YOLOv8s–YOLOv12s evidence: `::test_other_yolo_templates_build_and_backpropagate` covers the 30 detection
templates (5 architectures × 6 fusion modes). It checks `channels=5`, `[3, 1, 1]`, three fusion nodes of
arity 3 (none for IF), one Detect with three inputs, the architecture-native output shape
(`(1, 6, 84)` for Detect, `(1, 84, 6)` for YOLOv10's end2end `v10Detect`), and nonzero gradients in all
three modality stems. YOLOv12 has no nape tail after P5, so its EF, NIF and BF files describe the same
graph and differ only in their `fusion`/`fusion_points` metadata.

Scale coverage: every upstream scale now has its own six templates — `cfg/mmodels/<arch><scale>-mm3-<mode>.yaml`
with `yolov8` n/s/m/l/x, `yolov9` t/s/m/c/e, `yolov10` n/s/m/b/l/x, `yolo11` n/s/m/l/x and `yolo12` n/s/m/l/x
(156 files). YOLOv8/11/12 keep the upstream `scales:` block and get their scale from the filename
(`yaml_model_load` → `guess_model_scale`), so the header `scale:` line is documentation only. YOLOv9 files are
fixed-width per-scale configs with no `scales:` block, and YOLOv10 ships one file per scale whose `scales:`
block declares only that scale — `yolov10b` therefore falls back to the single declared key with an
"assuming scale='b'" warning. `::test_every_template_scale_builds_and_grows_with_the_scale` builds IF and BF
at every scale for all five architectures and asserts the parameter count strictly increases along the scale
order. Real-data smoke runs remain at `s` scale only.

YOLOv9e is the only covered backbone with multi-input layers: its auxiliary branch splits features with
`CBLinear` and merges them with `CBFuse`, so the generator keeps those layer sources instead of flattening
them to `-1`. That gives each modality branch two input convolutions (5-channel for IF, 3/1-channel for the
other modes), which is why the gradient checks accept any positive multiple of three stems instead of exactly
three. `::test_every_template_scale_builds_and_grows_with_the_scale` also runs a train-mode backward pass at
every scale.

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
- two-modality layouts `[3, 1]` (4 channels: RGB + thermal or depth) and `[1, 1]` (2 channels: thermal + depth)
  build, infer and backpropagate in all six modes
  (`::test_yolo26x_mm2_templates_match_the_dataset_modality_split`); the same mm2 file serves RGB-T and RGB-D,
  because both are 3+1;
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

### Architecture sweep (R-LiViT_mdet, v8–v12)

Same protocol as above: 1 epoch, `fraction=0.1` (170 train images), 640 px, batch 4,
`pretrained=False`, RTX 5090, template scale `s`. All 30 runs trained, validated (216 val images) and
reloaded `best.pt` for prediction on four val images. Precision, recall and mAP are `0` for every run —
expected from one epoch on random initialisation, not a quality signal. Inference is ms/image from the
validation pass.

| Architecture | Params IF→HF (M) | IF | EF | NIF | BF | NF | HF |
|---|---|---|---|---|---|---|---|
| YOLOv8s | 11.1 → 31.2 | 0.66 | 1.28 | 1.27 | 1.13 | 1.53 | 1.52 |
| YOLOv9s | 7.3 → 19.5 | 1.69 | 2.80 | 3.11 | 2.96 | 4.27 | 4.32 |
| YOLOv10s | 8.1 → 23.0 | 0.82 | 1.57 | 1.68 | 1.93 | 2.22 | 2.37 |
| YOLO11s | 9.4 → 29.3 | 1.06 | 1.69 | 1.75 | 1.92 | 2.54 | 2.63 |
| YOLO12s | 9.3 → 28.8 | 1.44 | 2.98 | 3.08 | 3.14 | 3.95 | 4.01 |

Training losses stayed in the same narrow band across modes (box 5.1–5.6, dfl 4.1–4.6; YOLOv10's
end2end classification loss is on a different scale, ~17–20). YOLOv8s and YOLOv9s NIF/BF have identical
parameter counts because their nape tail is a single layer; YOLO12s EF/NIF/BF are the same graph because
it has no nape tail at all. At `conf=0.001` the Detect head returns 35–52 boxes, while YOLOv10's end2end
`v10Detect` returns the 300-box `max_det` cap; all confidences are ≈0.0016–0.0017.

## Not yet covered

- Two-modality `data.yaml` subsets (B3–B5) and real-data train/predict runs for the `mm2` templates.
- Pretrained COCO transfer for YOLO26 and the v9/v10/v11/v12 families (`pretrained.py` accepts only
  `source: yolov8`, so those templates declare `pretrained: false`).
- Real-data train/predict for scales other than `s` in v8–v12 (build, forward, backward and parameter growth
  are covered for every scale).
- Real-data train/predict for the x-scale YOLOv8x templates and the YOLO11 segmentation template.
- Formal (full-epoch, multi-seed) training on R-LiViT / MID-3K and the B0–B8 baseline matrix.
