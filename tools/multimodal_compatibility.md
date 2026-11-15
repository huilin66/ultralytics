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
`multimodal.operator: concat`, `multimodal.share_weight: false` and `multimodal.pretrained: {source: auto}`.
Every template therefore keeps only what makes it different:

- the `mm2` files override `channels`/`input_sections` (`[1, 1]` → 2, `[3, 1]` → 4, `[3, 3]` → 6);
- the `mm1` file declares `channels: 1`, `input_sections: [1]` and must use `fusion: IF` — a
  single section has no fusion node (`parse_fusion_spec` rejects any other mode);
- every template inherits `pretrained: {source: auto}` and no longer spells out `copies`;
  `load_coco_pretrained` derives the copy plan from the donor and target graphs via
  `derive_copy_plan` (the mapping is not written into the YAML);
- a template can pin its donor family (`source: yolo11`) to reject a mismatched checkpoint, or set
  `pretrained: false` to disable the transfer.

All 183 templates accept a COCO checkpoint of their own architecture family and scale — `yolov8`,
`yolov9`, `yolov10`, `yolo11`, `yolo12` and `yolo26`. The plan is derived from layer declarations, so
the donor must use the same modules (a yolov8 `C2f` donor cannot map onto a yolo11 `C3k2` target); the
per-pair guard rejects a checkpoint whose widths do not match. `source: auto` therefore means "use the
checkpoint I give you if its graph maps, otherwise fail loudly".

`tests/test_multimodal.py::test_every_template_inherits_the_shared_contract` checks the config contract
for all 183 files (including that re-merging an already-merged config is idempotent), and
`::test_every_template_loads_its_family_coco_checkpoint` builds a donor for every family/scale pair
(27 combinations) and transfers it into every template, asserting that no target tensor is left
untouched.

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
| YOLO26x (det, 2 modalities) | `cfg/mmodels/yolo26x-mm2-{31,11}-{if,ef,nif,bf,nf,hf}.yaml` | ✅ | ✅ | ✅ | ✅ | — | build/forward/backward tested 2026-11-10; mm2 BF real-data train/val in the E2 3-epoch × 3-seed test 2026-11-14 |
| YOLO26x (det, 1 modality) | `cfg/mmodels/yolo26x-mm1-if.yaml` | ✅ | ✅ | ✅ | ✅ | ✅ | thermal + depth 1-epoch smoke 2026-11-13 |
| YOLO26x (det, learned fusion) | `cfg/mmodels/yolo26x-mm3-bf-{gated,conv}.yaml` | ✅ | ✅ | ✅ | ✅ | ✅ | BF-gated/BF-conv 1-epoch smoke 2026-11-13 |
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
  real source path (`tests/test_multimodal.py::test_multimodal_predictor_keeps_primary_paths_for_a_list_source`);
- one-modality `[1]` (1 channel: thermal-only or depth-only) builds, infers and backpropagates, and
  `derive_copy_plan` treats it as a non-RGB branch so COCO transfer averages the donor's RGB input
  channels into the 1-channel stem (`::test_mm1_template_builds_and_backpropagates` and
  `::test_mm1_coco_transfer_averages_the_first_convolution`; report `copied=1091, transformed=1,
  skipped=0` on `yolo26x.pt`);
- R-LiViT real-data 1-epoch smoke runs (thermal and depth, `fraction=0.1`, 640 px, batch 4, AdamW,
  seed 0) train, validate, save `best.pt` and reload it for a fused validation pass;
- the learned-fusion pair `yolo26x-mm3-bf-gated.yaml` (B8, per-channel `2*sigmoid(gate(x))`) and
  `yolo26x-mm3-bf-conv.yaml` (C1, plain residual control) share one `MultiModalFusion` implementation
  and differ only in the second 1×1 Conv per modality: both use identity projections plus a zero-init
  Conv, so each starts exactly at BF and has the same parameter count (`nc=2`, `scale: x`: BF
  121,817,412, B8 = C1 132,448,068, i.e. +10,630,656 each).
  `::test_gated_and_conv_templates_start_at_bf_with_matched_capacity` asserts bit-identical outputs
  after loading the BF state dict, and `::test_yolo26x_learned_fusion_templates_build_and_backpropagate`
  asserts nonzero gradients in every gate/residual Conv and every modality stem;
- R-LiViT B8/C1 1-epoch smoke runs (`fraction=0.1`, batch 4, AdamW, seed 0) train, validate, save,
  reload and predict on RGB-T-D companions; the gate and residual Convs leave their zero
  initialization as soon as the optimizer steps (checkpoint inspection). A `fraction=0.02` smoke never
  steps the optimizer because `accumulate=16` exceeds the 9 iterations in the epoch, so its untouched
  checkpoint is not evidence of frozen fusion.

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

## Efficiency (2026-11-15)

Measured on the 12 trained R-LiViT checkpoints: RTX 3090 24 GiB, `model.fuse()`, fp16, batch=1,
640×640. Each model gets 20 warm-up passes and then 3 rounds × 100 iterations timed with
`torch.cuda.Event`, keeping the median. Models are measured interleaved — a single sequential pass drifts
by up to 27% because the GPU clocks ramp up during the first model (B0 measured 16.5 ms first, 13.0 ms
last), while interleaved rounds stay within 3%. Latency covers the network forward only; decoding,
resizing and normalising the RGB/T/D files is excluded. GFLOPs comes from `thop.profile` on an fp32 input
with the correct stacked channel count (see Known issues for why `get_flops()` cannot do this).

| Model | Branches / fusion | Input channels | Params fused (M) | GFLOPs | Latency (ms) | FPS |
|---|---|---|---|---|---|---|
| B0 RGB (official YOLO26x) | 1 | 3 | 55.6 | 96.7 | 13.02 | 76.8 |
| B1 thermal (`mm1-if`) | 1 | 1 | 55.6 | 96.5 | 13.05 | 76.7 |
| B2 depth (`mm1-if`) | 1 | 1 | 55.6 | 96.5 | 13.08 | 76.4 |
| B3 RGB-T (`mm2-31-bf`) | 2, BF | 4 | 90.7 | 162.9 | 20.64 | 48.4 |
| B4 RGB-D (`mm2-31-bf`) | 2, BF | 4 | 90.7 | 162.9 | 20.89 | 47.9 |
| B5 T-D (`mm2-11-bf`) | 2, BF | 2 | 90.7 | 162.7 | 20.80 | 48.1 |
| A0 IF (`mm3-if`) | 1 (input stack) | 5 | 55.6 | 96.9 | 13.00 | 76.9 |
| A1 EF (`mm3-ef`) | 3, encoder | 5 | 109.1 | 218.2 | 23.40 | 42.7 |
| A2 NIF (`mm3-nif`) | 3, encoder+nape | 5 | 112.1 | 219.3 | 23.89 | 41.9 |
| A3/B7 BF (`mm3-bf`) | 3, backbone | 5 | 118.6 | 221.9 | 26.55 | 37.7 |
| A4 NF (`mm3-nf`) | 3, neck | 5 | 164.7 | 282.1 | 36.39 | 27.5 |
| A5 HF (`mm3-hf`) | 3, head | 5 | 168.6 | 288.5 | 36.96 | 27.1 |

Latency tracks GFLOPs linearly. Compared with a single branch, two-branch BF costs +38% latency
(76.8 → 48 FPS) and three-branch BF costs +29% on top of two branches (48 → 37.7 FPS). NF and HF spend
27–30% more FLOPs than BF and score lower on the frozen test split (`runs/e2_rlivit/test_eval/`), so
neither is defensible on efficiency grounds. Raw rows: `runs/e2_rlivit/flops_fps.json`.

## Known issues

- `get_flops()` returns 0.0 for every multimodal model. It sizes the dummy input from the channel count of
  the first parameter's convolution, but a multimodal graph starts with `ModalSplit`, which expects the
  stacked channel count (5 for `mm3`); the 3-channel probe raises `ValueError` inside the `try` and the
  zero is returned. Consequence: `yolo val` summaries never print GFLOPs for these models. The efficiency
  table above therefore calls `thop.profile` directly. The fix belongs in `get_flops`, which should take
  the model's declared input channels rather than the first convolution's.
- Checkpoints saved before 2026-11-13 15:30 UTC (`e406b4a26`) cannot forward on the current code.
  That commit added `projections` / `residuals` / `gates` to `MultiModalFusion` and `forward` references
  them unconditionally; instances pickled earlier carry no such attributes, so loading one raises
  `AttributeError: 'MultiModalFusion' object has no attribute 'projections'`. Every E1.5 checkpoint
  (A0–A5) is affected, including the ones whose test scores are published; E2 checkpoints are not.
  Running them requires restoring the three attributes to `None` (the correct value for the `concat`
  operator) — the efficiency rows for A1–A5 were measured that way. The fix belongs in the class's own
  deserialisation (restore defaults for older state), not as a `getattr` fallback inside `forward`.

## Not yet covered

- Two-modality `data.yaml` subsets (B3–B5): train/val are covered by the E2 3-epoch × 3-seed test
  (2026-11-14, `runs/e2_rlivit_3ep/`) and by the 200-epoch × 3-seed runs plus reload-and-evaluate on the
  frozen test split (`runs/e2_rlivit/{b3-rgb-t,b4-rgb-d,b5-t-d}-seed*/`, `runs/e2_rlivit/test_eval/`);
  reload-and-predict on images from the `mm2` checkpoints is still pending.
- Real-data fine-tuning from COCO weights for v9/v10/v11/v12/YOLO26 (the transfer itself is covered by
  `::test_every_template_loads_its_family_coco_checkpoint`; only the training run is missing).
- Real-data train/predict for scales other than `s` in v8–v12 (build, forward, backward and parameter growth
  are covered for every scale).
- Real-data train/predict for the x-scale YOLOv8x templates and the YOLO11 segmentation template.
- Formal (full-epoch, multi-seed) training on R-LiViT is done for the E2 B0–B7 matrix (200 epoch × 3 seeds,
  2026-11-14); the MID-3K matrix is still pending. B8/C1 learned-fusion templates remain reserved for the F3
  method ablation.
