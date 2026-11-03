# Dataset conversion (`tools/`)

Converts the five candidate multi-modal datasets into the layout consumed by
`ultralytics.data.multimodal` — one folder per dataset, named `<dataset>_mdet`.

Nothing here imports `ultralytics`; the only shared contract is the generated
`data.yaml`. Upstream source stays untouched.

## Why the layout looks like this

`ultralytics.data.utils.img2label_paths` derives the label path by replacing the
`images` path segment with `labels` and keeping everything after it, and
`Modalities.paths` derives every secondary modality by keeping the relative path
below the primary modality root. Both constraints force:

```text
<dataset>_mdet/
  images/rgb/<split>/<sample_id>.png        # primary (reference frame)
  images/thermal/<split>/<sample_id>.png
  images/depth/<split>/<sample_id>.png
  labels/<primary>/<split>/<sample_id>.txt  # mirrors images/<primary>/
  data.yaml
  metadata/
    samples.jsonl          # per-sample provenance, encoding and quality
    classes.json           # dataset-local class -> index
    split_manifest.csv     # sample_id, split, group, boxes, source
    conversion.json        # parameters, git sha, versions, seed
    exclusions.csv         # every skipped sample and why
    quality_report.json    # per-split counts, box stats, warnings
    derived/               # intermediate high-precision arrays, e.g. LiDAR depth
      depth_uint16/<split>/<sample_id>.png
      depth_projection_summary.json
  previews/                # RGB | thermal | depth panels with boxes drawn
```

The plan document describes `labels/<split>/`; the loader cannot express that
without a code change, so labels live under `labels/<primary>/<split>/`.

## Encoding contract

`Modalities.load` rejects anything that is not `uint8`. Every non-8-bit sensor is
therefore mapped **once, globally and explicitly**, and the mapping is recorded:

| Dataset | rgb | thermal | depth |
|---|---|---|---|
| MID-3K | `uint8` BGR 3ch | 3 identical channels → 1ch | `uint16` mm, clip 0–20 m, `64536` = no-return |
| InfraParis | Left RGBA → BGR | **omitted** (288×382 vs 1920×1080, no homography) | 8-bit PNG as shipped, unit unknown |
| R-LiViT | `uint8` BGR 3ch | already aligned 1ch | LiDAR → uint16 mm, 0 = no return; `uint8` affine 0–50 m, no interpolation |
| MM5 | `RGB8` 3ch | `T8` 1ch | `D16` uint16 mm, 0–2 m |
| VDT-2048 | `V` 3ch | palette-rendered, kept as 3ch | `uint16` mm, 0–3 m |

No per-frame min/max normalisation is applied anywhere: it would destroy
cross-frame radiometric comparability.

## Commands

```bash
# 1. extract the zipped downloads into <raw-root>/_extracted
python tools/extract_raw.py --dataset mid3k
python tools/extract_raw.py --dataset infraparis --only BBox Depth Infra   # skips the 60 GB Right archive
python tools/extract_raw.py --dataset r_livit --with-lidar   # RGB-T + LiDAR scans + calibration/mapping

# 2. inspect the plan without writing anything
python tools/convert_dataset.py --dataset mid3k --dry-run

# 3. convert (writes Y:/dataset/<name>_mdet)
python tools/convert_dataset.py --dataset mid3k
python tools/convert_dataset.py --dataset r_livit --limit 200 --out-root 'Y:\dataset\_smoke' --overwrite

# 4. verify the result with the real Ultralytics loader
python tools/validate_mdet.py --data 'Y:/dataset/MID-3K_mdet/data.yaml'

# 5. write cross-dataset review tables
python tools/summarize_datasets.py --root 'Y:/dataset' --out reports/data_statistics
```

Useful flags: `--splits train,val`, `--ratios 0.8,0.1,0.1`, `--seed`, `--previews 0`,
`--workers 4` (threaded encoding, deterministic output order), `--set key=value`
(adapter options such as `group_bucket_minutes` or `split_suffix`), `--overwrite`
(delete and rebuild), `--allow-existing` (write into a folder without clearing it),
`--no-sample-level-val` (refuse the sample-level val fallback).

`metadata/samples.jsonl` records, per sample and modality, the `dtype`, `shape`,
`units`, `scale`, `invalid_value`, `encoding`, `source_path`, `reference_camera`,
`alignment_status` and `aligned_to` fields required by the research plan, plus the
invalid/saturated pixel fractions observed during encoding.

`summarize_datasets.py` writes `dataset_summary.csv`, `class_distribution.csv`,
`modality_quality.csv`, `split_leakage_report.csv` and
`dataset_selection_report.md`. The leakage report flags any group id that appears
in more than one split; file-hash and perceptual-hash checks are still required
before the formal benchmark decision.

## Adapters

| key | source layout | annotations | splits |
|---|---|---|---|
| `mid3k` | `MID-3K-{rgb,thermal,depth}-main/{images,labels}` | official YOLO `person` boxes | none → grouped by 5-min collection bucket |
| `infraparis` | `Left/`, `Depth/`, `BBox/`, `Infra/` | Label Studio JSON rects, normalised | official train/val/test scenes |
| `r_livit` | `R-LiViT_RGB-T/{rgb,thermal,annotations}` + `R-LiViT_LiDAR/point_clouds` | Pascal VOC `person/car/...` | official train/test sequences, val carved from train groups |
| `mm5` | `MM5_ALIGNED/{RGB8,T8,D16,ANNO_INST,ANNO_CLASS}` | instance masks + class masks → boxes | `list_{train,eval}_f.txt` (all 324 samples) |
| `vdt2048` | `VDT-2048 dataset/{Train,Test}/{V,T,D,GT}` | binary saliency mask → **candidate** boxes | official Train/Test; val carved by sample-level hash (no scene metadata) |

`vdt2048` output is `annotation_status=candidate_only`: a salient region is not a
class label and touching objects merge into one component, so those boxes must not
enter a multi-class mAP table. Its thermal modality is a 256-colour palette
rendering, not a radiometric channel, so it stays 3-channel as shipped.

### R-LiViT depth (`r_livit`)

The release ships `calibration_params.json` (RGB `K`, distortion, `lidar_to_rgb`
4×4) and `rgb_to_lidar_sequence_mapping.csv` inside the full archive; with
`--with-lidar`, `extract_raw.py` pulls those two members out of
`R-LiViT/R-LiViT.zip` without unpacking the 16 GB tree. For every annotated RGB
frame the adapter:

1. matches the nearest LiDAR scan by timestamp (`RGB` XML timestamps are ns,
   LiDAR timestamps are s; default tolerance 100 ms);
2. reads the KITTI-style float32 `(x, y, z, intensity)` scan;
3. projects points with `T_lidar_to_rgb` and the pinhole `K`, keeping the nearest
   return per pixel (Z-buffer). Distortion is **not** applied because RGB/thermal
   are shipped undistorted;
4. writes `uint16` millimetres to `metadata/derived/depth_uint16/`, then the
   normal encoding step maps `[0, 50] m` to `uint8` with `0` = no return.

Depth is inherently sparse (a LiDAR scan covers ~1–2% of a 1280×720 frame). No
interpolation or hole filling is performed; `metadata/samples.jsonl` records
`coverage`, `points_in_image`, depth percentiles and the calibration fingerprint,
and `metadata/derived/depth_projection_summary.json` aggregates them. Use
`--set include_lidar_depth=false` to fall back to RGB-T only, and
`depth_min_m`/`depth_max_m`/`depth_encode_max_m`/`lidar_timestamp_tolerance_ms`
to override the defaults.

The depth cache is rendered serially even when `--workers > 1`: projecting
several large scans concurrently reproducibly segfaulted or hung the Python
process (cv2 5.0.0 + threads) after roughly 500 frames, while the same workload
is stable serially. `--workers` still parallelises the uint8 encoding step.

## Known blockers (do not paper over)

* **InfraParis thermal** — the infrared sensor is 288×382 raw against 1920×1080
  RGB and the release ships no RGB↔IR homography. Only reprojected IR *semantic
  masks* are published, not calibration; a homography fitted to those masks gives
  mean IoU ≈0.10–0.28 and parameters that vary by sequence, which is not a real
  alignment. The modality is omitted rather than resized into a fake alignment.
* **R-LiViT depth** — now converted from the optional LiDAR archive (see above);
  runs without it degrade to RGB-T and say so in the quality report.
* **VDT-2048 classes** — the dataset has 34 object kinds but no per-instance,
  per-class detection ground truth.

## Classification dataset from cropped images (`build_cls_dataset.py`)

`ydm_vis/crop` is already class-per-directory; Ultralytics classification expects
`<root>/{train,val,test}/<class>/` and is trained with `data=<root>` (no `data.yaml`):

```bash
python tools/build_cls_dataset.py \
  --src 'Y:/dataset/WHU_Infra2D_cube_yolo_shape_core/ydm_vis/crop' \
  --out 'Y:/dataset/WHU_Infra2D_cube_yolo_shape_core/ydm_vis/crop_split' \
  --split-dir 'Y:/dataset/WHU_Infra2D_cube_yolo_shape_core' --overwrite

yolo classify train data='Y:/dataset/WHU_Infra2D_cube_yolo_shape_core/ydm_vis/crop_split' model=yolo11n-cls.pt
```

Crop names are `<source_stem>_<instance>`, so splits are assigned per **source
image**: when `train.txt`/`val.txt` from the detection release are present they
are reused, and crops cut from one frame never straddle train/val. Without split
lists a deterministic hash split over source images is used. Class indices are
alphabetical (`circle, octagon, polygon, square, triangle`), independent of the
detection `class.txt` order. Every split must contain every class, because
Ultralytics derives the class list from `train/` and requires the same count in
val/test. `--mode symlink|hardlink` avoids duplicating the crop files where the
filesystem supports it. Provenance, per-split counts and warnings are written to
`classification_metadata.json` in the output root.

## Tests

```bash
python -m pytest tools/tests -q
```
