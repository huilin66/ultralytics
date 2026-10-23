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
| R-LiViT | `uint8` BGR 3ch | already aligned 1ch | not converted (LiDAR needs extrinsics) |
| MM5 | `RGB8` 3ch | `T8` 1ch | `D16` uint16 mm, 0–2 m |
| VDT-2048 | `V` 3ch | palette-rendered, kept as 3ch | `uint16` mm, 0–3 m |

No per-frame min/max normalisation is applied anywhere: it would destroy
cross-frame radiometric comparability.

## Commands

```bash
# 1. extract the zipped downloads into <raw-root>/_extracted
python tools/extract_raw.py --dataset mid3k
python tools/extract_raw.py --dataset infraparis --only BBox Depth Infra   # skips the 60 GB Right archive
python tools/extract_raw.py --dataset r_livit --with-lidar

# 2. inspect the plan without writing anything
python tools/convert_dataset.py --dataset mid3k --dry-run

# 3. convert (writes Y:/dataset/<name>_mdet)
python tools/convert_dataset.py --dataset mid3k
python tools/convert_dataset.py --dataset r_livit --limit 200 --out-root 'Y:\dataset\_smoke' --overwrite

# 4. verify the result with the real Ultralytics loader
python tools/validate_mdet.py --data 'Y:/dataset/MID-3K_mdet/data.yaml'
```

Useful flags: `--splits train,val`, `--ratios 0.8,0.1,0.1`, `--seed`, `--previews 0`,
`--set key=value` (adapter options such as `group_bucket_minutes` or `split_suffix`).

## Adapters

| key | source layout | annotations | splits |
|---|---|---|---|
| `mid3k` | `MID-3K-{rgb,thermal,depth}-main/{images,labels}` | official YOLO `person` boxes | none → grouped by 5-min collection bucket |
| `infraparis` | `Left/`, `Depth/`, `BBox/`, `Infra/` | Label Studio JSON rects, normalised | official train/val/test scenes |
| `r_livit` | `R-LiViT_RGB-T/{rgb,thermal,annotations}` | Pascal VOC `person/car/...` | official train/test sequences, val carved from train groups |
| `mm5` | `MM5_ALIGNED/{RGB8,T8,D16,ANNO_INST,ANNO_CLASS}` | instance masks + class masks → boxes | `list_{train,eval}_f.txt` (all 324 samples) |
| `vdt2048` | `VDT-2048 dataset/{Train,Test}/{V,T,D,GT}` | binary saliency mask → **candidate** boxes | official Train/Test |

`vdt2048` output is `annotation_status=candidate_only`: a salient region is not a
class label and touching objects merge into one component, so those boxes must not
enter a multi-class mAP table. Its thermal modality is a 256-colour palette
rendering, not a radiometric channel, so it stays 3-channel as shipped.

## Known blockers (do not paper over)

* **InfraParis thermal** — the infrared sensor is 288×382 raw against 1920×1080
  RGB and the release ships no RGB↔IR homography. The modality is omitted rather
  than resized into a fake alignment.
* **R-LiViT depth** — the LiDAR release needs the shipped extrinsics plus
  timestamp matching; the adapter converts RGB-T only.
* **VDT-2048 classes** — the dataset has 34 object kinds but no per-instance,
  per-class detection ground truth.

## Tests

```bash
python -m pytest tools/tests -q
```
