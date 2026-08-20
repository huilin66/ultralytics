# HMT Leakage-Only Loss Handoff

## Current Branch

- Branch: `det_loss_mask`
- Baseline: `main_demo`
- Feature baseline: `e0fa581f Fix leakage mask validation initialization`
- Scope: Ultralytics 8.4.91 three-class YOLO Detect training
- Classes: `0 Hollow Confirmed`, `1 Hollow Suspected`, `2 Leakage`

## Implemented Changes

| File | Changes |
| --- | --- |
| `ultralytics/utils/loss.py` | Adds `LEAKAGE_ONLY_LIST` loading during training loss initialization, accepts full paths by matching `Path(im_file).name`, validates missing files, empty lists, duplicate basenames, and non-class-2 labels, and logs the matched image count and masked IDs `[0, 1]`. |
| `ultralytics/utils/loss.py` | Keeps BCE unreduced, applies a broadcastable `batch_size x 1 x num_classes` mask by multiplication, and leaves class weights, Box loss, DFL loss, and TaskAlignedAssigner unchanged. |
| `ultralytics/utils/loss.py` | Does not load or validate the leakage-only list for evaluation models, so validation keeps the normal three-class loss and metrics. |
| `ultralytics/utils/leakage.py` | Provides one cached reader for the normalized leakage-only filename set, shared by the training dataset and loss criterion. |
| `ultralytics/data/base.py` | Caches leakage-only filenames on training datasets and exposes eligible indices for image-mixing sources. |
| `ultralytics/data/augment.py` | Skips Mosaic, MixUp, CutMix, and CopyPaste for listed primary images and excludes listed images from all multi-image augmentation sources. |
| `ultralytics/models/yolo/detect/train.py` | Keeps the configured augmentation probabilities and attaches complete training image/class metadata before criterion initialization. |
| `demo_base.py` | Adds `load_as_model` for direct checkpoint initialization and leaves caller-provided image-mixing probabilities unchanged. |
| `demo_det_hmt.py` | Sets the leakage-only list path in Python and currently invokes one official-weight run and one v2-weight run sequentially. |
| `tests/test_leakage_only_loss.py` | Adds tests for normal and leakage-only classification gradients, Box/DFL gradients, mixed batches, full POSIX and Windows-style paths, list validation, single-read behavior, per-image augmentation exclusion, and validation initialization. |

## Configuration

The list file is configured by `demo_det_hmt.py` or by the environment:

```python
os.environ["LEAKAGE_ONLY_LIST"] = "/absolute/path/leakage_only.txt"
```

Each non-comment line must identify a training image filename. Empty lines and lines beginning with `#` are ignored. Full paths copied from a training image list are accepted; only the basename is used for matching.

Leakage-only images must contain only class `2` labels. Images not in the list retain all three classification channels. Box and DFL losses are computed normally for every image.

## Weight Sources

The current working-tree version of `demo_det_hmt.py` does not read `HMT_WEIGHT_SOURCE`, `HMT_V2_WEIGHTS`, `HMT_V8_WEIGHTS`, or `HMT_RUN_NAME`. It hard-codes the leakage list path and runs two experiments sequentially:

1. Official `yolov8x.pt`: `yolov8x.yaml` plus the default `load_as_model=False` path. This builds the YAML model and transfers the official weights.
2. The v2 `best.pt` at `DEFAULT_V2_WEIGHTS`: this must use `load_as_model=True` so the existing three-class v2 classification head is preserved when the checkpoint is loaded directly.

The second call currently omits `load_as_model=True`; add it before running the v2 experiment. Give the two calls distinct `name` values so their output directories and logs cannot be confused.

`load_as_model` changes only the checkpoint initialization path. It does not change the dataset class count, leakage mask, augmentation policy, or model architecture. For official COCO `yolov8x.pt`, `load_as_model=False` is the intended path; its 80-class classification head is not directly transferred to the three-class HMT head.

## Training Requirements

Leakage-only masking depends on one `im_file` per original image. Normal images retain the configured augmentation probabilities. For every image in `LEAKAGE_ONLY_LIST`:

- `Mosaic`, `MixUp`, `CutMix`, and `CopyPaste` are skipped when the listed image is the primary sample.
- Listed images are excluded from the candidate sources used by those multi-image augmentations for normal samples.
- Other configured augmentations, such as affine and color transforms, remain active.

No pixel masking, label modification, class-count change, or model-structure change is performed. The original `close_mosaic` schedule remains active.

## Validation and Tests

- Validation does not apply the leakage-only classification mask.
- Test command:

```powershell
C:\ProgramData\anaconda3\python.exe -m pytest tests/test_leakage_only_loss.py -q
```

- Result at handoff: `12 passed`
- `compileall` and `git diff --check` passed.
- The shared list utility uses postponed annotations for Python 3.8 import compatibility.
- Full training was not started during implementation.

## Typical Start

Run the current two-experiment script without environment variables after applying the v2 loading fix above:

```powershell
python demo_det_hmt.py
```

The first call uses official `yolov8x.pt`; the second uses `DEFAULT_V2_WEIGHTS`. Before a real run, verify that the configured v2 checkpoint, dataset YAML, and `LEAKAGE_ONLY_LIST` paths exist on the machine running the training.
