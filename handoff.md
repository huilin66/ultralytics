# HMT Leakage-Only Loss Handoff

## Current Branch

- Branch: `det_loss_mask`
- Baseline: `main_demo`
- Latest commit: `e0fa581f Fix leakage mask validation initialization`
- Scope: Ultralytics 8.4.91 three-class YOLO Detect training
- Classes: `0 Hollow Confirmed`, `1 Hollow Suspected`, `2 Leakage`

## Implemented Changes

| File | Changes |
| --- | --- |
| `ultralytics/utils/loss.py` | Adds `LEAKAGE_ONLY_LIST` loading during training loss initialization, accepts full paths by matching `Path(im_file).name`, validates missing files, empty lists, duplicate basenames, and non-class-2 labels, and logs the matched image count and masked IDs `[0, 1]`. |
| `ultralytics/utils/loss.py` | Keeps BCE unreduced, applies a broadcastable `batch_size x 1 x num_classes` mask by multiplication, and leaves class weights, Box loss, DFL loss, and TaskAlignedAssigner unchanged. |
| `ultralytics/utils/loss.py` | Does not load or validate the leakage-only list for evaluation models, so validation keeps the normal three-class loss and metrics. |
| `ultralytics/models/yolo/detect/train.py` | Rejects non-zero `mosaic`, `mixup`, `cutmix`, or `copy_paste` when leakage-only masking is enabled, and attaches complete training image/class metadata before criterion initialization. |
| `demo_base.py` | Adds `load_as_model` for direct checkpoint initialization and forces all image-mixing augmentations to zero when `LEAKAGE_ONLY_LIST` is configured. |
| `demo_det_hmt.py` | Sets the leakage-only list path in Python, supports v2 and official `yolov8x.pt` weight sources, and provides default run names. |
| `tests/test_leakage_only_loss.py` | Adds tests for normal and leakage-only classification gradients, Box/DFL gradients, mixed batches, full POSIX and Windows-style paths, list validation, single-read behavior, augmentation checks, and validation initialization. |

## Configuration

The list file is configured by `demo_det_hmt.py` or by the environment:

```python
os.environ["LEAKAGE_ONLY_LIST"] = "/absolute/path/leakage_only.txt"
```

Each non-comment line must identify a training image filename. Empty lines and lines beginning with `#` are ignored. Full paths copied from a training image list are accepted; only the basename is used for matching.

Leakage-only images must contain only class `2` labels. Images not in the list retain all three classification channels. Box and DFL losses are computed normally for every image.

## Weight Sources

`demo_det_hmt.py` defaults to the v2 checkpoint configured in `DEFAULT_V2_WEIGHTS`.

To use the official `yolov8x.pt` source instead, set:

```powershell
$env:HMT_WEIGHT_SOURCE = "v8"
python demo_det_hmt.py
```

The script also contains Python defaults, so it can run without setting environment variables. `HMT_V2_WEIGHTS`, `HMT_V8_WEIGHTS`, and `HMT_RUN_NAME` can override the defaults when needed.

## Training Requirements

Leakage-only masking depends on one `im_file` per original image. The following must all be zero:

```text
mosaic=0.0
mixup=0.0
cutmix=0.0
copy_paste=0.0
```

If any is non-zero, training raises an error. No pixel masking, label modification, class-count change, or model-structure change is performed.

## Validation and Tests

- Validation does not apply the leakage-only classification mask.
- Test command:

```powershell
C:\ProgramData\anaconda3\python.exe -m pytest tests/test_leakage_only_loss.py -q
```

- Result at handoff: `11 passed`
- `compileall` and `git diff --check` passed.
- Full training was not started during implementation.

## Typical Start

Run with the Python defaults:

```powershell
python demo_det_hmt.py
```

The default path starts from the configured v2 `best.pt`. To run the official-weight experiment:

```powershell
$env:HMT_WEIGHT_SOURCE = "v8"
python demo_det_hmt.py
```

Before a real run, verify that the configured `DEFAULT_V2_WEIGHTS`, dataset YAML, and `LEAKAGE_ONLY_LIST` paths exist on the machine running the training.
