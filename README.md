# MAYOLO

## A Hierarchical Multi-Attribute Detection Network Integrating Global Information and Graph Category Attention for Signboard Inspection

This repository contains the research implementation of MAYOLO, a
multi-attribute object-detection framework for signboard inspection. MAYOLO
extends the Ultralytics detection framework with a multi-attribute detection
head and introduces:

- **Global Information Aggregation (GIA)** for incorporating global visual
  context into the feature hierarchy;
- **Graph Category Attention (GCA)** for propagating information between
  correlated attribute categories using a training-set co-occurrence matrix;
- **YOLOv10-style one-to-one and one-to-many detection branches** with
  independently parameterized attribute heads;
- a two-stage training protocol in which Stage 1 trains the full model and
  Stage 2 freezes the detection part and fine-tunes the attribute head.

The repository is based on and extends
[Ultralytics](https://github.com/ultralytics/ultralytics). It is intended for
research reproducibility rather than as a drop-in replacement for the upstream
Ultralytics package.

## Method overview

The default MAYOLO signboard task contains two object classes and ten binary
attributes. The reported MAYOLO configuration combines YOLOv10 with
GIA-v2.5.7, Cross-GIN Graph Category Attention, and the one-to-many inference
strategy.
The available configurations also include different GIA placements, graph
operators, the Cross co-occurrence matrix, model scales, RT-DETR baselines, a true
multi-label detector, and a two-stage detector/classifier pipeline.

For graph experiments, the co-occurrence matrix is constructed from the
training split only. The reproducible protocol uses the Cross matrix and the
following graph operators:

GCN, GAT, GraphSAGE, and GIN.

The exact experiment definitions and reported results are documented in:

- [scripts/EXPERIMENTS.md](scripts/EXPERIMENTS.md)
- [scripts/EXPERIMENTS_results.md](scripts/EXPERIMENTS_results.md)
- [run_experiment.sh](run_experiment.sh)

## Repository structure

~~~text
ultralytics/
├── ultralytics/                         # Ultralytics core and MAYOLO extensions
│   ├── nn/modules/block.py              # GIA and feature modules
│   ├── nn/modules/head.py               # multi-attribute head and graph operators
│   ├── models/yolo/mdetect/             # multi-attribute detection task
│   └── cfg/models/                      # model YAML configurations
├── scripts/
│   ├── train_mdet_experiments.py        # two-stage mdet experiments
│   ├── eval_mdet_experiments.py         # Test evaluation and HO inference
│   ├── train_mayolo_final_sizes.py      # MAYOLO size experiments
│   ├── benchmark_performance.py         # parameters, FLOPs, memory and latency
│   ├── convert_mdet_to_multilabel.py    # mdet to multi-label detection data
│   ├── prepare_two_stage_crops.py       # crop data for the E6 pipeline
│   ├── eval_e5_e6.py                    # E5/E6 Test evaluation
│   ├── generate_robustness_variants.py  # deterministic Test perturbations
│   ├── eval_robustness.py               # robustness evaluation
│   └── eval_attribute_*.py              # attribute-level diagnostics
├── run_experiment.sh                    # independent reproducibility launcher
├── ultralytics/cfg/mayolo_r1/           # dataset configuration
└── runs/experiments/                    # generated checkpoints and summaries
~~~

## Installation

Use a Python environment with a CUDA-enabled PyTorch installation compatible
with the target GPU. From the repository root:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
~~~

For Conda or a system-specific PyTorch installation, install PyTorch first and
then run python -m pip install -e . The repository was developed and tested
with the Ultralytics research environment and NVIDIA GPUs.

## Dataset and annotation format

The default dataset configuration is
[ultralytics/cfg/mayolo_r1/mayolo_v3.yaml](ultralytics/cfg/mayolo_r1/mayolo_v3.yaml).
Before training, replace its machine-specific path, train, val, and test
entries with paths valid on the current machine.

The dataset YAML contains object classes and attribute levels:

~~~yaml
path: /path/to/mayolo_v3
train: train.txt
val: val.txt
test: test.txt

names:
  0: projecting_signboard
  1: wall_signboard

attributes:
  surface_missing: [no, yes]
  surface_incomplete: [no, yes]
  surface_corroded: [no, yes]
  frame_corroded: [no, yes]
  surface_peeling: [no, yes]
  surface_fade: [no, yes]
  surface_deformed: [no, yes]
  frame_deformed: [no, yes]
  disconnected: [no, yes]
  added_billboard: [no, yes]
~~~

Each object label uses the repository's multi-attribute detection format:

~~~text
class attribute_count attr_0 ... attr_N-1 x_center y_center width height
~~~

All coordinates are normalized to [0, 1]. The shared parser and conversion
utilities are implemented in
[scripts/attribute_dataset_utils.py](scripts/attribute_dataset_utils.py).

### Annotation and data-management tools

Annotations can be created and reviewed with
[multianno](https://github.com/huilin66/multianno).
Dataset organization, format conversion, and YOLO-style data management can be
performed with
[yolo_data_manager](https://github.com/huilin66/yolo_data_manager).

These tools are maintained separately from this repository.

## Training

The recommended entry point is the independent reproducibility launcher:

~~~bash
# Inspect all available experiment codes.
bash run_experiment.sh help

# Check paths and required inputs without starting training.
bash run_experiment.sh preflight --dry-run

# Baseline/GIA five-seed Stage 1 + Stage 2 training.
PYTHON_BIN=/path/to/python bash run_experiment.sh e2.0

# Pure Baseline + GCA, using the Cross matrix and the four graph operators.
PYTHON_BIN=/path/to/python bash run_experiment.sh e2.2

# GIA + GCA Stage 2.
PYTHON_BIN=/path/to/python bash run_experiment.sh e2.5
~~~

The launcher prints every underlying Python command before execution. It does
not call run.sh. Common settings can be overridden through environment
variables:

~~~bash
DEVICE=0
BATCH=16
WORKERS=8
IMGSZ=640
STAGE1_EPOCHS=100
STAGE2_EPOCHS=100
SEEDS="0 1 2 3 4"
~~~

The standard protocol is:

1. **Stage 1:** train the complete detector and attribute head;
2. **Stage 2:** load the Stage 1 best.pt, freeze the detection part, and
   fine-tune the attribute head or the added GCA module.

The co-occurrence matrices are supplied explicitly, for example:

~~~bash
COM_CROSS=/path/to/co_occurrence_matrix_train.csv \
bash run_experiment.sh e2.2
~~~

## Evaluation

Detection quality is primarily reported with **Test mAP50**. Test mAP50-95
is reported as a supplementary localization metric. Attribute metrics are
computed only on one-to-one detection--ground-truth matches that satisfy both
of the following conditions:

1. the predicted object class is correct; and
2. the matched box has IoU at least 0.5.

The main attribute metric is **Macro-F1 at IoU 0.5**. Attribute-level
precision, recall, and overall accuracy are reported as auxiliary measures.
Because each attribute is a mutually exclusive softmax task, pooling the
attribute-level decisions makes overall accuracy equivalent to the
corresponding pooled Micro-F1.

Validation results are used during training to select best.pt. The reported
scientific comparisons use the held-out Test split. Summary files contain the
test metrics and are written below the corresponding directory in
runs/experiments/.

Typical evaluation commands are:

~~~bash
# Evaluate a trained multi-attribute detector.
python scripts/eval_mdet_experiments.py --help

# Inspect attribute-level and per-level Test metrics.
python scripts/eval_attribute_levels.py --help

# Measure model complexity, memory, latency, and FPS.
python scripts/benchmark_performance.py --help
~~~

## E5 and E6 comparison pipelines

The repository also contains two comparison baselines:

- **E5: true multi-label object detection.** Each object has one bounding
  box with its object class and all attribute labels. The converted task is
  evaluated as a multi-label detector.
- **E6: two-stage detection plus multi-label classification.** A detector
  first produces object boxes; cropped object regions are then classified
  using the multi-label classification model.

The data preparation and evaluation entry points are:

~~~bash
python scripts/convert_mdet_to_multilabel.py --help
python scripts/prepare_two_stage_crops.py --help
python scripts/train_multilabel.py --help
python scripts/train_two_stage.py --help
python scripts/eval_e5_e6.py --help
~~~

The corresponding reproducibility launcher commands are:

~~~bash
bash run_experiment.sh e5
bash run_experiment.sh e5.eval
bash run_experiment.sh e6
bash run_experiment.sh e6.eval
~~~

## Robustness and attribute diagnostics

Robustness variants are generated only for Test-time evaluation. They are
not added to the training set. The supported deterministic perturbation
families include illumination changes, image degradation, and viewpoint or
scale transformations. Geometric transformations update and clip the
corresponding ground-truth boxes.

~~~bash
bash run_experiment.sh e5.1
bash run_experiment.sh e6.1

# Attribute calibration and confusion-matrix diagnostics.
python scripts/eval_attribute_calibration.py --help
python scripts/plot_attribute_confusion_matrices.py --help
~~~

## Results and reproducibility

Experiment definitions are listed in
[scripts/EXPERIMENTS.md](scripts/EXPERIMENTS.md), while the reported Test
results and their summary paths are maintained in
[scripts/EXPERIMENTS_results.md](scripts/EXPERIMENTS_results.md).
Generated checkpoints, predictions, visualizations, and CSV summaries are
stored under runs/experiments/ and are intentionally not tracked as source
code.

For a reproducible run:

1. use the same dataset split files and label parser;
2. use the same model YAML and pretrained checkpoint;
3. keep the seed, image size, batch size, optimizer schedule, and matrix
   construction protocol fixed;
4. record the command printed by run_experiment.sh; and
5. compare the generated summary.csv and mean_std_test.csv files.

Dataset paths, checkpoints, and co-occurrence matrices are machine-specific
and are therefore not bundled in this repository. Replace the example paths
in the YAML and launcher before running the experiments.

## Citation and license

This repository is the implementation associated with the MAYOLO signboard
inspection study. Citation information will be added when the manuscript is
publicly released.

The project follows the licensing terms of the upstream
[Ultralytics repository](https://github.com/ultralytics/ultralytics). Please
read the repository license before redistributing code, checkpoints, or
derived datasets.
