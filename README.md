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

## Running experiments

The training and evaluation entry points are provided through
[`run_experiment.sh`](run_experiment.sh). Use the launcher to inspect the
available experiment codes and run the corresponding training or validation
pipeline:

~~~bash
bash run_experiment.sh help
bash run_experiment.sh e2.0       # training
bash run_experiment.sh e2.0.eval  # evaluation
~~~

The launcher prints the underlying commands and supports common settings such
as `DEVICE`, `BATCH`, `WORKERS`, `IMGSZ`, and `SEEDS` through environment
variables.

## Other tools

The project uses the following companion tools for dataset preparation and
annotation:

- [`yolo_data_manager`](https://github.com/huilin66/yolo_data_manager) provides
  dataset organization, format conversion, and related data-processing
  utilities.
- [`multianno`](https://github.com/huilin66/multianno) provides image
  annotation and review tools for object categories and multiple attributes.

These tools are maintained in separate repositories.

## Citation and license

This repository is the implementation associated with the MAYOLO signboard
inspection study. Citation information will be added when the manuscript is
publicly released.

The project follows the licensing terms of the upstream
[Ultralytics repository](https://github.com/ultralytics/ultralytics). Please
read the repository license before redistributing code, checkpoints, or
derived datasets.
