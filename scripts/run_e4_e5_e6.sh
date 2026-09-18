#!/usr/bin/env bash
set -euo pipefail

# Final E4/E5/E6 launch script.
# Run from the repository root or let the script resolve its own root:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_e4_e5_e6.sh
#
# DEVICE is the logical CUDA index inside CUDA_VISIBLE_DEVICES.  The defaults
# below match the remote yolov8 environment and the generated remote datasets,
# but every input can be overridden from the shell.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN=${PYTHON_BIN:-/home/23039356r/.conda/envs/yolov8/bin/python}
DEVICE=${DEVICE:-0}
BATCH=${BATCH:-16}
WORKERS=${WORKERS:-8}
SEED=${SEED:-0}

RUN_E4=${RUN_E4:-1}
RUN_E5=${RUN_E5:-1}
RUN_E6=${RUN_E6:-1}

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [ "$RUN_E4" = "1" ]; then
  E4_DATA=${E4_DATA:-ultralytics/cfg/mayolo_r1/mayolo_v3.yaml}
  E4_PROJECT=${E4_PROJECT:-runs/experiments/E4_rtdetr_LX}
  E4_L_CONFIG=${E4_L_CONFIG:-ultralytics/cfg/models/rt-detr/rtdetr-l-md.yaml}
  E4_X_CONFIG=${E4_X_CONFIG:-ultralytics/cfg/models/rt-detr/rtdetr-x.yaml}
  E4_L_WEIGHT=${E4_L_WEIGHT:-rtdetr-l.pt}
  E4_X_WEIGHT=${E4_X_WEIGHT:-rtdetr-x.pt}

  for REQUIRED_FILE in "$E4_DATA" "$E4_L_CONFIG" "$E4_X_CONFIG" "$E4_L_WEIGHT" "$E4_X_WEIGHT"; do
    if [ ! -f "$REQUIRED_FILE" ]; then
      echo "Missing E4 input: $REQUIRED_FILE" >&2
      exit 1
    fi
  done

  "$PYTHON_BIN" scripts/train_mdet_experiments.py rtdetr \
    --label E4_rtdetr_LX \
    --data "$E4_DATA" \
    --variant rtdetr_l="$E4_L_CONFIG" \
    --variant rtdetr_x="$E4_X_CONFIG" \
    --pretrain-map rtdetr_l="$E4_L_WEIGHT" \
    --pretrain-map rtdetr_x="$E4_X_WEIGHT" \
    --stage1-epochs 100 \
    --stage2-epochs 100 \
    --imgsz 640 \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --w4 0.5 \
    --skip-existing \
    --project "$E4_PROJECT"
fi

if [ "$RUN_E5" = "1" ]; then
  E5_DATA=${E5_DATA:-/localnvme/data/billboard/mayolo_v3_multilabel/data.yaml}
  E5_MODEL=${E5_MODEL:-yolov10x.pt}
  E5_PROJECT=${E5_PROJECT:-runs/experiments/E5_multilabel}

  for REQUIRED_FILE in "$E5_DATA" "$E5_MODEL"; do
    if [ ! -f "$REQUIRED_FILE" ]; then
      echo "Missing E5 input: $REQUIRED_FILE" >&2
      exit 1
    fi
  done

  "$PYTHON_BIN" scripts/train_multilabel.py \
    --model "$E5_MODEL" \
    --data "$E5_DATA" \
    --epochs 100 \
    --imgsz 640 \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --project "$E5_PROJECT" \
    --name yolov10x \
    --seed "$SEED"
fi

if [ "$RUN_E6" = "1" ]; then
  # The detector checkpoint is the completed E3 YOLOv10x 100+100 run.  E6
  # trains only the crop classifier for 100 epochs on the generated crops.
  E6_DETECTOR_CHECKPOINT=${E6_DETECTOR_CHECKPOINT:-runs/experiments/E3_versions/E3_versions_yolov10x_w4_0p5_seed_0_stage2/weights/best.pt}
  E6_DATA=${E6_DATA:-/localnvme/data/billboard/mayolo_v3_two_stage_crops/data.yaml}
  E6_MODEL=${E6_MODEL:-ultralytics/cfg/models/v10/yolov10x-cls.yaml}
  E6_PROJECT=${E6_PROJECT:-runs/experiments/E6_two_stage_yolov10x}

  for REQUIRED_FILE in "$E6_DETECTOR_CHECKPOINT" "$E6_DATA" "$E6_MODEL"; do
    if [ ! -f "$REQUIRED_FILE" ]; then
      echo "Missing E6 input: $REQUIRED_FILE" >&2
      exit 1
    fi
  done

  # There is no official yolov10x-cls.pt.  Transfer the compatible backbone
  # parameters from the official YOLOv10x detector checkpoint and initialize
  # only the new Classify head.
  E6_PRETRAIN=${E6_PRETRAIN:-yolov10x.pt}
  if [ ! -f "$E6_PRETRAIN" ]; then
    echo "Missing E6 classifier pretrain: $E6_PRETRAIN" >&2
    exit 1
  fi

  "$PYTHON_BIN" scripts/train_two_stage.py \
    --detector-checkpoint "$E6_DETECTOR_CHECKPOINT" \
    --model "$E6_MODEL" \
    --pretrain "$E6_PRETRAIN" \
    --data "$E6_DATA" \
    --epochs 100 \
    --imgsz 224 \
    --batch "$BATCH" \
    --workers "$WORKERS" \
    --device "$DEVICE" \
    --project "$E6_PROJECT" \
    --name detector_yolov10x_classifier_yolov10x_cls \
    --seed "$SEED"
fi
