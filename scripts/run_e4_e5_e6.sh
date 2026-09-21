#!/usr/bin/env bash
set -euo pipefail

# Final E4/E5/E6 launch script.
# Run from the repository root or let the script resolve its own root:
#   DEVICE=1 bash scripts/run_e4_e5_e6.sh
#
# DEVICE is the physical CUDA index passed to Ultralytics.  When running two
# jobs in parallel, use DEVICE=0 and DEVICE=1 in separate processes and do not
# set a conflicting CUDA_VISIBLE_DEVICES mask.  The defaults below match the
# remote yolov8 environment and the generated remote datasets, but every input
# can be overridden from the shell.

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
  E5_PROJECT=${E5_PROJECT:-runs/experiments/E5_multilabel_200}
  E5_EPOCHS=${E5_EPOCHS:-200}
  E5_SEEDS=${E5_SEEDS:-"0 1 2 3 4"}
  E5_NAME_PREFIX=${E5_NAME_PREFIX:-yolov10x}
  E5_SKIP_EXISTING=${E5_SKIP_EXISTING:-1}

  for REQUIRED_FILE in "$E5_DATA" "$E5_MODEL"; do
    if [ ! -f "$REQUIRED_FILE" ]; then
      echo "Missing E5 input: $REQUIRED_FILE" >&2
      exit 1
    fi
  done

  read -r -a E5_SEED_LIST <<< "$E5_SEEDS"
  for E5_SEED in "${E5_SEED_LIST[@]}"; do
    E5_RUN_NAME="${E5_NAME_PREFIX}_seed${E5_SEED}"
    E5_RUN_DIR="$E5_PROJECT/$E5_RUN_NAME"
    if [ "$E5_SKIP_EXISTING" = "1" ] && [ -f "$E5_RUN_DIR/weights/best.pt" ]; then
      echo "Skipping existing E5 seed $E5_SEED: $E5_RUN_DIR/weights/best.pt"
      continue
    fi

    echo "Starting E5 seed=$E5_SEED for $E5_EPOCHS epochs"
    "$PYTHON_BIN" scripts/train_multilabel.py \
      --model "$E5_MODEL" \
      --data "$E5_DATA" \
      --epochs "$E5_EPOCHS" \
      --imgsz 640 \
      --batch "$BATCH" \
      --device "$DEVICE" \
      --project "$E5_PROJECT" \
      --name "$E5_RUN_NAME" \
      --seed "$E5_SEED"
  done
fi

if [ "$RUN_E6" = "1" ]; then
  # E6 is a matched-seed two-stage pipeline.  For each seed, use the
  # corresponding E3 YOLOv10x detector (its detector Stage1+Stage2 checkpoint)
  # and train the E6 crop classifier with the same seed.  The classifier is
  # trained on the shared GT-aligned crop dataset; the matching detector is
  # used again during the final predicted-crop Test evaluation.
  E6_DETECTOR_ROOT=${E6_DETECTOR_ROOT:-runs/experiments/E3_versions}
  E6_DETECTOR_PREFIX=${E6_DETECTOR_PREFIX:-E3_versions_yolov10x_w4_0p5_seed_}
  E6_DATA=${E6_DATA:-/localnvme/data/billboard/mayolo_v3_two_stage_crops/data.yaml}
  E6_MODEL=${E6_MODEL:-ultralytics/cfg/models/v10/yolov10x-cls.yaml}
  E6_PROJECT=${E6_PROJECT:-runs/experiments/E6_two_stage_yolov10x}
  E6_SEEDS=${E6_SEEDS:-"0 1 2 3 4"}
  E6_EPOCHS=${E6_EPOCHS:-100}
  E6_NAME=${E6_NAME:-detector_yolov10x_classifier_yolov10x_cls}

  for REQUIRED_FILE in "$E6_DATA" "$E6_MODEL"; do
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

  read -r -a E6_SEED_LIST <<< "$E6_SEEDS"
  for E6_SEED in "${E6_SEED_LIST[@]}"; do
    E6_DETECTOR_CHECKPOINT="$E6_DETECTOR_ROOT/${E6_DETECTOR_PREFIX}${E6_SEED}_stage2/weights/best.pt"
    if [ ! -f "$E6_DETECTOR_CHECKPOINT" ]; then
      echo "Missing E6 detector checkpoint for seed $E6_SEED: $E6_DETECTOR_CHECKPOINT" >&2
      exit 1
    fi

    # Keep seed=0 compatible with the original E6 path; use isolated projects
    # for the additional seeds so manifests and classifier weights cannot clash.
    if [ "$E6_SEED" = "0" ]; then
      E6_SEED_PROJECT="$E6_PROJECT"
    else
      E6_SEED_PROJECT="${E6_PROJECT}_seed${E6_SEED}"
    fi
    E6_RUN_DIR="$E6_SEED_PROJECT/$E6_NAME"
    if [ "${E6_SKIP_EXISTING:-1}" = "1" ] && [ -f "$E6_RUN_DIR/weights/best.pt" ]; then
      echo "Skipping existing E6 seed $E6_SEED: $E6_RUN_DIR/weights/best.pt"
      continue
    fi

    echo "Starting E6 seed=$E6_SEED with detector=$E6_DETECTOR_CHECKPOINT"
    "$PYTHON_BIN" scripts/train_two_stage.py \
      --detector-checkpoint "$E6_DETECTOR_CHECKPOINT" \
      --model "$E6_MODEL" \
      --pretrain "$E6_PRETRAIN" \
      --data "$E6_DATA" \
      --epochs "$E6_EPOCHS" \
      --imgsz 224 \
      --batch "$BATCH" \
      --workers "$WORKERS" \
      --device "$DEVICE" \
      --project "$E6_SEED_PROJECT" \
      --name "$E6_NAME" \
      --seed "$E6_SEED"
  done
fi
