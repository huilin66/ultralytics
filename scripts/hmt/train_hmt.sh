#!/usr/bin/env bash
set -euo pipefail

# Train the three HMT clean-v1 models (bdd_hmt_clean_v1).
# Override HMT_DATA_ROOT when the dataset lives elsewhere (e.g. the Windows
# machine: HMT_DATA_ROOT='D:/bdd_hmt_clean_v1').
#
# On Windows there is no bash; call tools/hmt_train.py directly, e.g.:
#   C:\Users\ftri\.conda\envs\yolo\python.exe tools\hmt_train.py \
#       --repo-root . --model yolov8x.pt --run-root D:\runs\hmt_clean_v1 \
#       --data-root D:\bdd_hmt_clean_v1 --dataset sua_t --device 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-yolo}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${HMT_DATA_ROOT:-/data/huilin/data/bdd_hmt_clean_v1}"

if command -v conda > /dev/null 2>&1; then
  run_python() {
    conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" "$@"
  }
else
  echo "[train_hmt] conda was not found; using ${PYTHON_BIN} from the active environment" >&2
  run_python() {
    "${PYTHON_BIN}" "$@"
  }
fi

MODEL="${MODEL:-${REPO_ROOT}/yolov8x.pt}"
RUN_ROOT="${HMT_RUN_ROOT:-${REPO_ROOT}/runs/hmt_clean_v1}"
DATASET="${HMT_DATASET:-all}"

run_python "${REPO_ROOT}/tools/hmt_train.py" \
  --repo-root "${REPO_ROOT}" \
  --model "${MODEL}" \
  --run-root "${RUN_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --dataset "${DATASET}" \
  --seed "${HMT_SEED:-233}" \
  --workers "${HMT_WORKERS:-4}" \
  --device "${HMT_DEVICE:-0}" \
  --epochs-sua_t "${HMT_EPOCHS_T:-200}" \
  --epochs-sua_rgb "${HMT_EPOCHS_RGB:-200}" \
  --epochs-bp_cube "${HMT_EPOCHS_CUBE:-300}" \
  --imgsz-sua_t "${HMT_IMGSZ_T:-640}" \
  --imgsz-sua_rgb "${HMT_IMGSZ_RGB:-1280}" \
  --imgsz-bp_cube "${HMT_IMGSZ_CUBE:-960}" \
  --batch-sua_t "${HMT_BATCH_T:--1}" \
  --batch-sua_rgb "${HMT_BATCH_RGB:--1}" \
  --batch-bp_cube "${HMT_BATCH_CUBE:--1}" \
  "$@"
