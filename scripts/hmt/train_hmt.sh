#!/usr/bin/env bash
set -euo pipefail

# Train the three HMT models in yolov8_deploy. Run from Git Bash.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-yolov8_deploy}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if command -v conda >/dev/null 2>&1; then
    run_python() {
        conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" "$@"
    }
else
    echo "[train_hmt] conda was not found; using ${PYTHON_BIN} from the active Git Bash environment" >&2
    run_python() {
        "${PYTHON_BIN}" "$@"
    }
fi

MODEL="${MODEL:-${REPO_ROOT}/yolov8x.pt}"
DATA_ROOT="${HMT_UPDATE_ROOT:-//158.132.186.40/isds/huilin/tp/0803_hmt_data_check/bdd_hmt_update}"
RUN_ROOT="${HMT_RUN_ROOT:-//158.132.186.40/isds/huilin/tp/0803_hmt_data_check/hmt_update_runs}"
DATASET="${HMT_DATASET:-all}"

run_python "${REPO_ROOT}/tools/hmt_train.py" \
    --repo-root "${REPO_ROOT}" \
    --model "${MODEL}" \
    --data-root "${DATA_ROOT}" \
    --run-root "${RUN_ROOT}" \
    --dataset "${DATASET}" \
    --seed "${HMT_SEED:-233}" \
    --workers "${HMT_WORKERS:-4}" \
    --device "${HMT_DEVICE:-0}" \
    --epochs-t "${HMT_EPOCHS_T:-240}" \
    --epochs-rgb "${HMT_EPOCHS_RGB:-240}" \
    --epochs-cube "${HMT_EPOCHS_CUBE:-300}" \
    --imgsz-t "${HMT_IMGSZ_T:-640}" \
    --imgsz-rgb "${HMT_IMGSZ_RGB:-768}" \
    --imgsz-cube "${HMT_IMGSZ_CUBE:-1024}" \
    --batch-t "${HMT_BATCH_T:--1}" \
    --batch-rgb "${HMT_BATCH_RGB:--1}" \
    --batch-cube "${HMT_BATCH_CUBE:--1}" \
    "$@"
