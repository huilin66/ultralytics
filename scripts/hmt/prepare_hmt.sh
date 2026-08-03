#!/usr/bin/env bash
set -euo pipefail

# Build immutable HMT update datasets. Run this file from Git Bash.
# Override HMT_DATA_ROOT/HMT_UPDATE_ROOT when the share is mounted elsewhere.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_TOOL="${REPO_ROOT}/tools/hmt_prepare_dataset.py"

HMT_DATA_ROOT="${HMT_DATA_ROOT:-//158.132.186.40/isds/huilin/tp/0803_hmt_data_check/bdd_hmt}"
HMT_UPDATE_ROOT="${HMT_UPDATE_ROOT:-//158.132.186.40/isds/huilin/tp/0803_hmt_data_check/bdd_hmt_update}"
CONDA_ENV="${CONDA_ENV:-common_py312}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if command -v conda >/dev/null 2>&1; then
    run_python() {
        conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" "$@"
    }
else
    echo "[prepare_hmt] conda was not found; using ${PYTHON_BIN} from the active Git Bash environment" >&2
    run_python() {
        "${PYTHON_BIN}" "$@"
    }
fi

run_ydm() {
    run_python -m yolo_data_manager.cli "$@"
}

FORCE_ARGS=()
if [[ "${HMT_REBUILD:-0}" == "1" ]]; then
    FORCE_ARGS+=(--force)
fi

for spec in t rgb cube; do
    case "${spec}" in
        t) source_name="sua_t"; output_name="sua_t_update" ;;
        rgb) source_name="sua_rgb"; output_name="sua_rgb_update" ;;
        cube) source_name="bp_cube"; output_name="bp_cube_update" ;;
    esac
    source_dir="${HMT_DATA_ROOT}/${source_name}"
    output_dir="${HMT_UPDATE_ROOT}/${output_name}"

    echo "[prepare_hmt] ${spec}: ${source_dir} -> ${output_dir}"
    run_python "${PYTHON_TOOL}" \
        --dataset "${spec}" \
        --source-root "${HMT_DATA_ROOT}" \
        --output-root "${HMT_UPDATE_ROOT}" \
        --split-mode "${HMT_SPLIT_MODE:-group}" \
        --seed "${HMT_SEED:-233}" \
        --val-ratio "${HMT_VAL_RATIO:-0.1}" \
        --test-ratio "${HMT_TEST_RATIO:-0.1}" \
        --sequence-group-size "${HMT_SEQUENCE_GROUP_SIZE:-12}" \
        --empty-train-ratio "${HMT_EMPTY_TRAIN_RATIO:-0.25}" \
        --max-repeat "${HMT_MAX_REPEAT:-4}" \
        "${FORCE_ARGS[@]}"

    run_python "${REPO_ROOT}/tools/hmt_absolute_lists.py" --dataset-root "${output_dir}"

    # yolo_data_manager owns standard validation and statistics. Explicitly
    # select flat layout so absolute source split-list paths cannot confuse
    # local validation on the Windows share.
    run_ydm check \
        --root "${output_dir}" \
        --layout flat \
        --images-dir images \
        --labels-dir labels \
        --class-file class.txt \
        --out "${output_dir}/reports/ydm_check.json" \
        --no-progress
    run_ydm stats \
        --root "${output_dir}" \
        --layout flat \
        --images-dir images \
        --labels-dir labels \
        --class-file class.txt \
        --out "${output_dir}/reports/ydm_stats.json" \
        --ann-csv "${output_dir}/reports/annotations.csv" \
        --no-progress
done

echo "[prepare_hmt] all HMT update datasets are ready under ${HMT_UPDATE_ROOT}"
