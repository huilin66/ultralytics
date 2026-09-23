#!/usr/bin/env bash
# Reproducibility launcher for the experiments reported in
# scripts/EXPERIMENTS_results.md.
#
# Nothing is started by default. Select one or more sections explicitly:
#
#   RUN_E0=1 bash run.sh                         # protocol / HSV / epochs
#   RUN_E2=1 RUN_E2_EVAL=1 bash run.sh           # GIA/GCA/HO
#   RUN_E2_GCA=1 bash run.sh                     # corrected pure GCA/FGA only
#   RUN_SECTION2_8=1 bash run.sh                  # E2 performance profile
#   RUN_SECTION3_PERF=1 bash run.sh               # E3/E4 model performance profile
#   RUN_E3=1 RUN_E3_STABILITY=1 RUN_E3_EVAL=1 bash run.sh
#   RUN_ALL=1 bash run.sh                        # complete reproduction
#
# All paths and training settings can be overridden by environment variables.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA="${DATA:-ultralytics/cfg/mayolo_r1/mayolo_v3.yaml}"
MD_MODEL="${MD_MODEL:-ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml}"
PRETRAIN="${PRETRAIN:-yolov10x.pt}"
COM_PATH="${COM_PATH:-/localnvme/data/billboard/mayolo_v3/co_occurrence_matrix_train.csv}"
COM_CONDITIONAL_PATH="${COM_CONDITIONAL_PATH:-/localnvme/data/billboard/mayolo_v3/co_occurrence_matrix_train_conditional.csv}"

DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-8}"
IMGSZ="${IMGSZ:-640}"
W4="${W4:-0.5}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-100}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-100}"
HSV_H="${HSV_H:-0}"
HSV_S="${HSV_S:-0.2}"
HSV_V="${HSV_V:-0.2}"
SEEDS="${SEEDS:-0 1 2 3 4}"
MAYOLO_SIZES="${MAYOLO_SIZES:-n s m l b}"
W4_TAG="${W4//./p}"
E3_PERF_SEED="${E3_PERF_SEED:-0}"
E3_PERF_BATCH="${E3_PERF_BATCH:-1}"
E3_PERF_WARMUP="${E3_PERF_WARMUP:-50}"
E3_PERF_ITERATIONS="${E3_PERF_ITERATIONS:-200}"
E3_PERF_PRECISION="${E3_PERF_PRECISION:-fp16}"
E3_PERF_ROOT="${E3_PERF_ROOT:-runs/experiments/performance_profile}"

# Pass the selected interpreter/device settings to the E4/E5/E6 wrapper.
export PYTHON_BIN DEVICE BATCH WORKERS

# CPU-thread limits used by the final remote experiments.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

RUN_ALL="${RUN_ALL:-0}"
RUN_E0="${RUN_E0:-0}"
RUN_E0_EVAL="${RUN_E0_EVAL:-0}"
RUN_E1="${RUN_E1:-0}"
RUN_E1_EVAL="${RUN_E1_EVAL:-0}"
RUN_E2="${RUN_E2:-0}"
RUN_E2_EVAL="${RUN_E2_EVAL:-0}"
RUN_E2_GCA="${RUN_E2_GCA:-0}"
E2_GCA_MATRIX="${E2_GCA_MATRIX:-both}"
RUN_E3="${RUN_E3:-0}"
RUN_E3_STABILITY="${RUN_E3_STABILITY:-0}"
RUN_E3_EVAL="${RUN_E3_EVAL:-0}"
RUN_E4="${RUN_E4:-0}"
RUN_E5="${RUN_E5:-0}"
RUN_E6="${RUN_E6:-0}"
RUN_SECTION2_8="${RUN_SECTION2_8:-0}"
RUN_SECTION3_PERF="${RUN_SECTION3_PERF:-0}"
RUN_ROBUSTNESS="${RUN_ROBUSTNESS:-0}"
RUN_SECTION6="${RUN_SECTION6:-0}"

if [[ "$RUN_ALL" == "1" ]]; then
  RUN_E0=1; RUN_E0_EVAL=1
  RUN_E1=1; RUN_E1_EVAL=1
  RUN_E2=1; RUN_E2_EVAL=1
  RUN_E3=1; RUN_E3_STABILITY=1; RUN_E3_EVAL=1
  RUN_E4=1; RUN_E5=1; RUN_E6=1
  RUN_SECTION2_8=1
  RUN_SECTION3_PERF=1
  RUN_ROBUSTNESS=1; RUN_SECTION6=1
fi

if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || { echo "Python executable is not executable: $PYTHON_BIN" >&2; exit 1; }
else
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "Python executable was not found: $PYTHON_BIN" >&2
    exit 1
  }
fi

py() { "$PYTHON_BIN" "$@"; }
die() { echo "[run.sh] $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "required file does not exist: $1"; }

require_pure_gca_config() {
  local config="$1"
  require_file "$config"
  [[ "$config" != *GIA* ]] || die "pure Baseline+GCA config must not contain GIA: $config"
  if grep -qE 'SCDown, \[[^]]*, True, True, True\]' "$config"; then
    die "pure Baseline+GCA config contains GIA-enabled SCDown layers: $config"
  fi
}

require_checkpoint_ref() {
  local value="$1"
  # A bare official checkpoint name may be downloaded by Ultralytics. An
  # explicit path must already exist so a reviewer gets an immediate error.
  if [[ "$value" == */* || "$value" == ./* || "$value" == ../* ]]; then
    require_file "$value"
  fi
}

read -r -a SEED_LIST <<< "$SEEDS"
MD_ARGS=(
  --data "$DATA" --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS"
  --device "$DEVICE" --w4 "$W4"
  --hsv-h "$HSV_H" --hsv-s "$HSV_S" --hsv-v "$HSV_V"
  --skip-existing
)

echo_usage() {
  cat <<'EOF'
Usage examples:
  RUN_E0=1 bash run.sh
  RUN_E2=1 RUN_E2_EVAL=1 bash run.sh
  RUN_E2_GCA=1 bash run.sh
  RUN_SECTION2_8=1 bash run.sh
  RUN_SECTION3_PERF=1 bash run.sh
  RUN_E3=1 RUN_E3_STABILITY=1 RUN_E3_EVAL=1 bash run.sh
  RUN_E4=1 RUN_E5=1 RUN_E6=1 bash run.sh
  RUN_ROBUSTNESS=1 RUN_SECTION6=1 bash run.sh
  RUN_ALL=1 bash run.sh

Useful overrides:
  PYTHON_BIN=/home/.../envs/yolov8/bin/python
  DEVICE=0, BATCH=16, WORKERS=8, SEEDS="0 1 2 3 4"
  DATA=/path/to/mayolo_v3.yaml
EOF
}

# ----------------------------- E0 / E1 ----------------------------------

run_e0() {
  echo "[E0] protocol, HSV, Stage1 and Stage2 sweeps"
  require_file "$DATA"; require_file "$MD_MODEL"; require_checkpoint_ref "$PRETRAIN"

  py scripts/train_mdet_experiments.py hsv-ablation \
    "${MD_ARGS[@]}" --label E0_hsv_ablation --model "$MD_MODEL" --pretrain "$PRETRAIN" \
    --epochs "$STAGE1_EPOCHS" --project runs/experiments/E0_hsv_ablation

  py scripts/train_mdet_experiments.py stage1-sweep \
    "${MD_ARGS[@]}" --label E0_stage1_sweep --model "$MD_MODEL" --pretrain "$PRETRAIN" \
    --stage1-values 100 200 300 400 500 --project runs/experiments/E0_stage1_sweep

  local stage1_checkpoint="${E0_STAGE1_CHECKPOINT:-runs/experiments/E0_stage1_sweep/E0_stage1_sweep_stage1_100_w4_${W4_TAG}_seed_0/weights/best.pt}"
  require_file "$stage1_checkpoint"
  py scripts/train_mdet_experiments.py stage2-sweep \
    "${MD_ARGS[@]}" --label E0_stage2_sweep --model "$MD_MODEL" \
    --stage1-checkpoint "$stage1_checkpoint" --stage1-epochs "$STAGE1_EPOCHS" \
    --stage2-values 50 100 150 200 --project runs/experiments/E0_stage2_sweep

  [[ "$RUN_E0_EVAL" == "1" ]] && eval_e0
}

run_e1() {
  echo "[E1] w4 scan"
  require_file "$DATA"; require_file "$MD_MODEL"; require_checkpoint_ref "$PRETRAIN"
  py scripts/train_mdet_experiments.py w4 \
    "${MD_ARGS[@]}" --label E1_w4 --model "$MD_MODEL" --pretrain "$PRETRAIN" \
    --w4-values 0.25 0.5 0.75 1.0 1.25 1.5 --project runs/experiments/E1_w4
  [[ "$RUN_E1_EVAL" == "1" ]] && eval_e1
}

# ----------------------------- E2 ---------------------------------------

E2_27_ROOT="${E2_27_ROOT:-runs/experiments/E2_27_baseline_gia_seed5}"
E2_27_LABEL="${E2_27_LABEL:-E2_27_baseline_gia_seed5}"
E2_BASELINE_STAGE1_PREFIX="${E2_BASELINE_STAGE1_PREFIX:-${E2_27_LABEL}_baseline_w4_${W4_TAG}_seed_}"
E2_GIA_STAGE1_PREFIX="${E2_GIA_STAGE1_PREFIX:-${E2_27_LABEL}_gia_v2_5_7_w4_${W4_TAG}_seed_}"
E2_BASELINE_GCA_CONFIG="${E2_BASELINE_GCA_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GCA_margin_residual.yaml}"
E2_GIA_GCA_CONFIG="${E2_GIA_GCA_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_5_7_GCA_margin_residual.yaml}"
# E2.29 previously used a GIA+GCA YAML by mistake. Keep those old artifacts
# untouched and write the corrected pure-Baseline+GCA runs elsewhere.
E2_BASELINE_GCA_LABEL="${E2_BASELINE_GCA_LABEL:-E2_29_Baseline_GCA_pure_margin_residual_5seed}"
E2_BASELINE_GCA_CROSS_LABEL="${E2_BASELINE_GCA_CROSS_LABEL:-${E2_BASELINE_GCA_LABEL}_cross}"
E2_BASELINE_GCA_CONDITIONAL_LABEL="${E2_BASELINE_GCA_CONDITIONAL_LABEL:-${E2_BASELINE_GCA_LABEL}_conditional}"
E2_BASELINE_GCA_CROSS_ROOT="${E2_BASELINE_GCA_CROSS_ROOT:-runs/experiments/${E2_BASELINE_GCA_CROSS_LABEL}}"
E2_BASELINE_GCA_CONDITIONAL_ROOT="${E2_BASELINE_GCA_CONDITIONAL_ROOT:-runs/experiments/${E2_BASELINE_GCA_CONDITIONAL_LABEL}}"
E2_BASELINE_GCA_HO_ROOT="${E2_BASELINE_GCA_HO_ROOT:-runs/experiments/E2_6_Baseline_GCA_pure_HO_test}"
GIA_CONFIG="${GIA_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_5_7.yaml}"
GCA_GNN_TYPES=(fga gcn gat graphsage gin)
E2_POSITION_VARIANTS=(
  "gia_v2_6=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_6.yaml"
  "gia_v2_7=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_7.yaml"
  "gia_v2_8=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_8.yaml"
  "gia_v2_9=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_9.yaml"
  "gia_v2_10=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_10.yaml"
  "gia_v2_13=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_13.yaml"
  "gia_v2_16=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_16.yaml"
  "gia_v2_19=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_19.yaml"
  "gia_v2_22=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_22.yaml"
  "gia_v2_5_7=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_5_7.yaml"
)

stage1_checkpoint() {
  local prefix="$1" seed="$2"
  echo "$E2_27_ROOT/${prefix}${seed}_stage1/weights/best.pt"
}

stage2_checkpoint() {
  local root="$1" prefix="$2" operator="$3" seed="$4"
  echo "$root/${prefix}_${operator}_margin_residual_stage1_${STAGE1_EPOCHS}_stage2_${STAGE2_EPOCHS}_w4_${W4_TAG}_seed_${seed}/weights/best.pt"
}

run_e2_baseline_gia() {
  local seed
  for seed in "${SEED_LIST[@]}"; do
    py scripts/train_mdet_experiments.py variants \
      "${MD_ARGS[@]}" --label "$E2_27_LABEL" \
      --stage1-epochs "$STAGE1_EPOCHS" --stage2-epochs "$STAGE2_EPOCHS" \
      --variant "baseline=$MD_MODEL" --variant "gia_v2_5_7=$GIA_CONFIG" \
      --pretrain "$PRETRAIN" --seed "$seed" --project "$E2_27_ROOT"
  done
}

run_e2_position() {
  local seed variant
  for seed in "${SEED_LIST[@]}"; do
    local -a cmd=(
      "$PYTHON_BIN" scripts/train_mdet_experiments.py gia-position
      "${MD_ARGS[@]}" --label E2_1_GIA_v2_position
      --stage1-epochs "$STAGE1_EPOCHS" --stage2-epochs "$STAGE2_EPOCHS"
      --stage1-only --model "$MD_MODEL" --pretrain "$PRETRAIN" --seed "$seed"
      --project runs/experiments/E2_1_GIA_v2_position
    )
    for variant in "${E2_POSITION_VARIANTS[@]}"; do cmd+=(--variant "$variant"); done
    "${cmd[@]}"
  done
}

run_gca_matrix() {
  local label="$1" project="$2" stage1_prefix="$3" config="$4" matrix_path="$5"
  local -a checkpoint_args=()
  local seed checkpoint
  require_file "$matrix_path"
  if [[ "$label" == *Baseline_GCA_pure* ]]; then
    require_pure_gca_config "$config"
  else
    require_file "$config"
  fi
  for seed in "${SEED_LIST[@]}"; do
    checkpoint="$(stage1_checkpoint "$stage1_prefix" "$seed")"
    require_file "$checkpoint"
    checkpoint_args+=(--stage1-checkpoint-map "${seed}=${checkpoint}")
  done
  py scripts/train_mdet_experiments.py gca-stage2-seeds \
    "${MD_ARGS[@]}" --label "$label" \
    --stage1-epochs "$STAGE1_EPOCHS" --stage2-epochs "$STAGE2_EPOCHS" \
    --variant margin_residual="$config" --gnn-types "${GCA_GNN_TYPES[@]}" \
    --seeds "${SEED_LIST[@]}" --com-path "$matrix_path" \
    "${checkpoint_args[@]}" --project "$project"
}

run_e2_training() {
  echo "[E2] baseline/GIA stability, GCA matrices and matched Stage2 studies"
  require_file "$DATA"; require_file "$MD_MODEL"; require_checkpoint_ref "$PRETRAIN"
  require_file "$GIA_CONFIG"; require_file "$COM_PATH"; require_file "$COM_CONDITIONAL_PATH"
  run_e2_baseline_gia
  run_e2_position
  run_gca_matrix "$E2_BASELINE_GCA_CROSS_LABEL" \
    "$E2_BASELINE_GCA_CROSS_ROOT" \
    "$E2_BASELINE_STAGE1_PREFIX" "$E2_BASELINE_GCA_CONFIG" "$COM_PATH"
  run_gca_matrix "$E2_BASELINE_GCA_CONDITIONAL_LABEL" \
    "$E2_BASELINE_GCA_CONDITIONAL_ROOT" \
    "$E2_BASELINE_STAGE1_PREFIX" "$E2_BASELINE_GCA_CONFIG" "$COM_CONDITIONAL_PATH"
  run_gca_matrix E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross \
    runs/experiments/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross \
    "$E2_GIA_STAGE1_PREFIX" "$E2_GIA_GCA_CONFIG" "$COM_PATH"
  run_gca_matrix E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_conditional \
    runs/experiments/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_conditional \
    "$E2_GIA_STAGE1_PREFIX" "$E2_GIA_GCA_CONFIG" "$COM_CONDITIONAL_PATH"
}

run_e2_pure_gca() {
  echo "[E2.2] corrected pure Baseline+GCA/FGA: 5 graph operators x 5 seeds x Cross/Conditional"
  require_file "$DATA"
  require_file "$COM_PATH"
  require_file "$COM_CONDITIONAL_PATH"
  require_pure_gca_config "$E2_BASELINE_GCA_CONFIG"

  case "$E2_GCA_MATRIX" in
    cross)
      run_gca_matrix "$E2_BASELINE_GCA_CROSS_LABEL" \
        "$E2_BASELINE_GCA_CROSS_ROOT" \
        "$E2_BASELINE_STAGE1_PREFIX" "$E2_BASELINE_GCA_CONFIG" "$COM_PATH"
      ;;
    conditional)
      run_gca_matrix "$E2_BASELINE_GCA_CONDITIONAL_LABEL" \
        "$E2_BASELINE_GCA_CONDITIONAL_ROOT" \
        "$E2_BASELINE_STAGE1_PREFIX" "$E2_BASELINE_GCA_CONFIG" "$COM_CONDITIONAL_PATH"
      ;;
    both)
      run_gca_matrix "$E2_BASELINE_GCA_CROSS_LABEL" \
        "$E2_BASELINE_GCA_CROSS_ROOT" \
        "$E2_BASELINE_STAGE1_PREFIX" "$E2_BASELINE_GCA_CONFIG" "$COM_PATH"
      run_gca_matrix "$E2_BASELINE_GCA_CONDITIONAL_LABEL" \
        "$E2_BASELINE_GCA_CONDITIONAL_ROOT" \
        "$E2_BASELINE_STAGE1_PREFIX" "$E2_BASELINE_GCA_CONFIG" "$COM_CONDITIONAL_PATH"
      ;;
    *)
      die "E2_GCA_MATRIX must be cross, conditional, or both; got: $E2_GCA_MATRIX"
      ;;
  esac
}

eval_mdet_set() {
  local summary="$1" name="$2" mode="$3"
  shift 3
  local parent="${summary%/*}"
  [[ "$parent" == "$summary" ]] && parent="."
  local -a cmd=(
    "$PYTHON_BIN" scripts/eval_mdet_experiments.py ho
    --data "$DATA" --mode "$mode" --split test --device "$DEVICE"
    --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS"
    --project "$parent" --name "$name" --summary "$summary" --no-plots
  )
  local weight
  for weight in "$@"; do require_file "$weight"; cmd+=(--weights "$weight"); done
  "${cmd[@]}"
}

collect_stage2_weights() {
  local -n output="$1"
  local prefix="$2"
  local seed
  output=()
  for seed in "${SEED_LIST[@]}"; do
    output+=("$E2_27_ROOT/${prefix}${seed}_stage2/weights/best.pt")
  done
}

collect_gca_weights() {
  local -n output="$1"
  local root="$2" prefix="$3"
  local operator seed
  output=()
  for operator in "${GCA_GNN_TYPES[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
      output+=("$(stage2_checkpoint "$root" "$prefix" "$operator" "$seed")")
    done
  done
}

run_e2_evaluation() {
  echo "[E2] generating Test summaries"
  local -a baseline=() gia=() baseline_gca=() gia_gca=()
  collect_stage2_weights baseline "$E2_BASELINE_STAGE1_PREFIX"
  collect_stage2_weights gia "$E2_GIA_STAGE1_PREFIX"

  eval_mdet_set runs/experiments/E2_27_HO/summary.csv E2_27_HO both "${baseline[@]}" "${gia[@]}"

  collect_gca_weights baseline_gca \
    "$E2_BASELINE_GCA_CONDITIONAL_ROOT" "$E2_BASELINE_GCA_CONDITIONAL_LABEL"
  eval_mdet_set "$E2_BASELINE_GCA_CONDITIONAL_ROOT/summary.csv" \
    E2_29_Baseline_GCA_conditional native "${baseline[@]}" "${baseline_gca[@]}"

  collect_gca_weights baseline_gca \
    "$E2_BASELINE_GCA_CROSS_ROOT" "$E2_BASELINE_GCA_CROSS_LABEL"
  eval_mdet_set "$E2_BASELINE_GCA_CROSS_ROOT/summary.csv" \
    E2_29_Baseline_GCA_cross native "${baseline[@]}" "${baseline_gca[@]}"

  collect_gca_weights gia_gca \
    runs/experiments/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_conditional \
    E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_conditional
  eval_mdet_set runs/experiments/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_conditional/summary.csv \
    E2_28_GIA_GCA_conditional native "${gia[@]}" "${gia_gca[@]}"

  collect_gca_weights gia_gca \
    runs/experiments/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross \
    E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross
  eval_mdet_set runs/experiments/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross/summary.csv \
    E2_28_GIA_GCA_cross native "${gia[@]}" "${gia_gca[@]}"

  eval_mdet_set "$E2_BASELINE_GCA_HO_ROOT/summary.csv" \
    E2_6_Baseline_GCA_pure_HO both "${baseline[@]}" "${baseline_gca[@]}"
  eval_mdet_set runs/experiments/E2_7_Baseline_GIA_GCA_HO_test/summary.csv \
    E2_7_Baseline_GIA_GCA_HO both "${gia[@]}" "${gia_gca[@]}"
}

run_section2_8() {
  echo "[2.8] E2 seed-0 performance profile"

  # The pure Baseline+GCA checkpoint comes from the corrected E2.29 run.
  # The old E2.29 directory contained generated YAML bound to GIA+GCA.
  local baseline="${E2_27_ROOT}/${E2_BASELINE_STAGE1_PREFIX}0_stage2/weights/best.pt"
  local gia="${E2_27_ROOT}/${E2_GIA_STAGE1_PREFIX}0_stage2/weights/best.pt"
  local pure_gca_root="${E2_PURE_GCA_ROOT:-$E2_BASELINE_GCA_CROSS_ROOT}"
  local pure_gca_prefix="${E2_PURE_GCA_PREFIX:-$E2_BASELINE_GCA_CROSS_LABEL}"
  local pure_gca
  pure_gca="$(stage2_checkpoint "$pure_gca_root" "$pure_gca_prefix" gin 0)"
  local gia_gca_root="runs/experiments/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross"
  local gia_gca_prefix="E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross"
  local gia_gca
  gia_gca="$(stage2_checkpoint "$gia_gca_root" "$gia_gca_prefix" gin 0)"
  local output_root="${E2_PERFORMANCE_ROOT:-runs/experiments/performance_profile}"

  require_file "$baseline"
  require_file "$gia"
  require_file "$pure_gca"
  require_file "$gia_gca"

  local -a native_labels=(
    E2.0_Baseline_seed0
    E2.1_GIA_v2_5_7_seed0
    E2.2_Baseline+GCA_cross_GIN_seed0
    E2.5_GIA+GCA_cross_GIN_seed0
  )
  local -a native_weights=("$baseline" "$gia" "$pure_gca" "$gia_gca")
  py scripts/benchmark_performance.py \
    --weights "${native_weights[@]}" \
    --labels "${native_labels[@]}" \
    --task mdetect --head-mode native --device "$DEVICE" --precision fp16 \
    --batch 1 --imgsz "$IMGSZ" --warmup "${E2_PERF_WARMUP:-50}" \
    --iterations "${E2_PERF_ITERATIONS:-200}" \
    --output "$output_root/E2_seed0_native.csv"

  local -a one2many_labels=(
    E2.3_Baseline_one2many_seed0
    E2.4_GIA_one2many_seed0
    E2.6_Baseline+GCA+HO_cross_GIN_seed0
    E2.7_GIA+GCA+HO_cross_GIN_seed0
  )
  py scripts/benchmark_performance.py \
    --weights "${native_weights[@]}" \
    --labels "${one2many_labels[@]}" \
    --task mdetect --head-mode one2many --device "$DEVICE" --precision fp16 \
    --batch 1 --imgsz "$IMGSZ" --warmup "${E2_PERF_WARMUP:-50}" \
    --iterations "${E2_PERF_ITERATIONS:-200}" \
    --output "$output_root/E2_seed0_one2many.csv"
}

run_section3_performance() {
  echo "[3] E3/E4 seed-${E3_PERF_SEED} performance profile"
  require_file "$DATA"
  mkdir -p "$E3_PERF_ROOT"

  # E3 native YOLO families.  The family/size list is shared with the E3
  # training/evaluation launcher, so this benchmark cannot silently omit a
  # size that is reported in Section 3.
  local family size name label weight
  local -a native_weights=() native_labels=()
  for family in yolov8 yolov9 yolov10 yolov11 yolov12 yolov13 yolov26; do
    for size in $(e3_sizes "$family"); do
      name="${family}${size}"
      label="YOLO${family#yolo}${size}"
      weight="runs/experiments/E3_versions/E3_versions_${name}_w4_${W4_TAG}_seed_${E3_PERF_SEED}_stage2/weights/best.pt"
      require_file "$weight"
      native_weights+=("$weight")
      native_labels+=("$label")
    done
  done
  py scripts/benchmark_performance.py \
    --weights "${native_weights[@]}" --labels "${native_labels[@]}" \
    --network yolo --task mdetect --head-mode native --device "$DEVICE" \
    --precision "$E3_PERF_PRECISION" --batch "$E3_PERF_BATCH" --imgsz "$IMGSZ" \
    --warmup "$E3_PERF_WARMUP" --iterations "$E3_PERF_ITERATIONS" \
    --output "$E3_PERF_ROOT/E3_seed${E3_PERF_SEED}_native.csv"

  # MAYOLO uses the one-to-many inference head.  MAYOLOx is the final
  # GIA+GCA+HO model from E2.28 and is included beside the E3 size sweep.
  local -a mayolo_weights=() mayolo_labels=()
  for size in ${E3_PERF_MAYOLO_SIZES:-n s m l b}; do
    weight="runs/experiments/E3_versions/E3_MAYOLO_final_mayolo${size}_w4_${W4_TAG}_seed_${E3_PERF_SEED}_stage2/weights/best.pt"
    require_file "$weight"
    mayolo_weights+=("$weight")
    mayolo_labels+=("MAYOLO${size}")
  done
  weight="runs/experiments/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross_gin_margin_residual_stage1_${STAGE1_EPOCHS}_stage2_${STAGE2_EPOCHS}_w4_${W4_TAG}_seed_${E3_PERF_SEED}/weights/best.pt"
  require_file "$weight"
  mayolo_weights+=("$weight")
  mayolo_labels+=("MAYOLOx")
  py scripts/benchmark_performance.py \
    --weights "${mayolo_weights[@]}" --labels "${mayolo_labels[@]}" \
    --network yolo --task mdetect --head-mode one2many --device "$DEVICE" \
    --precision "$E3_PERF_PRECISION" --batch "$E3_PERF_BATCH" --imgsz "$IMGSZ" \
    --warmup "$E3_PERF_WARMUP" --iterations "$E3_PERF_ITERATIONS" \
    --output "$E3_PERF_ROOT/E3_seed${E3_PERF_SEED}_mayolo.csv"

  # RT-DETR checkpoints use the RTDETR API rather than YOLO(..., task=mdetect).
  local rtdetr_root="runs/experiments/E4_rtdetr_LX"
  local -a rtdetr_weights=(
    "$rtdetr_root/E4_rtdetr_LX_rtdetr_l_w4_${W4_TAG}_seed_${E3_PERF_SEED}_stage2/weights/best.pt"
    "$rtdetr_root/E4_rtdetr_LX_rtdetr_x_w4_${W4_TAG}_seed_${E3_PERF_SEED}_stage2/weights/best.pt"
  )
  local -a rtdetr_labels=("RT-DETR-L" "RT-DETR-X")
  for weight in "${rtdetr_weights[@]}"; do require_file "$weight"; done
  py scripts/benchmark_performance.py \
    --weights "${rtdetr_weights[@]}" --labels "${rtdetr_labels[@]}" \
    --network rtdetr --head-mode native --device "$DEVICE" \
    --precision "$E3_PERF_PRECISION" --batch "$E3_PERF_BATCH" --imgsz "$IMGSZ" \
    --warmup "$E3_PERF_WARMUP" --iterations "$E3_PERF_ITERATIONS" \
    --output "$E3_PERF_ROOT/E4_seed${E3_PERF_SEED}_rtdetr.csv"
}

# ----------------------------- E3 ---------------------------------------

e3_sizes() {
  case "$1" in
    yolov8) echo "n s m l x" ;;
    yolov9) echo "t s m c e" ;;
    yolov10) echo "n s m b l x" ;;
    yolov11|yolov12) echo "n s m l x" ;;
    yolov13) echo "n s l x" ;;
    yolov26) echo "n s m l x" ;;
    *) die "unknown E3 family: $1" ;;
  esac
}

e3_pretrain_name() {
  local family="$1" size="$2"
  case "$family" in
    yolov11|yolov12) echo "yolo${family#yolov}${size}.pt" ;;
    yolov13) echo "yolov13${size}.pt" ;;
    yolov26) echo "yolo26${size}.pt" ;;
    *) echo "${family}${size}.pt" ;;
  esac
}

run_e3_native_training() {
  local family size name config pretrain
  for family in yolov8 yolov9 yolov10 yolov11 yolov12 yolov13 yolov26; do
    for size in $(e3_sizes "$family"); do
      name="${family}${size}"
      config="ultralytics/cfg/models/experiments/${name}-mdetect.yaml"
      pretrain="$(e3_pretrain_name "$family" "$size")"
      [[ -n "${E3_PRETRAIN_DIR:-}" ]] && pretrain="${E3_PRETRAIN_DIR%/}/${pretrain}"
      require_file "$config"; require_checkpoint_ref "$pretrain"
      py scripts/train_mdet_experiments.py versions \
        "${MD_ARGS[@]}" --label E3_versions \
        --stage1-epochs "$STAGE1_EPOCHS" --stage2-epochs "$STAGE2_EPOCHS" \
        --variant "${name}=${config}" --pretrain-map "${name}=${pretrain}" \
        --project runs/experiments/E3_versions
    done
  done
}

run_e3_mayolo_training() {
  local mayolo_sizes="$MAYOLO_SIZES"
  py scripts/train_mayolo_final_sizes.py \
    --sizes $mayolo_sizes --device "$DEVICE" --data "$DATA" \
    --project runs/experiments/E3_versions --label E3_MAYOLO_final \
    --stage1-epochs "$STAGE1_EPOCHS" --stage2-epochs "$STAGE2_EPOCHS" \
    --batch "$BATCH" --imgsz "$IMGSZ" --workers "$WORKERS" --w4 "$W4" --seed 0 \
    --hsv-h "$HSV_H" --hsv-s "$HSV_S" --hsv-v "$HSV_V" \
    --com-path "$COM_PATH" --skip-existing
}

run_e3_stability_training() {
  local family size name config pretrain seed
  local stability_seeds="${E3_STABILITY_SEEDS:-1 2 3 4}"
  read -r -a E3_SEED_LIST <<< "$stability_seeds"
  for family in yolov8 yolov9 yolov10 yolov11 yolov12 yolov13 yolov26; do
    for size in $(e3_sizes "$family"); do
      name="${family}${size}"
      config="ultralytics/cfg/models/experiments/${name}-mdetect.yaml"
      pretrain="$(e3_pretrain_name "$family" "$size")"
      [[ -n "${E3_PRETRAIN_DIR:-}" ]] && pretrain="${E3_PRETRAIN_DIR%/}/${pretrain}"
      for seed in "${E3_SEED_LIST[@]}"; do
        py scripts/train_mdet_experiments.py stability \
          "${MD_ARGS[@]}" --label E3_versions \
          --stage1-epochs "$STAGE1_EPOCHS" --stage2-epochs "$STAGE2_EPOCHS" \
          --variant "${name}=${config}" --pretrain-map "${name}=${pretrain}" \
          --seeds "$seed" --project runs/experiments/E3_versions
      done
    done
  done
  local mayolo_stability_sizes="${E3_MAYOLO_STABILITY_SIZES:-$MAYOLO_SIZES}"
  for seed in "${E3_SEED_LIST[@]}"; do
    py scripts/train_mayolo_final_sizes.py \
      --sizes $mayolo_stability_sizes --device "$DEVICE" --data "$DATA" \
      --project runs/experiments/E3_versions --label E3_MAYOLO_final \
      --stage1-epochs "$STAGE1_EPOCHS" --stage2-epochs "$STAGE2_EPOCHS" \
      --batch "$BATCH" --imgsz "$IMGSZ" --workers "$WORKERS" --w4 "$W4" --seed "$seed" \
      --hsv-h "$HSV_H" --hsv-s "$HSV_S" --hsv-v "$HSV_V" \
      --com-path "$COM_PATH" --skip-existing
  done
}

run_e3_evaluation() {
  echo "[E3] generating seed-0 Test summary"
  local -a native_weights=() mayolo_weights=() summaries=()
  local family size name
  for family in yolov8 yolov9 yolov10 yolov11 yolov12 yolov13 yolov26; do
    for size in $(e3_sizes "$family"); do
      name="${family}${size}"
      native_weights+=("runs/experiments/E3_versions/E3_versions_${name}_w4_${W4_TAG}_seed_0_stage2/weights/best.pt")
    done
  done
  for size in $MAYOLO_SIZES; do
    mayolo_weights+=("runs/experiments/E3_versions/E3_MAYOLO_final_mayolo${size}_w4_${W4_TAG}_seed_0_stage2/weights/best.pt")
  done
  # Native baselines and final MAYOLO models use different head modes.
  local native_summary="runs/experiments/E3_versions/summary_native_seed0.csv"
  local mayolo_summary="runs/experiments/E3_versions/summary_mayolo_seed0.csv"
  eval_mdet_set "$native_summary" E3_versions_native_seed0 native "${native_weights[@]}"
  eval_mdet_set "$mayolo_summary" E3_versions_mayolo_seed0 one2many "${mayolo_weights[@]}"
  local mayolox="runs/experiments/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross_gin_margin_residual_stage1_${STAGE1_EPOCHS}_stage2_${STAGE2_EPOCHS}_w4_${W4_TAG}_seed_0/weights/best.pt"
  require_file "$mayolox"
  local mayolox_summary="runs/experiments/E3_versions/summary_mayolox_seed0.csv"
  eval_mdet_set "$mayolox_summary" E3_versions_mayolox_seed0 one2many "$mayolox"
  summaries=("$native_summary" "$mayolo_summary" "$mayolox_summary")
  merge_csvs runs/experiments/E3_versions/summary.csv "${summaries[@]}"
}

run_e3() {
  echo "[E3] native model/size matrix and MAYOLO sizes"
  require_file "$DATA"; require_file "$MD_MODEL"
  [[ "${E3_NATIVE:-1}" == "1" ]] && run_e3_native_training
  [[ "${E3_MAYOLO:-1}" == "1" ]] && run_e3_mayolo_training
  [[ "$RUN_E3_STABILITY" == "1" ]] && run_e3_stability_training
  [[ "$RUN_E3_EVAL" == "1" ]] && run_e3_evaluation
}

# ----------------------------- E4/E5/E6 ---------------------------------

run_e4() {
  echo "[E4] RT-DETR-L/X five-seed training"
  local seeds="${E4_SEEDS:-$SEEDS}" seed
  for seed in $seeds; do
    SEED="$seed" RUN_E4=1 RUN_E5=0 RUN_E6=0 E4_DATA="$DATA" \
      E4_PROJECT="${E4_PROJECT:-runs/experiments/E4_rtdetr_LX}" \
      bash scripts/run_e4_e5_e6.sh
  done
}

ensure_e5_data() {
  E5_DATA="${E5_DATA:-/localnvme/data/billboard/mayolo_v3_multilabel/data.yaml}"
  if [[ ! -f "$E5_DATA" ]]; then
    E5_GENERATED_ROOT="${E5_GENERATED_ROOT:-/localnvme/data/billboard/mayolo_v3_multilabel}"
    py scripts/convert_mdet_to_multilabel.py \
      --data "$DATA" --output "$E5_GENERATED_ROOT" --image-mode symlink --exist-ok
  fi
  require_file "$E5_DATA"
}

merge_csvs() {
  local output="$1"; shift
  "$PYTHON_BIN" - "$output" "$@" <<'PY'
import csv, re, sys
from pathlib import Path
output = Path(sys.argv[1])
rows, fields = [], []
for raw in sys.argv[2:]:
    path = Path(raw)
    if not path.is_file():
        continue
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            match = re.search(r"seed(\d+)", path.name)
            if match:
                row = {"seed": match.group(1), **row}
            rows.append(row)
            for key in row:
                if key not in fields:
                    fields.append(key)
if not rows:
    raise SystemExit(f"no CSV rows found for {output}")
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
print(f"[summary] {output}")
PY
}

run_e5() {
  echo "[E5] 200-epoch YOLOv10x multi-label detection"
  ensure_e5_data
  local project="${E5_PROJECT:-runs/experiments/E5_multilabel_200_seedfix_final}"
  RUN_E4=0 RUN_E5=1 RUN_E6=0 E5_DATA="$E5_DATA" E5_PROJECT="$project" \
    E5_EPOCHS="${E5_EPOCHS:-200}" E5_SEEDS="${E5_SEEDS:-$SEEDS}" E5_SKIP_EXISTING=1 \
    bash scripts/run_e4_e5_e6.sh
  if [[ "${RUN_E5_EVAL:-$RUN_ALL}" == "1" ]]; then
    local summary_dir="$project/test_summary_200" seed weight output
    local -a summaries=()
    for seed in ${E5_SEEDS:-$SEEDS}; do
      weight="$project/yolov10x_seed${seed}/weights/best.pt"; output="$summary_dir/seed${seed}.csv"
      require_file "$weight"
      py scripts/eval_e5_e6.py e5 --device "$DEVICE" --batch "$BATCH" --workers "$WORKERS" \
        --project "$summary_dir" --name "seed${seed}" --summary "$output" \
        --weights "$weight" --data "$E5_DATA"
      summaries+=("$output")
    done
    merge_csvs "$summary_dir/summary.csv" "${summaries[@]}"
  fi
}

run_e6() {
  echo "[E6] two-stage detector plus multi-label classifier"
  E6_DATA="${E6_DATA:-/localnvme/data/billboard/mayolo_v3_two_stage_crops/data.yaml}"
  if [[ ! -f "$E6_DATA" ]]; then
    E6_GENERATED_ROOT="${E6_GENERATED_ROOT:-/localnvme/data/billboard/mayolo_v3_two_stage_crops}"
    py scripts/prepare_two_stage_crops.py --data "$DATA" --output "$E6_GENERATED_ROOT" --exist-ok
  fi
  require_file "$E6_DATA"
  local project="${E6_PROJECT:-runs/experiments/E6_two_stage_yolov10x}"
  RUN_E4=0 RUN_E5=0 RUN_E6=1 E6_DATA="$E6_DATA" E6_PROJECT="$project" \
    E6_SEEDS="${E6_SEEDS:-$SEEDS}" E6_SKIP_EXISTING=1 bash scripts/run_e4_e5_e6.sh
  if [[ "${RUN_E6_EVAL:-$RUN_ALL}" == "1" ]]; then
    local seed classifier detector output seed_project
    local -a summaries=()
    for seed in ${E6_SEEDS:-$SEEDS}; do
      detector="runs/experiments/E3_versions/E3_versions_yolov10x_w4_${W4_TAG}_seed_${seed}_stage2/weights/best.pt"
      if [[ "$seed" == "0" ]]; then seed_project="$project"; else seed_project="${project}_seed${seed}"; fi
      classifier="$seed_project/detector_yolov10x_classifier_yolov10x_cls/weights/best.pt"
      output="$project/test_summary_seed${seed}.csv"
      require_file "$detector"; require_file "$classifier"
      py scripts/eval_e5_e6.py e6 --device "$DEVICE" --batch "$BATCH" --workers "$WORKERS" \
        --project "$project" --name "test_seed${seed}" --summary "$output" \
        --source-data "$DATA" --detector-weights "$detector" --classifier-weights "$classifier"
      summaries+=("$output")
    done
    merge_csvs "$project/summary.csv" "${summaries[@]}"
  fi
}

# -------------------------- robustness / section 6 ----------------------

run_robustness() {
  echo "[5.1] deterministic test-set robustness variants"
  require_file "$DATA"
  local root="${ROBUSTNESS_ROOT:-runs/experiments/E5_robustness/seed0_v1}" generated="$root/data"
  if [[ "${ROBUSTNESS_REGENERATE:-0}" == "1" || ! -f "$generated/manifest.json" ]]; then
    local -a generator=("$PYTHON_BIN" scripts/generate_robustness_variants.py
      --data "$DATA" --output "$generated" --split test --seed 0 --exist-ok)
    [[ "${ROBUSTNESS_INCLUDE_OPTIONAL:-1}" == "1" ]] && generator+=(--include-optional)
    "${generator[@]}"
  fi
  require_file "$generated/manifest.json"
  local e3root="runs/experiments/E3_versions" rtdetrroot="runs/experiments/E4_rtdetr_LX"
  local e2root="runs/experiments/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross"
  local -a models=(
    "YOLOv8x=$e3root/E3_versions_yolov8x_w4_${W4_TAG}_seed_0_stage2/weights/best.pt"
    "YOLOv9e=$e3root/E3_versions_yolov9e_w4_${W4_TAG}_seed_0_stage2/weights/best.pt"
    "YOLOv10x=$e3root/E3_versions_yolov10x_w4_${W4_TAG}_seed_0_stage2/weights/best.pt"
    "YOLOv11x=$e3root/E3_versions_yolov11x_w4_${W4_TAG}_seed_0_stage2/weights/best.pt"
    "YOLOv12x=$e3root/E3_versions_yolov12x_w4_${W4_TAG}_seed_0_stage2/weights/best.pt"
    "YOLOv13x=$e3root/E3_versions_yolov13x_w4_${W4_TAG}_seed_0_stage2/weights/best.pt"
    "YOLO26x=$e3root/E3_versions_yolov26x_w4_${W4_TAG}_seed_0_stage2/weights/best.pt"
    "RT-DETR-x=$rtdetrroot/E4_rtdetr_LX_rtdetr_x_w4_${W4_TAG}_seed_0_stage2/weights/best.pt"
    "MAYOLOx=$e2root/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross_gin_margin_residual_stage1_${STAGE1_EPOCHS}_stage2_${STAGE2_EPOCHS}_w4_${W4_TAG}_seed_0/weights/best.pt::one2many"
  )
  local -a cmd=("$PYTHON_BIN" scripts/eval_robustness.py --data "$DATA"
    --manifest "$generated/manifest.json" --device "${ROBUSTNESS_DEVICE:-$DEVICE}"
    --imgsz "$IMGSZ" --batch "${ROBUSTNESS_BATCH:-4}" --workers 0
    --project "$root" --name seed0_full --seed 0 --resume)
  local model model_spec model_weight
  for model in "${models[@]}"; do
    model_spec="${model%%::one2many}"; model_weight="${model_spec#*=}"
    require_file "$model_weight"; cmd+=(--model "$model")
  done
  "${cmd[@]}"
}

run_section6() {
  echo "[6] per-class, per-attribute and per-level Test metrics"
  require_file "$DATA"
  local yolov10x="runs/experiments/E3_versions/E3_versions_yolov10x_w4_${W4_TAG}_seed_0_stage2/weights/best.pt"
  local mayolox="runs/experiments/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross_gin_margin_residual_stage1_${STAGE1_EPOCHS}_stage2_${STAGE2_EPOCHS}_w4_${W4_TAG}_seed_0/weights/best.pt"
  require_file "$yolov10x"; require_file "$mayolox"
  py scripts/eval_attribute_levels.py --data "$DATA" \
    --model "YOLOv10x=${yolov10x}::native" --model "MAYOLOx=${mayolox}::one2many" \
    --device "$DEVICE" --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS" \
    --project runs/experiments/E3_risk_level_test --name smoke_v4
}

# ------------------------------ summaries --------------------------------

eval_e0() {
  local -a weights=()
  local variant stage
  for variant in current disabled reduced; do
    weights+=("runs/experiments/E0_hsv_ablation/E0_hsv_ablation_${variant}_stage1_${STAGE1_EPOCHS}_w4_${W4_TAG}_seed_0/weights/best.pt")
  done
  eval_mdet_set runs/experiments/E0_hsv_ablation/summary.csv E0_hsv_ablation native "${weights[@]}"

  weights=()
  for stage in stage1_100 stage1_200 stage1_300 stage1_400 stage1_500; do
    weights+=("runs/experiments/E0_stage1_sweep/E0_stage1_sweep_${stage}_w4_${W4_TAG}_seed_0/weights/best.pt")
  done
  eval_mdet_set runs/experiments/E0_stage1_sweep/summary.csv E0_stage1_sweep native "${weights[@]}"

  weights=()
  for stage in stage1_${STAGE1_EPOCHS}_stage2_50 stage1_${STAGE1_EPOCHS}_stage2_100 stage1_${STAGE1_EPOCHS}_stage2_150 stage1_${STAGE1_EPOCHS}_stage2_200; do
    weights+=("runs/experiments/E0_stage2_sweep/E0_stage2_sweep_${stage}_w4_${W4_TAG}_seed_0/weights/best.pt")
  done
  eval_mdet_set runs/experiments/E0_stage2_sweep/summary.csv E0_stage2_sweep native "${weights[@]}"
}

eval_e1() {
  local -a weights=()
  local value
  for value in 0p25 0p5 0p75 1p0 1p25 1p5; do
    weights+=("runs/experiments/E1_w4/E1_w4_base_w4_${value}_seed_0_stage1/weights/best.pt")
    weights+=("runs/experiments/E1_w4/E1_w4_base_w4_${value}_seed_0_stage2/weights/best.pt")
  done
  eval_mdet_set runs/experiments/E1_w4/summary.csv E1_w4 native "${weights[@]}"
}

if [[ "$RUN_E0" == "1" ]]; then
  run_e0
elif [[ "$RUN_E0_EVAL" == "1" ]]; then
  eval_e0
fi
if [[ "$RUN_E1" == "1" ]]; then
  run_e1
elif [[ "$RUN_E1_EVAL" == "1" ]]; then
  eval_e1
fi
if [[ "$RUN_E2" == "1" ]]; then run_e2_training; fi
if [[ "$RUN_E2_EVAL" == "1" ]]; then run_e2_evaluation; fi
if [[ "$RUN_E2_GCA" == "1" ]]; then run_e2_pure_gca; fi
if [[ "$RUN_SECTION2_8" == "1" ]]; then run_section2_8; fi
if [[ "$RUN_SECTION3_PERF" == "1" ]]; then run_section3_performance; fi
if [[ "$RUN_E3" == "1" ]]; then run_e3; fi
if [[ "$RUN_E4" == "1" ]]; then run_e4; fi
if [[ "$RUN_E5" == "1" ]]; then run_e5; fi
if [[ "$RUN_E6" == "1" ]]; then run_e6; fi
if [[ "$RUN_ROBUSTNESS" == "1" ]]; then run_robustness; fi
if [[ "$RUN_SECTION6" == "1" ]]; then run_section6; fi

if [[ "$RUN_ALL" != "1" && "$RUN_E0" == "0" && "$RUN_E0_EVAL" == "0" && "$RUN_E1" == "0" && "$RUN_E1_EVAL" == "0" && "$RUN_E2" == "0" && "$RUN_E2_EVAL" == "0" && "$RUN_E2_GCA" == "0" && "$RUN_SECTION2_8" == "0" && "$RUN_SECTION3_PERF" == "0" && "$RUN_E3" == "0" && "$RUN_E4" == "0" && "$RUN_E5" == "0" && "$RUN_E6" == "0" && "$RUN_ROBUSTNESS" == "0" && "$RUN_SECTION6" == "0" ]]; then
  echo "No experiment selected; nothing was started."
  echo_usage
fi
