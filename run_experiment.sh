#!/usr/bin/env bash
# Reproducibility launcher for scripts/EXPERIMENTS_results.md.
# It is intentionally independent of run.sh: every Python entry point is
# called here directly, and every command is printed before execution.
#
# Examples:
#   bash run_experiment.sh help
#   bash run_experiment.sh e2.0
#   DEVICE=1 bash run_experiment.sh e2.2
#   bash run_experiment.sh e2.8 --dry-run

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA="${DATA:-ultralytics/cfg/mayolo_r1/mayolo_v3.yaml}"
MD_MODEL="${MD_MODEL:-ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml}"
PRETRAIN="${PRETRAIN:-yolov10x.pt}"
COM_CROSS="${COM_CROSS:-/localnvme/data/billboard/mayolo_v3/co_occurrence_matrix_train.csv}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-8}"
IMGSZ="${IMGSZ:-640}"
W4="${W4:-0.5}"
W4_SWEEP_VALUES="${W4_SWEEP_VALUES:-0 0.25 0.5 0.75 1.0}"
W4_SWEEP_ROOT="${W4_SWEEP_ROOT:-runs/experiments/E1_w4_5seed}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-100}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-100}"
SEEDS="${SEEDS:-0 1 2 3 4}"
DRY_RUN="${DRY_RUN:-0}"

CODE="${1:-help}"
shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --device) DEVICE="$2"; shift ;;
    --seeds) SEEDS="$2"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

read -r -a SEED_LIST <<< "$SEEDS"
W4_TAG="${W4//./p}"

E2_ROOT="${E2_ROOT:-runs/experiments/E2_27_baseline_gia_seed5}"
E2_LABEL="${E2_LABEL:-E2_27_baseline_gia_seed5}"
BASE_STAGE1_PREFIX="${E2_LABEL}_baseline_w4_${W4_TAG}_seed_"
GIA_STAGE1_PREFIX="${E2_LABEL}_gia_v2_5_7_w4_${W4_TAG}_seed_"
PURE_GCA_CONFIG="${PURE_GCA_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GCA_margin_residual.yaml}"
GIA_GCA_CONFIG="${GIA_GCA_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_5_7_GCA_margin_residual.yaml}"
GIA_CONFIG="${GIA_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_5_7.yaml}"

GNN_TYPES=(gcn gat graphsage gin)
GIA_POSITION_VARIANTS=(
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

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

usage() {
  cat <<'EOF'
Usage: bash run_experiment.sh CODE [options]

Codes:
  preflight  Check repository inputs and paths.
  e1         E1 w4 validation scan.
  e1.w4-5seed  E1 w4 sweep with five seeds (0, 0.25, 0.5, 0.75, 1.0).
  e2.0       Baseline/GIA five-seed Stage1+Stage2 training.
  e2.1       GIA-v2 position Stage1 sweep.
  e2.2       Pure Baseline+GCA, 4 graph operators x 5 seeds.
  e2.3       Baseline HO Test evaluation.
  e2.4       GIA HO Test evaluation.
  e2.5       GIA+GCA Stage2, 4 graph operators x 5 seeds.
  e2.6       Baseline+GCA HO Test evaluation.
  e2.7       GIA+GCA HO Test evaluation (MAYOLO).
  e2.8       E2 performance profile for eight ablation models x five seeds.
  e3.train   Native YOLO/YOLO26 and MAYOLO size training.
  e3.eval    E3/E4 five-seed performance profile.
  e4         RT-DETR-L/X five-seed training.
  e5         200-epoch multi-label detector training and Test inference.
  e5.eval    Test inference only for existing E5 checkpoints.
  e6         Two-stage detector + multi-label classifier training/Test inference.
  e6.eval    Test inference only for existing E6 checkpoints.
  e5.1       Offline robustness-variant generation and evaluation.
  e6.1       Per-attribute/level metrics, calibration and confusion matrices.
  all        Run automated sections in dependency order.

Options:
  --dry-run              Print commands without executing them.
  --device CUDA_DEVICE   Override DEVICE (default: 0).
  --seeds "0 1 2"        Override the seed list.

Common environment overrides: PYTHON_BIN, DATA, PRETRAIN, COM_CROSS, BATCH,
WORKERS, IMGSZ, W4_SWEEP_VALUES, W4_SWEEP_ROOT, STAGE1_EPOCHS, STAGE2_EPOCHS, E2_PERFORMANCE_ROOT,
E2_PERF_SEEDS, E3_PERF_ROOT and E3_PERF_SEEDS.
EOF
}

run() {
  printf "\n[command]"
  printf " %q" "$@"
  printf "\n"
  if [[ "$DRY_RUN" != "1" ]]; then "$@"; fi
}

py() { run "$PYTHON_BIN" "$@"; }

need() {
  [[ "$DRY_RUN" == "1" || -f "$1" ]] || {
    echo "Missing required file: $1" >&2
    exit 1
  }
}

stage1_weight() {
  local prefix="$1" seed="$2"
  echo "$E2_ROOT/${prefix}${seed}_stage1/weights/best.pt"
}

stage2_plain_weight() {
  local prefix="$1" seed="$2"
  echo "$E2_ROOT/${prefix}${seed}_stage2/weights/best.pt"
}

stage2_weight() {
  local root="$1" label="$2" operator="$3" seed="$4"
  echo "$root/${label}_${operator}_margin_residual_stage1_${STAGE1_EPOCHS}_stage2_${STAGE2_EPOCHS}_w4_${W4_TAG}_seed_${seed}/weights/best.pt"
}

preflight() {
  echo "[preflight] repository=$ROOT"
  echo "[preflight] python=$PYTHON_BIN device=$DEVICE batch=$BATCH workers=$WORKERS"
  echo "[preflight] seeds=$SEEDS stage=${STAGE1_EPOCHS}+${STAGE2_EPOCHS} w4=$W4"
  need "$DATA"; need "$MD_MODEL"; need "$COM_CROSS"
  echo "[preflight] inputs are present"
}

e1() {
  echo "[E1] w4 validation scan"
  need "$DATA"; need "$MD_MODEL"
  py scripts/train_mdet_experiments.py w4 \
    --data "$DATA" --model "$MD_MODEL" --pretrain "$PRETRAIN" \
    --w4-values 0.25 0.5 0.75 1.0 1.25 1.5 \
    --stage1-epochs "$STAGE1_EPOCHS" --stage2-epochs "$STAGE2_EPOCHS" \
    --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS" --device "$DEVICE" \
    --project runs/experiments/E1_w4 --label E1_w4 --skip-existing
}

e1_w4_5seed() {
  echo "[E1] w4 five-seed sweep: values=${W4_SWEEP_VALUES}, seeds=${SEEDS}"
  need "$DATA"; need "$MD_MODEL"; need "$PRETRAIN"
  local -a w4_values=()
  read -r -a w4_values <<< "$W4_SWEEP_VALUES"
  if [[ "${#w4_values[@]}" -eq 0 ]]; then
    echo "W4_SWEEP_VALUES must contain at least one value" >&2
    exit 2
  fi
  py scripts/train_mdet_experiments.py w4-seeds \
    --data "$DATA" --model "$MD_MODEL" --pretrain "$PRETRAIN" \
    --w4-values "${w4_values[@]}" --seeds "${SEED_LIST[@]}" \
    --stage1-epochs "$STAGE1_EPOCHS" --stage2-epochs "$STAGE2_EPOCHS" \
    --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS" --device "$DEVICE" \
    --project "$W4_SWEEP_ROOT" --label E1_w4_5seed --skip-existing
}

e20() {
  echo "[E2.0] baseline and GIA five-seed Stage1+Stage2"
  need "$DATA"; need "$MD_MODEL"; need "$GIA_CONFIG"
  local seed
  for seed in "${SEED_LIST[@]}"; do
    py scripts/train_mdet_experiments.py variants \
      --data "$DATA" --project "$E2_ROOT" --imgsz "$IMGSZ" --batch "$BATCH" \
      --workers "$WORKERS" --device "$DEVICE" --seed "$seed" --w4 "$W4" \
      --stage1-epochs "$STAGE1_EPOCHS" --stage2-epochs "$STAGE2_EPOCHS" \
      --label "$E2_LABEL" --variant "baseline=$MD_MODEL" \
      --variant "gia_v2_5_7=$GIA_CONFIG" --pretrain "$PRETRAIN" --skip-existing
  done
}

e21() {
  echo "[E2.1] GIA-v2 position Stage1 sweep"
  need "$DATA"
  local seed variant
  for seed in "${SEED_LIST[@]}"; do
    local -a cmd=(
      "$PYTHON_BIN" scripts/train_mdet_experiments.py gia-position
      --data "$DATA" --project runs/experiments/E2_1_GIA_v2_position
      --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS" --device "$DEVICE"
      --seed "$seed" --w4 "$W4"
      --stage1-epochs "$STAGE1_EPOCHS" --stage2-epochs "$STAGE2_EPOCHS"
      --label E2_1_GIA_v2_position --pretrain "$PRETRAIN" --stage1-only --skip-existing
    )
    for variant in "${GIA_POSITION_VARIANTS[@]}"; do cmd+=(--variant "$variant"); done
    run "${cmd[@]}"
  done
}

gca_train() {
  local label="$1" root="$2" config="$3" matrix_path="$4" prefix="$5"
  need "$DATA"; need "$config"; need "$matrix_path"
  local -a cmd=(
    "$PYTHON_BIN" scripts/train_mdet_experiments.py gca-stage2-seeds
    --label "$label" --data "$DATA" --project "$root"
    --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS" --device "$DEVICE"
    --w4 "$W4"
    --stage1-epochs "$STAGE1_EPOCHS" --stage2-epochs "$STAGE2_EPOCHS"
    --variant "margin_residual=$config" --gnn-types "${GNN_TYPES[@]}"
    --seeds "${SEED_LIST[@]}" --com-path "$matrix_path" --skip-existing
  )
  local seed checkpoint
  for seed in "${SEED_LIST[@]}"; do
    checkpoint="$(stage1_weight "$prefix" "$seed")"
    need "$checkpoint"
    cmd+=(--stage1-checkpoint-map "${seed}=${checkpoint}")
  done
  run "${cmd[@]}"
}

e22() {
  echo "[E2.2] pure Baseline+GCA: Cross matrix and four graph operators"
  gca_train E2_29_Baseline_GCA_pure_margin_residual_5seed_cross \
    runs/experiments/E2_29_Baseline_GCA_pure_margin_residual_5seed_cross \
    "$PURE_GCA_CONFIG" "$COM_CROSS" "$BASE_STAGE1_PREFIX"
}

e25() {
  echo "[E2.5] GIA+GCA Stage2: Cross matrix and four graph operators"
  gca_train E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross \
    runs/experiments/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross \
    "$GIA_GCA_CONFIG" "$COM_CROSS" "$GIA_STAGE1_PREFIX"
}

eval_ho() {
  local summary="$1" name="$2" mode="$3"
  shift 3
  local parent="${summary%/*}"
  local -a cmd=(
    "$PYTHON_BIN" scripts/eval_mdet_experiments.py ho
    --data "$DATA" --mode "$mode" --split test --device "$DEVICE"
    --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS"
    --project "$parent" --name "$name" --summary "$summary" --no-plots
  )
  local weight
  for weight in "$@"; do need "$weight"; cmd+=(--weights "$weight"); done
  run "${cmd[@]}"
}

collect_stage2() {
  local prefix="$1"
  COLLECTED=()
  local seed
  for seed in "${SEED_LIST[@]}"; do
    COLLECTED+=("$E2_ROOT/${prefix}${seed}_stage2/weights/best.pt")
  done
}

collect_gca() {
  local root="$1" label="$2"
  COLLECTED=()
  local operator seed
  for operator in "${GNN_TYPES[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
      COLLECTED+=("$(stage2_weight "$root" "$label" "$operator" "$seed")")
    done
  done
}

e23() {
  echo "[E2.3] Baseline HO Test evaluation"
  collect_stage2 "$BASE_STAGE1_PREFIX"
  eval_ho runs/experiments/E2_27_HO/summary.csv E2_27_HO one2many "${COLLECTED[@]}"
}

e24() {
  echo "[E2.4] GIA HO Test evaluation"
  collect_stage2 "$GIA_STAGE1_PREFIX"
  eval_ho runs/experiments/E2_27_GIA_v2_5_7_stage2_recheck/one2many_summary.csv \
    E2_27_GIA_v2_5_7_stage2_recheck one2many "${COLLECTED[@]}"
}

e26() {
  echo "[E2.6] Baseline+GCA HO Test evaluation"
  local root="runs/experiments/E2_29_Baseline_GCA_pure_margin_residual_5seed_cross"
  local label="E2_29_Baseline_GCA_pure_margin_residual_5seed_cross"
  collect_gca "$root" "$label"
  eval_ho "$root/ho_summary.csv" "E2_6_Baseline_GCA_cross_HO" one2many \
    "${COLLECTED[@]}"
}

e27() {
  echo "[E2.7] GIA+GCA HO Test evaluation (MAYOLO)"
  local root="runs/experiments/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross"
  local label="E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross"
  collect_gca "$root" "$label"
  eval_ho "$root/ho_summary.csv" "E2_7_MAYOLO_cross_HO" one2many \
    "${COLLECTED[@]}"
}

e28() {
  echo "[E2.8] ablation/performance profile for eight models x five seeds"
  local root="${E2_PERFORMANCE_ROOT:-runs/experiments/performance_profile/E2_8_5seed_repro}"
  local perf_seeds="${E2_PERF_SEEDS:-0 1 2 3 4}"
  read -r -a PERF_SEED_LIST <<< "$perf_seeds"
  local seed baseline gia gca gia_gca
  for seed in "${PERF_SEED_LIST[@]}"; do
    baseline="$(stage2_plain_weight "$BASE_STAGE1_PREFIX" "$seed")"
    gia="$(stage2_plain_weight "$GIA_STAGE1_PREFIX" "$seed")"
    gca="$(stage2_weight runs/experiments/E2_29_Baseline_GCA_pure_margin_residual_5seed_cross \
      E2_29_Baseline_GCA_pure_margin_residual_5seed_cross gin "$seed")"
    gia_gca="$(stage2_weight runs/experiments/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross \
      E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross gin "$seed")"
    need "$baseline"; need "$gia"; need "$gca"; need "$gia_gca"

    run "$PYTHON_BIN" scripts/benchmark_performance.py \
      --weights "$baseline" "$gia" "$gca" "$gia_gca" \
      --labels Baseline GIA GCA-GIN GIA+GCA-GIN \
      --network yolo --task mdetect --head-mode native --device "$DEVICE" \
      --precision "${PERF_PRECISION:-fp16}" --batch "${PERF_BATCH:-1}" --imgsz "$IMGSZ" \
      --warmup "${PERF_WARMUP:-50}" --iterations "${PERF_ITERATIONS:-200}" \
      --output "$root/seed${seed}_native.csv"

    run "$PYTHON_BIN" scripts/benchmark_performance.py \
      --weights "$baseline" "$gia" "$gca" "$gia_gca" \
      --labels HO GIA+HO GCA-GIN+HO MAYOLOx \
      --network yolo --task mdetect --head-mode one2many --device "$DEVICE" \
      --precision "${PERF_PRECISION:-fp16}" --batch "${PERF_BATCH:-1}" --imgsz "$IMGSZ" \
      --warmup "${PERF_WARMUP:-50}" --iterations "${PERF_ITERATIONS:-200}" \
      --output "$root/seed${seed}_one2many.csv"
  done
}

e3_sizes() {
  case "$1" in
    yolov8) echo "n s m l x" ;;
    yolov9) echo "t s m c e" ;;
    yolov10) echo "n s m b l x" ;;
    yolov11|yolov12) echo "n s m l x" ;;
    yolov13) echo "n s l x" ;;
    yolov26) echo "n s m l x" ;;
    *) echo "Unknown model family: $1" >&2; exit 2 ;;
  esac
}

e3_pretrain() {
  local family="$1" size="$2"
  case "$family" in
    yolov11|yolov12) echo "yolo${family#yolov}${size}.pt" ;;
    yolov13) echo "yolov13${size}.pt" ;;
    yolov26) echo "yolo26${size}.pt" ;;
    *) echo "${family}${size}.pt" ;;
  esac
}

e3_train_native() {
  local family size name config pretrain seed
  local e3_seeds="${E3_SEEDS:-0 1 2 3 4}"
  read -r -a E3_SEED_LIST <<< "$e3_seeds"
  for seed in "${E3_SEED_LIST[@]}"; do
    for family in yolov8 yolov9 yolov10 yolov11 yolov12 yolov13 yolov26; do
      for size in $(e3_sizes "$family"); do
        name="${family}${size}"
        config="ultralytics/cfg/models/experiments/${name}-mdetect.yaml"
        pretrain="$(e3_pretrain "$family" "$size")"
        need "$config"
        py scripts/train_mdet_experiments.py versions \
          --data "$DATA" --project runs/experiments/E3_versions \
          --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS" --device "$DEVICE" \
          --seed "$seed" --w4 "$W4" \
          --stage1-epochs "$STAGE1_EPOCHS" --stage2-epochs "$STAGE2_EPOCHS" \
          --label E3_versions --variant "$name=$config" --pretrain "$pretrain" \
          --skip-existing
      done
    done
  done
}

e3_train_mayolo() {
  local e3_seeds="${E3_SEEDS:-0 1 2 3 4}" seed
  read -r -a E3_SEED_LIST <<< "$e3_seeds"
  for seed in "${E3_SEED_LIST[@]}"; do
    py scripts/train_mayolo_final_sizes.py \
      --sizes ${MAYOLO_SIZES:-n s m l b x} --device "$DEVICE" --data "$DATA" \
      --project runs/experiments/E3_versions --label E3_MAYOLO_final \
      --stage1-epochs "$STAGE1_EPOCHS" --stage2-epochs "$STAGE2_EPOCHS" \
      --batch "$BATCH" --imgsz "$IMGSZ" --workers "$WORKERS" --w4 "$W4" --seed "$seed" \
      --com-path "$COM_CROSS" --skip-existing
  done
}

e3_train() {
  echo "[E3] native model/size and MAYOLO size training"
  need "$DATA"
  [[ "${E3_NATIVE:-1}" == "1" ]] && e3_train_native
  [[ "${E3_MAYOLO:-1}" == "1" ]] && e3_train_mayolo
}

e3_perf() {
  echo "[E3/E4] five-seed performance profile"
  local perf_seeds="${E3_PERF_SEEDS:-0 1 2 3 4}"
  read -r -a PERF_SEED_LIST <<< "$perf_seeds"
  local root="${E3_PERF_ROOT:-runs/experiments/performance_profile/E3_E4_5seed}"
  local seed family size name weight
  local -a native_weights=() native_labels=()
  local -a mayolo_weights=() mayolo_labels=()
  local -a rtdetr_weights=()
  local rtdetr_root="runs/experiments/E4_rtdetr_LX"

  for seed in "${PERF_SEED_LIST[@]}"; do
    echo "[E3/E4] performance seed=${seed}"
    native_weights=()
    native_labels=()
    mayolo_weights=()
    mayolo_labels=()
    rtdetr_weights=()

    for family in yolov8 yolov9 yolov10 yolov11 yolov12 yolov13 yolov26; do
      for size in $(e3_sizes "$family"); do
        name="${family}${size}"
        weight="runs/experiments/E3_versions/E3_versions_${name}_w4_${W4_TAG}_seed_${seed}_stage2/weights/best.pt"
        need "$weight"
        native_weights+=("$weight"); native_labels+=("YOLO${family#yolo}${size}")
      done
    done
    run "$PYTHON_BIN" scripts/benchmark_performance.py \
      --weights "${native_weights[@]}" --labels "${native_labels[@]}" \
      --network yolo --task mdetect --head-mode native --device "$DEVICE" \
      --precision "${PERF_PRECISION:-fp16}" --batch "${PERF_BATCH:-1}" --imgsz "$IMGSZ" \
      --warmup "${PERF_WARMUP:-50}" --iterations "${PERF_ITERATIONS:-200}" \
      --output "$root/E3_seed${seed}_native.csv"

    for size in ${MAYOLO_SIZES:-n s m l b}; do
      weight="runs/experiments/E3_versions/E3_MAYOLO_final_mayolo${size}_w4_${W4_TAG}_seed_${seed}_stage2/weights/best.pt"
      need "$weight"
      mayolo_weights+=("$weight"); mayolo_labels+=("MAYOLO$size")
    done
    weight="runs/experiments/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross_gin_margin_residual_stage1_${STAGE1_EPOCHS}_stage2_${STAGE2_EPOCHS}_w4_${W4_TAG}_seed_${seed}/weights/best.pt"
    need "$weight"
    mayolo_weights+=("$weight"); mayolo_labels+=("MAYOLOx")
    run "$PYTHON_BIN" scripts/benchmark_performance.py \
      --weights "${mayolo_weights[@]}" --labels "${mayolo_labels[@]}" \
      --network yolo --task mdetect --head-mode one2many --device "$DEVICE" \
      --precision "${PERF_PRECISION:-fp16}" --batch "${PERF_BATCH:-1}" --imgsz "$IMGSZ" \
      --warmup "${PERF_WARMUP:-50}" --iterations "${PERF_ITERATIONS:-200}" \
      --output "$root/E3_seed${seed}_mayolo.csv"

    rtdetr_weights=(
      "$rtdetr_root/E4_rtdetr_LX_rtdetr_l_w4_${W4_TAG}_seed_${seed}_stage2/weights/best.pt"
      "$rtdetr_root/E4_rtdetr_LX_rtdetr_x_w4_${W4_TAG}_seed_${seed}_stage2/weights/best.pt"
    )
    for weight in "${rtdetr_weights[@]}"; do need "$weight"; done
    run "$PYTHON_BIN" scripts/benchmark_performance.py \
      --weights "${rtdetr_weights[@]}" --labels RT-DETR-L RT-DETR-X \
      --network rtdetr --head-mode native --device "$DEVICE" \
      --precision "${PERF_PRECISION:-fp16}" --batch "${PERF_BATCH:-1}" --imgsz "$IMGSZ" \
      --warmup "${PERF_WARMUP:-50}" --iterations "${PERF_ITERATIONS:-200}" \
      --output "$root/E4_seed${seed}_rtdetr.csv"
  done
}

e3_eval() {
  echo "[E3/E4] five-seed performance profile"
  e3_perf
}

e4() {
  echo "[E4] RT-DETR-L/X five-seed training"
  need "$DATA"
  local seed
  for seed in "${SEED_LIST[@]}"; do
    py scripts/train_mdet_experiments.py rtdetr \
      --label E4_rtdetr_LX --data "$DATA" --project runs/experiments/E4_rtdetr_LX \
      --variant rtdetr_l=ultralytics/cfg/models/rt-detr/rtdetr-l-md.yaml \
      --variant rtdetr_x=ultralytics/cfg/models/rt-detr/rtdetr-x.yaml \
      --pretrain-map rtdetr_l=rtdetr-l.pt --pretrain-map rtdetr_x=rtdetr-x.pt \
      --stage1-epochs "$STAGE1_EPOCHS" --stage2-epochs "$STAGE2_EPOCHS" \
      --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS" --device "$DEVICE" \
      --seed "$seed" --w4 "$W4" --skip-existing
  done
}

e5() {
  echo "[E5] 200-epoch YOLOv10x multi-label detector"
  local data="${E5_DATA:-/localnvme/data/billboard/mayolo_v3_multilabel/data.yaml}"
  local generated="${E5_GENERATED_ROOT:-/localnvme/data/billboard/mayolo_v3_multilabel}"
  local model="${E5_MODEL:-yolov10x.pt}"
  local project="${E5_PROJECT:-runs/experiments/E5_multilabel_200_seedfix_final}"
  local epochs="${E5_EPOCHS:-200}" seed
  if [[ ! -f "$data" ]]; then
    py scripts/convert_mdet_to_multilabel.py --data "$DATA" --output "$generated" \
      --image-mode symlink --exist-ok
  fi
  need "$data"; need "$model"
  for seed in "${SEED_LIST[@]}"; do
    if [[ "${E5_SKIP_EXISTING:-1}" == "1" && -f "$project/yolov10x_seed$seed/weights/best.pt" ]]; then
      echo "[skip] $project/yolov10x_seed$seed"
      continue
    fi
    run "$PYTHON_BIN" scripts/train_multilabel.py \
      --model "$model" --data "$data" --epochs "$epochs" --imgsz "$IMGSZ" \
      --batch "$BATCH" --workers "$WORKERS" --device "$DEVICE" \
      --project "$project" --name "yolov10x_seed$seed" --seed "$seed"
  done
}

e6() {
  echo "[E6] two-stage detector plus multi-label classifier"
  local data="${E6_DATA:-/localnvme/data/billboard/mayolo_v3_two_stage_crops/data.yaml}"
  local generated="${E6_GENERATED_ROOT:-/localnvme/data/billboard/mayolo_v3_two_stage_crops}"
  local model="${E6_MODEL:-ultralytics/cfg/models/v10/yolov10x-cls.yaml}"
  local pretrain="${E6_PRETRAIN:-yolov10x.pt}"
  local project="${E6_PROJECT:-runs/experiments/E6_two_stage_yolov10x}"
  local detector_root="${E6_DETECTOR_ROOT:-runs/experiments/E3_versions}"
  local detector_prefix="${E6_DETECTOR_PREFIX:-E3_versions_yolov10x_w4_${W4_TAG}_seed_}"
  local seed seed_project detector
  if [[ ! -f "$data" ]]; then
    py scripts/prepare_two_stage_crops.py --data "$DATA" --output "$generated" --exist-ok
  fi
  need "$data"; need "$model"; need "$pretrain"
  for seed in "${SEED_LIST[@]}"; do
    detector="$detector_root/${detector_prefix}${seed}_stage2/weights/best.pt"
    need "$detector"
    if [[ "$seed" == "0" ]]; then seed_project="$project"; else seed_project="${project}_matched_seed${seed}"; fi
    if [[ "${E6_SKIP_EXISTING:-1}" == "1" && -f "$seed_project/${E6_NAME:-detector_yolov10x_classifier_yolov10x_cls}/weights/best.pt" ]]; then
      echo "[skip] $seed_project"
      continue
    fi
    run "$PYTHON_BIN" scripts/train_two_stage.py \
      --detector-checkpoint "$detector" --model "$model" --pretrain "$pretrain" \
      --data "$data" --epochs "${E6_EPOCHS:-100}" --imgsz 224 --batch "$BATCH" \
      --workers "$WORKERS" --device "$DEVICE" --project "$seed_project" \
      --name "${E6_NAME:-detector_yolov10x_classifier_yolov10x_cls}" --seed "$seed"
  done
}

e5_eval() {
  echo "[E5] Test inference for each seed"
  local data="${E5_DATA:-/localnvme/data/billboard/mayolo_v3_multilabel/data.yaml}"
  local project="${E5_PROJECT:-runs/experiments/E5_multilabel_200_seedfix_final}"
  local seed weight
  need "$data"
  for seed in "${SEED_LIST[@]}"; do
    weight="$project/yolov10x_seed$seed/weights/best.pt"
    need "$weight"
    run "$PYTHON_BIN" scripts/eval_e5_e6.py e5 --device "$DEVICE" --batch "$BATCH" \
      --workers "$WORKERS" --imgsz "$IMGSZ" --project "$project/test_summary_200" \
      --name "seed$seed" --summary "$project/test_summary_200/seed$seed.csv" \
      --weights "$weight" --data "$data"
  done
}

e6_eval() {
  echo "[E6] Test inference for each matched seed"
  local source_data="$DATA"
  local project="${E6_PROJECT:-runs/experiments/E6_two_stage_yolov10x}"
  local detector_root="${E6_DETECTOR_ROOT:-runs/experiments/E3_versions}"
  local detector_prefix="${E6_DETECTOR_PREFIX:-E3_versions_yolov10x_w4_${W4_TAG}_seed_}"
  local name="${E6_NAME:-detector_yolov10x_classifier_yolov10x_cls}"
  local seed detector seed_project classifier
  for seed in "${SEED_LIST[@]}"; do
    detector="$detector_root/${detector_prefix}${seed}_stage2/weights/best.pt"
    if [[ "$seed" == "0" ]]; then seed_project="$project"; else seed_project="${project}_matched_seed${seed}"; fi
    classifier="$seed_project/$name/weights/best.pt"
    need "$detector"; need "$classifier"
    run "$PYTHON_BIN" scripts/eval_e5_e6.py e6 --device "$DEVICE" --batch "$BATCH" \
      --workers "$WORKERS" --imgsz "$IMGSZ" --project "$project" \
      --name "test_seed$seed" --summary "$project/test_summary_seed$seed.csv" \
      --source-data "$source_data" --detector-weights "$detector" \
      --classifier-weights "$classifier"
  done
}

e51() {
  echo "[E5.1] deterministic test-set robustness evaluation"
  local root="${ROBUSTNESS_ROOT:-runs/experiments/E5_robustness/seed0_v1}"
  local generated="$root/data"
  if [[ ! -f "$generated/manifest.json" ]]; then
    py scripts/generate_robustness_variants.py --data "$DATA" --output "$generated" \
      --split test --seed 0 --include-optional --exist-ok
  fi
  need "$generated/manifest.json"

  local e3root="runs/experiments/E3_versions"
  local rtdetrroot="runs/experiments/E4_rtdetr_LX"
  local mayolox="runs/experiments/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross_gin_margin_residual_stage1_${STAGE1_EPOCHS}_stage2_${STAGE2_EPOCHS}_w4_${W4_TAG}_seed_0/weights/best.pt"
  local -a specs=()
  local label weight
  for label in YOLOv8x YOLOv9e YOLOv10x YOLOv11x YOLOv12x YOLOv13x YOLO26x; do
    case "$label" in
      YOLOv8x) weight="$e3root/E3_versions_yolov8x_w4_${W4_TAG}_seed_0_stage2/weights/best.pt" ;;
      YOLOv9e) weight="$e3root/E3_versions_yolov9e_w4_${W4_TAG}_seed_0_stage2/weights/best.pt" ;;
      YOLOv10x) weight="$e3root/E3_versions_yolov10x_w4_${W4_TAG}_seed_0_stage2/weights/best.pt" ;;
      YOLOv11x) weight="$e3root/E3_versions_yolov11x_w4_${W4_TAG}_seed_0_stage2/weights/best.pt" ;;
      YOLOv12x) weight="$e3root/E3_versions_yolov12x_w4_${W4_TAG}_seed_0_stage2/weights/best.pt" ;;
      YOLOv13x) weight="$e3root/E3_versions_yolov13x_w4_${W4_TAG}_seed_0_stage2/weights/best.pt" ;;
      YOLO26x) weight="$e3root/E3_versions_yolov26x_w4_${W4_TAG}_seed_0_stage2/weights/best.pt" ;;
    esac
    need "$weight"; specs+=(--model "$label=$weight")
  done
  weight="$rtdetrroot/E4_rtdetr_LX_rtdetr_x_w4_${W4_TAG}_seed_0_stage2/weights/best.pt"
  need "$weight"; specs+=(--model "RT-DETR-x=$weight")
  need "$mayolox"; specs+=(--model "MAYOLOx=$mayolox::one2many")
  run "$PYTHON_BIN" scripts/eval_robustness.py --data "$DATA" \
    --manifest "$generated/manifest.json" --device "$DEVICE" --imgsz "$IMGSZ" \
    --batch "${ROBUSTNESS_BATCH:-4}" --workers 0 --project "$root" --name seed0_full \
    --seed 0 --resume "${specs[@]}"
}

e61() {
  echo "[E6.1] per-attribute/level metrics, calibration and confusion matrices"
  local yolov10x="runs/experiments/E3_versions/E3_versions_yolov10x_w4_${W4_TAG}_seed_0_stage2/weights/best.pt"
  local mayolox="runs/experiments/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross_gin_margin_residual_stage1_${STAGE1_EPOCHS}_stage2_${STAGE2_EPOCHS}_w4_${W4_TAG}_seed_0/weights/best.pt"
  need "$yolov10x"; need "$mayolox"
  local project="runs/experiments/E3_risk_level_test"
  run "$PYTHON_BIN" scripts/eval_attribute_levels.py --data "$DATA" \
    --model "YOLOv10x=$yolov10x::native" --model "MAYOLOx=$mayolox::one2many" \
    --device "$DEVICE" --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS" \
    --project "$project" --name comparison
  run "$PYTHON_BIN" scripts/eval_attribute_calibration.py --data "$DATA" \
    --model "YOLOv10x=$yolov10x::native" --model "MAYOLOx=$mayolox::one2many" \
    --device "$DEVICE" --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS" \
    --project runs/experiments/E3_attribute_quality --name mayolox_vs_yolov10x
  run "$PYTHON_BIN" scripts/plot_attribute_confusion_matrices.py \
    --confusion "$project/comparison/confusion_test.csv" --data "$DATA" \
    --models YOLOv10x MAYOLOx --head-mode all \
    --output "$project/comparison/confusion_figures"
}

all_experiments() {
  preflight
  e20
  e22
  e23
  e24
  e25
  e26
  e27
  e28
  e3_train
  e4
  e5
  e5_eval
  e6
  e6_eval
  e3_eval
  e51
  e61
}

case "$CODE" in
  help|-h|--help) usage ;;
  preflight) preflight ;;
  e1) e1 ;;
  e1.w4-5seed) e1_w4_5seed ;;
  e2.0) e20 ;;
  e2.1) e21 ;;
  e2.2) e22 ;;
  e2.3) e23 ;;
  e2.4) e24 ;;
  e2.5) e25 ;;
  e2.6) e26 ;;
  e2.7) e27 ;;
  e2.8) e28 ;;
  e3.train) e3_train ;;
  e3.eval) e3_eval ;;
  e4) e4 ;;
  e5) e5 ;;
  e5.eval) e5_eval ;;
  e6) e6 ;;
  e6.eval) e6_eval ;;
  e5.1) e51 ;;
  e6.1) e61 ;;
  all) all_experiments ;;
  *) echo "Unknown experiment code: $CODE" >&2; usage; exit 2 ;;
esac
