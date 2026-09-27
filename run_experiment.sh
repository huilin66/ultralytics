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
#   bash run_experiment.sh eigencam --device 0

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
STAGE1_EPOCHS="${STAGE1_EPOCHS:-100}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-100}"
SEEDS="${SEEDS:-0 1 2 3 4}"
FINE_SEED="${FINE_SEED:-1}"
DRY_RUN="${DRY_RUN:-0}"
WEIGHTS_MANIFEST="${WEIGHTS_MANIFEST:-}"
EIGENCAM_MAYOLO_WEIGHT="${EIGENCAM_MAYOLO_WEIGHT:-}"
EIGENCAM_YOLOV10_WEIGHT="${EIGENCAM_YOLOV10_WEIGHT:-}"
EIGENCAM_IMAGES="${EIGENCAM_IMAGES:-/localnvme/data/billboard/mayolo_v3/heatmap_demo}"
EIGENCAM_IMAGE_NAMES="${EIGENCAM_IMAGE_NAMES:-}"
EIGENCAM_OUTPUT="${EIGENCAM_OUTPUT:-runs/experiments/E3_final_test/heatmaps_eigencam}"
EIGENCAM_LAYER="${EIGENCAM_LAYER:-22}"
EIGENCAM_CONF="${EIGENCAM_CONF:-0.5}"
EIGENCAM_IOU="${EIGENCAM_IOU:-0.7}"
EIGENCAM_METHODS="${EIGENCAM_METHODS:-EigenCAM}"

CODE="${1:-help}"
shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --device) DEVICE="$2"; shift ;;
    --seeds) SEEDS="$2"; shift ;;
    --fine-seed) FINE_SEED="$2"; shift ;;
    --weights-manifest) WEIGHTS_MANIFEST="$2"; shift ;;
    --mayolo-weight) EIGENCAM_MAYOLO_WEIGHT="$2"; shift ;;
    --yolov10-weight) EIGENCAM_YOLOV10_WEIGHT="$2"; shift ;;
    --images|--image|--image-path) EIGENCAM_IMAGES="$2"; shift ;;
    --image-names)
      shift
      EIGENCAM_IMAGE_NAMES=""
      while [[ $# -gt 0 && "$1" != --* ]]; do
        if [[ -n "$EIGENCAM_IMAGE_NAMES" ]]; then
          EIGENCAM_IMAGE_NAMES+=" "
        fi
        EIGENCAM_IMAGE_NAMES+="$1"
        shift
      done
      continue
      ;;
    --output) EIGENCAM_OUTPUT="$2"; shift ;;
    --layer) EIGENCAM_LAYER="$2"; shift ;;
    --iou) EIGENCAM_IOU="$2"; shift ;;
    --all-methods) EIGENCAM_METHODS="GradCAM GradCAMPlusPlus XGradCAM EigenCAM HiResCAM LayerCAM RandomCAM EigenGradCAM" ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

read -r -a SEED_LIST <<< "$SEEDS"
W4_TAG="${W4//./p}"

E2_OUTPUT_ROOT="${E2_OUTPUT_ROOT:-runs/experiments/E2_27_baseline_gia_seed5}"
E2_LABEL="${E2_LABEL:-E2_27_baseline_gia_seed5}"
PURE_GCA_CONFIG="${PURE_GCA_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GCA_margin_residual.yaml}"
GIA_GCA_CONFIG="${GIA_GCA_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_5_7_GCA_margin_residual.yaml}"
GIA_CONFIG="${GIA_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_5_7.yaml}"

GNN_TYPES=(gcn gat graphsage gin)
GIA_POSITION_VARIANTS=(
  "gia_6=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_6.yaml"
  "gia_7=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_7.yaml"
  "gia_8=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_8.yaml"
  "gia_9=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_9.yaml"
  "gia_10=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_10.yaml"
  "gia_13=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_13.yaml"
  "gia_16=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_16.yaml"
  "gia_19=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_19.yaml"
  "gia_22=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_22.yaml"
  "gia=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_5_7.yaml"
)

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

usage() {
  cat <<'EOF'
Usage: bash run_experiment.sh CODE [options]

EigenCAM requires explicit checkpoints:
  bash run_experiment.sh eigencam --device 0 --mayolo-weight PATH --yolov10-weight PATH

Codes:
  preflight  Check repository inputs and paths.
  e1         E1 w4 validation scan.
  e2.0       Baseline/GIA five-seed Stage1+Stage2 training.
  e2.1       GIA position Stage1+Stage2 sweep.
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
  e7         Offline robustness-variant generation and evaluation.
  e8         Per-attribute/level metrics, calibration and confusion matrices.
  eigencam    Generate EigenCAM visualizations for MAYOLOx and YOLOv10x and side-by-side images.
  all        Run automated sections in dependency order.

Options:
  --dry-run              Print commands without executing them.
  --device CUDA_DEVICE   Override DEVICE (default: 0).
  --seeds "0 1 2"        Override the seed list.
  --fine-seed SEED        Seed used by e8 fine-grained Test evaluation (default: 1).
  --weights-manifest PATH Checkpoint manifest for existing-weight commands.
  --mayolo-weight PATH   MAYOLO checkpoint for eigencam.
  --yolov10-weight PATH  YOLOv10 checkpoint for eigencam.
  --images PATH           Input image or image directory for eigencam.
  --image-names NAMES      Optional names when --images is a directory; accepts
                           "A B" or separate names: A B.
  --output PATH           CAM output directory.
  --layer INDEX           CAM target layer (default: 22).
  --iou VALUE             NMS IoU threshold (default: 0.7).
  --all-methods           Generate all eight supported CAM methods.

Commands that evaluate existing checkpoints (e2.2--e2.8, e3.eval, e5.eval,
e6, e6.eval, e7 and e8) require --weights-manifest or WEIGHTS_MANIFEST.
The manifest is tab-separated with one entry per line:
  KEY<TAB>SEED<TAB>CHECKPOINT_PATH
Blank lines and lines beginning with # are ignored.  Required keys are
documented by each command's error message; common keys include baseline,
gia, gca_gin, gia_gca_gin, YOLOv10x, MAYOLOx, E5 and E6-classifier.

Common environment overrides: PYTHON_BIN, DATA, PRETRAIN, COM_CROSS, BATCH,
WORKERS, IMGSZ, STAGE1_EPOCHS, STAGE2_EPOCHS, E2_OUTPUT_ROOT,
E2_PERFORMANCE_ROOT, E2_PERF_SEEDS, E3_PERF_ROOT, E3_PERF_SEEDS, FINE_SEED,
FINE_PROJECT_ROOT, EIGENCAM_LAYER, EIGENCAM_CONF, EIGENCAM_IOU,
EIGENCAM_METHODS and EIGENCAM_IMAGE_NAMES.
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

require_weights_manifest() {
  [[ -n "$WEIGHTS_MANIFEST" ]] || {
    echo "This command requires --weights-manifest PATH (or WEIGHTS_MANIFEST)." >&2
    exit 2
  }
  need "$WEIGHTS_MANIFEST"
}

manifest_weight() {
  local key="$1" seed="$2" path
  require_weights_manifest
  path="$(awk -F $'\t' -v wanted_key="$key" -v wanted_seed="$seed" \
    '$0 !~ /^[[:space:]]*#/ && NF >= 3 && $1 == wanted_key && $2 == wanted_seed { print $3; exit }' \
    "$WEIGHTS_MANIFEST")"
  [[ -n "$path" ]] || {
    echo "Missing manifest entry: key=$key seed=$seed in $WEIGHTS_MANIFEST" >&2
    exit 1
  }
  need "$path"
  printf '%s\n' "$path"
}

collect_manifest_group() {
  local key="$1" seed
  COLLECTED=()
  for seed in "${SEED_LIST[@]}"; do
    COLLECTED+=("$(manifest_weight "$key" "$seed")")
  done
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

e20() {
  echo "[E2.0] baseline and GIA five-seed Stage1+Stage2"
  need "$DATA"; need "$MD_MODEL"; need "$GIA_CONFIG"
  local seed
  for seed in "${SEED_LIST[@]}"; do
    py scripts/train_mdet_experiments.py variants \
      --data "$DATA" --project "$E2_OUTPUT_ROOT" --imgsz "$IMGSZ" --batch "$BATCH" \
      --workers "$WORKERS" --device "$DEVICE" --seed "$seed" --w4 "$W4" \
      --stage1-epochs "$STAGE1_EPOCHS" --stage2-epochs "$STAGE2_EPOCHS" \
      --label "$E2_LABEL" --variant "baseline=$MD_MODEL" \
      --variant "gia_v2_5_7=$GIA_CONFIG" --pretrain "$PRETRAIN" --skip-existing
  done
}

e21() {
  echo "[E2.1] GIA position Stage1+Stage2 sweep"
  need "$DATA"
  local seed variant
  for seed in "${SEED_LIST[@]}"; do
    local -a cmd=(
      "$PYTHON_BIN" scripts/train_mdet_experiments.py gia-position
      --data "$DATA" --project runs/experiments/E2_1_GIA_position
      --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS" --device "$DEVICE"
      --seed "$seed" --w4 "$W4"
      --stage1-epochs "$STAGE1_EPOCHS" --stage2-epochs "$STAGE2_EPOCHS"
      --label E2_1_GIA_position --pretrain "$PRETRAIN" --skip-existing
    )
    for variant in "${GIA_POSITION_VARIANTS[@]}"; do cmd+=(--variant "$variant"); done
    run "${cmd[@]}"
  done
}

gca_train() {
  local label="$1" root="$2" config="$3" matrix_path="$4" stage1_key="$5"
  need "$DATA"; need "$config"; need "$matrix_path"
  require_weights_manifest
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
    checkpoint="$(manifest_weight "$stage1_key" "$seed")"
    cmd+=(--stage1-checkpoint-map "${seed}=${checkpoint}")
  done
  run "${cmd[@]}"
}

e22() {
  echo "[E2.2] pure Baseline+GCA: Cross matrix and four graph operators"
  gca_train E2_29_Baseline_GCA_pure_margin_residual_5seed_cross \
    runs/experiments/E2_29_Baseline_GCA_pure_margin_residual_5seed_cross \
    "$PURE_GCA_CONFIG" "$COM_CROSS" baseline_stage1
}

e25() {
  echo "[E2.5] GIA+GCA Stage2: Cross matrix and four graph operators"
  gca_train E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross \
    runs/experiments/E2_28_GIA_v2_5_7_GCA_margin_residual_5seed_cross \
    "$GIA_GCA_CONFIG" "$COM_CROSS" gia_stage1
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

e23() {
  echo "[E2.3] Baseline HO Test evaluation"
  collect_manifest_group baseline
  eval_ho runs/experiments/E2_27_HO/summary.csv E2_27_HO one2many "${COLLECTED[@]}"
}

e24() {
  echo "[E2.4] GIA HO Test evaluation"
  collect_manifest_group gia
  eval_ho runs/experiments/E2_27_GIA_v2_5_7_stage2_recheck/one2many_summary.csv \
    E2_27_GIA_v2_5_7_stage2_recheck one2many "${COLLECTED[@]}"
}

e26() {
  echo "[E2.6] Baseline+GCA HO Test evaluation"
  require_weights_manifest
  local root="${E2_GCA_HO_ROOT:-runs/experiments/E2_6_Baseline_GCA_HO}"
  COLLECTED=()
  local operator seed
  for operator in "${GNN_TYPES[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
      COLLECTED+=("$(manifest_weight "gca_${operator}" "$seed")")
    done
  done
  eval_ho "$root/ho_summary.csv" "E2_6_Baseline_GCA_cross_HO" one2many \
    "${COLLECTED[@]}"
}

e27() {
  echo "[E2.7] GIA+GCA HO Test evaluation (MAYOLO)"
  require_weights_manifest
  local root="${E2_GIA_GCA_HO_ROOT:-runs/experiments/E2_7_MAYOLO_HO}"
  COLLECTED=()
  local operator seed
  for operator in "${GNN_TYPES[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
      COLLECTED+=("$(manifest_weight "gia_gca_${operator}" "$seed")")
    done
  done
  eval_ho "$root/ho_summary.csv" "E2_7_MAYOLO_cross_HO" one2many \
    "${COLLECTED[@]}"
}

e28() {
  echo "[E2.8] ablation/performance profile for eight models x five seeds"
  require_weights_manifest
  local root="${E2_PERFORMANCE_ROOT:-runs/experiments/performance_profile/E2_8_5seed_repro}"
  local perf_seeds="${E2_PERF_SEEDS:-0 1 2 3 4}"
  read -r -a PERF_SEED_LIST <<< "$perf_seeds"
  local seed baseline gia gca gia_gca
  for seed in "${PERF_SEED_LIST[@]}"; do
    baseline="$(manifest_weight baseline "$seed")"
    gia="$(manifest_weight gia "$seed")"
    gca="$(manifest_weight gca_gin "$seed")"
    gia_gca="$(manifest_weight gia_gca_gin "$seed")"

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
  require_weights_manifest
  local perf_seeds="${E3_PERF_SEEDS:-0 1 2 3 4}"
  read -r -a PERF_SEED_LIST <<< "$perf_seeds"
  local root="${E3_PERF_ROOT:-runs/experiments/performance_profile/E3_E4_5seed}"
  local seed family size name weight
  local -a native_weights=() native_labels=()
  local -a mayolo_weights=() mayolo_labels=()
  local -a rtdetr_weights=()

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
        name="YOLO${family#yolo}${size}"
        weight="$(manifest_weight "$name" "$seed")"
        native_weights+=("$weight"); native_labels+=("$name")
      done
    done
    run "$PYTHON_BIN" scripts/benchmark_performance.py \
      --weights "${native_weights[@]}" --labels "${native_labels[@]}" \
      --network yolo --task mdetect --head-mode native --device "$DEVICE" \
      --precision "${PERF_PRECISION:-fp16}" --batch "${PERF_BATCH:-1}" --imgsz "$IMGSZ" \
      --warmup "${PERF_WARMUP:-50}" --iterations "${PERF_ITERATIONS:-200}" \
      --output "$root/E3_seed${seed}_native.csv"

    for size in ${MAYOLO_SIZES:-n s m l b}; do
      name="MAYOLO$size"
      weight="$(manifest_weight "$name" "$seed")"
      mayolo_weights+=("$weight"); mayolo_labels+=("$name")
    done
    weight="$(manifest_weight MAYOLOx "$seed")"
    mayolo_weights+=("$weight"); mayolo_labels+=("MAYOLOx")
    run "$PYTHON_BIN" scripts/benchmark_performance.py \
      --weights "${mayolo_weights[@]}" --labels "${mayolo_labels[@]}" \
      --network yolo --task mdetect --head-mode one2many --device "$DEVICE" \
      --precision "${PERF_PRECISION:-fp16}" --batch "${PERF_BATCH:-1}" --imgsz "$IMGSZ" \
      --warmup "${PERF_WARMUP:-50}" --iterations "${PERF_ITERATIONS:-200}" \
      --output "$root/E3_seed${seed}_mayolo.csv"

    rtdetr_weights=(
      "$(manifest_weight RT-DETR-L "$seed")"
      "$(manifest_weight RT-DETR-X "$seed")"
    )
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
  local detector_key="${E6_DETECTOR_KEY:-YOLOv10x}"
  local seed seed_project detector
  if [[ ! -f "$data" ]]; then
    py scripts/prepare_two_stage_crops.py --data "$DATA" --output "$generated" --exist-ok
  fi
  need "$data"; need "$model"; need "$pretrain"; require_weights_manifest
  for seed in "${SEED_LIST[@]}"; do
    detector="$(manifest_weight "$detector_key" "$seed")"
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
  need "$data"; require_weights_manifest
  for seed in "${SEED_LIST[@]}"; do
    weight="$(manifest_weight E5 "$seed")"
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
  local detector_key="${E6_DETECTOR_KEY:-YOLOv10x}"
  local classifier_key="${E6_CLASSIFIER_KEY:-E6-classifier}"
  local name="${E6_NAME:-detector_yolov10x_classifier_yolov10x_cls}"
  local seed detector seed_project classifier
  require_weights_manifest
  for seed in "${SEED_LIST[@]}"; do
    detector="$(manifest_weight "$detector_key" "$seed")"
    if [[ "$seed" == "0" ]]; then seed_project="$project"; else seed_project="${project}_matched_seed${seed}"; fi
    classifier="$(manifest_weight "$classifier_key" "$seed")"
    run "$PYTHON_BIN" scripts/eval_e5_e6.py e6 --device "$DEVICE" --batch "$BATCH" \
      --workers "$WORKERS" --imgsz "$IMGSZ" --project "$project" \
      --name "test_seed$seed" --summary "$project/test_summary_seed$seed.csv" \
      --source-data "$source_data" --detector-weights "$detector" \
      --classifier-weights "$classifier"
  done
}

e7() {
  echo "[E7] deterministic test-set robustness evaluation"
  require_weights_manifest
  local root="${ROBUSTNESS_ROOT:-runs/experiments/E5_robustness/seed0_v1}"
  local generated="$root/data"
  if [[ ! -f "$generated/manifest.json" ]]; then
    py scripts/generate_robustness_variants.py --data "$DATA" --output "$generated" \
      --split test --seed 0 --include-optional --exist-ok
  fi
  need "$generated/manifest.json"

  local -a specs=()
  local label weight
  for label in YOLOv8x YOLOv9e YOLOv10x YOLOv11x YOLOv12x YOLOv13x YOLO26x; do
    weight="$(manifest_weight "$label" 0)"
    specs+=(--model "$label=$weight")
  done
  weight="$(manifest_weight RT-DETR-X 0)"
  specs+=(--model "RT-DETR-x=$weight")
  weight="$(manifest_weight MAYOLOx 0)"
  specs+=(--model "MAYOLOx=$weight::one2many")
  run "$PYTHON_BIN" scripts/eval_robustness.py --data "$DATA" \
    --manifest "$generated/manifest.json" --device "$DEVICE" --imgsz "$IMGSZ" \
    --batch "${ROBUSTNESS_BATCH:-4}" --workers 0 --project "$root" --name seed0_full \
    --seed 0 --resume "${specs[@]}"
}

e8() {
  echo "[E8] per-attribute/level metrics, calibration and confusion matrices"
  local seed="${FINE_SEED}"
  local fine_root="${FINE_PROJECT_ROOT:-runs/experiments/E4_4_10_finegrained_seed${seed}}"
  require_weights_manifest
  local yolov10x="$(manifest_weight YOLOv10x "$seed")"
  local mayolox="$(manifest_weight MAYOLOx "$seed")"
  run "$PYTHON_BIN" scripts/eval_attribute_levels.py --data "$DATA" \
    --model "YOLOv10x=$yolov10x::native" --model "MAYOLOx=$mayolox::one2many" \
    --device "$DEVICE" --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS" \
    --project "$fine_root" --name levels
  run "$PYTHON_BIN" scripts/eval_attribute_calibration.py --data "$DATA" \
    --model "YOLOv10x=$yolov10x::native" --model "MAYOLOx=$mayolox::one2many" \
    --device "$DEVICE" --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS" \
    --project "$fine_root" --name calibration
  run "$PYTHON_BIN" scripts/plot_attribute_confusion_matrices.py \
    --confusion "$fine_root/levels/confusion_test.csv" --data "$DATA" \
    --models YOLOv10x MAYOLOx --head-mode all \
    --output "$fine_root/confusion"
  run "$PYTHON_BIN" scripts/plot_box_confusion_matrix.py \
    --confusion "$fine_root/levels/box_confusion_test.csv" \
    --models YOLOv10x MAYOLOx --head-mode all \
    --output "$fine_root/box_confusion"
}

eigencam() {
  echo "[CAM historical-2026-09-22] MAYOLOx vs YOLOv10x"
  need scripts/generate_eigencam.py
  [[ -n "$EIGENCAM_MAYOLO_WEIGHT" && -n "$EIGENCAM_YOLOV10_WEIGHT" ]] || {
    echo "eigencam requires --mayolo-weight PATH and --yolov10-weight PATH." >&2
    exit 2
  }
  need "$EIGENCAM_MAYOLO_WEIGHT"
  need "$EIGENCAM_YOLOV10_WEIGHT"
  if [[ "$DRY_RUN" != "1" && ! -e "$EIGENCAM_IMAGES" ]]; then
    echo "Missing image file or directory: $EIGENCAM_IMAGES" >&2
    exit 1
  fi

  if [[ -z "$EIGENCAM_LAYER" ]]; then
    echo "EIGENCAM_LAYER must contain one layer index" >&2
    exit 2
  fi

  local -a methods=()
  read -r -a methods <<< "$EIGENCAM_METHODS"
  if [[ "${#methods[@]}" -eq 0 ]]; then
    echo "EIGENCAM_METHODS must contain at least one CAM method" >&2
    exit 2
  fi

  local -a image_names=()
  if [[ -n "$EIGENCAM_IMAGE_NAMES" ]]; then
    read -r -a image_names <<< "$EIGENCAM_IMAGE_NAMES"
  fi

  local -a cmd=(
    "$PYTHON_BIN" scripts/generate_eigencam.py
    --mayolo-weight "$EIGENCAM_MAYOLO_WEIGHT"
    --yolov10-weight "$EIGENCAM_YOLOV10_WEIGHT"
    --images "$EIGENCAM_IMAGES"
    --output "$EIGENCAM_OUTPUT"
    --device "$DEVICE"
    --layer "$EIGENCAM_LAYER"
    --conf "$EIGENCAM_CONF"
    --iou "$EIGENCAM_IOU"
    --methods "${methods[@]}"
  )
  if [[ "${#image_names[@]}" -gt 0 ]]; then
    cmd+=(--image-names "${image_names[@]}")
  fi
  run "${cmd[@]}"
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
  e7
  e8
}

case "$CODE" in
  help|-h|--help) usage ;;
  preflight) preflight ;;
  e1) e1 ;;
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
  e7) e7 ;;
  e8) e8 ;;
  eigencam) eigencam ;;
  all) all_experiments ;;
  *) echo "Unknown experiment code: $CODE" >&2; usage; exit 2 ;;
esac
