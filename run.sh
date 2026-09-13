DATA=ultralytics/cfg/mayolo_r1/mayolo_v3.yaml

# python scripts/gpu_memory_smoke_test.py \
#   --data "$DATA" \
#   --device 0 \
#   --imgsz 640 \
#   --batch 16 \
#   --epochs 2 \
#   --project runs/gpu_memory_smoke

# python scripts/train_mdet_experiments.py hsv-ablation \
#   --data "$DATA" \
#   --model ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml \
#   --pretrain yolov10x.pt \
#   --epochs 100 \
#   --project runs/experiments/E0_hsv_ablation

# python scripts/train_mdet_experiments.py stage1-sweep \
#   --data "$DATA" \
#   --model ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml \
#   --pretrain yolov10x.pt \
#   --stage1-values 100 200 300 400 500 \
#   --project runs/experiments/E0_stage1_sweep


# python scripts/train_mdet_experiments.py stage2-sweep \
#   --data "$DATA" \
#   --model ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml \
#   --stage1-checkpoint runs/experiments/E0_stage1_sweep/E0_stage1_stage1_100_w4_0p5_seed_0/weights/best.pt \
#   --stage1-epochs 100 \
#   --stage2-values 50 100 150 200 \
#   --project runs/experiments/E0_stage2_sweep


# python scripts/train_mdet_experiments.py w4 \
#   --data "$DATA" \
#   --model ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml \
#   --pretrain yolov10x.pt \
#   --w4-values 0.25 0.5 1.0 1.25 1.5 \
#   --project runs/experiments/E1_w4




# E2.1 residual-GIA smoke test: run each Res configuration for one epoch
# before launching the full position ablation.
# python scripts/gpu_memory_smoke_test.py \
#   --data "$DATA" \
#   --models e2_1_gia5_res e2_1_gia7_res e2_1_gia8_res e2_1_gia9_res e2_1_gia10_res e2_1_gia5_7_res \
#   --device 0 \
#   --imgsz 640 \
#   --batch 16 \
#   --epochs 1 \
#   --project runs/gpu_memory_smoke/E2_1_GIA_position_res \
#   --name e2_1_res

# E2.1 GIA-v2 smoke test: the zero-gated upgraded GIA is identity-initialized,
# so verify memory and construction before the 100-epoch position ablation.
# python scripts/gpu_memory_smoke_test.py \
#   --data "$DATA" \
#   --models e2_1_gia_v2_6 e2_1_gia_v2_7 e2_1_gia_v2_8 e2_1_gia_v2_9 e2_1_gia_v2_10 e2_1_gia_v2_13 e2_1_gia_v2_16 e2_1_gia_v2_19 e2_1_gia_v2_22 e2_1_gia_v2_5_7 \
#   --device 0 \
#   --imgsz 640 \
#   --batch 16 \
#   --epochs 1 \
#   --project runs/gpu_memory_smoke/E2_1_GIA_v2_position \
#   --name e2_1_gia_v2

# E2.1 GIA-v2 position ablation: the previously selected backbone positions
# 7/8/9/10 and 5+7, plus the suggested neck/head positions 6/13/16/19/22.
# The baseline and the old GIA position runs are not repeated.
# python scripts/train_mdet_experiments.py gia-position \
#   --label E2_1_GIA_v2_position \
#   --data "$DATA" \
#   --pretrain yolov10x.pt \
#   --stage1-only \
#   --stage1-epochs 100 \
#   --variant gia_v2_6=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_6.yaml \
#   --variant gia_v2_7=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_7.yaml \
#   --variant gia_v2_8=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_8.yaml \
#   --variant gia_v2_9=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_9.yaml \
#   --variant gia_v2_10=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_10.yaml \
#   --variant gia_v2_13=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_13.yaml \
#   --variant gia_v2_16=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_16.yaml \
#   --variant gia_v2_19=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_19.yaml \
#   --variant gia_v2_22=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_22.yaml \
#   --variant gia_v2_5_7=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_5_7.yaml \
#   --project runs/experiments/E2_1_GIA_v2_position


# E2.1 confirmation: compare baseline and the promising GIA@5+7 candidate
# under three real random seeds. Each run is stage 1 only (100 epochs),
# batch=16, and uses the default mdet w4=0.5. The matrix is complete, so it
# is opt-in to avoid repeating six expensive jobs when running E2.2 below.

# for SEED in 0 1 2; do
#   python scripts/train_mdet_experiments.py gia-position \
#     --label E2_1_GIA_v2_confirm_seedfix \
#     --data "$DATA" \
#     --pretrain yolov10x.pt \
#     --seed "$SEED" \
#     --batch 16 \
#     --stage1-only \
#     --stage1-epochs 100 \
#     --variant baseline=ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml \
#     --variant gia_v2_5_7=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_5_7.yaml \
#     --project runs/experiments/E2_1_GIA_v2_confirm_seedfix
# done


# Override the train-only matrices when they are stored elsewhere:
#   COM_PATH=/path/to/co_occurrence_matrix_train.csv \
#   COM_CONDITIONAL_PATH=/path/to/co_occurrence_matrix_train_conditional.csv bash run.sh
COM_PATH=${COM_PATH:-/localnvme/data/billboard/mayolo_v3/co_occurrence_matrix_train.csv}
COM_CONDITIONAL_PATH=${COM_CONDITIONAL_PATH:-/localnvme/data/billboard/mayolo_v3/co_occurrence_matrix_train_conditional.csv}

# Previous active E2.2 multiclass-aware GCA/GNN comparison.  This block is
# intentionally retained as comments for visual comparison and is not run.
# STAGE1_CKPT=runs/experiments/E1_w4/E1_w4_base_w4_0p5_seed_0_stage1/weights/best.pt
# python scripts/train_mdet_experiments.py gca-stage2 \
#   --label E2_2_GCA_GNN_margin_residual \
#   --data "$DATA" \
#   --stage1-checkpoint "$STAGE1_CKPT" \
#   --stage1-epochs 100 \
#   --stage2-epochs 100 \
#   --variant gca_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_margin_residual.yaml \
#   --variant gcn_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCN_margin_residual.yaml \
#   --variant gat_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GAT_margin_residual.yaml \
#   --variant graphsage_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GraphSAGE_margin_residual.yaml \
#   --variant gin_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GIN_margin_residual.yaml \
#   --w4 0.5 \
#   --batch 16 \
#   --seed 0 \
#   --com-path "$COM_PATH" \
#   --project runs/experiments/E2_2_GCA_GNN_margin_residual


# # E2.2 matrix-controlled 5x5 study.  The five previously designed variant
# # entries are crossed with five GNN operators twice: once with the original
# # train-only matrix and once with the train-only conditional matrix.  Each
# # invocation of run_gca_5x5 expands to 25 Stage2-only runs.  The old commands
# # above remain commented for visual comparison.
#
# STAGE1_CKPT=runs/experiments/E1_w4/E1_w4_base_w4_0p5_seed_0_stage1/weights/best.pt
# if [ ! -f "$STAGE1_CKPT" ]; then
#   echo "Missing Stage1 checkpoint: $STAGE1_CKPT" >&2
#   exit 1
# fi
# if [ ! -f "$COM_PATH" ]; then
#   echo "Missing cross-normalized COM: $COM_PATH" >&2
#   exit 1
# fi
# if [ ! -f "$COM_CONDITIONAL_PATH" ]; then
#   echo "Missing conditional COM: $COM_CONDITIONAL_PATH" >&2
#   echo "Generate it with generate_com.py --split train --mode conditional --smoothing 1.0" >&2
#   exit 1
# fi
#
# run_gca_5x5() {
#   local label="$1"
#   local matrix_path="$2"
#   local project="runs/experiments/${label}"
#
#   python scripts/train_mdet_experiments.py gca-stage2 \
#     --label "$label" \
#     --data "$DATA" \
#     --stage1-checkpoint "$STAGE1_CKPT" \
#     --stage1-epochs 100 \
#     --stage2-epochs 100 \
#     --variant context_cross=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_context_residual.yaml \
#     --variant context_conditional=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_context_residual.yaml \
#     --variant adaptive=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_adaptive_residual.yaml \
#     --variant twohop=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_twohop_residual.yaml \
#     --variant conv_adapter=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_conv_adapter_residual.yaml \
#     --gnn-types gca gcn gat graphsage gin \
#     --w4 0.5 \
#     --batch 16 \
#     --seed 0 \
#     --com-path "$matrix_path" \
#     --project "$project"
# }
#
# # 25 runs: the original train-only cross-normalized matrix.
# run_gca_5x5 E2_2_GCA5x5_cross "$COM_PATH"
#
# # 25 runs: the train-only directed conditional matrix generated separately.
# run_gca_5x5 E2_2_GCA5x5_conditional "$COM_CONDITIONAL_PATH"
#
# # Supplementary conditional-matrix comparison using the five previously
# # completed margin-residual GNN configurations.  These are kept separate from
# # the new 5x5 table because their fusion mechanism is different.
# python scripts/train_mdet_experiments.py gca-stage2 \
#   --label E2_2_GCA_GNN_margin_residual_conditional \
#   --data "$DATA" \
#   --stage1-checkpoint "$STAGE1_CKPT" \
#   --stage1-epochs 100 \
#   --stage2-epochs 100 \
#   --variant gca_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_margin_residual.yaml \
#   --variant gcn_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCN_margin_residual.yaml \
#   --variant gat_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GAT_margin_residual.yaml \
#   --variant graphsage_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GraphSAGE_margin_residual.yaml \
#   --variant gin_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GIN_margin_residual.yaml \
#   --w4 0.5 \
#   --batch 16 \
#   --seed 0 \
#   --com-path "$COM_CONDITIONAL_PATH" \
#   --project runs/experiments/E2_2_GCA_GNN_margin_residual_conditional
#
#
# # The completed 10-position GIA-v2 matrix is retained below for reference;
# # leave it commented to avoid retraining those experiments.

# E2.9 label-correlation classifier batch.
# Inspired by ML-GCN, these heads turn learnable label nodes into spatial
# attribute classifiers.  They test fixed label GCN, thresholded propagation,
# prior-plus-learnable adjacency, image-conditioned graph propagation, and
# label-aware attention.  The existing Stage2 freeze policy (freeze=23 for
# YOLOv10/MAYOLO) remains active; no backbone or neck parameters are unfrozen.
STAGE1_CKPT=runs/experiments/E1_w4/E1_w4_base_w4_0p5_seed_0_stage1/weights/best.pt
PRIOR_MODEL=${PRIOR_MODEL:-ultralytics/cfg/models/exp_ablation/yolov10x_com_prior.yaml}
PRIOR_LABEL=${PRIOR_LABEL:-E2_9_label_gcn}
PRIOR_PROJECT=${PRIOR_PROJECT:-runs/experiments/E2_9_label_gcn}
PRIOR_STAGE1_EPOCHS=${PRIOR_STAGE1_EPOCHS:-100}
PRIOR_STAGE2_EPOCHS=${PRIOR_STAGE2_EPOCHS:-100}
PRIOR_TYPES=${PRIOR_TYPES:-"label_gcn label_gcn_threshold adaptive_label_gcn dynamic_label_gcn label_attention"}
# Run both matrices by default. Set PRIOR_MATRIX_MODES="cross" for five jobs.
PRIOR_MATRIX_MODES=${PRIOR_MATRIX_MODES:-"cross conditional"}
PRIOR_W4=${PRIOR_W4:-0.5}
PRIOR_BATCH=${PRIOR_BATCH:-16}
PRIOR_SEED=${PRIOR_SEED:-0}
RUN_PRIOR_BATCH=${RUN_PRIOR_BATCH:-1}

if [ "$RUN_PRIOR_BATCH" = "1" ]; then
  if [ ! -f "$STAGE1_CKPT" ]; then
    echo "Missing Stage1 checkpoint: $STAGE1_CKPT" >&2
    exit 1
  fi

  for MATRIX_MODE in $PRIOR_MATRIX_MODES; do
    case "$MATRIX_MODE" in
      cross)
        MATRIX_PATH="$COM_PATH"
        ;;
      conditional)
        MATRIX_PATH="$COM_CONDITIONAL_PATH"
        ;;
      *)
        echo "Unsupported PRIOR_MATRIX_MODES value: $MATRIX_MODE" >&2
        exit 1
        ;;
    esac

    if [ ! -f "$MATRIX_PATH" ]; then
      echo "Missing $MATRIX_MODE co-occurrence matrix: $MATRIX_PATH" >&2
      exit 1
    fi

    python scripts/train_mdet_experiments.py prior-stage2 \
      --label "$PRIOR_LABEL" \
      --model "$PRIOR_MODEL" \
      --data "$DATA" \
      --stage1-checkpoint "$STAGE1_CKPT" \
      --stage1-epochs "$PRIOR_STAGE1_EPOCHS" \
      --stage2-epochs "$PRIOR_STAGE2_EPOCHS" \
      --prior-types $PRIOR_TYPES \
      --matrix-modes "$MATRIX_MODE" \
      --w4 "$PRIOR_W4" \
      --batch "$PRIOR_BATCH" \
      --seed "$PRIOR_SEED" \
      --com-path "$MATRIX_PATH" \
      --project "$PRIOR_PROJECT" || exit $?
  done
fi

# Feature graph direction is closed after the gain screen. The block below is
# retained only for exact reproduction of archived runs and is opt-in.
if [ "${ENABLE_FEATURE_GRAPH:-0}" = "1" ]; then
  echo "Running archived feature graph reproduction because ENABLE_FEATURE_GRAPH=1."
  # Set FEATURE_EPOCHS=1 and FEATURE_PROJECT=... for a separate smoke run.
  for REQUIRED_FILE in "$STAGE1_CKPT" "$COM_PATH"; do
    if [ ! -f "$REQUIRED_FILE" ]; then
      echo "Missing input: $REQUIRED_FILE" >&2
      exit 1
    fi
  done

  FEATURE_EPOCHS=${FEATURE_EPOCHS:-100}
  FEATURE_PROJECT=${FEATURE_PROJECT:-runs/experiments/E2_2_feature_gain}
  FEATURE_GAIN_VALUES=${FEATURE_GAIN_VALUES:-"2 4 8"}
  FEATURE_LOCAL_GAIN=${FEATURE_LOCAL_GAIN:-4}

# Gain=1 is the completed feature_gca_cross reference.  Reproduce it with
# FEATURE_GAIN_VALUES="1 2 4 8" when the reference is not available locally.
for GAIN in $FEATURE_GAIN_VALUES; do
  python scripts/train_mdet_experiments.py gca-stage2 \
    --label "feature_gca_cross_gain_${GAIN}" \
    --data "$DATA" \
    --stage1-checkpoint "$STAGE1_CKPT" \
    --stage1-epochs 100 \
    --stage2-epochs "$FEATURE_EPOCHS" \
    --variant "gca_cross_gain_${GAIN}=ultralytics/cfg/models/exp_ablation/yolov10x_feature_gca_cross.yaml" \
    --feature-gain "$GAIN" \
    --w4 0.5 \
    --batch 16 \
    --seed 0 \
    --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
      --com-path "$COM_PATH" \
      --project "$FEATURE_PROJECT" || exit $?
done

# A no-graph local adapter control separates a useful residual amplitude from
# a gain that only compensates for an ineffective graph message.
python scripts/train_mdet_experiments.py gca-stage2 \
  --label "feature_local_cross_gain_${FEATURE_LOCAL_GAIN}" \
  --data "$DATA" \
  --stage1-checkpoint "$STAGE1_CKPT" \
  --stage1-epochs 100 \
  --stage2-epochs "$FEATURE_EPOCHS" \
  --variant "local_cross_gain_${FEATURE_LOCAL_GAIN}=ultralytics/cfg/models/exp_ablation/yolov10x_feature_local_cross.yaml" \
  --feature-gain "$FEATURE_LOCAL_GAIN" \
  --w4 0.5 \
  --batch 16 \
  --seed 0 \
  --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
  --com-path "$COM_PATH" \
  --project "$FEATURE_PROJECT" || exit $?
fi
