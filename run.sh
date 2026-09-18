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
PYTHON_BIN=${PYTHON_BIN:-python3}

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

# E2.11 graph-refinement batch.
# E2.10 showed that a direct ML-GCN classifier-weight path is not sufficient
# for hard-F1 gains.  These variants retain the visual margin as the unary
# prediction and use the co-occurrence graph as a learnable refinement:
# graph-masked label attention, GraphSAGE label aggregation, a label
# Transformer, two-step mean-field refinement, and a pixel-wise ML-GCN MoE.
# The existing Stage2 freeze policy (freeze=23 for YOLOv10/MAYOLO) remains
# active; no backbone or neck parameters are unfrozen.
# STAGE1_CKPT=runs/experiments/E1_w4/E1_w4_base_w4_0p5_seed_0_stage1/weights/best.pt
# PRIOR_MODEL=${PRIOR_MODEL:-ultralytics/cfg/models/exp_ablation/yolov10x_com_prior.yaml}
# PRIOR_LABEL=${PRIOR_LABEL:-E2_11_graph_refine}
# PRIOR_PROJECT=${PRIOR_PROJECT:-runs/experiments/E2_11_graph_refine}
# PRIOR_STAGE1_EPOCHS=${PRIOR_STAGE1_EPOCHS:-100}
# PRIOR_STAGE2_EPOCHS=${PRIOR_STAGE2_EPOCHS:-100}
# PRIOR_TYPES=${PRIOR_TYPES:-"mlgat mlsage mltransformer graph_mean_field mlgcn_moe"}
# # Run both matrices by default. Set PRIOR_MATRIX_MODES="cross" for five jobs.
# PRIOR_MATRIX_MODES=${PRIOR_MATRIX_MODES:-"cross conditional"}
# PRIOR_W4=${PRIOR_W4:-0.5}
# PRIOR_BATCH=${PRIOR_BATCH:-16}
# PRIOR_SEED=${PRIOR_SEED:-0}
# RUN_PRIOR_BATCH=${RUN_PRIOR_BATCH:-0}

# if [ "$RUN_PRIOR_BATCH" = "1" ]; then
#   if [ ! -f "$STAGE1_CKPT" ]; then
#     echo "Missing Stage1 checkpoint: $STAGE1_CKPT" >&2
#     exit 1
#   fi

#   for MATRIX_MODE in $PRIOR_MATRIX_MODES; do
#     case "$MATRIX_MODE" in
#       cross)
#         MATRIX_PATH="$COM_PATH"
#         ;;
#       conditional)
#         MATRIX_PATH="$COM_CONDITIONAL_PATH"
#         ;;
#       *)
#         echo "Unsupported PRIOR_MATRIX_MODES value: $MATRIX_MODE" >&2
#         exit 1
#         ;;
#     esac

#     if [ ! -f "$MATRIX_PATH" ]; then
#       echo "Missing $MATRIX_MODE co-occurrence matrix: $MATRIX_PATH" >&2
#       exit 1
#     fi

#     "$PYTHON_BIN" scripts/train_mdet_experiments.py prior-stage2 \
#       --label "$PRIOR_LABEL" \
#       --model "$PRIOR_MODEL" \
#       --data "$DATA" \
#       --stage1-checkpoint "$STAGE1_CKPT" \
#       --stage1-epochs "$PRIOR_STAGE1_EPOCHS" \
#       --stage2-epochs "$PRIOR_STAGE2_EPOCHS" \
#       --prior-types $PRIOR_TYPES \
#       --matrix-modes "$MATRIX_MODE" \
#       --w4 "$PRIOR_W4" \
#       --batch "$PRIOR_BATCH" \
#       --seed "$PRIOR_SEED" \
#       --com-path "$MATRIX_PATH" \
#       --project "$PRIOR_PROJECT" || exit $?
#   done
# fi

# # E2.12 GIA-transfer selection batch.
# # This batch is intentionally opt-in because it launches 26 long runs:
# # 13 selected structures under two initialization protocols.
# #   (1) Stage2-only, initialized from the best GIA-v2 Test checkpoint.
# #   (2) Full stage1=100 + stage2=100, initialized from yolov10x.pt.
# # The selected structures are fixed by the remote E2.2 summary ranking:
# # three Test-improved structures plus ten highest Val-F1 configurations.
# if [ "${RUN_GCA_GIA_TRANSFER_BATCH:-0}" = "1" ]; then
#   GIA_BEST_STAGE1_CKPT=${GIA_BEST_STAGE1_CKPT:-runs/experiments/E2_1_GIA_v2_position/E2_1_GIA_v2_position_gia_v2_9_stage1_100_w4_0p5_seed_0/weights/best.pt}
#   GIA_TRANSFER_STAGE2_PROJECT=${GIA_TRANSFER_STAGE2_PROJECT:-runs/experiments/E2_12_GCA_GIA_transfer_stage2}
#   GIA_TRANSFER_FULL_PROJECT=${GIA_TRANSFER_FULL_PROJECT:-runs/experiments/E2_12_GCA_GIA_transfer_full}
#   GIA_TRANSFER_W4=${GIA_TRANSFER_W4:-0.5}
#   GIA_TRANSFER_BATCH=${GIA_TRANSFER_BATCH:-16}
#   GIA_TRANSFER_SEED=${GIA_TRANSFER_SEED:-0}

#   for REQUIRED_FILE in "$GIA_BEST_STAGE1_CKPT" "$COM_PATH" "$COM_CONDITIONAL_PATH" yolov10x.pt; do
#     if [ ! -f "$REQUIRED_FILE" ]; then
#       echo "Missing GIA-transfer input: $REQUIRED_FILE" >&2
#       exit 1
#     fi
#   done

#   run_gia_transfer_stage2() {
#     local matrix_path="$1"
#     shift
#     "$PYTHON_BIN" scripts/train_mdet_experiments.py gca-stage2 \
#       --label E2_12_GCA_GIA_transfer_stage2 \
#       --data "$DATA" \
#       --stage1-checkpoint "$GIA_BEST_STAGE1_CKPT" \
#       --stage1-epochs 100 \
#       --stage2-epochs 100 \
#       "$@" \
#       --skip-existing \
#       --w4 "$GIA_TRANSFER_W4" \
#       --batch "$GIA_TRANSFER_BATCH" \
#       --seed "$GIA_TRANSFER_SEED" \
#       --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
#       --com-path "$matrix_path" \
#       --project "$GIA_TRANSFER_STAGE2_PROJECT" || exit $?
#   }

#   run_gia_transfer_full() {
#     local matrix_path="$1"
#     shift
#     "$PYTHON_BIN" scripts/train_mdet_experiments.py gca-structure \
#       --label E2_12_GCA_GIA_transfer_full \
#       --data "$DATA" \
#       --pretrain yolov10x.pt \
#       --stage1-epochs 100 \
#       --stage2-epochs 100 \
#       "$@" \
#       --skip-existing \
#       --w4 "$GIA_TRANSFER_W4" \
#       --batch "$GIA_TRANSFER_BATCH" \
#       --seed "$GIA_TRANSFER_SEED" \
#       --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
#       --com-path "$matrix_path" \
#       --project "$GIA_TRANSFER_FULL_PROJECT" || exit $?
#   }

#   # Test-improved: GraphSAGE adaptive (cross), GCN margin residual (cross),
#   # and GCN margin residual (conditional).
#   run_gia_transfer_stage2 "$COM_PATH" \
#     --variant adaptive=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_adaptive_residual.yaml \
#     --variant context_conditional=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_context_residual.yaml \
#     --variant context_cross=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_context_residual.yaml \
#     --gnn-types graphsage

#   run_gia_transfer_stage2 "$COM_PATH" \
#     --variant gca_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_margin_residual.yaml \
#     --variant gin_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GIN_margin_residual.yaml \
#     --variant gcn_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCN_margin_residual.yaml

#   run_gia_transfer_stage2 "$COM_CONDITIONAL_PATH" \
#     --variant adaptive=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_adaptive_residual.yaml \
#     --variant context_conditional=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_context_residual.yaml \
#     --variant context_cross=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_context_residual.yaml \
#     --variant conv_adapter=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_conv_adapter_residual.yaml \
#     --gnn-types gat

#   run_gia_transfer_stage2 "$COM_CONDITIONAL_PATH" \
#     --variant gca_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_margin_residual.yaml \
#     --variant gin_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GIN_margin_residual.yaml \
#     --variant gcn_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCN_margin_residual.yaml

#   # Full stage1+stage2 equivalents. The gnn-type option materializes the
#   # operator into a generated YAML without changing checked-in configs.
#   run_gia_transfer_full "$COM_PATH" \
#     --variant cross_graphsage_adaptive=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_adaptive_residual.yaml \
#     --variant cross_graphsage_context_conditional=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_context_residual.yaml \
#     --variant cross_graphsage_context_cross=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_context_residual.yaml \
#     --gnn-type graphsage

#   run_gia_transfer_full "$COM_PATH" \
#     --variant cross_gca_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_margin_residual.yaml \
#     --variant cross_gin_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GIN_margin_residual.yaml \
#     --variant cross_gcn_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCN_margin_residual.yaml

#   run_gia_transfer_full "$COM_CONDITIONAL_PATH" \
#     --variant conditional_gat_adaptive=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_adaptive_residual.yaml \
#     --variant conditional_gat_context_conditional=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_context_residual.yaml \
#     --variant conditional_gat_context_cross=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_context_residual.yaml \
#     --variant conditional_gat_conv_adapter=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_conv_adapter_residual.yaml \
#     --gnn-type gat

#   run_gia_transfer_full "$COM_CONDITIONAL_PATH" \
#     --variant conditional_gca_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_margin_residual.yaml \
#     --variant conditional_gin_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GIN_margin_residual.yaml \
#     --variant conditional_gcn_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCN_margin_residual.yaml
# fi

# # E2.13: GIA-initialized repeat of the three selected GCA/GNN structures.
# # The matching no-GCA/GNN GIA Stage2 control is defined in E2.15 below.
# if [ "${RUN_GCA_GIA_TEST3_SEEDS:-0}" = "1" ]; then
#   GIA_TEST3_STAGE1_CKPT=${GIA_TEST3_STAGE1_CKPT:-runs/experiments/E2_1_GIA_v2_position/E2_1_GIA_v2_position_gia_v2_9_stage1_100_w4_0p5_seed_0/weights/best.pt}
#   GIA_TEST3_PROJECT=${GIA_TEST3_PROJECT:-runs/experiments/E2_13_GCA_GIA_test3_seed}
#   GIA_TEST3_W4=${GIA_TEST3_W4:-0.5}
#   GIA_TEST3_BATCH=${GIA_TEST3_BATCH:-16}
#   GIA_TEST3_SEEDS=${GIA_TEST3_SEEDS:-"0 1 2"}

#   for REQUIRED_FILE in "$GIA_TEST3_STAGE1_CKPT" "$COM_PATH" "$COM_CONDITIONAL_PATH"; do
#     if [ ! -f "$REQUIRED_FILE" ]; then
#       echo "Missing E2.13 input: $REQUIRED_FILE" >&2
#       exit 1
#     fi
#   done

#   for TEST3_SEED in $GIA_TEST3_SEEDS; do
#     "$PYTHON_BIN" scripts/train_mdet_experiments.py gca-stage2 \
#       --label E2_13_GCA_GIA_test3_seed \
#       --data "$DATA" \
#       --stage1-checkpoint "$GIA_TEST3_STAGE1_CKPT" \
#       --stage1-epochs 100 \
#       --stage2-epochs 100 \
#       --variant adaptive=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_adaptive_residual.yaml \
#       --gnn-types graphsage \
#       --skip-existing \
#       --w4 "$GIA_TEST3_W4" \
#       --batch "$GIA_TEST3_BATCH" \
#       --seed "$TEST3_SEED" \
#       --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
#       --com-path "$COM_PATH" \
#       --project "$GIA_TEST3_PROJECT" || exit $?

#     "$PYTHON_BIN" scripts/train_mdet_experiments.py gca-stage2 \
#       --label E2_13_GCA_GIA_test3_seed \
#       --data "$DATA" \
#       --stage1-checkpoint "$GIA_TEST3_STAGE1_CKPT" \
#       --stage1-epochs 100 \
#       --stage2-epochs 100 \
#       --variant cross_gcn_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCN_margin_residual.yaml \
#       --skip-existing \
#       --w4 "$GIA_TEST3_W4" \
#       --batch "$GIA_TEST3_BATCH" \
#       --seed "$TEST3_SEED" \
#       --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
#       --com-path "$COM_PATH" \
#       --project "$GIA_TEST3_PROJECT" || exit $?

#     "$PYTHON_BIN" scripts/train_mdet_experiments.py gca-stage2 \
#       --label E2_13_GCA_GIA_test3_seed \
#       --data "$DATA" \
#       --stage1-checkpoint "$GIA_TEST3_STAGE1_CKPT" \
#       --stage1-epochs 100 \
#       --stage2-epochs 100 \
#       --variant conditional_gcn_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCN_margin_residual.yaml \
#       --skip-existing \
#       --w4 "$GIA_TEST3_W4" \
#       --batch "$GIA_TEST3_BATCH" \
#       --seed "$TEST3_SEED" \
#       --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
#       --com-path "$COM_CONDITIONAL_PATH" \
#       --project "$GIA_TEST3_PROJECT" || exit $?
#   done
# fi

# # E2.14: reproduce the three Test-improved E2.2 GCA/GNN structures with two
# # additional Stage2 seeds, using the same plain E1 Stage1 checkpoint as E2.2.
# # This is the no-GIA repeat batch requested after correcting E2.13.  Set
# # RUN_GCA_GNN_REPEAT_SEEDS=1 to launch the six sequential Stage2-only runs.
# if [ "${RUN_GCA_GNN_REPEAT_SEEDS:-0}" = "1" ]; then
#   GCA_REPEAT_STAGE1_CKPT=${GCA_REPEAT_STAGE1_CKPT:-runs/experiments/E1_w4/E1_w4_base_w4_0p5_seed_0_stage1/weights/best.pt}
#   GCA_REPEAT_PROJECT=${GCA_REPEAT_PROJECT:-runs/experiments/E2_14_GCA_GNN_repeat_no_gia}
#   GCA_REPEAT_W4=${GCA_REPEAT_W4:-0.5}
#   GCA_REPEAT_BATCH=${GCA_REPEAT_BATCH:-16}
#   GCA_REPEAT_SEEDS=${GCA_REPEAT_SEEDS:-"1 2"}

#   for REQUIRED_FILE in "$GCA_REPEAT_STAGE1_CKPT" "$COM_PATH" "$COM_CONDITIONAL_PATH"; do
#     if [ ! -f "$REQUIRED_FILE" ]; then
#       echo "Missing E2.14 input: $REQUIRED_FILE" >&2
#       exit 1
#     fi
#   done

#   for REPEAT_SEED in $GCA_REPEAT_SEEDS; do
#     "$PYTHON_BIN" scripts/train_mdet_experiments.py gca-stage2 \
#       --label E2_14_GCA_GNN_repeat_no_gia \
#       --data "$DATA" \
#       --stage1-checkpoint "$GCA_REPEAT_STAGE1_CKPT" \
#       --stage1-epochs 100 \
#       --stage2-epochs 100 \
#       --variant adaptive=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_adaptive_residual.yaml \
#       --gnn-types graphsage \
#       --skip-existing \
#       --w4 "$GCA_REPEAT_W4" \
#       --batch "$GCA_REPEAT_BATCH" \
#       --seed "$REPEAT_SEED" \
#       --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
#       --com-path "$COM_PATH" \
#       --project "$GCA_REPEAT_PROJECT" || exit $?

#     "$PYTHON_BIN" scripts/train_mdet_experiments.py gca-stage2 \
#       --label E2_14_GCA_GNN_repeat_no_gia \
#       --data "$DATA" \
#       --stage1-checkpoint "$GCA_REPEAT_STAGE1_CKPT" \
#       --stage1-epochs 100 \
#       --stage2-epochs 100 \
#       --variant cross_gcn_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCN_margin_residual.yaml \
#       --skip-existing \
#       --w4 "$GCA_REPEAT_W4" \
#       --batch "$GCA_REPEAT_BATCH" \
#       --seed "$REPEAT_SEED" \
#       --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
#       --com-path "$COM_PATH" \
#       --project "$GCA_REPEAT_PROJECT" || exit $?

#     "$PYTHON_BIN" scripts/train_mdet_experiments.py gca-stage2 \
#       --label E2_14_GCA_GNN_repeat_no_gia \
#       --data "$DATA" \
#       --stage1-checkpoint "$GCA_REPEAT_STAGE1_CKPT" \
#       --stage1-epochs 100 \
#       --stage2-epochs 100 \
#       --variant conditional_gcn_margin_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCN_margin_residual.yaml \
#       --skip-existing \
#       --w4 "$GCA_REPEAT_W4" \
#       --batch "$GCA_REPEAT_BATCH" \
#       --seed "$REPEAT_SEED" \
#       --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
#       --com-path "$COM_CONDITIONAL_PATH" \
#       --project "$GCA_REPEAT_PROJECT" || exit $?
#   done
# fi

# # E2.15: matched control for E2.13.  Keep the best GIA-v2 Stage1 model and
# # train its ordinary attribute head in Stage2 without adding GCA or any GNN.
# # Use the same three seeds as E2.13 so the structural runs have a direct
# # no-GCA/GNN reference.  Set RUN_GIA_STAGE2_CONTROL=1 to launch the runs.
# if [ "${RUN_GIA_STAGE2_CONTROL:-0}" = "1" ]; then
#   GIA_CONTROL_STAGE1_CKPT=${GIA_CONTROL_STAGE1_CKPT:-runs/experiments/E2_1_GIA_v2_position/E2_1_GIA_v2_position_gia_v2_9_stage1_100_w4_0p5_seed_0/weights/best.pt}
#   GIA_CONTROL_PROJECT=${GIA_CONTROL_PROJECT:-runs/experiments/E2_15_GIA_stage2_control}
#   GIA_CONTROL_W4=${GIA_CONTROL_W4:-0.5}
#   GIA_CONTROL_BATCH=${GIA_CONTROL_BATCH:-16}
#   GIA_CONTROL_SEEDS=${GIA_CONTROL_SEEDS:-"0 1 2"}

#   if [ ! -f "$GIA_CONTROL_STAGE1_CKPT" ]; then
#     echo "Missing E2.15 input: $GIA_CONTROL_STAGE1_CKPT" >&2
#     exit 1
#   fi

#   for CONTROL_SEED in $GIA_CONTROL_SEEDS; do
#     "$PYTHON_BIN" scripts/train_mdet_experiments.py gca-stage2 \
#       --label E2_15_GIA_stage2_control \
#       --data "$DATA" \
#       --stage1-checkpoint "$GIA_CONTROL_STAGE1_CKPT" \
#       --stage1-epochs 100 \
#       --stage2-epochs 100 \
#       --variant gia_v2_9=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_9.yaml \
#       --skip-existing \
#       --w4 "$GIA_CONTROL_W4" \
#       --batch "$GIA_CONTROL_BATCH" \
#       --seed "$CONTROL_SEED" \
#       --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
#       --project "$GIA_CONTROL_PROJECT" || exit $?
#   done
# fi

# # Feature graph direction is closed after the gain screen. The block below is
# # retained only for exact reproduction of archived runs and is opt-in.
# if [ "${ENABLE_FEATURE_GRAPH:-0}" = "1" ]; then
#   echo "Running archived feature graph reproduction because ENABLE_FEATURE_GRAPH=1."
#   # Set FEATURE_EPOCHS=1 and FEATURE_PROJECT=... for a separate smoke run.
#   for REQUIRED_FILE in "$STAGE1_CKPT" "$COM_PATH"; do
#     if [ ! -f "$REQUIRED_FILE" ]; then
#       echo "Missing input: $REQUIRED_FILE" >&2
#       exit 1
#     fi
#   done

#   FEATURE_EPOCHS=${FEATURE_EPOCHS:-100}
#   FEATURE_PROJECT=${FEATURE_PROJECT:-runs/experiments/E2_2_feature_gain}
#   FEATURE_GAIN_VALUES=${FEATURE_GAIN_VALUES:-"2 4 8"}
#   FEATURE_LOCAL_GAIN=${FEATURE_LOCAL_GAIN:-4}

# # Gain=1 is the completed feature_gca_cross reference.  Reproduce it with
# # FEATURE_GAIN_VALUES="1 2 4 8" when the reference is not available locally.
# for GAIN in $FEATURE_GAIN_VALUES; do
#   "$PYTHON_BIN" scripts/train_mdet_experiments.py gca-stage2 \
#     --label "feature_gca_cross_gain_${GAIN}" \
#     --data "$DATA" \
#     --stage1-checkpoint "$STAGE1_CKPT" \
#     --stage1-epochs 100 \
#     --stage2-epochs "$FEATURE_EPOCHS" \
#     --variant "gca_cross_gain_${GAIN}=ultralytics/cfg/models/exp_ablation/yolov10x_feature_gca_cross.yaml" \
#     --feature-gain "$GAIN" \
#     --w4 0.5 \
#     --batch 16 \
#     --seed 0 \
#     --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
#       --com-path "$COM_PATH" \
#       --project "$FEATURE_PROJECT" || exit $?
# done

# # A no-graph local adapter control separates a useful residual amplitude from
# # a gain that only compensates for an ineffective graph message.
# "$PYTHON_BIN" scripts/train_mdet_experiments.py gca-stage2 \
#   --label "feature_local_cross_gain_${FEATURE_LOCAL_GAIN}" \
#   --data "$DATA" \
#   --stage1-checkpoint "$STAGE1_CKPT" \
#   --stage1-epochs 100 \
#   --stage2-epochs "$FEATURE_EPOCHS" \
#   --variant "local_cross_gain_${FEATURE_LOCAL_GAIN}=ultralytics/cfg/models/exp_ablation/yolov10x_feature_local_cross.yaml" \
#   --feature-gain "$FEATURE_LOCAL_GAIN" \
#   --w4 0.5 \
#   --batch 16 \
#   --seed 0 \
#   --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
#   --com-path "$COM_PATH" \
#   --project "$FEATURE_PROJECT" || exit $?
# fi

# E3.1 model-version/scale matrix.  This is opt-in and intentionally contains
# only native YOLO versions: no GIA, GCA/GNN, HO, or co-occurrence matrix is
# used.  One subprocess is launched per variant so an OOM or missing asset in
# one size does not prevent the remaining sizes from running.
E3_LABEL=${E3_LABEL:-E3_versions}
E3_PROJECT=${E3_PROJECT:-runs/experiments/E3_versions}
E3_STAGE1_EPOCHS=${E3_STAGE1_EPOCHS:-100}
E3_STAGE2_EPOCHS=${E3_STAGE2_EPOCHS:-100}
E3_W4=${E3_W4:-0.5}
E3_BATCH=${E3_BATCH:-16}
E3_SEED=${E3_SEED:-0}
E3_DEVICE=${E3_DEVICE:-0}
E3_IMGSZ=${E3_IMGSZ:-640}
E3_WORKERS=${E3_WORKERS:-2}
E3_HSV_H=${E3_HSV_H:-0}
E3_HSV_S=${E3_HSV_S:-0.2}
E3_HSV_V=${E3_HSV_V:-0.2}
E3_PRETRAIN_DIR=${E3_PRETRAIN_DIR:-}

# v8/v10/v11/v12/v13/YOLO26 use n/s/m/l/x here.  YOLOv9's native scale names
# are t/s/m/c/e, YOLOv10 additionally provides the b scale, and YOLOv13's
# official checkpoints are n/s/l/x only.
E3_SIZES=${E3_SIZES:-"n s m l x"}
E3_YOLOV9_SIZES=${E3_YOLOV9_SIZES:-"t s m c e"}
E3_YOLOV10_SIZES=${E3_YOLOV10_SIZES:-"n s m b l x"}
E3_YOLOV13_SIZES=${E3_YOLOV13_SIZES:-"n s l x"}
E3_FAMILIES=${E3_FAMILIES:-"yolov8 yolov9 yolov10 yolov11 yolov12 yolov13 yolov26"}

if [ "${RUN_E3_VERSIONS:-0}" = "1" ]; then
  E3_BASE_ARGS=(
    versions
    --label "$E3_LABEL"
    --data "$DATA"
    --stage1-epochs "$E3_STAGE1_EPOCHS"
    --stage2-epochs "$E3_STAGE2_EPOCHS"
    --w4 "$E3_W4"
    --batch "$E3_BATCH"
    --seed "$E3_SEED"
    --device "$E3_DEVICE"
    --imgsz "$E3_IMGSZ"
    --workers "$E3_WORKERS"
    --hsv-h "$E3_HSV_H"
    --hsv-s "$E3_HSV_S"
    --hsv-v "$E3_HSV_V"
    --skip-existing
  )
  E3_FAILED=()
  E3_TOTAL=0

  for E3_FAMILY in $E3_FAMILIES; do
    case "$E3_FAMILY" in
      yolov9)
        E3_FAMILY_SIZES="$E3_YOLOV9_SIZES"
        ;;
      yolov10)
        E3_FAMILY_SIZES="$E3_YOLOV10_SIZES"
        ;;
      yolov13)
        E3_FAMILY_SIZES="$E3_YOLOV13_SIZES"
        ;;
      yolov8|yolov11|yolov12|yolov26)
        E3_FAMILY_SIZES="$E3_SIZES"
        ;;
      *)
        echo "Unsupported E3 family: $E3_FAMILY" >&2
        exit 1
        ;;
    esac

    for E3_SIZE in $E3_FAMILY_SIZES; do
      E3_NAME="${E3_FAMILY}${E3_SIZE}"
      E3_CONFIG="ultralytics/cfg/models/experiments/${E3_NAME}-mdetect.yaml"
      E3_PRETRAIN="${E3_NAME}.pt"
      case "$E3_FAMILY" in
        yolov11)
          E3_PRETRAIN="yolo11${E3_SIZE}.pt"
          ;;
        yolov12)
          E3_PRETRAIN="yolo12${E3_SIZE}.pt"
          ;;
        yolov26)
          E3_PRETRAIN="yolo26${E3_SIZE}.pt"
          ;;
      esac
      if [ -n "$E3_PRETRAIN_DIR" ]; then
        E3_PRETRAIN="${E3_PRETRAIN_DIR%/}/${E3_PRETRAIN}"
      fi

      if [ ! -f "$E3_CONFIG" ]; then
        echo "Missing E3 config: $E3_CONFIG" >&2
        exit 1
      fi

      E3_TOTAL=$((E3_TOTAL + 1))
      echo "[E3 ${E3_TOTAL}] ${E3_NAME}: config=${E3_CONFIG}, pretrain=${E3_PRETRAIN}"
      if ! "$PYTHON_BIN" scripts/train_mdet_experiments.py \
        "${E3_BASE_ARGS[@]}" \
        --variant "${E3_NAME}=${E3_CONFIG}" \
        --pretrain-map "${E3_NAME}=${E3_PRETRAIN}" \
        --project "$E3_PROJECT"; then
        echo "[E3 failed] ${E3_NAME}; continuing with the remaining variants." >&2
        E3_FAILED+=("$E3_NAME")
      fi
    done
  done

  echo "E3 submitted ${E3_TOTAL} variants."
  if [ "${#E3_FAILED[@]}" -gt 0 ]; then
    echo "E3 failed variants: ${E3_FAILED[*]}" >&2
    exit 1
  fi
fi

# E2.18: corrected matched GIA+GCA comparison from the Test-mAP50-best GIA
# checkpoint (gia_v2_5_7).  E2.17 is retained as an archived partial-transfer
# sanity check; its GCA YAMLs did not contain the GIA-v2 backbone blocks.  The
# E2.18 YAMLs preserve GIA-v2 at layers 5/7 and replace only the attribute head.
# This batch is opt-in and runs nine corrected GCA Stage2 jobs plus three
# matched GIA-only controls.
if [ "${RUN_GIA_257_GCA_BATCH:-0}" = "1" ]; then
  GIA_257_STAGE1_CKPT=${GIA_257_STAGE1_CKPT:-runs/experiments/E2_1_GIA_v2_position/E2_1_GIA_v2_position_gia_v2_5_7_stage1_100_w4_0p5_seed_0/weights/best.pt}
  GIA_257_GCA_PROJECT=${GIA_257_GCA_PROJECT:-runs/experiments/E2_18_GIA_v2_5_7_GCA_corrected}
  GIA_257_CONTROL_PROJECT=${GIA_257_CONTROL_PROJECT:-runs/experiments/E2_18_GIA_v2_5_7_control}
  GIA_257_ADAPTIVE_CONFIG=${GIA_257_ADAPTIVE_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_5_7_GCA_adaptive_residual.yaml}
  GIA_257_GCN_CONFIG=${GIA_257_GCN_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_5_7_GCN_margin_residual.yaml}
  GIA_257_W4=${GIA_257_W4:-0.5}
  GIA_257_BATCH=${GIA_257_BATCH:-16}
  GIA_257_SEEDS=${GIA_257_SEEDS:-"0 1 2"}

  for REQUIRED_FILE in "$GIA_257_STAGE1_CKPT" "$GIA_257_ADAPTIVE_CONFIG" "$GIA_257_GCN_CONFIG" "$COM_PATH" "$COM_CONDITIONAL_PATH"; do
    if [ ! -f "$REQUIRED_FILE" ]; then
      echo "Missing E2.18 input: $REQUIRED_FILE" >&2
      exit 1
    fi
  done

  for GIA_257_SEED in $GIA_257_SEEDS; do
    "$PYTHON_BIN" scripts/train_mdet_experiments.py gca-stage2 \
      --label E2_18_GIA_v2_5_7_GCA_corrected \
      --data "$DATA" \
      --stage1-checkpoint "$GIA_257_STAGE1_CKPT" \
      --stage1-epochs 100 \
      --stage2-epochs 100 \
      --variant adaptive="$GIA_257_ADAPTIVE_CONFIG" \
      --gnn-types graphsage \
      --skip-existing \
      --w4 "$GIA_257_W4" \
      --batch "$GIA_257_BATCH" \
      --seed "$GIA_257_SEED" \
      --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
      --com-path "$COM_PATH" \
      --project "$GIA_257_GCA_PROJECT" || exit $?

    "$PYTHON_BIN" scripts/train_mdet_experiments.py gca-stage2 \
      --label E2_18_GIA_v2_5_7_GCA_corrected \
      --data "$DATA" \
      --stage1-checkpoint "$GIA_257_STAGE1_CKPT" \
      --stage1-epochs 100 \
      --stage2-epochs 100 \
      --variant cross_gcn_margin_residual="$GIA_257_GCN_CONFIG" \
      --skip-existing \
      --w4 "$GIA_257_W4" \
      --batch "$GIA_257_BATCH" \
      --seed "$GIA_257_SEED" \
      --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
      --com-path "$COM_PATH" \
      --project "$GIA_257_GCA_PROJECT" || exit $?

    "$PYTHON_BIN" scripts/train_mdet_experiments.py gca-stage2 \
      --label E2_18_GIA_v2_5_7_GCA_corrected \
      --data "$DATA" \
      --stage1-checkpoint "$GIA_257_STAGE1_CKPT" \
      --stage1-epochs 100 \
      --stage2-epochs 100 \
      --variant conditional_gcn_margin_residual="$GIA_257_GCN_CONFIG" \
      --skip-existing \
      --w4 "$GIA_257_W4" \
      --batch "$GIA_257_BATCH" \
      --seed "$GIA_257_SEED" \
      --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
      --com-path "$COM_CONDITIONAL_PATH" \
      --project "$GIA_257_GCA_PROJECT" || exit $?

    "$PYTHON_BIN" scripts/train_mdet_experiments.py gca-stage2 \
      --label E2_18_GIA_v2_5_7_control \
      --data "$DATA" \
      --stage1-checkpoint "$GIA_257_STAGE1_CKPT" \
      --stage1-epochs 100 \
      --stage2-epochs 100 \
      --variant gia_v2_5_7=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_5_7.yaml \
      --skip-existing \
      --w4 "$GIA_257_W4" \
      --batch "$GIA_257_BATCH" \
      --seed "$GIA_257_SEED" \
      --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
      --project "$GIA_257_CONTROL_PROJECT" || exit $?
  done
fi

# E2.3: corrected GIA-v2.5.7 seed-0 initialized 5x6 GCA/GNN matrix.
# The previous historical E2.3 entry is archived as E2_3_old in the
# experiment-summary helpers.  This batch is Stage2-only and preserves the
# complete GIA-v2.5.7 backbone from the fixed seed-0 Stage1 checkpoint.
# Conditional is intentionally run before Cross.  The five GNN operators are
# crossed with the five existing structural variants plus margin_residual,
# giving 30 jobs per matrix and 60 jobs in total.
if [ "${RUN_E2_3_GIA_GCA_BATCH:-0}" = "1" ]; then
  E2_3_STAGE1_CKPT=${E2_3_STAGE1_CKPT:-runs/experiments/E2_1_GIA_v2_position/E2_1_GIA_v2_position_gia_v2_5_7_stage1_100_w4_0p5_seed_0/weights/best.pt}
  E2_3_CONDITIONAL_PROJECT=${E2_3_CONDITIONAL_PROJECT:-runs/experiments/E2_3_GIA_v2_5_7_GCA5x6_conditional}
  E2_3_CROSS_PROJECT=${E2_3_CROSS_PROJECT:-runs/experiments/E2_3_GIA_v2_5_7_GCA5x6_cross}
  E2_3_CONTEXT_CONFIG=${E2_3_CONTEXT_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_5_7_GCA_context_residual.yaml}
  E2_3_ADAPTIVE_CONFIG=${E2_3_ADAPTIVE_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_5_7_GCA_adaptive_residual.yaml}
  E2_3_TWOHOP_CONFIG=${E2_3_TWOHOP_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_5_7_GCA_twohop_residual.yaml}
  E2_3_CONV_CONFIG=${E2_3_CONV_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_5_7_GCA_conv_adapter_residual.yaml}
  E2_3_MARGIN_CONFIG=${E2_3_MARGIN_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_5_7_GCA_margin_residual.yaml}
  E2_3_W4=${E2_3_W4:-0.5}
  E2_3_BATCH=${E2_3_BATCH:-16}
  E2_3_SEED=${E2_3_SEED:-0}
  E2_3_GNN_TYPES=${E2_3_GNN_TYPES:-"gca gcn gat graphsage gin"}

  # If a real old E2.3 output exists, preserve it under the requested archive
  # name before creating the new E2.3 projects; never overwrite an archive.
  E2_3_OLD_SOURCE=${E2_3_OLD_SOURCE:-runs/experiments/E2_3_GIA_GCA}
  E2_3_OLD_TARGET=${E2_3_OLD_TARGET:-runs/experiments/E2_3_old_GIA_GCA}
  if [ -d "$E2_3_OLD_SOURCE" ] && [ ! -e "$E2_3_OLD_TARGET" ]; then
    mv "$E2_3_OLD_SOURCE" "$E2_3_OLD_TARGET"
  elif [ -d "$E2_3_OLD_SOURCE" ] && [ -e "$E2_3_OLD_TARGET" ]; then
    echo "Both old E2.3 paths exist; refusing to overwrite: $E2_3_OLD_TARGET" >&2
    exit 1
  fi

  for REQUIRED_FILE in "$E2_3_STAGE1_CKPT" "$E2_3_CONTEXT_CONFIG" "$E2_3_ADAPTIVE_CONFIG" "$E2_3_TWOHOP_CONFIG" "$E2_3_CONV_CONFIG" "$E2_3_MARGIN_CONFIG" "$COM_CONDITIONAL_PATH" "$COM_PATH"; do
    if [ ! -f "$REQUIRED_FILE" ]; then
      echo "Missing E2.3 input: $REQUIRED_FILE" >&2
      exit 1
    fi
  done

  run_e2_3_matrix() {
    local matrix_name="$1"
    local matrix_path="$2"
    local project="$3"

    "$PYTHON_BIN" scripts/train_mdet_experiments.py gca-stage2 \
      --label "E2_3_GIA_v2_5_7_GCA5x6_${matrix_name}" \
      --data "$DATA" \
      --stage1-checkpoint "$E2_3_STAGE1_CKPT" \
      --stage1-epochs 100 \
      --stage2-epochs 100 \
      --variant context_cross="$E2_3_CONTEXT_CONFIG" \
      --variant context_conditional="$E2_3_CONTEXT_CONFIG" \
      --variant adaptive="$E2_3_ADAPTIVE_CONFIG" \
      --variant twohop="$E2_3_TWOHOP_CONFIG" \
      --variant conv_adapter="$E2_3_CONV_CONFIG" \
      --variant margin_residual="$E2_3_MARGIN_CONFIG" \
      --gnn-types $E2_3_GNN_TYPES \
      --skip-existing \
      --w4 "$E2_3_W4" \
      --batch "$E2_3_BATCH" \
      --seed "$E2_3_SEED" \
      --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
      --com-path "$matrix_path" \
      --project "$project" || exit $?
  }

  run_e2_3_matrix conditional "$COM_CONDITIONAL_PATH" "$E2_3_CONDITIONAL_PROJECT"
  run_e2_3_matrix cross "$COM_PATH" "$E2_3_CROSS_PROJECT"
fi

# E2.19: corrected GIA-v2.9 initialized 5x6 GCA/GNN matrix.
# The five GNN operators are crossed with the five existing structural
# variants plus the new multiclass-aware margin-residual variant. Cross and
# conditional train-only matrices are run in separate projects, for 30 jobs
# per matrix and 60 Stage2-only jobs in total. Every config below preserves
# the full GIA-v2.9 backbone; only the attribute head is changed.
if [ "${RUN_GIA_GCA_5X6_BATCH:-0}" = "1" ]; then
  GIA_5X6_STAGE1_CKPT=${GIA_5X6_STAGE1_CKPT:-runs/experiments/E2_1_GIA_v2_position/E2_1_GIA_v2_position_gia_v2_9_stage1_100_w4_0p5_seed_0/weights/best.pt}
  GIA_5X6_CROSS_PROJECT=${GIA_5X6_CROSS_PROJECT:-runs/experiments/E2_19_GIA_GCA5x6_cross}
  GIA_5X6_CONDITIONAL_PROJECT=${GIA_5X6_CONDITIONAL_PROJECT:-runs/experiments/E2_19_GIA_GCA5x6_conditional}
  GIA_5X6_CONTEXT_CONFIG=${GIA_5X6_CONTEXT_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_9_GCA_context_residual.yaml}
  GIA_5X6_ADAPTIVE_CONFIG=${GIA_5X6_ADAPTIVE_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_9_GCA_adaptive_residual.yaml}
  GIA_5X6_TWOHOP_CONFIG=${GIA_5X6_TWOHOP_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_9_GCA_twohop_residual.yaml}
  GIA_5X6_CONV_CONFIG=${GIA_5X6_CONV_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_9_GCA_conv_adapter_residual.yaml}
  GIA_5X6_MARGIN_CONFIG=${GIA_5X6_MARGIN_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_9_GCA_margin_residual.yaml}
  GIA_5X6_W4=${GIA_5X6_W4:-0.5}
  GIA_5X6_BATCH=${GIA_5X6_BATCH:-16}
  GIA_5X6_SEED=${GIA_5X6_SEED:-0}
  GIA_5X6_GNN_TYPES=${GIA_5X6_GNN_TYPES:-"gca gcn gat graphsage gin"}

  for REQUIRED_FILE in "$GIA_5X6_STAGE1_CKPT" "$GIA_5X6_CONTEXT_CONFIG" "$GIA_5X6_ADAPTIVE_CONFIG" "$GIA_5X6_TWOHOP_CONFIG" "$GIA_5X6_CONV_CONFIG" "$GIA_5X6_MARGIN_CONFIG" "$COM_PATH" "$COM_CONDITIONAL_PATH"; do
    if [ ! -f "$REQUIRED_FILE" ]; then
      echo "Missing E2.19 input: $REQUIRED_FILE" >&2
      exit 1
    fi
  done

  run_gia_5x6_matrix() {
    local matrix_name="$1"
    local matrix_path="$2"
    local project="$3"

    "$PYTHON_BIN" scripts/train_mdet_experiments.py gca-stage2 \
      --label "E2_19_GIA_GCA5x6_${matrix_name}" \
      --data "$DATA" \
      --stage1-checkpoint "$GIA_5X6_STAGE1_CKPT" \
      --stage1-epochs 100 \
      --stage2-epochs 100 \
      --variant context_cross="$GIA_5X6_CONTEXT_CONFIG" \
      --variant context_conditional="$GIA_5X6_CONTEXT_CONFIG" \
      --variant adaptive="$GIA_5X6_ADAPTIVE_CONFIG" \
      --variant twohop="$GIA_5X6_TWOHOP_CONFIG" \
      --variant conv_adapter="$GIA_5X6_CONV_CONFIG" \
      --variant margin_residual="$GIA_5X6_MARGIN_CONFIG" \
      --gnn-types $GIA_5X6_GNN_TYPES \
      --skip-existing \
      --w4 "$GIA_5X6_W4" \
      --batch "$GIA_5X6_BATCH" \
      --seed "$GIA_5X6_SEED" \
      --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
      --com-path "$matrix_path" \
      --project "$project" || exit $?
  }

  run_gia_5x6_matrix cross "$COM_PATH" "$GIA_5X6_CROSS_PROJECT"
  run_gia_5x6_matrix conditional "$COM_CONDITIONAL_PATH" "$GIA_5X6_CONDITIONAL_PROJECT"
fi

# E2.20: stability repeats for the five tied Test-F1 leaders from E2.19.
# Keep the GIA-v2.9 seed-0 Stage-1 checkpoint fixed and repeat only Stage 2
# with seeds 1 and 2.  Each helper call selects one exact GNN x structure
# combination, so this block launches 5 x 2 = 10 jobs rather than a Cartesian
# product containing unselected combinations.
if [ "${RUN_GIA_GCA_TOP5_STABILITY:-0}" = "1" ]; then
  GIA_TOP5_STAGE1_CKPT=${GIA_TOP5_STAGE1_CKPT:-runs/experiments/E2_1_GIA_v2_position/E2_1_GIA_v2_position_gia_v2_9_stage1_100_w4_0p5_seed_0/weights/best.pt}
  GIA_TOP5_PROJECT=${GIA_TOP5_PROJECT:-runs/experiments/E2_20_GIA_GCA_top5_stability}
  GIA_TOP5_CONTEXT_CONFIG=${GIA_TOP5_CONTEXT_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_9_GCA_context_residual.yaml}
  GIA_TOP5_TWOHOP_CONFIG=${GIA_TOP5_TWOHOP_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_9_GCA_twohop_residual.yaml}
  GIA_TOP5_ADAPTIVE_CONFIG=${GIA_TOP5_ADAPTIVE_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_9_GCA_adaptive_residual.yaml}
  GIA_TOP5_W4=${GIA_TOP5_W4:-0.5}
  GIA_TOP5_BATCH=${GIA_TOP5_BATCH:-16}
  GIA_TOP5_SEEDS=${GIA_TOP5_SEEDS:-"1 2"}

  for REQUIRED_FILE in "$GIA_TOP5_STAGE1_CKPT" "$GIA_TOP5_CONTEXT_CONFIG" "$GIA_TOP5_TWOHOP_CONFIG" "$GIA_TOP5_ADAPTIVE_CONFIG" "$COM_CONDITIONAL_PATH"; do
    if [ ! -f "$REQUIRED_FILE" ]; then
      echo "Missing E2.20 input: $REQUIRED_FILE" >&2
      exit 1
    fi
  done

  run_gia_top5_one() {
    local seed="$1"
    local variant_name="$2"
    local config="$3"
    local gnn_type="$4"

    "$PYTHON_BIN" scripts/train_mdet_experiments.py gca-stage2 \
      --label E2_20_GIA_GCA_top5_stability \
      --data "$DATA" \
      --stage1-checkpoint "$GIA_TOP5_STAGE1_CKPT" \
      --stage1-epochs 100 \
      --stage2-epochs 100 \
      --variant "$variant_name=$config" \
      --gnn-types "$gnn_type" \
      --skip-existing \
      --w4 "$GIA_TOP5_W4" \
      --batch "$GIA_TOP5_BATCH" \
      --seed "$seed" \
      --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
      --com-path "$COM_CONDITIONAL_PATH" \
      --project "$GIA_TOP5_PROJECT" || exit $?
  }

  for GIA_TOP5_SEED in $GIA_TOP5_SEEDS; do
    run_gia_top5_one "$GIA_TOP5_SEED" context_conditional "$GIA_TOP5_CONTEXT_CONFIG" gca
    run_gia_top5_one "$GIA_TOP5_SEED" context_cross "$GIA_TOP5_CONTEXT_CONFIG" gca
    run_gia_top5_one "$GIA_TOP5_SEED" twohop "$GIA_TOP5_TWOHOP_CONFIG" gca
    run_gia_top5_one "$GIA_TOP5_SEED" adaptive "$GIA_TOP5_ADAPTIVE_CONFIG" gin
    run_gia_top5_one "$GIA_TOP5_SEED" twohop "$GIA_TOP5_TWOHOP_CONFIG" graphsage
  done
fi

# E2.23: conditional margin-residual GCA staged ablation.
# Each run first trains only the original attribute head (cv4/one2one_cv4) for
# k1 epochs with the GCA residual fixed at identity, then freezes the complete
# original path and trains only the GCA/GNN residual heads for k2 epochs.  The
# five margin-residual operators are evaluated for k1/k2 = 50/50 and 66/34,
# giving 10 final checkpoints.  This block is opt-in.
if [ "${RUN_E2_23_GCA_WARMUP_BATCH:-0}" = "1" ]; then
  E2_23_STAGE1_CKPT=${E2_23_STAGE1_CKPT:-runs/experiments/E2_1_GIA_v2_position/E2_1_GIA_v2_position_gia_v2_5_7_stage1_100_w4_0p5_seed_0/weights/best.pt}
  E2_23_PROJECT=${E2_23_PROJECT:-runs/experiments/E2_23_GIA_v2_5_7_GCA_attr_warmup_conditional}
  E2_23_MARGIN_CONFIG=${E2_23_MARGIN_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_5_7_GCA_margin_residual.yaml}
  E2_23_W4=${E2_23_W4:-0.5}
  E2_23_BATCH=${E2_23_BATCH:-16}
  E2_23_GNN_TYPES=${E2_23_GNN_TYPES:-"gca gcn gat graphsage gin"}

  for REQUIRED_FILE in "$E2_23_STAGE1_CKPT" "$E2_23_MARGIN_CONFIG" "$COM_CONDITIONAL_PATH"; do
    if [ ! -f "$REQUIRED_FILE" ]; then
      echo "Missing E2.23 input: $REQUIRED_FILE" >&2
      exit 1
    fi
  done

  for E2_23_SCHEDULE in 50:50 66:34; do
    E2_23_K1=${E2_23_SCHEDULE%%:*}
    E2_23_K2=${E2_23_SCHEDULE##*:}
    "$PYTHON_BIN" scripts/train_mdet_experiments.py gca-warmup \
      --label E2_23_GIA_v2_5_7_GCA_attr_warmup_conditional \
      --data "$DATA" \
      --stage1-checkpoint "$E2_23_STAGE1_CKPT" \
      --k1-epochs "$E2_23_K1" \
      --k2-epochs "$E2_23_K2" \
      --variant margin_residual="$E2_23_MARGIN_CONFIG" \
      --gnn-types $E2_23_GNN_TYPES \
      --skip-existing \
      --w4 "$E2_23_W4" \
      --batch "$E2_23_BATCH" \
      --seed 0 \
      --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
      --com-path "$COM_CONDITIONAL_PATH" \
      --project "$E2_23_PROJECT" || exit $?
  done
fi

# E2.24: direct Stage2 training of MHA + conditional margin-residual graphs.
# Each job starts from the fixed GIA-v2.5.7 seed-0 Stage1 best checkpoint and
# directly trains the attribute/GCA branch for 100 epochs.  Multi-head
# attention mixes the per-pixel attribute margins before one of the five graph
# operators (GCA/GCN/GAT/GraphSAGE/GIN) applies the conditional
# margin-residual propagation.  This block produces exactly five jobs and is
# opt-in.
if [ "${RUN_E2_24_GCA_MHA_BATCH:-0}" = "1" ]; then
  E2_24_STAGE1_CKPT=${E2_24_STAGE1_CKPT:-runs/experiments/E2_1_GIA_v2_position/E2_1_GIA_v2_position_gia_v2_5_7_stage1_100_w4_0p5_seed_0/weights/best.pt}
  E2_24_PROJECT=${E2_24_PROJECT:-runs/experiments/E2_24_GIA_v2_5_7_GCA_MHA_margin_residual_conditional}
  E2_24_MHA_MARGIN_CONFIG=${E2_24_MHA_MARGIN_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_5_7_GCA_mha_margin_residual.yaml}
  E2_24_DEVICE=${E2_24_DEVICE:-0}
  E2_24_W4=${E2_24_W4:-0.5}
  E2_24_BATCH=${E2_24_BATCH:-16}
  E2_24_GNN_TYPES=${E2_24_GNN_TYPES:-"gca gcn gat graphsage gin"}

  for REQUIRED_FILE in "$E2_24_STAGE1_CKPT" "$E2_24_MHA_MARGIN_CONFIG" "$COM_CONDITIONAL_PATH"; do
    if [ ! -f "$REQUIRED_FILE" ]; then
      echo "Missing E2.24 input: $REQUIRED_FILE" >&2
      exit 1
    fi
  done

  "$PYTHON_BIN" scripts/train_mdet_experiments.py gca-stage2 \
    --label E2_24_GIA_v2_5_7_GCA_MHA_margin_residual_conditional \
    --data "$DATA" \
    --stage1-checkpoint "$E2_24_STAGE1_CKPT" \
    --stage1-epochs 100 \
    --stage2-epochs 100 \
    --device "$E2_24_DEVICE" \
    --variant mha_margin_residual="$E2_24_MHA_MARGIN_CONFIG" \
    --gnn-types $E2_24_GNN_TYPES \
    --skip-existing \
    --w4 "$E2_24_W4" \
    --batch "$E2_24_BATCH" \
    --seed 0 \
    --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
    --com-path "$COM_CONDITIONAL_PATH" \
    --project "$E2_24_PROJECT" || exit $?
fi

# E2.25: direct Stage2 training of feature-logit MHA + conditional
# margin-residual graphs.  This is the feature-aware companion to E2.24:
# attribute features provide the queries and attribute logits provide the
# keys/values before the same five graph operators are applied.  It produces
# five additional jobs and is opt-in.
if [ "${RUN_E2_25_GCA_FEATURE_LOGIT_MHA_BATCH:-0}" = "1" ]; then
  E2_25_STAGE1_CKPT=${E2_25_STAGE1_CKPT:-runs/experiments/E2_1_GIA_v2_position/E2_1_GIA_v2_position_gia_v2_5_7_stage1_100_w4_0p5_seed_0/weights/best.pt}
  E2_25_PROJECT=${E2_25_PROJECT:-runs/experiments/E2_25_GIA_v2_5_7_GCA_feature_logit_MHA_margin_residual_conditional}
  E2_25_CONFIG=${E2_25_CONFIG:-ultralytics/cfg/models/exp_ablation/yolov10x_GIA_v2_5_7_GCA_feature_logit_mha_margin_residual.yaml}
  E2_25_DEVICE=${E2_25_DEVICE:-1}
  E2_25_W4=${E2_25_W4:-0.5}
  E2_25_BATCH=${E2_25_BATCH:-16}
  E2_25_GNN_TYPES=${E2_25_GNN_TYPES:-"gca gcn gat graphsage gin"}

  for REQUIRED_FILE in "$E2_25_STAGE1_CKPT" "$E2_25_CONFIG" "$COM_CONDITIONAL_PATH"; do
    if [ ! -f "$REQUIRED_FILE" ]; then
      echo "Missing E2.25 input: $REQUIRED_FILE" >&2
      exit 1
    fi
  done

  "$PYTHON_BIN" scripts/train_mdet_experiments.py gca-stage2 \
    --label E2_25_GIA_v2_5_7_GCA_feature_logit_MHA_margin_residual_conditional \
    --data "$DATA" \
    --stage1-checkpoint "$E2_25_STAGE1_CKPT" \
    --stage1-epochs 100 \
    --stage2-epochs 100 \
    --device "$E2_25_DEVICE" \
    --variant feature_logit_mha_margin_residual="$E2_25_CONFIG" \
    --gnn-types $E2_25_GNN_TYPES \
    --skip-existing \
    --w4 "$E2_25_W4" \
    --batch "$E2_25_BATCH" \
    --seed 0 \
    --hsv-h 0 --hsv-s 0.2 --hsv-v 0.2 \
    --com-path "$COM_CONDITIONAL_PATH" \
    --project "$E2_25_PROJECT" || exit $?
fi
