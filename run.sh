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


# The GCA YAML file contains the original Linux matrix path. Override it with
# an environment variable when the matrix is stored elsewhere:
#   COM_PATH=/path/to/co_occurrence_matrix_train.csv bash run.sh
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

# Previous E2.2 structural follow-up.  Retained as comments for visual
# comparison; these jobs are superseded by the 5x5 matrix below and will not
# be retrained by this script.
# STAGE1_CKPT=runs/experiments/E1_w4/E1_w4_base_w4_0p5_seed_0_stage1/weights/best.pt
# if [ ! -f "$COM_CONDITIONAL_PATH" ]; then
#   echo "Missing conditional COM: $COM_CONDITIONAL_PATH" >&2
#   exit 1
# fi
#
# python scripts/train_mdet_experiments.py gca-stage2 \
#   --label E2_2_GCA_context_residual \
#   --data "$DATA" \
#   --stage1-checkpoint "$STAGE1_CKPT" \
#   --stage1-epochs 100 \
#   --stage2-epochs 100 \
#   --variant gca_context_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_context_residual.yaml \
#   --w4 0.5 \
#   --batch 16 \
#   --seed 0 \
#   --com-path "$COM_PATH" \
#   --project runs/experiments/E2_2_GCA_context_residual
#
# python scripts/train_mdet_experiments.py gca-stage2 \
#   --label E2_2_GCA_context_conditional \
#   --data "$DATA" \
#   --stage1-checkpoint "$STAGE1_CKPT" \
#   --stage1-epochs 100 \
#   --stage2-epochs 100 \
#   --variant gca_context_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_context_residual.yaml \
#   --w4 0.5 \
#   --batch 16 \
#   --seed 0 \
#   --com-path "$COM_CONDITIONAL_PATH" \
#   --project runs/experiments/E2_2_GCA_context_conditional
#
# python scripts/train_mdet_experiments.py gca-stage2 \
#   --label E2_2_GCA_adaptive_conditional \
#   --data "$DATA" \
#   --stage1-checkpoint "$STAGE1_CKPT" \
#   --stage1-epochs 100 \
#   --stage2-epochs 100 \
#   --variant gca_adaptive_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_adaptive_residual.yaml \
#   --w4 0.5 \
#   --batch 16 \
#   --seed 0 \
#   --com-path "$COM_CONDITIONAL_PATH" \
#   --project runs/experiments/E2_2_GCA_adaptive_conditional
#
# python scripts/train_mdet_experiments.py gca-stage2 \
#   --label E2_2_GCA_twohop_conditional \
#   --data "$DATA" \
#   --stage1-checkpoint "$STAGE1_CKPT" \
#   --stage1-epochs 100 \
#   --stage2-epochs 100 \
#   --variant gca_twohop_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_twohop_residual.yaml \
#   --w4 0.5 \
#   --batch 16 \
#   --seed 0 \
#   --com-path "$COM_CONDITIONAL_PATH" \
#   --project runs/experiments/E2_2_GCA_twohop_conditional
#
# python scripts/train_mdet_experiments.py gca-stage2 \
#   --label E2_2_GCA_conv_adapter_conditional \
#   --data "$DATA" \
#   --stage1-checkpoint "$STAGE1_CKPT" \
#   --stage1-epochs 100 \
#   --stage2-epochs 100 \
#   --variant gca_conv_adapter_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_conv_adapter_residual.yaml \
#   --w4 0.5 \
#   --batch 16 \
#   --seed 0 \
#   --com-path "$COM_CONDITIONAL_PATH" \
#   --project runs/experiments/E2_2_GCA_conv_adapter_conditional
#
# python scripts/train_mdet_experiments.py gca-stage2 \
#   --label E2_2_GCA_context_GNN_conditional \
#   --data "$DATA" \
#   --stage1-checkpoint "$STAGE1_CKPT" \
#   --stage1-epochs 100 \
#   --stage2-epochs 100 \
#   --variant gca_context=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_context_residual.yaml \
#   --variant gcn_context=ultralytics/cfg/models/exp_ablation/yolov10x_GCN_context_residual.yaml \
#   --variant gat_context=ultralytics/cfg/models/exp_ablation/yolov10x_GAT_context_residual.yaml \
#   --variant graphsage_context=ultralytics/cfg/models/exp_ablation/yolov10x_GraphSAGE_context_residual.yaml \
#   --variant gin_context=ultralytics/cfg/models/exp_ablation/yolov10x_GIN_context_residual.yaml \
#   --w4 0.5 \
#   --batch 16 \
#   --seed 0 \
#   --com-path "$COM_CONDITIONAL_PATH" \
#   --project runs/experiments/E2_2_GCA_context_GNN_conditional

# E2.2 true 5x5 matrix: five structural variants crossed with five GNN
# operators.  Each command below expands to five Stage2-only runs through
# --gnn-types: GCA, GCN, GAT, GraphSAGE, and GIN.  All 25 runs use the same
# fixed Stage1 checkpoint, 100-epoch Stage2 budget, w4, batch size, seed, and
# train-only co-occurrence matrix policy.  Only mdet code is involved.

STAGE1_CKPT=runs/experiments/E1_w4/E1_w4_base_w4_0p5_seed_0_stage1/weights/best.pt
if [ ! -f "$STAGE1_CKPT" ]; then
  echo "Missing Stage1 checkpoint: $STAGE1_CKPT" >&2
  exit 1
fi
if [ ! -f "$COM_PATH" ]; then
  echo "Missing cross-normalized COM: $COM_PATH" >&2
  exit 1
fi
if [ ! -f "$COM_CONDITIONAL_PATH" ]; then
  echo "Missing conditional COM: $COM_CONDITIONAL_PATH" >&2
  echo "Generate it with generate_com.py --split train --mode conditional --smoothing 1.0" >&2
  exit 1
fi

python scripts/train_mdet_experiments.py gca-stage2 \
  --label E2_2_GCA5x5_context_cross \
  --data "$DATA" \
  --stage1-checkpoint "$STAGE1_CKPT" \
  --stage1-epochs 100 \
  --stage2-epochs 100 \
  --variant context_cross=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_context_residual.yaml \
  --gnn-types gca gcn gat graphsage gin \
  --w4 0.5 \
  --batch 16 \
  --seed 0 \
  --com-path "$COM_PATH" \
  --project runs/experiments/E2_2_GCA5x5_context_cross

python scripts/train_mdet_experiments.py gca-stage2 \
  --label E2_2_GCA5x5_context_conditional \
  --data "$DATA" \
  --stage1-checkpoint "$STAGE1_CKPT" \
  --stage1-epochs 100 \
  --stage2-epochs 100 \
  --variant context_conditional=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_context_residual.yaml \
  --gnn-types gca gcn gat graphsage gin \
  --w4 0.5 \
  --batch 16 \
  --seed 0 \
  --com-path "$COM_CONDITIONAL_PATH" \
  --project runs/experiments/E2_2_GCA5x5_context_conditional

python scripts/train_mdet_experiments.py gca-stage2 \
  --label E2_2_GCA5x5_adaptive_conditional \
  --data "$DATA" \
  --stage1-checkpoint "$STAGE1_CKPT" \
  --stage1-epochs 100 \
  --stage2-epochs 100 \
  --variant adaptive_conditional=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_adaptive_residual.yaml \
  --gnn-types gca gcn gat graphsage gin \
  --w4 0.5 \
  --batch 16 \
  --seed 0 \
  --com-path "$COM_CONDITIONAL_PATH" \
  --project runs/experiments/E2_2_GCA5x5_adaptive_conditional

python scripts/train_mdet_experiments.py gca-stage2 \
  --label E2_2_GCA5x5_twohop_conditional \
  --data "$DATA" \
  --stage1-checkpoint "$STAGE1_CKPT" \
  --stage1-epochs 100 \
  --stage2-epochs 100 \
  --variant twohop_conditional=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_twohop_residual.yaml \
  --gnn-types gca gcn gat graphsage gin \
  --w4 0.5 \
  --batch 16 \
  --seed 0 \
  --com-path "$COM_CONDITIONAL_PATH" \
  --project runs/experiments/E2_2_GCA5x5_twohop_conditional

python scripts/train_mdet_experiments.py gca-stage2 \
  --label E2_2_GCA5x5_conv_adapter_conditional \
  --data "$DATA" \
  --stage1-checkpoint "$STAGE1_CKPT" \
  --stage1-epochs 100 \
  --stage2-epochs 100 \
  --variant conv_adapter_conditional=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_conv_adapter_residual.yaml \
  --gnn-types gca gcn gat graphsage gin \
  --w4 0.5 \
  --batch 16 \
  --seed 0 \
  --com-path "$COM_CONDITIONAL_PATH" \
  --project runs/experiments/E2_2_GCA5x5_conv_adapter_conditional


# The completed 10-position GIA-v2 matrix is retained below for reference;
# leave it commented to avoid retraining those experiments.
