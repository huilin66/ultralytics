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
#   --w4-values 0.25 0.5 1.0 \
#   --project runs/experiments/E1_w4


# The GCA YAML files contain the original Linux matrix path. Override it with
# an environment variable when the matrix is stored elsewhere:
#   COM_PATH=/path/to/co_occurrence_matrix6.csv bash run.sh
COM_PATH=${COM_PATH:-/localnvme/data/billboard/mayolo_v3/co_occurrence_matrix_train.csv}

# E2.1 GIA-position ablation is not launched here because the current branch
# has no current-compatible GIA-only or multi-position YAML configuration.

# E2.2 GCA structure ablation: baseline vs GCA.
python scripts/train_mdet_experiments.py gca-structure \
  --label E2_2_GCA_structure \
  --data "$DATA" \
  --pretrain yolov10x.pt \
  --variant baseline=ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml \
  --variant gca=ultralytics/cfg/models/exp_ablation/yolov10x_GCA.yaml \
  --com-path "$COM_PATH" \
  --project runs/experiments/E2_2_GCA

# E2.3 joint GIA+GCA ablation: baseline vs GIA+GCA.
python scripts/train_mdet_experiments.py gia-gca \
  --label E2_3_GIA_GCA \
  --data "$DATA" \
  --pretrain yolov10x.pt \
  --variant baseline=ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml \
  --variant gia_gca=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_GCA.yaml \
  --com-path "$COM_PATH" \
  --project runs/experiments/E2_3_GIA_GCA

# E2.4 HO training: train the available HO+GCA configuration.
python scripts/train_mdet_experiments.py ho \
  --label E2_4_HO \
  --data "$DATA" \
  --pretrain yolov10x.pt \
  --variant ho_gca=ultralytics/cfg/models/exp_ablation/yolov10x_HO_GCA.yaml \
  --com-path "$COM_PATH" \
  --project runs/experiments/E2_4_HO

# Compare the trained HO checkpoint with native and one-to-many inference.
python scripts/eval_mdet_experiments.py ho \
  --weights runs/experiments/E2_4_HO/E2_4_HO_ho_gca_w4_0p5_seed_0_stage2/weights/best.pt \
  --data "$DATA" \
  --mode both \
  --project runs/experiments/E2_4_HO_eval
