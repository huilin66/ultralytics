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

# E2.1 GIA-position ablation: single positions 5/7/8/9/10 and the 5+7
# combination, each with the ordinary and residual GIA variants.  The
# baseline is intentionally not repeated here; it is already trained by the
# other ablation jobs.
python scripts/train_mdet_experiments.py gia-position \
  --label E2_1_GIA_position \
  --data "$DATA" \
  --pretrain yolov10x.pt \
  --variant gia5=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_5.yaml \
  --variant gia5_res=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_5_Res.yaml \
  --variant gia7=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_7.yaml \
  --variant gia7_res=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_7_Res.yaml \
  --variant gia8=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_8.yaml \
  --variant gia8_res=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_8_Res.yaml \
  --variant gia9=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_9.yaml \
  --variant gia9_res=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_9_Res.yaml \
  --variant gia10=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_10.yaml \
  --variant gia10_res=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_10_Res.yaml \
  --variant gia5_7=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_5_7.yaml \
  --variant gia5_7_res=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_5_7_Res.yaml \
  --project runs/experiments/E2_1_GIA_position

