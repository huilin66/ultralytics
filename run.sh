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
#   --w4 0.5 \
#   --project runs/experiments/E0_hsv_ablation

# python scripts/train_mdet_experiments.py stage1-sweep \
#   --data "$DATA" \
#   --model ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml \
#   --pretrain yolov10x.pt \
#   --stage1-values 100 200 300 400 500 \
#   --w4 0.5 \
#   --project runs/experiments/E0_stage1_sweep


python scripts/train_mdet_experiments.py stage2-sweep \
  --data ultralytics/cfg/mayolo_r1/mayolo_v3.yaml \
  --model ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml \
  --stage1-checkpoint runs/experiments/E0_stage1_sweep/E0_stage1_stage1_100_w4_0p5_seed_0/weights/best.pt \
  --stage1-epochs 100 \
  --stage2-values 50 100 150 200 \
  --w4 0.5 \
  --project runs/experiments/E0_stage2_sweep