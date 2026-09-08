DATA=ultralytics/cfg/mayolo_r1/mayolo_v3.yaml

python scripts/gpu_memory_smoke_test.py \
  --data "$DATA" \
  --device 0 \
  --imgsz 640 \
  --batch 1 \
  --epochs 2 \
  --project runs/gpu_memory_smoke


# python scripts/train_mdet_experiments.py stage1-sweep \
#   --data "$DATA" \
#   --model ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml \
#   --pretrain yolov10x.pt \
#   --stage1-values 100 200 300 400 500 \
#   --w4 0.5 \
#   --project runs/experiments/E0_stage1_sweep