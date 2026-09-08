python scripts/train_mdet_experiments.py stage1-sweep \
  --data path/to/billboard_mdet.yaml \
  --model ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml \
  --pretrain yolov10x.pt \
  --stage1-values 100 200 300 400 500 \
  --w4 0.5 \
  --project runs/experiments/E0_stage1_sweep