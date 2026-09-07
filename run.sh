python scripts/train_mdet_experiments.py w4 \
  --data /localnvme/project/ultralytics/ultralytics/cfg/mayolo_r1/mayolo_v2.yaml \
  --model ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml \
  --pretrain yolov10x.pt \
  --w4-values 0.25 \
  --project runs/experiments/E1_w4_test \
  --stage1-epochs 10 \
  --stage2-epochs 10