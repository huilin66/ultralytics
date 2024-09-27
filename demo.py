import torch
w1_path = r'runs/mdetect/exp_yolo10x_m34/weights/best.pt'
weights1 = torch.load(w1_path)
print(weights1.keys())