import os

import yaml
from tqdm import tqdm
label_dir = r'/data/huilin/scrinvme/huilin/PVD/Thermal PV Panel Detection Dataset for UAV Inspection/train/labels'
sta_dict = {}
for filename in tqdm(os.listdir(label_dir)):
    file_path = os.path.join(label_dir, filename)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split(' ')
            class_id = int(parts[0])
            if class_id in sta_dict:
                sta_dict[class_id] += 1
            else:
                sta_dict[class_id] = 1

print(sta_dict)