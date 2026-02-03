import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def vis2(img_path, input_dir1, input_dir2, feature_name, feature_idx):
    # 1. load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    # 2. load features
    feat1 = np.load(os.path.join(input_dir1, feature_name))
    feat2 = np.load(os.path.join(input_dir2, feature_name))

    # 去掉 batch 维
    if feat1.ndim == 4:
        feat1 = feat1[0]
        feat2 = feat2[0]

    # 3. select channel
    f1 = feat1[idx]
    f2 = feat2[idx]

    # 4. normalize (必须一致)
    def norm(x):
        x = x - x.min()
        return x / (x.max() + 1e-6)

    f1 = norm(f1)
    f2 = norm(f2)

    # 5. resize to image size
    f1 = cv2.resize(f1, (W, H))
    f2 = cv2.resize(f2, (W, H))

    # 6. visualization
    save_path1 = os.path.join(input_dir1, feature_name.replace('.npy', f'_{feature_idx}.png'))
    save_path2 = os.path.join(input_dir2, feature_name.replace('.npy', f'_{feature_idx}.png'))

    cv2.imwrite(save_path1, np.uint8(f1 * 255))
    print(f'save to {save_path1}')
    cv2.imwrite(save_path2, np.uint8(f2 * 255))
    print(f'save to {save_path2}')


if __name__ == '__main__':
    pass
    img_path = r'/scrinvme/huilin/tp/FLIR1444_img.png'
    input_dir1 = r'/scrinvme/huilin/tp/FLIR1444_img_yolo10/FLIR1444_img'
    input_dir2 = r'/scrinvme/huilin/tp/FLIR1444_img_mayolo/FLIR1444_img'
    feature_name = r'stage6_C2fCIB_features.npy'
    idx = 13
    vis2(img_path, input_dir1, input_dir2, feature_name, idx)
