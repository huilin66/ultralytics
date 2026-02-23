import shutil

import cv2
import numpy as np
import os
from tqdm import tqdm


def img_concate(img_path_list, output_path):
    img_list = [cv2.imread(img_path) for img_path in img_path_list]
    img_concat = np.concatenate(img_list, axis=1)
    cv2.imwrite(output_path, img_concat)
    print(f'save to {output_path}')

def imgs_concate(img_dir_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    img_list = os.listdir(img_dir_list[0])
    for img_name in tqdm(img_list):
        if not img_name.endswith('.jpg'):
            continue
        img_path_list = [os.path.join(img_dir, img_name) for img_dir in img_dir_list]
        output_path = os.path.join(output_dir, img_name)
        img_concate(img_path_list, output_path)
    print(f'save to {output_dir}')

def select_imgs(img_dir_list, output_dir, select_name, img_dir_suffix):
    os.makedirs(output_dir, exist_ok=True)
    for i, input_dir in enumerate(img_dir_list):
        input_path = os.path.join(input_dir, select_name)
        output_path = os.path.join(output_dir, select_name.replace('.jpg', f'_{img_dir_suffix[i]}.jpg'))
        shutil.copyfile(input_path, output_path)

if __name__ == '__main__':
    src_dir = r'/nfsv4/23039356r/data/billboard/data0806_m/yolo_rgb_detection5_10_c'
    brightness_list = [
        0.5,
        0.75,
        1.25,
        1.5,
    ]
    img_dir_list = [os.path.join(src_dir,  'image_infer_vis')]
    for brightness in brightness_list:
        dst_dir = os.path.join(f'{src_dir}_b{int(brightness*100)}', 'image_infer_vis')
        img_dir_list.append(dst_dir)
    output_dir = src_dir+'_val_infer_cat'
    # imgs_concate(img_dir_list, output_dir)
    select_name = 'FLIR0862.jpg'
    output_dir = src_dir+'_val_infer_select'
    select_imgs(img_dir_list, output_dir, select_name,
                ['b100', 'b050', 'b075', 'b125', 'b150'])