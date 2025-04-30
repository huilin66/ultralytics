import os
import shutil

from tqdm import tqdm
from pathlib import Path

def mseg2seg(input_dir, output_dir, cp_img=True):
    input_label_dir = os.path.join(input_dir, 'labels')
    output_label_dir = os.path.join(output_dir, 'labels')
    input_image_dir = os.path.join(input_dir, 'images')
    output_image_dir = os.path.join(output_dir, 'images')
    image_list = os.listdir(input_image_dir)
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    for image_name in tqdm(image_list):
        label_name = Path(image_name).stem + '.txt'
        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)
        with open(input_label_path, 'r') as f_in, open(output_label_path, 'w+') as f_out:
            lines = f_in.readlines()
            for line in lines:
                num_list = line.split(' ')
                attribute_num = int(num_list[1])
                num_list_seg = num_list[:1]+num_list[1+attribute_num+1:]
                line_seg = ' '.join(num_list_seg)
                f_out.write(line_seg)
        if cp_img:
            input_image_path = os.path.join(input_image_dir, image_name)
            output_image_path = os.path.join(output_image_dir, image_name)
            shutil.copy(input_image_path, output_image_path)

def class_update(input_dir, output_dir, cp_img=True):
    input_label_dir = os.path.join(input_dir, 'labels')
    output_label_dir = os.path.join(output_dir, 'labels')
    input_image_dir = os.path.join(input_dir, 'images')
    output_image_dir = os.path.join(output_dir, 'images')
    image_list = os.listdir(input_image_dir)
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    for image_name in tqdm(image_list):
        label_name = Path(image_name).stem + '.txt'
        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)

        with open(input_label_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if line[0] == '0':
                    continue
                else:
                    lines[idx] = str(int(lines[idx][0])-1)+lines[idx][1:]
        with open(output_label_path, 'w') as f:
            f.writelines(lines)
        if cp_img:
            input_image_path = os.path.join(input_image_dir, image_name)
            output_image_path = os.path.join(output_image_dir, image_name)
            shutil.copy(input_image_path, output_image_path)

if __name__ == '__main__':
    pass
    # mseg2seg(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data307',
    #          output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data307_det', cp_img=True)
    class_update(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data307_det',
                 output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data307_det_c6', cp_img=True)