import os
import shutil

from tqdm import tqdm
import numpy as np

def random_split(input_dir, output_dir, train_ratio=0.9, random_seed=1010):
    pass
    level_list= os.listdir(input_dir)
    if 'train' in level_list:
        level_list.remove('train')
    if 'val' in level_list:
        level_list.remove('val')
    for level_name in level_list:
        level_dir = os.path.join(input_dir, level_name)
        level_dir_train = os.path.join(output_dir, 'train', level_name)
        level_dir_val = os.path.join(output_dir, 'val', level_name)
        os.makedirs(level_dir_train, exist_ok=True)
        os.makedirs(level_dir_val, exist_ok=True)

        file_list = os.listdir(level_dir)
        np.random.seed(random_seed)
        np.random.shuffle(file_list)
        train_num = int(len(file_list) * train_ratio)
        for idx, file_name in enumerate(tqdm(file_list)):
            src_path = os.path.join(level_dir, file_name)
            if idx<train_num:
                dst_path = os.path.join(level_dir_train, file_name)
            else:
                dst_path = os.path.join(level_dir_val, file_name)
            shutil.copyfile(src_path, dst_path)
        print(f'\ncopy {train_num} files to {level_dir_train}'
              f'\ncopy {len(file_list) - train_num} files to {level_dir_val}\n')


if __name__ == '__main__':
    pass
    risk_a_dir = r'/localnvme/data/billboard/bd_data/data626_mseg_f001/images_crop_box/abandonment'
    risk_b_dir = r'/localnvme/data/billboard/bd_data/data626_mseg_f001/images_crop_box/broken'
    risk_c_dir = r'/localnvme/data/billboard/bd_data/data626_mseg_f001/images_crop_box/corrosion'
    risk_d_dir = r'/localnvme/data/billboard/bd_data/data626_mseg_f001/images_crop_box/deformation'
    random_split(risk_a_dir, risk_a_dir)
    random_split(risk_b_dir, risk_b_dir)
    random_split(risk_c_dir, risk_c_dir)
    random_split(risk_d_dir, risk_d_dir)