import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np

def is_image_file(filename):
    filesuffix = Path(filename.lower()).suffix
    if filesuffix in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        return True
    return False

def random_select(data_dir, save_dir=None, train_ratio=0.9, random_seed=1010, full_path=True, suffix='', skip_list=[]):
    image_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')
    file_list = [f for f in os.listdir(image_dir) if f not in skip_list and is_image_file(f)]

    if label_dir is not None:
        label_list = os.listdir(label_dir)
        label_list = [Path(label_name).stem for label_name in label_list]
        file_list_check = []
        for img_name in tqdm(file_list, desc='img check', total=len(file_list)):
            name = Path(img_name).stem
            if name in label_list:
                file_list_check.append(img_name)
        file_list = file_list_check
    if save_dir is None:
        save_dir = os.path.dirname(image_dir)

    np.random.seed(random_seed)
    np.random.shuffle(file_list)
    val_num = int(len(file_list)*(1-train_ratio))

    val_list = file_list[:val_num]
    train_list = [file_name for file_name in file_list if file_name not in val_list]

    if full_path:
        train_list = [os.path.join(image_dir, name) for name in train_list]
        val_list = [os.path.join(image_dir, name) for name in val_list]

    df_train = pd.DataFrame({'filename': train_list})
    df_val = pd.DataFrame({'filename': val_list})
    df_all = pd.DataFrame({'filename': train_list+val_list})
    df_train.to_csv(os.path.join(save_dir, f'train{suffix}.txt'), header=None, index=None)
    df_val.to_csv(os.path.join(save_dir, f'val{suffix}.txt'), header=None, index=None)
    df_all.to_csv(os.path.join(save_dir, 'all.txt'), header=None, index=None)
    print('%d save to %s,\n%d save to %s!'%(len(train_list), os.path.join(save_dir, f'train{suffix}.txt'),
                                           len(val_list), os.path.join(save_dir, f'val{suffix}.txt')))


if __name__ == '__main__':
    pass
    data_dir = r'/data/huilin/data/BDD/cubit-det'
    skip_list = ['hk2562.jpg']
    random_select(data_dir, skip_list=skip_list)
