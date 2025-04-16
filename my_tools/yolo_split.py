import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def random_select(img_dir, label_dir=None, save_dir=None, train_ratio=0.9, random_seed=1010, full_path=True):
    file_list = os.listdir(img_dir)
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
        save_dir = os.path.dirname(img_dir)
    if full_path:
        file_list = [os.path.join(img_dir, filename) for filename in file_list]
    np.random.seed(random_seed)
    np.random.shuffle(file_list)
    train_num = int(len(file_list)*train_ratio)


    train_list = file_list[:train_num]
    val_list = file_list[train_num:]

    df_train = pd.DataFrame({'filename': train_list})
    df_val = pd.DataFrame({'filename': val_list})
    df_all = pd.DataFrame({'filename': train_list+val_list})
    df_train.to_csv(os.path.join(save_dir, 'train.txt'), header=None, index=None)
    df_val.to_csv(os.path.join(save_dir, 'val.txt'), header=None, index=None)
    df_all.to_csv(os.path.join(save_dir, 'all.txt'), header=None, index=None)
    print('%d save to %s,\n%d save to %s!'%(len(train_list), os.path.join(save_dir, 'train.txt'),
                                           len(val_list), os.path.join(save_dir, 'val.txt')))

def random_kfold(img_dir, k, label_dir=None, save_dir=None, random_seed=1010, full_path=True):
    file_list = os.listdir(img_dir)
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
        save_dir = os.path.dirname(img_dir)
    if full_path:
        file_list = [os.path.join(img_dir, filename) for filename in file_list]

    np.random.seed(random_seed)
    np.random.shuffle(file_list)

    total_images = len(file_list)
    fold_size = [total_images // k] * k
    for i in range(total_images % k):
        fold_size[i] += 1

    start_idx = 0
    for fold in range(k):
        end_idx = start_idx + fold_size[fold]
        val_ids = list(range(start_idx, end_idx))
        train_ids = [i for i in range(total_images) if i not in val_ids]
        train_list = [file_list[i] for i in train_ids]
        val_list = [file_list[i] for i in val_ids]

        df_train = pd.DataFrame({'filename': train_list})
        df_val = pd.DataFrame({'filename': val_list})
        train_path = os.path.join(save_dir, f'train_{fold}.txt')
        val_path = os.path.join(save_dir, f'val_{fold}.txt')
        df_train.to_csv(train_path, header=None, index=None)
        df_val.to_csv(val_path, header=None, index=None)
        print('%d save to %s,\n%d save to %s!' % (len(train_list), train_path,
                                                  len(val_list), val_path))

if __name__ == '__main__':
    pass
    random_select(r'/nfsv4/23039356r/data/billboard/bd_data/data127/images',
                  r'/nfsv4/23039356r/data/billboard/bd_data/data127/labels',
                  train_ratio=0.9)
    # random_kfold(r'/nfsv4/23039356r/data/billboard/bd_data/data127/images', 5,
    #               r'/nfsv4/23039356r/data/billboard/bd_data/data127/labels',)