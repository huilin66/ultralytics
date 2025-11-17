import os
import cv2
from tqdm import tqdm
import numpy as np
def resize_imgs(input_dir, ouput_dir):
    os.makedirs(ouput_dir, exist_ok=True)
    file_list = os.listdir(input_dir)
    for file_name in tqdm(file_list):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(ouput_dir, file_name)
        # img = cv2.imread(input_path)
        img = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[1]==1024:
            if img.shape[0]==615:
                pass
            else:
                img = cv2.resize(img, (1024, 615))
        elif img.shape[1]==4096:
            if img.shape[0]==2460:
                pass
            else:
                print(f'{file_name} {img.shape} --> (4096, 2460)')
                img = cv2.resize(img, (4096, 2460), )
        else:
            img = cv2.resize(img, (960, 608))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
if __name__ == '__main__':
    pass
    resize_imgs(r'/localnvme/data/billboard/all_data/mseg_c5_l2/data80_v17/images',
                r'/localnvme/data/billboard/all_data/mseg_c5_l2/data80_v17/images_rs')