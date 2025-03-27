import os
from tqdm import tqdm

def mseg2seg(input_dir, output_dir):
    file_list = os.listdir(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    for file_name in tqdm(file_list):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        with open(input_path, 'r') as f_in, open(output_path, 'w+') as f_out:
            lines = f_in.readlines()
            for line in lines:
                num_list = line.split(' ')
                num_list_seg = num_list[:1]+num_list[11:]
                line_seg = ' '.join(num_list_seg)
                f_out.write(line_seg)



if __name__ == '__main__':
    pass
    mseg2seg(input_dir=r'/nfsv4/23039356r/data/billboard/data0521_m/yolo_rgb_segmentation2_seg/labels_src',
             output_dir=r'/nfsv4/23039356r/data/billboard/data0521_m/yolo_rgb_segmentation2_seg/labels')