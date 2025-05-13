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
def mseg_class_update(input_dir, output_dir, cp_img=True):
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


def seg_class_update(input_dir, output_dir, cp_img=True):
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

def seg_filter_small(input_dir, output_dir, threshold, class_list, with_attribute):
    input_image_dir = os.path.join(input_dir, 'images')
    input_label_dir = os.path.join(input_dir, 'labels')
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    image_list = os.listdir(input_image_dir)
    for image_name in tqdm(image_list):
        label_name = Path(image_name).stem + '.txt'
        input_image_path = os.path.join(input_image_dir, image_name)
        output_image_path = os.path.join(output_image_dir, image_name)
        input_label_path = os.path.join(input_label_dir, label_name)
        output_label_path = os.path.join(output_label_dir, label_name)

        shutil.copy(input_image_path, output_image_path)

        filter_yolo_segmentation(input_label_path, output_label_path, threshold=threshold,
                                 class_list=class_list, with_attribute=with_attribute)


def filter_yolo_segmentation(input_file, output_file, threshold, with_attribute=False, class_list=[]):
    def calculate_polygon_area(points):
        n = len(points)
        area = 0.0
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            area += (x1 * y2) - (x2 * y1)
        return abs(area) / 2.0
    def clalulate_bbox_area(points):
        x_points = [p[0] for p in points]
        y_points = [p[1] for p in points]
        x_min, x_max = min(x_points), max(x_points)
        y_min, y_max = min(y_points), max(y_points)
        width = x_max - x_min
        height = y_max - y_min
        area = width * height
        return area

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        src_count, dst_count = 0, 0
        for line in f_in:
            src_count += 1
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if not parts:
                continue

            class_id = int(parts[0])
            if class_id not in class_list:
                f_out.write(line + '\n')
                dst_count += 1
                continue
            if with_attribute:
                att_len= int(parts[1])
                coords = list(map(float, parts[2+att_len:]))
            else:
                coords = list(map(float, parts[1:]))


            normalized_points = [(coords[i], coords[i + 1])
                                 for i in range(0, len(coords), 2)]

            area = clalulate_bbox_area(normalized_points)
            if area >= threshold:
                f_out.write(line + '\n')
                dst_count += 1
        if src_count != dst_count:
            print(f'{os.path.basename(input_file)} change from {src_count} --> {dst_count}')


if __name__ == '__main__':
    pass
    # mseg2seg(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389',
    #          output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_seg', cp_img=True)
    # seg_class_update(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_seg',
    #              output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_seg_c6', cp_img=True)
    # mseg_class_update(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389',
    #              output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_c6', cp_img=True)

    # seg_filter_small(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389',
    #                  output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter001',
    #                  threshold=0.01, class_list=[2, 4, 5, 7], with_attribute=True)
    # seg_filter_small(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389',
    #                  output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter005',
    #                  threshold=0.05, class_list=[2, 4, 5, 7], with_attribute=True)
    # seg_filter_small(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389',
    #                  output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter010',
    #                  threshold=0.10, class_list=[2, 4, 5, 7], with_attribute=True)

    # mseg_class_update(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter001',
    #              output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter001_c6', cp_img=True)
    # mseg_class_update(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter005',
    #              output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter005_c6', cp_img=True)
    # mseg_class_update(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter010',
    #              output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter010_c6', cp_img=True)
    # mseg2seg(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter001_c6',
    #          output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter001_c6_seg', cp_img=True)
    # mseg2seg(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter005_c6',
    #          output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter005_c6_seg', cp_img=True)
    # mseg2seg(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter010_c6',
    #          output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data389_filter010_c6_seg', cp_img=True)


    # mseg2seg(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data611',
    #          output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data611_seg', cp_img=True)

    mseg2seg(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data618',
             output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data618_seg', cp_img=True)
    seg_class_update(input_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data618_seg',
                 output_dir=r'/nfsv4/23039356r/data/billboard/bd_data/data618_seg_c6', cp_img=True)