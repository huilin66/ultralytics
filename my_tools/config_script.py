import os.path
import sys

import yaml

def config_change1(input_path, output_path, dst_com_path):
    pass
    with open(input_path, 'r') as file:
        data = yaml.safe_load(file)
    data['head'][12][3][2][4] = data['head'][12][3][2][4].replace('6.csv', dst_com_path)
    with open(output_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    print('save to', os.path.basename(output_path))

if __name__ == '__main__':
    root_dir = r'/ultralytics/cfg/models/expyolo10x_head'

    input_path = os.path.join(root_dir, r'yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom6.yaml')
    for i in range(1, 6):
        output_path = os.path.join(root_dir, r'yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom%d.yaml'%i)
        dst_com_path = '%d.csv'%i
        config_change1(input_path, output_path, dst_com_path)

    input_path = os.path.join(root_dir, r'yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom6_res.yaml')
    for i in range(1, 6):
        output_path = os.path.join(root_dir, r'yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom%d_res.yaml'%i)
        dst_com_path = '%d.csv'%i
        config_change1(input_path, output_path, dst_com_path)
    input_path = os.path.join(root_dir, r'yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom6_nosf.yaml')
    for i in range(1, 6):
        output_path = os.path.join(root_dir, r'yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom%d_nosf.yaml'%i)
        dst_com_path = '%d.csv'%i
        config_change1(input_path, output_path, dst_com_path)
    input_path = os.path.join(root_dir, r'yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom6_res_nosf.yaml')
    for i in range(1, 6):
        output_path = os.path.join(root_dir, r'yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom%d_res_nosf.yaml'%i)
        dst_com_path = '%d.csv'%i
        config_change1(input_path, output_path, dst_com_path)
    input_path = os.path.join(root_dir, r'yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom6_pure.yaml')
    for i in range(1, 6):
        output_path = os.path.join(root_dir, r'yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom%d_pure.yaml'%i)
        dst_com_path = '%d.csv'%i
        config_change1(input_path, output_path, dst_com_path)
    input_path = os.path.join(root_dir, r'yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom6_res_pure.yaml')
    for i in range(1, 6):
        output_path = os.path.join(root_dir, r'yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom%d_res_pure.yaml'%i)
        dst_com_path = '%d.csv'%i
        config_change1(input_path, output_path, dst_com_path)
    input_path = os.path.join(root_dir, r'yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom6_nosf_pure.yaml')
    for i in range(1, 6):
        output_path = os.path.join(root_dir, r'yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom%d_nosf_pure.yaml'%i)
        dst_com_path = '%d.csv'%i
        config_change1(input_path, output_path, dst_com_path)
    input_path = os.path.join(root_dir, r'yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom6_res_nosf_pure.yaml')
    for i in range(1, 6):
        output_path = os.path.join(root_dir, r'yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcom%d_res_nosf_pure.yaml'%i)
        dst_com_path = '%d.csv'%i
        config_change1(input_path, output_path, dst_com_path)