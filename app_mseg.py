import argparse
import os
import sys

import demo_base

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='', help='Path to the model file')
    parser.add_argument('--img_dir', type=str, default='images', help='Directory of images to infer')
    parser.add_argument('--save_dir', type=str, default='inference', help='Directory to save inference results')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for predictions')
    opt = parser.parse_args()
    return opt

def main():
    '''
    Main function to run the model prediction.
    Example usage:
    python app_mseg.py \
        --model_path /localnvme/project/ultralytics/runs/msegment/fusedata7961_mseg_c5_l2_1023_80p_ref-[yolov10x-mseg-dlka3res-7-unet]/weights/best.pt \
        --img_dir /localnvme/data/added_data/test_data1121/images \
        --save_dir /localnvme/data/added_data/test_data1121/images_infer0518 \
        --conf 0.5
    '''
    opt = parse_opt()

    demo_base.model_predict(opt.model_path, img_dir=opt.img_dir, name=opt.save_dir, conf=opt.conf, weight_name=False)
    print(f"Inference complete. Results saved to '{opt.save_dir}'.")

if __name__ == "__main__":
    main()