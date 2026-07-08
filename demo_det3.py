import demo_base
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
demo_base.TASK = 'detect'
demo_base.EPOCHS = 300
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device('cuda:1')
demo_base.BATCH_SIZE = 16
# demo_base.DATA = ".yaml"
# demo_base.CONF = 0.5
demo_base.CONF_VAL = 0.5

if __name__ == '__main__':
    pass
    NAME = None

    csv_path = r'summary_conf_0.5.csv'

    demo_base.model_val_summary('hmt_bp_cube-[yolov8x]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_bp_cube-[yolov9e]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_bp_cube-[yolov10x]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_bp_cube-[yolo11x]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_bp_cube-[yolo12x]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_bp_cube-[yolo26x]', csv_path=csv_path)


    demo_base.model_val_summary('hmt_t-[yolov8x]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_t-[yolov9e]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_t-[yolov10x]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_t-[yolo11x]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_t-[yolo12x]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_t-[yolo26x]', csv_path=csv_path)

    demo_base.model_val_summary('hmt_t_rgbt-[yolov8x]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_t_rgbt-[yolov9e]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_t_rgbt-[yolov10x]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_t_rgbt-[yolo11x]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_t_rgbt-[yolo12x]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_t_rgbt-[yolo26x]', csv_path=csv_path)

    demo_base.model_val_summary('hmt_rgb-[yolov8x]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_rgb-[yolov9e]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_rgb-[yolov10x]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_rgb-[yolo11x]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_rgb-[yolo12x]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_rgb-[yolo26x]', csv_path=csv_path)

    demo_base.model_val_summary('hmt_rgb_rgbt-[yolov8x]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_rgb_rgbt-[yolov9e]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_rgb_rgbt-[yolov10x]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_rgb_rgbt-[yolo11x]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_rgb_rgbt-[yolo12x]', csv_path=csv_path)
    demo_base.model_val_summary('hmt_rgb_rgbt-[yolo26x]', csv_path=csv_path)
