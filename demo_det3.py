import demo_base
import torch

demo_base.TASK = 'detect'
demo_base.EPOCHS = 300
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device('cuda:1')
demo_base.BATCH_SIZE = 8
# demo_base.DATA = ".yaml"
demo_base.CONF = 0.5

if __name__ == '__main__':
    pass
    # demo_base.model_val('hmt_rgb_p12-[yolov8x]5', batch=32)
    # demo_base.model_val('hmt_rgb_p12-[yolov8x]6', batch=32)
    #
    # demo_base.model_val('hmt_rgb_p12_s640-[yolov8x]5', batch=32)
    # demo_base.model_val('hmt_rgb_p12_s640-[yolov8x]6', batch=32)
    #
    # demo_base.model_val('hmt_rgb_p12_s960-[yolov8x]3', batch=32)
    # demo_base.model_val('hmt_rgb_p12_s960-[yolov8x]4', batch=32)
    #
    # demo_base.model_val('hmt_rgb_p12_s1280-[yolov8x]3', batch=32)
    # demo_base.model_val('hmt_rgb_p12_s1280-[yolov8x]4', batch=32)
    #
    #
    # demo_base.model_val('hmt_rgb_p12_v2-[yolov8x]', batch=32)
    # demo_base.model_val('hmt_rgb_p12_v2-[yolov8x]2', batch=32)
    #
    # demo_base.model_val('hmt_rgb_p12_v2_s640-[yolov8x]', batch=32)
    # demo_base.model_val('hmt_rgb_p12_v2_s640-[yolov8x]2', batch=32)
    # demo_base.model_val('hmt_rgb_p12_v2_s640-[yolov8x]3', batch=32)
    #
    # demo_base.model_val('hmt_rgb_p12_v2_s960-[yolov8x]', batch=32)
    # demo_base.model_val('hmt_rgb_p12_v2_s960-[yolov8x]2', batch=32)
    # demo_base.model_val('hmt_rgb_p12_v2_s960-[yolov8x]3', batch=32)
    #
    # demo_base.model_val('hmt_rgb_p12_v2_s1280-[yolov8x]', batch=32)
    # demo_base.model_val('hmt_rgb_p12_v2_s1280-[yolov8x]2', batch=32)
    # demo_base.model_val('hmt_rgb_p12_v2_s1280-[yolov8x]3', batch=32)


    # demo_base.model_val('hmt_t_p123_v41-[yolov8x]5')
    # demo_base.model_val('hmt_t_p123_v41-[yolo11x]')
    # demo_base.model_val('hmt_t_p123_v41-[yolov9e]')
    # demo_base.model_val('hmt_t_p123_v41-[yolov10x]')
    # demo_base.model_val('hmt_t_p123_v41-[yolo12x]7')
    #
    demo_base.model_val('hmt_rgb_p12_v2_s640-[yolov8x]5')
    # demo_base.model_val('hmt_rgb_p12_v2_s640-[yolo11x]')
    # demo_base.model_val('hmt_rgb_p12_v2_s640-[yolov9e]')
    # demo_base.model_val('hmt_rgb_p12_v2_s640-[yolov10x]')
    # demo_base.model_val('hmt_rgb_p12_v2_s640-[yolo12x]')


    demo_base.model_val('hmt_rgb_p12_v3-[yolov8x]2')
    demo_base.model_val('hmt_rgb_p12_v4-[yolov8x]')
    demo_base.model_val('hmt_rgb_p12_v3_s640-[yolov8x]')
    demo_base.model_val('hmt_rgb_p12_v4_s640-[yolov8x]')

    # demo_base.model_val('hmt_t_p12_v4-[yolov8x]')
    #
    # demo_base.model_val('hmt_t_p123_v4-[yolov8x]')
    # demo_base.model_val('hmt_t_p123_v4-[yolov8x]2')
    # demo_base.model_val('hmt_t_p123_v4-[yolov8x]3')
    # demo_base.model_val('hmt_t_p123_v4-[yolov8x]4')

    # demo_base.model_val('hmt_t_p123_v41-[yolov8x]2')

    # import os
    # root_dir = r'/scrinvme/huilin/bdd/collected_data/HMT_data/data_split/thermal_views'
    # infer_dir = root_dir + '_infer3'
    # data_list = os.listdir(root_dir)
    # os.makedirs(infer_dir, exist_ok=True)
    # for data_name in data_list:
    #     data_path = os.path.join(root_dir, data_name)
    #     infer_path = os.path.join(infer_dir, data_name)
    #     if os.path.isdir(data_path) and len(os.listdir(data_path)) > 0:
    #         demo_base.model_predict('hmt_t_p123_v41-[yolo11x]',
    #                                 data_path,
    #                                 name=infer_path,
    #                                 batch=32, save_conf=True)