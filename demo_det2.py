import demo_base
import torch

demo_base.TASK = 'detect'
demo_base.EPOCHS = 300
demo_base.IMGSZ = 2560
demo_base.DEVICE = torch.device('cuda:1')
demo_base.BATCH_SIZE = 16
demo_base.DATA = "hmt_rgb_p12.yaml"
# demo_base.CONF = 0.5

if __name__ == '__main__':
    pass
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p12_v4.yaml', imgsz=640)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p12_v4_s640.yaml', imgsz=640)

    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p12_v2_s640.yaml', imgsz=640)
    # demo_base.yolo11('yolo11x.yaml', auto_optim=False, data='hmt_rgb_p12_v2_s640.yaml', imgsz=640)
    # demo_base.yolo9('yolov9e.yaml', auto_optim=False, data='hmt_rgb_p12_v2_s640.yaml', imgsz=640)
    # demo_base.yolo10('yolov10x.yaml', auto_optim=False, data='hmt_rgb_p12_v2_s640.yaml', imgsz=640)
    # demo_base.yolo12('yolo12x.yaml', auto_optim=False, data='hmt_rgb_p12_v2_s640.yaml', imgsz=640)

    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p3_s640.yaml', imgsz=960)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p3_v2_s640.yaml', imgsz=960)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p3_v3_s640.yaml', imgsz=960)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p3_s960.yaml', imgsz=960)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p3_v2_s960.yaml', imgsz=960)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p3_v3_s960.yaml', imgsz=960)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p3.yaml', imgsz=960)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p3_v2.yaml', imgsz=960)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p3_v3.yaml', imgsz=960)
    
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data="hmt_t_p12_v3.yaml", imgsz=960)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data="hmt_t_p123_v4.yaml", imgsz=960)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data="hmt_t_p3_v5.yaml", imgsz=960)

    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data="hmt_t_p123_v4.yaml", imgsz=1280)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data="hmt_t_p123_v4.yaml", imgsz=960)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data="hmt_t_p123_v4.yaml", imgsz=960)


    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data="hmt_t_p12_v3.yaml", imgsz=960)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data="hmt_t_p12_v3.yaml", imgsz=960)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data="hmt_t_p12_v3.yaml", imgsz=1280)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p12_s960.yaml', imgsz=960)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p12_s1280.yaml', imgsz=1280)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, imgsz=2560)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False,  imgsz=1280)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=True, imgsz=2560)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=True, imgsz=1280)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data="dreality_1c_fv2_v3.yaml")
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data="dreality_1c_fv2.yaml")
    demo_base.model_val('hmt_t_0211_extendv1-[yolov8x]',)
    demo_base.model_val('hmt_t_0211_extendv1-[yolov9e]',)
    # demo_base.model_predict('hmt_t_p12-[yolov8x]2',
    #                         r'/scrinvme/huilin/bdd/collected_data/HMT_data/dataset/thermal_selected_4_p12/val/images',
    #                         name=r'/scrinvme/huilin/bdd/collected_data/HMT_data/dataset/thermal_selected_4_p12/result_analysis/val_infer',
    #                         batch=32,)