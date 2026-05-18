import torch

import demo_base

demo_base.TASK = "detect"
demo_base.EPOCHS = 500
demo_base.IMGSZ = 640
demo_base.DEVICE = torch.device("cuda:0")
demo_base.BATCH_SIZE = 16
# demo_base.DATA = ".yaml"
# demo_base.CONF = 0.5

if __name__ == "__main__":
    pass
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p12_v3.yaml', imgsz=640)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p12_v3_s640.yaml', imgsz=640)

    demo_base.yolo8("yolov8x.yaml", auto_optim=False, data="hmt_t_0211_extendv1.yaml", imgsz=960)
    demo_base.yolo9("yolov9e.yaml", auto_optim=False, data="hmt_t_0211_extendv1.yaml", imgsz=960)
    demo_base.yolo10("yolov10x.yaml", auto_optim=False, data="hmt_t_0211_extendv1.yaml", imgsz=960)
    demo_base.yolo11("yolo11x.yaml", auto_optim=False, data="hmt_t_0211_extendv1.yaml", imgsz=960)
    demo_base.yolo12("yolo12x.yaml", auto_optim=False, data="hmt_t_0211_extendv1.yaml", imgsz=960, batch=8)

    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p3_s640.yaml', imgsz=640)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p3_v2_s640.yaml', imgsz=640)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p3_v3_s640.yaml', imgsz=640)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p3_s960.yaml', imgsz=640)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p3_v2_s960.yaml', imgsz=640)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p3_v3_s960.yaml', imgsz=640)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p3.yaml', imgsz=640)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p3_v2.yaml', imgsz=640)
    # demo_base.yolo8('yolov8x.yaml', auto_optim=False, data='hmt_rgb_p3_v3.yaml', imgsz=640)

    # demo_base.model_predict('hmt_t_p12-[yolov8x]3',
    #                         r'/scrinvme/huilin/bdd/collected_data/HMT_data/data/thermal',
    #                         name=r'/scrinvme/huilin/bdd/collected_data/HMT_data/data/thermal_infer',
    #                         batch=32, save_conf=True)
    # import os
    # root_dir = r'/scrinvme/huilin/bdd/collected_data/HMT_data/data_split/thermal_views'
    # infer_dir = root_dir + '_infer'
    # data_list = os.listdir(root_dir)
    # os.makedirs(infer_dir, exist_ok=True)
    # for data_name in data_list:
    #     data_path = os.path.join(root_dir, data_name)
    #     infer_path = os.path.join(infer_dir, data_name)
    #     if os.path.isdir(data_path) and len(os.listdir(data_path)) > 0:
    #         demo_base.model_predict('hmt_t_p123_v4-[yolov8x]',
    #                                 data_path,
    #                                 name=infer_path,
    #                                 batch=32, save_conf=True)

    # model_path = r'C:\Users\USER\Downloads\model.pt'
    # data_path = r'C:\Users\USER\Downloads\demo'
    # demo_base.model_predict(model_path, data_path, weight_name=False)

    import os

    root_dir = r"\\158.132.186.40\isds\huilin\bdd\collected_data\20260211_HMT_data\data_split\visible_views_align"
    infer_dir = root_dir + "_infer"
    view_list = os.listdir(root_dir)
    os.makedirs(infer_dir, exist_ok=True)
    for view_name in view_list:
        view_path = os.path.join(root_dir, view_name)
        infer_path = os.path.join(infer_dir, view_name)
        if os.path.isdir(view_path) and len(os.listdir(view_path)) > 0:
            demo_base.model_predict(
                r"\\158.132.186.40\isds\huilin\bdd\collected_data\20260211_HMT_data\best.pt",
                view_path,
                name=infer_path,
                weight_name=False,
                batch=2,
                save_conf=True,
            )
