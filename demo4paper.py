import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import torch
import numpy as np
from fiftyone.brain.visualization import visualize
from seaborn import heatmap

from ultralytics import YOLO
BATCH_SIZE = 32
EPOCHS = 500
IMGSZ = 640
CONF = 0.5
TASK = 'mdetect'
DEVICE = torch.device('cuda:1')
DATA = "billboard_mdet5_10_c_0806m.yaml"
FREEZE_NUMS = {
    'yolov8' : 22,
    'yolov9e': 42,
    'yolov9' : 22,
    'yolov10': 23,
    'mayolo': 23,
}
# MLOSS_ENLARGE = 0.3
# region meta tools

def myolo_train(cfg_path, pretrain_path, network=YOLO, auto_optim=False, retrain=False, **kwargs):
    model = network(cfg_path, task=TASK)
    model.load(pretrain_path)

    train_params = {
        'data': DATA,
        'device': DEVICE,
        'epochs': EPOCHS,
        'imgsz': IMGSZ,
        'val': True,
        'batch': BATCH_SIZE,
        'patience': EPOCHS,
    }

    if not auto_optim:
        train_params.update({
            'optimizer': 'AdamW',
            'lr0': 0.0001
        })

    if retrain:
        train_params.update(
            {
                'freeze':get_freeze_num(cfg_path),
                'freeze_head':['.cv2', '.cv3'] if 'yolov10' not in cfg_path and 'mayolo' not in cfg_path else ['.cv2', '.cv3', '.one2one_cv2', '.one2one_cv3'],
                'freeze_bn':True,
            }
        )
    train_params.update(kwargs)
    model.train(**train_params)

def model_val(weight_path, network=YOLO, **kwargs):
    model = network(weight_path, task=TASK)
    print(weight_path)
    print(model.info(detailed=False))
    val_params = {
        'data': DATA,
        'device': DEVICE,
    }
    val_params.update(kwargs)
    return model.val(**val_params)

def model_gat_val(weight_path, com_path, network=YOLO):
    model = network(weight_path, task=TASK)
    model.model.model[23].added_gat_head(com_path)
    print(weight_path)
    print(model.info(detailed=False))
    model.val(data=DATA, device=DEVICE)

def model_val_single(weight_path, network=YOLO):
    model = network(weight_path, task=TASK)
    model.model.model[23].use_one2many_head()
    print(weight_path)
    print(model.info(detailed=False))
    model.val(data=DATA, device=DEVICE)

def model_predict(weight_path, img_dir, network=YOLO, name=None, visualize=False):
    model = network(weight_path, task=TASK)
    model.predict(
        img_dir,
        save=False,
        conf=CONF,
        device=DEVICE,
        imgsz=IMGSZ,
        save_txt=True,
        save_conf=True,
        name=name,
        visualize=visualize,
    )

def model_export(weight_path, format='onnx', network=YOLO):
    model = network(weight_path, task=TASK)
    model.export(format=format)

# endregion


# region other tools

def get_freeze_num(cfg_path):
    for k,v in FREEZE_NUMS.items():
        if k in cfg_path:
            return v
    print('freeze num error for cfg_path {}'.format(cfg_path))
    return None

# endregion


# region run tools

def myolo8(cfg_path, weight_path='yolov8x.pt', auto_optim=False, retrain=False, **kwargs):
    assert 'yolov8' in cfg_path, ValueError(cfg_path, 'is not yolov8 config!')
    myolo_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def myolo9(cfg_path, weight_path='yolov9e.pt', auto_optim=False, retrain=False, **kwargs):
    assert 'yolov9' in cfg_path, ValueError(cfg_path, 'is not yolov9 config!')
    myolo_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def myolo10(cfg_path, weight_path='yolov10x.pt', auto_optim=False, retrain=False, **kwargs):
    assert 'yolov10' in cfg_path, ValueError(cfg_path, 'is not yolov10 config!')
    myolo_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def mayolo(cfg_path, weight_path='yolov10x.pt', auto_optim=False, retrain=False, **kwargs):
    kwargs['mloss_mask'] = True
    myolo_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

# endregion

if __name__ == '__main__':
    pass
    # model_val(r'runs/exp_results/exp_yolo10x/weights/best.pt')
    # model_val(r'runs/exp_results/exp_mayolox_/weights/best.pt')


    # # # region object detection results of different models
    # model_val(r'runs/exp_results/exp_yolo8n/weights/best.pt')
    # model_val(r'runs/exp_results/exp_yolo8s/weights/best.pt')
    # model_val(r'runs/exp_results/exp_yolo8m/weights/best.pt')
    # model_val(r'runs/exp_results/exp_yolo8l/weights/best.pt')
    # model_val(r'runs/exp_results/exp_yolo8x/weights/best.pt')
    #
    # model_val(r'runs/exp_results/exp_yolo9s/weights/best.pt')
    # model_val(r'runs/exp_results/exp_yolo9m/weights/best.pt')
    # model_val(r'runs/exp_results/exp_yolo9c/weights/best.pt')
    # model_val(r'runs/exp_results/exp_yolo9e/weights/best.pt')
    # #
    # model_val(r'runs/exp_results/exp_yolo10n/weights/best.pt')
    # model_val(r'runs/exp_results/exp_yolo10s/weights/best.pt')
    # model_val(r'runs/exp_results/exp_yolo10m/weights/best.pt')
    # model_val(r'runs/exp_results/exp_yolo10b/weights/best.pt')
    # model_val(r'runs/exp_results/exp_yolo10l/weights/best.pt')
    # model_val(r'runs/exp_results/exp_yolo10x/weights/best.pt')
    # #
    # model_val(r'runs/exp_results/exp_mayolon_/weights/best.pt')
    # model_val(r'runs/exp_results/exp_mayolos_/weights/best.pt')
    # model_val(r'runs/exp_results/exp_mayolom_/weights/best.pt')
    # model_val(r'runs/exp_results/exp_mayolob_/weights/best.pt')
    # model_val(r'runs/exp_results/exp_mayolol_/weights/best.pt')
    # model_val(r'runs/exp_results/exp_mayolox_/weights/best.pt')
    # # # endregion
    #
    # # region scale comparison
    # # early stage, no sigmoid
    # model_val(r'runs/exp_results/exp_mloss_enlarge/weights/best.pt')
    # model_val(r'runs/exp_results/exp_mloss_enlarge3/weights/best.pt')
    # model_val(r'runs/exp_results/exp_mloss_enlarge5/weights/best.pt')
    # model_val(r'runs/exp_results/exp_mloss_enlarge7/weights/best.pt')
    # model_val(r'runs/exp_results/exp_mloss_enlarge9/weights/best.pt')
    # model_val(r'runs/exp_results/exp_mloss_enlarge2/weights/best.pt')
    # model_val(r'runs/exp_results/exp_mloss_enlarge4/weights/best.pt')
    # model_val(r'runs/exp_results/exp_mloss_enlarge6/weights/best.pt')
    # model_val(r'runs/exp_results/exp_mloss_enlarge8/weights/best.pt')
    # model_val(r'runs/exp_results/exp_mloss_enlarge10/weights/best.pt')
    # # # endregion
    #
    # # # region ablation experiment
    # model_val('runs/exp_results/exp_ablation0107_GIA/weights/best.pt')
    # model_val_single('runs/exp_results/exp_yolo10x/weights/best.pt')
    # model_val(r'runs/exp_results/exp_ablation0107_GCA/weights/best.pt')
    #
    # model_val(r'runs/exp_results/exp_ablation0107_HO_GCA/weights/best.pt')
    # model_val(r'runs/exp_results/exp_ablation0107_GIA_GCA/weights/best.pt')
    # model_val_single('runs/exp_results/exp_ablation0107_GIA/weights/best.pt')
    # # # endregion

    # model_val(r'runs/exp_results/exp_yolo10x/weights/best.pt')

    # img_path = r'/scrinvme/huilin/tp/FLIR1444_img.png'
    # output_dir = r'/scrinvme/huilin/tp/FLIR1444_img_yolo10'
    # model_path = r'runs/exp_results/exp_yolo10x/weights/best.pt'
    # output_dir = r'/scrinvme/huilin/tp/FLIR1444_img_mayolo'
    # model_path = r'runs/exp_results/exp_mayolox_/weights/best.pt'
    # model_predict(
    #     model_path,
    #     img_path,
    #     name=output_dir,
    #     visualize=True,
    # )







    # model_val(r'runs/exp_results/exp_mayolox_/weights/best.pt')
    # model_val(r'runs/exp_results/exp_mayolox_/weights/best.pt', conf=CONF)
    #
    # for i in np.arange(0.2, 0.8, 0.05):
    #     model_val(r'runs/exp_results/exp_mayolox_/weights/best.pt', conf=CONF, att_threshold=i)


    # IMG_DIR = r'/nfsv4/23039356r/data/billboard/data0806_m/paper_demo/images'
    #
    # model_predict(
    #     r'runs/exp_results/exp_yolo8x/weights/best.pt',
    #     IMG_DIR,
    #     name=r'prediction_yolo8x'
    # )
    # model_predict(
    #     r'runs/exp_results/exp_yolo9e/weights/best.pt',
    #     IMG_DIR,
    #     name=r'prediction_yolo9e'
    # )
    # model_predict(
    #     r'runs/exp_results/exp_yolo10x/weights/best.pt',
    #     IMG_DIR,
    #     name=r'prediction_yolo10x'
    # )
    # model_predict(
    #     r'runs/exp_results/exp_mayolox_/weights/best.pt',
    #     IMG_DIR,
    #     name=r'prediction_mayolox'
    # )

        # myolo8(cfg_path='yolov8n.yaml', weight_path='yolov8n.pt', name='exp_ml')
    # myolo8(cfg_path='yolov8s.yaml', weight_path='yolov8s.pt', name='exp_ml')
    # myolo8(cfg_path='yolov8m.yaml', weight_path='yolov8m.pt', name='exp_ml')
    # myolo8(cfg_path='yolov8l.yaml', weight_path='yolov8l.pt', name='exp_ml')
    # myolo8(cfg_path='yolov8x.yaml', weight_path='yolov8x.pt', name='exp_ml')
    #
    #
    # myolo9(cfg_path='yolov9s.yaml', weight_path='yolov9s.pt', name='exp_ml')
    # myolo9(cfg_path='yolov9m.yaml', weight_path='yolov9m.pt', name='exp_ml')
    # myolo9(cfg_path='yolov9c.yaml', weight_path='yolov9c.pt', name='exp_ml')
    # myolo9(cfg_path='yolov9e.yaml', weight_path='yolov9e.pt', name='exp_ml')
    #
    #
    # myolo10(cfg_path='yolov10n.yaml', weight_path='yolov10n.pt', name='exp_ml')
    # myolo10(cfg_path='yolov10s.yaml', weight_path='yolov10s.pt', name='exp_ml')
    # myolo10(cfg_path='yolov10m.yaml', weight_path='yolov10m.pt', name='exp_ml')
    # myolo10(cfg_path='yolov10l.yaml', weight_path='yolov10l.pt', name='exp_ml')
    # myolo10(cfg_path='yolov10x.yaml', weight_path='yolov10x.pt', name='exp_ml')



    # val_data_list = [
    #     'billboard_mdet5_10_c_0806m.yaml',
    #     'billboard_mdet5_10_c_0806m_b50.yaml',
    #     'billboard_mdet5_10_c_0806m_b75.yaml',
    #     'billboard_mdet5_10_c_0806m_b125.yaml',
    #     'billboard_mdet5_10_c_0806m_b150.yaml',
    #     # 'billboard_mdet5_10_c_0806m_b200.yaml',
    # ]
    # for val_data in val_data_list:
    #     model_val(
    #         r'runs/exp_results/exp_mayolox_/weights/best.pt',
    #         data=val_data,
    #     )


    src_dir = r'/nfsv4/23039356r/data/billboard/data0806_m/yolo_rgb_detection5_10_c'
    brightness_list = [
        0.5,
        0.75,
        1.25,
        1.5,
    ]
    for brightness in brightness_list:
        dst_dir = f'{src_dir}_b{int(brightness*100)}'
        model_predict(
            r'runs/exp_results/exp_mayolox_/weights/best.pt',
            os.path.join(dst_dir, 'val', 'images'),
            name=os.path.join(dst_dir, 'val', 'images_infer'),
        )