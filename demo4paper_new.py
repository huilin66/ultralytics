import torch
import numpy as np
from seaborn import heatmap

from ultralytics import YOLO
BATCH_SIZE = 32
EPOCHS = 300
IMGSZ = 640
CONF = 0.5
TASK = 'mdetect'
DEVICE = torch.device('cuda:0')
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
        freeze_num = get_freeze_num(cfg_path)
        train_params.update(
            {
                'freeze':freeze_num,
                'freeze_head':[f'{freeze_num}.cv2', f'{freeze_num}.cv3', f'{freeze_num}.cv4', f'{freeze_num}.proto'],
                'freeze_att_head': [f'{freeze_num}.cva.{[freeze_att_num]}' for freeze_att_num in kwargs['freeze_att_nums']] if 'freeze_att_nums' in kwargs else None,
                'freeze_bn':True,
                'box': 0,
                'seg': 0,
                'cls': 0,
                'dfl': 0,
                'mdet': 100,
                'close_mosaic': 30,
            }
        )
    train_params.update(kwargs)
    model.train(**train_params)

def model_val(weight_path, network=YOLO, **kwargs):
    model = network(weight_path, task=TASK)
    print(weight_path)
    print(model.info(detailed=False))
    return model.val(data=DATA, device=DEVICE, **kwargs)

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

def model_predict(weight_path, img_dir, network=YOLO, name=None):
    model = network(weight_path, task=TASK)
    model.predict(
        img_dir,
        save=True,
        conf=CONF,
        device=DEVICE,
        imgsz=IMGSZ,
        save_txt=True,
        save_conf=True,
        name=name,
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
    # myolo10(
    #     'yolov10x-mdetect.yaml', weight_path='runs/mdetect/debug4/weights/best.pt', auto_optim=False, retrain=False,
    #      mloss_enlarge=100, name='debug', mdet= 100,
    # )
    # for mloss_enlarge in range(1, 11):
    #     print(f'train with mloss_enlarge {mloss_enlarge}')
    #     myolo10(
    #         'yolov10x-mdetect.yaml', weight_path='runs/mdetect/debug4/weights/best.pt', auto_optim=False, retrain=True,
    #          mloss_enlarge=100, name='debug',
    #     )
    # myolo10(
    #     'yolov10x-mdetect.yaml', weight_path='runs/mdetect/debug4/weights/best.pt', auto_optim=False, retrain=False,
    #      mloss_enlarge=100, name='debug', mdet= 100,
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


