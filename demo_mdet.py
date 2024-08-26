import torch
from ultralytics import YOLO
BATCH_SIZE = 2
EPOCHS = 5
IMGSZ = 640
CONF = 0.5
DEVICE = torch.device('cuda:0')
# DATA = "billboard_mdet5_10_0806m.yaml"
DATA = "billboard_mdet5_10.yaml"
FREEZE_NUMS = {
    'yolov8' : 22,
    'yolov9e': 42,
    'yolov9' : 22,
    'yolov10': 23,
}

# region meta tools

def myolo_train(cfg_path, pretrain_path, auto_optim=True, retrain=False, **kwargs):
    model = YOLO(cfg_path, task='mdetect')
    model.load(pretrain_path)

    train_params = {
        'data': DATA,
        'device': DEVICE,
        'epochs': EPOCHS,
        'imgsz': IMGSZ,
        'val': True,
        'batch': BATCH_SIZE,
        'patience': EPOCHS
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
                'freeze_head':['.cv2', '.cv3'],
                'freeze_bn':True,
            }
        )

    train_params.update(kwargs)
    model.train(**train_params)

def model_val(network, weight_path):
    model = network(weight_path, task='mdetect')
    model.val(data=DATA, device=DEVICE)

def model_predict(network, weight_path, img_dir):
    model = network(weight_path, task='mdetect')
    model.predict(
        img_dir,
        save=True,
        conf=CONF,
        device=DEVICE,
        imgsz=IMGSZ,
    )

def model_export(network, weight_path):
    model = network(weight_path, task='mdetect')
    model.export(format='onnx')

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

def myolo8x(cfg_path, weight_path='yolov8x.pt', auto_optim=True, retrain=False, **kwargs):
    myolo_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def myolo9e(cfg_path, weight_path='yolov9e.pt', auto_optim=True, retrain=False, **kwargs):
    myolo_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def myolo10x(cfg_path, weight_path='yolov10x.pt', auto_optim=True, retrain=False, **kwargs):
    myolo_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

# endregion

if __name__ == '__main__':
    pass
    myolo8x(r'yolov8x-mdetect.yaml')