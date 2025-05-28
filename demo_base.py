import torch
from ultralytics import YOLO

BATCH_SIZE = 32
EPOCHS = 500
IMGSZ = 640
CONF = 0.5
TASK = 'msegment'
DEVICE = torch.device('cuda:0')
DATA = "billboard_mseg_389.yaml"
FREEZE_NUMS = {
    'yolov8' : 22,
    'yolov9e': 42,
    'yolov9' : 22,
    'yolov10': 23,
    'yolov11': 23,
    'yolo12': 21,
}

# region meta tools

def model_train(cfg_path, pretrain_path, network=YOLO, auto_optim=True, retrain=False, **kwargs):
    model = network(cfg_path, task=TASK)
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
        freeze_num = get_freeze_num(cfg_path)
        train_params.update(
            {
                'freeze':freeze_num,
                'freeze_head':[f'{freeze_num}.cv2', f'{freeze_num}.cv3', f'{freeze_num}.cv4', f'{freeze_num}.proto'],
                'freeze_bn':True,
            }
        )

    train_params.update(kwargs)
    if "name" not in train_params:
        train_params["name"] = f"{train_params['data'].replace('.yaml', '')}-[{cfg_path.replace('.yaml', '')}]"
    model.train(**train_params)

def model_val(weight_path, network=YOLO, **kwargs):
    model = network(weight_path, task=TASK)
    model.val(device=DEVICE, **kwargs)

def model_predict(weight_path, img_dir, network=YOLO, save=True, save_txt=True,**kwargs):
    model = network(weight_path, task=TASK)
    result = model.predict(
        img_dir,
        save=save,
        save_txt=save_txt,
        conf=CONF,
        device=DEVICE,
        imgsz=IMGSZ,
        **kwargs,
    )
    for _ in result: pass

def model_export(weight_path, format='onnx', network=YOLO, **kwargs):
    model = network(weight_path, task=TASK)
    model.export(format=format, **kwargs)

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

def yolo8(cfg_path, weight_path='yolov8x.pt', auto_optim=True, retrain=False, **kwargs):
    assert 'yolov8' in cfg_path or 'yolo8' in cfg_path, ValueError(cfg_path, 'is not yolov8 config!')
    model_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def yolo9(cfg_path, weight_path='yolov9e.pt', auto_optim=True, retrain=False, **kwargs):
    assert 'yolov9' in cfg_path or 'yolo9' in cfg_path, ValueError(cfg_path, 'is not yolov9 config!')
    model_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def yolo10(cfg_path, weight_path='yolov10x.pt', auto_optim=True, retrain=False, **kwargs):
    assert 'yolov10' in cfg_path or 'yolo10' in cfg_path, ValueError(cfg_path, 'is not yolov10 config!')
    model_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def yolo11(cfg_path, weight_path='yolo11x.pt', auto_optim=True, retrain=False, **kwargs):
    assert 'yolov11' in cfg_path or 'yolo11' in cfg_path, ValueError(cfg_path, 'is not yolov11 config!')
    model_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def yolo12(cfg_path, weight_path='yolo12x.pt', auto_optim=True, retrain=False, **kwargs):
    assert 'yolov12' in cfg_path or 'yolo12' in cfg_path, ValueError(cfg_path, 'is not yolov12 config!')
    model_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)
# endregion

if __name__ == '__main__':
    pass
    # yolo8x('yolov8x-mseg.yaml', auto_optim=False, name=f'debug', retrain=True,task='msegment',
    #        weight_path=r'runs/segment/billboard_seg_3895/weights/best.pt')