import torch
from ultralytics import YOLO
BATCH_SIZE = 32
EPOCHS = 500
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
}

# region meta tools

def myolo_train(cfg_path, pretrain_path, network=YOLO, auto_optim=True, retrain=False, **kwargs):
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
        # 'runs_dir': r'/nfsv4/23039356r/repository/ultralytics/runs/mdetect1'
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

def model_val(weight_path, network=YOLO):
    model = network(weight_path, task=TASK)
    print(weight_path)
    print(model.info(detailed=False))
    model.val(data=DATA, device=DEVICE)

def model_predict(weight_path, img_dir, network=YOLO):
    model = network(weight_path, task=TASK)
    model.predict(
        img_dir,
        save=True,
        conf=CONF,
        device=DEVICE,
        imgsz=IMGSZ,
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

def myolo8x(cfg_path, weight_path='yolov8x.pt', auto_optim=True, retrain=False, **kwargs):
    myolo_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def myolo9e(cfg_path, weight_path='yolov9e.pt', auto_optim=True, retrain=False, **kwargs):
    myolo_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def myolo10x(cfg_path, weight_path='yolov10x.pt', auto_optim=True, retrain=False, **kwargs):
    myolo_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def myolo8(cfg_path, weight_path='yolov8x.pt', auto_optim=True, retrain=False, **kwargs):
    myolo_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def myolo9(cfg_path, weight_path='yolov9e.pt', auto_optim=True, retrain=False, **kwargs):
    myolo_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def myolo10(cfg_path, weight_path='yolov10x.pt', auto_optim=True, retrain=False, **kwargs):
    myolo_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)


# endregion

if __name__ == '__main__':
    pass
    myolo8('yolov8n-mdetect.yaml', 'yolov8n.pt',auto_optim=False, name='exp_yolov8n')
    myolo8('yolov8s-mdetect.yaml', 'yolov8s.pt',auto_optim=False, name='exp_yolov8s')
    myolo8('yolov8m-mdetect.yaml', 'yolov8m.pt',auto_optim=False, name='exp_yolov8m')
    myolo8('yolov8l-mdetect.yaml', 'yolov8l.pt',auto_optim=False, name='exp_yolov8l')

    myolo9('yolov9s-mdetect.yaml', 'yolov9s.pt',auto_optim=False, name='exp_yolov9s')
    myolo9('yolov9m-mdetect.yaml', 'yolov9m.pt',auto_optim=False, name='exp_yolov9m')
    myolo9('yolov9c-mdetect.yaml', 'yolov9c.pt',auto_optim=False, name='exp_yolov9c')
    
    myolo10('yolov10n-mdetect.yaml', 'yolov10n.pt',auto_optim=False, name='exp_yolov10n')
    myolo10('yolov10s-mdetect.yaml', 'yolov10s.pt', auto_optim=False, name='exp_yolov10s')
    myolo10('yolov10m-mdetect.yaml', 'yolov10m.pt', auto_optim=False, name='exp_yolov10m')
    myolo10('yolov10b-mdetect.yaml', 'yolov10b.pt', auto_optim=False, name='exp_yolov10b')
    myolo10('yolov10l-mdetect.yaml', 'yolov10l.pt', auto_optim=False, name='exp_yolov10l')