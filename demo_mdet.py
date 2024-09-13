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

def myolo8(cfg_path, weight_path='yolov8x.pt', auto_optim=True, retrain=False, **kwargs):
    assert 'yolov8' in cfg_path, ValueError(cfg_path, 'is not yolov8 config!')
    myolo_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def myolo9(cfg_path, weight_path='yolov9e.pt', auto_optim=True, retrain=False, **kwargs):
    assert 'yolov9' in cfg_path, ValueError(cfg_path, 'is not yolov9 config!')
    myolo_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def myolo10(cfg_path, weight_path='yolov10x.pt', auto_optim=True, retrain=False, **kwargs):
    assert 'yolov10' in cfg_path, ValueError(cfg_path, 'is not yolov10 config!')
    myolo_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)


# endregion

if __name__ == '__main__':
    pass
    # myolo8('yolov8x-mdetect.yaml', 'yolov8x.pt', auto_optim=False, name='debug', mloss_enlarge=0.4)
    # myolo9('yolov9e-mdetect.yaml', 'yolov9e.pt',auto_optim=False, name='debug', mloss_enlarge=0.4)
    myolo10('yolov10x-mdetect.yaml', 'yolov10x.pt',auto_optim=False, name='debug', mloss_enlarge=0.4)
    # myolo10('yolov10n-mdetect.yaml', 'yolov10n.pt',auto_optim=False, name='debug', mloss_enlarge=0.4)

    # myolo8('yolov8n-mdetect.yaml', 'yolov8n.pt',auto_optim=False, name='debug', mloss_enlarge=0.4)
    # myolo8('yolov8s-mdetect.yaml', 'yolov8s.pt',auto_optim=False, name='exp_yolov8s', mloss_enlarge=0.4)
    # myolo8('yolov8m-mdetect.yaml', 'yolov8m.pt',auto_optim=False, name='exp_yolov8m', mloss_enlarge=0.4)
    # myolo8('yolov8l-mdetect.yaml', 'yolov8l.pt',auto_optim=False, name='exp_yolov8l', mloss_enlarge=0.4)
    #
    # myolo9('yolov9s-mdetect.yaml', 'yolov9s.pt',auto_optim=False, name='exp_yolov9s', mloss_enlarge=0.4)
    # myolo9('yolov9m-mdetect.yaml', 'yolov9m.pt',auto_optim=False, name='exp_yolov9m', mloss_enlarge=0.4)
    # myolo9('yolov9c-mdetect.yaml', 'yolov9c.pt',auto_optim=False, name='exp_yolov9c', mloss_enlarge=0.4)
    #
    # myolo10('yolov10n-mdetect.yaml', 'yolov10n.pt',auto_optim=False, name='exp_yolov10n', mloss_enlarge=0.4)
    # myolo10('yolov10s-mdetect.yaml', 'yolov10s.pt', auto_optim=False, name='exp_yolov10s', mloss_enlarge=0.4)
    # myolo10('yolov10m-mdetect.yaml', 'yolov10m.pt', auto_optim=False, name='exp_yolov10m', mloss_enlarge=0.4)
    # myolo10('yolov10b-mdetect.yaml', 'yolov10b.pt', auto_optim=False, name='exp_yolov10b', mloss_enlarge=0.4)
    # myolo10('yolov10l-mdetect.yaml', 'yolov10l.pt', auto_optim=False, name='exp_yolov10l', mloss_enlarge=0.4)

    # myolo8('yolov8x-mdetect.yaml', 'yolov8x.pt',auto_optim=False, name='exp_yolov8x', mloss_enlarge=0.4)
    # myolo9('yolov9e-mdetect.yaml', 'yolov9e.pt',auto_optim=False, name='exp_yolov9e', mloss_enlarge=0.4)
    # myolo10('yolov10x-mdetect.yaml', 'yolov10x.pt',auto_optim=False, name='exp_yolov10x', mloss_enlarge=0.4)
    # myolo10('yolov10x-mdetect-psa_c3str_1_res.yaml', 'yolov10x.pt', auto_optim=False, name='exp_mayolox',
    #         mloss_enlarge=0.4)
    # myolo10('yolov10n-mdetect-psa_c3strpp_1_res.yaml', 'yolov10n.pt', auto_optim=False, name='debug',
    #         mloss_enlarge=0.4)
    # myolo10('yolov10s-mdetect-psa_c3strpp_1_res.yaml', 'yolov10s.pt', auto_optim=False, name='exp_mayolops',
    #         mloss_enlarge=0.4)
    # myolo10('yolov10m-mdetect-psa_c3strpp_1_res.yaml', 'yolov10m.pt', auto_optim=False, name='exp_mayolopm',
    #         mloss_enlarge=0.4)
    # myolo10('yolov10b-mdetect-psa_c3strpp_1_res.yaml', 'yolov10b.pt', auto_optim=False, name='exp_mayolopb',
    #         mloss_enlarge=0.4)
    # myolo10('yolov10l-mdetect-psa_c3strpp_1_res.yaml', 'yolov10l.pt', auto_optim=False, name='exp_mayolopl',
    #         mloss_enlarge=0.4)

    # model_val(r'runs/mdetect/exp_yolov8n/weights/best.pt')
    # model_val(r'runs/mdetect/exp_yolov8s/weights/best.pt')
    # model_val(r'runs/mdetect/exp_yolov8m/weights/best.pt')
    # model_val(r'runs/mdetect/exp_yolov8l/weights/best.pt')
    #
    # model_val(r'runs/mdetect/exp_yolov9s/weights/best.pt')
    # model_val(r'runs/mdetect/exp_yolov9m/weights/best.pt')
    # model_val(r'runs/mdetect/exp_yolov9c/weights/best.pt')
    #
    # model_val(r'runs/mdetect/exp_yolov10n/weights/best.pt')
    # model_val(r'runs/mdetect/exp_yolov10s/weights/best.pt')
    # model_val(r'runs/mdetect/exp_yolov10m/weights/best.pt')
    # model_val(r'runs/mdetect/exp_yolov10b/weights/best.pt')
    # model_val(r'runs/mdetect/exp_yolov10l/weights/best.pt')
