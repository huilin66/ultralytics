import torch
from ultralytics import YOLO, RTDETR
BATCH_SIZE = 8
EPOCHS = 500
IMGSZ = 1280
CONF = 0.0
TASK = 'detect'
DEVICE = torch.device('cuda:0')
# DATA = "billboard_mdet5_10_det_0806m.yaml"
# DATA = "billboard_mdet5_10_c_det_0806m.yaml"
# DATA = "trafficsign.yaml"
DATA = "road_veg_1115.yaml"
FREEZE_NUMS = {
    'yolov8' : 22,
    'yolov9e': 42,
    'yolov9' : 22,
    'yolov10': 23,
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
        train_params.update(
            {
                'freeze':get_freeze_num(cfg_path),
                'freeze_head':['.cv2', '.cv3'],
                'freeze_bn':True,
            }
        )

    train_params.update(kwargs)
    model.train(**train_params)

def model_val(weight_path, network=YOLO, **kwargs):
    model = network(weight_path, task=TASK)
    model.val(data=DATA, device=DEVICE, **kwargs)

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

def yolo8x(cfg_path, weight_path='yolov8x.pt', auto_optim=True, retrain=False, **kwargs):
    model_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def yolo9e(cfg_path, weight_path='yolov9e.pt', auto_optim=True, retrain=False, **kwargs):
    model_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def yolo10x(cfg_path, weight_path='yolov10x.pt', auto_optim=True, retrain=False, **kwargs):
    model_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def rtdetrx(cfg_path, weight_path='rtdetr-x.pt', auto_optim=True, retrain=False, **kwargs):
    model_train(cfg_path, pretrain_path=weight_path, network=RTDETR, auto_optim=auto_optim, retrain=retrain, **kwargs)


def yolo8(cfg_path, weight_path='yolov8x.pt', auto_optim=True, retrain=False, **kwargs):
    model_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def yolo9(cfg_path, weight_path='yolov9e.pt', auto_optim=True, retrain=False, **kwargs):
    model_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def yolo10(cfg_path, weight_path='yolov10x.pt', auto_optim=True, retrain=False, **kwargs):
    model_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def rtdetr(cfg_path, weight_path='rtdetr-x.pt', auto_optim=True, retrain=False, **kwargs):
    model_train(cfg_path, pretrain_path=weight_path, network=RTDETR, auto_optim=auto_optim, retrain=retrain, **kwargs)

# endregion


if __name__ == '__main__':
    pass
    yolo8x('yolov8x.yaml', weight_path='yolov8x.pt', auto_optim=False, name='road_veg_1115_')
    yolo8x('yolov8x-p2.yaml', weight_path='yolov8x.pt', auto_optim=False, name='road_veg_1115_')
    yolo8x('yolov8x-p6.yaml', weight_path='yolov8x.pt', auto_optim=False, name='road_veg_1115_')
    # yolo8x('yolov8n.yaml', weight_path='yolov8n.pt', auto_optim=False)
    # yolo8x('yolov8n-p2.yaml', weight_path='yolov8n.pt', auto_optim=False)
    # yolo8x('yolov8n-p6.yaml', weight_path='yolov8n.pt', auto_optim=False)
    # model_val(r'runs/detect/train73/weights/best.pt')
    # model_predict(r'runs/detect/train73/weights/best.pt',
    #               r'/nfsv4/23039356r/data/traffsign/road_veg_1114/images')


    # yolo8x('yolov8x.yaml', auto_optim=False)
    # yolo9e('yolov9e.yaml', auto_optim=False)
    # yolo10x('yolov10x.yaml', auto_optim=False)
    # rtdetrx('rtdetr-x.yaml', auto_optim=False)
    # yolo10('yolov8n.yaml', 'yolov8n.pt', auto_optim=False, name='debug', mloss_enlarge=0.4)
    # yolo10('yolov10n.yaml', 'yolov10n.pt', auto_optim=False, name='debug', mloss_enlarge=0.4)

    # model_val(r'runs/detect/train71/weights/best.pt')
    # model_predict(r'runs/detect/train71/weights/best.pt',
    #               '/nfsv4/23039356r/data/traffsign/traff_sign_yolo/demo',)

    # import shutil
    # shutil.copy('/nfsv4/23039356r/data/traffsign/traff_sign_yolo/images/21452.png',
    #             '/nfsv4/23039356r/data/traffsign/traff_sign_yolo/demo/21452.png',
    #             )