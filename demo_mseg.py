import torch
from ultralytics import YOLO, RTDETR
BATCH_SIZE = 32
EPOCHS = 500
IMGSZ = 640
CONF = 0.5
TASK = 'msegment'
DEVICE = torch.device('cuda:0')
DATA = "billboard_mseg_127.yaml"
DATA_TRAIN = DATA.replace('.yaml', '_train.yaml')
DATA_ALL = DATA.replace('.yaml', '_all.yaml')
FREEZE_NUMS = {
    'yolov8' : 22,
    'yolov9e': 42,
    'yolov9' : 22,
    'yolov10': 23,
}

# region meta tools

def model_train(cfg_path, pretrain_path, network=YOLO, auto_optim=True, retrain=False, **kwargs):
    model = network(cfg_path, task=TASK)
    # save_ckpt_path = '/nfsv4/23039356r/repository/ultralytics/my_tools/ckpt_mseg1.pth'
    # torch.save(model.model.state_dict(), save_ckpt_path)
    # return
    model.load(pretrain_path)
    # save_ckpt_path = '/nfsv4/23039356r/repository/ultralytics/my_tools/ckpt_mseg2.pth'
    # torch.save(model.model.state_dict(), save_ckpt_path)

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
    save_ckpt_path = '/nfsv4/23039356r/repository/ultralytics/my_tools/ckpt_mseg3.pth'
    torch.save(model.model.state_dict(), save_ckpt_path)
    # return
    train_params.update(kwargs)
    model.train(**train_params)

def model_val(weight_path, network=YOLO, **kwargs):
    model = network(weight_path, task=TASK)
    model.val(data=DATA, device=DEVICE, **kwargs)

def model_val_train(weight_path, network=YOLO, **kwargs):
    model = network(weight_path, task=TASK)
    model.val(data=DATA_TRAIN, device=DEVICE, **kwargs)

def model_val_all(weight_path, network=YOLO, **kwargs):
    model = network(weight_path, task=TASK)
    model.val(data=DATA_ALL, device=DEVICE, **kwargs)

def model_predict(weight_path, img_dir, network=YOLO):
    model = network(weight_path, task=TASK)
    model.predict(
        img_dir,
        save=True,
        conf=CONF,
        device=DEVICE,
        imgsz=IMGSZ,
        save_txt=True,
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

# endregion




if __name__ == '__main__':
    pass
    # yolo8x('yolov8x-mseg.yaml', auto_optim=False, name=f'billboard_mseg_127')
    # model_val(r'runs/msegment/billboard_mseg_127/weights/best.pt', save_txt=True)
    # model_val_train(r'runs/msegment/billboard_mseg_127/weights/best.pt', save_txt=True)
    # model_val_all(r'runs/msegment/billboard_mseg_127/weights/best.pt', save_txt=True)
    # model_predict(r'runs/msegment/billboard_mseg_127/weights/best.pt',
    #               # '/nfsv4/23039356r/data/billboard/bd_data/data127/images/20210325_172330.jpg',
    #               r'/nfsv4/23039356r/data/billboard/bd_data/data127/images'
    #               )
    model_predict(r'runs/msegment/billboard_mseg_127/weights/best.pt',
                  r'/nfsv4/23039356r/data/billboard/bd_data/data127/val.txt'
                  )
    # yolo9e('yolov9e.yaml', auto_optim=False)
    # yolo10x('yolov10x.yaml', auto_optim=False)
    # rtdetrx('rtdetr-x.yaml', auto_optim=False)
    # model_val(r'runs/msegment/train4/weights/best.pt')
    # model_predict(r'runs/segment/train3/weights/best.pt',
    #               r'/nfsv4/23039356r/data/billboard/data0521_m/yolo_rgb_segmentation1/images')
    # model_predict(r'runs/msegment/debug108/weights/best.pt',
    #               r'/nfsv4/23039356r/data/billboard/bd_data/selected_sample/images/48.JPG')

    # model_export(r'runs/msegment/debug108/weights/best.pt')