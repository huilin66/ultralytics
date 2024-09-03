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

# endregion

if __name__ == '__main__':
    pass
    # myolo8x(r'yolov8x-mdetect.yaml', auto_optim=False)
    # myolo9e(r'yolov9e-mdetect.yaml', auto_optim=False)
    # myolo10x(r'yolov10x-mdetect.yaml', auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-elantf1.yaml', auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-elantf1res.yaml', auto_optim=False)
    # myolo10x(r'yolov10x-mdetect.yaml', auto_optim=False)
    # myolo10x(r'yolov10x-mdetect.yaml', auto_optim=False)
    # myolo10x(r'yolov10x-mdetect.yaml', auto_optim=False)
    # myolo10x(r'yolov10x-mdetect.yaml', auto_optim=False)

    # myolo10x(r'yolov10x-mdetect-psa_c3tr_1.yaml', auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3tr_1_res.yaml', auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1.yaml', auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res.yaml', auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3ghost_1.yaml', auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3ghost_1_res.yaml', auto_optim=False)

    # myolo8x(r'yolov8x-mdetect.yaml', auto_optim=False)
    # myolo9e(r'yolov9e-mdetect.yaml', auto_optim=False)
    # myolo10x(r'yolov10x-mdetect.yaml', auto_optim=False)

    # myolo8x(r'yolov8x-mdetect.yaml', auto_optim=False)
    # myolo9e(r'yolov9e-mdetect.yaml', auto_optim=False)
    # myolo10x(r'yolov10x-mdetect.yaml', auto_optim=False)
    # myolo8x(r'yolov8x-mdetect.yaml', auto_optim=False)
    # myolo9e(r'yolov9e-mdetect.yaml', auto_optim=False)
    # myolo10x(r'yolov10x-mdetect.yaml', auto_optim=False)
    # myolo8x(r'yolov8x-mdetect.yaml', auto_optim=False)
    # myolo9e(r'yolov9e-mdetect.yaml', auto_optim=False)
    # myolo10x(r'yolov10x-mdetect.yaml', auto_optim=False)


    # model_val(weight_path=r'runs/mdetect/train204/weights/best.pt')
    # model_val(weight_path=r'runs/mdetect/train205/weights/best.pt')
    # model_val(weight_path=r'runs/mdetect/train207/weights/best.pt')
    # model_val(weight_path=r'runs/mdetect/train208/weights/best.pt')
    # model_val(weight_path=r'runs/mdetect/train155/weights/best.pt')


    myolo10x(r'yolov10x-mdetect-psa_c3str_1_res.yaml',
               weight_path='runs/mdetect/train188/weights/best.pt',
               retrain=True, auto_optim=False)
    myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead.yaml',
               weight_path='runs/mdetect/train188/weights/best.pt',
               retrain=True, auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat11.yaml',
    #            weight_path='runs/mdetect/train188/weights/best.pt',
    #            retrain=True, auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat12.yaml',
    #            weight_path='runs/mdetect/train188/weights/best.pt',
    #            retrain=True, auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat13.yaml',
    #            weight_path='runs/mdetect/train188/weights/best.pt',
    #            retrain=True, auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat14.yaml',
    #            weight_path='runs/mdetect/train188/weights/best.pt',
    #            retrain=True, auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat15.yaml',
    #            weight_path='runs/mdetect/train188/weights/best.pt',
    #            retrain=True, auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat21.yaml',
    #            weight_path='runs/mdetect/train188/weights/best.pt',
    #            retrain=True, auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat22.yaml',
    #            weight_path='runs/mdetect/train188/weights/best.pt',
    #            retrain=True, auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat23.yaml',
    #            weight_path='runs/mdetect/train188/weights/best.pt',
    #            retrain=True, auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat24.yaml',
    #            weight_path='runs/mdetect/train188/weights/best.pt',
    #            retrain=True, auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat25.yaml',
    #            weight_path='runs/mdetect/train188/weights/best.pt',
    #            retrain=True, auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat31.yaml',
    #            weight_path='runs/mdetect/train188/weights/best.pt',
    #            retrain=True, auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat32.yaml',
    #            weight_path='runs/mdetect/train188/weights/best.pt',
    #            retrain=True, auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat33.yaml',
    #            weight_path='runs/mdetect/train188/weights/best.pt',
    #            retrain=True, auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat34.yaml',
    #            weight_path='runs/mdetect/train188/weights/best.pt',
    #            retrain=True, auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat35.yaml',
    #            weight_path='runs/mdetect/train188/weights/best.pt',
    #            retrain=True, auto_optim=False)
    myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat41.yaml',
               weight_path='runs/mdetect/train188/weights/best.pt',
               retrain=True, auto_optim=False)
    myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat42.yaml',
               weight_path='runs/mdetect/train188/weights/best.pt',
               retrain=True, auto_optim=False)
    myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat43.yaml',
               weight_path='runs/mdetect/train188/weights/best.pt',
               retrain=True, auto_optim=False)
    myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat44.yaml',
               weight_path='runs/mdetect/train188/weights/best.pt',
               retrain=True, auto_optim=False)
    myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat45.yaml',
               weight_path='runs/mdetect/train188/weights/best.pt',
               retrain=True, auto_optim=False)

    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead.yaml', auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat11.yaml', auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat12.yaml', auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat13.yaml', auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat21.yaml', auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat22.yaml', auto_optim=False)
    # myolo10x(r'yolov10x-mdetect-psa_c3str_1_res-sephead-gat23.yaml', auto_optim=False)