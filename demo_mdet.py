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
# MLOSS_ENLARGE = 0.3
# region meta tools

def myolo_train(cfg_path, pretrain_path, network=YOLO, auto_optim=False, retrain=False, fine_tune=False, **kwargs):
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
                'freeze_head':['.cv2', '.cv3'] if 'yolov10' not in cfg_path else ['.cv2', '.cv3', '.one2one_cv2', '.one2one_cv3'],
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

def myolo8(cfg_path, weight_path='yolov8x.pt', auto_optim=False, retrain=False, **kwargs):
    assert 'yolov8' in cfg_path, ValueError(cfg_path, 'is not yolov8 config!')
    myolo_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def myolo9(cfg_path, weight_path='yolov9e.pt', auto_optim=False, retrain=False, **kwargs):
    assert 'yolov9' in cfg_path, ValueError(cfg_path, 'is not yolov9 config!')
    myolo_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

def myolo10(cfg_path, weight_path='yolov10x.pt', auto_optim=False, retrain=False, **kwargs):
    assert 'yolov10' in cfg_path, ValueError(cfg_path, 'is not yolov10 config!')
    myolo_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)


# endregion

if __name__ == '__main__':
    pass
    # myolo10('yolov10x-mdetect.yaml', 'yolov10x.pt', name='exp_mloss_enlarge', mloss_enlarge=0.00)
    # myolo10('yolov10x-mdetect.yaml', 'yolov10x.pt', name='exp_mloss_enlarge', mloss_enlarge=0.05)
    # myolo10('yolov10x-mdetect.yaml', 'yolov10x.pt', name='exp_mloss_enlarge', mloss_enlarge=0.10)
    # myolo10('yolov10x-mdetect.yaml', 'yolov10x.pt', name='exp_mloss_enlarge', mloss_enlarge=0.15)
    # myolo10('yolov10x-mdetect.yaml', 'yolov10x.pt', name='exp_mloss_enlarge', mloss_enlarge=0.20)

    # myolo8('yolov8n-mdetect.yaml', 'yolov8n.pt', name='exp_yolo8n', mloss_enlarge=0.3)
    # myolo8('yolov8s-mdetect.yaml', 'yolov8s.pt', name='exp_yolo8s', mloss_enlarge=0.3)
    # myolo8('yolov8m-mdetect.yaml', 'yolov8m.pt', name='exp_yolo8m', mloss_enlarge=0.3)
    # myolo8('yolov8l-mdetect.yaml', 'yolov8l.pt', name='exp_yolo8l', mloss_enlarge=0.3)
    # myolo8('yolov8x-mdetect.yaml', 'yolov8x.pt', name='exp_yolo8x', mloss_enlarge=0.3)
    #
    # myolo9('yolov9m-mdetect.yaml', 'yolov9m.pt', name='exp_yolo9m', mloss_enlarge=0.3)
    # myolo9('yolov9s-mdetect.yaml', 'yolov9s.pt', name='exp_yolo9s', mloss_enlarge=0.3)
    #
    #
    #
    # myolo10('yolov10x-mdetect-psa_c3tr_1.yaml',  'yolov10x.pt',  name='exp_yolo10x_m1', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c3str_1.yaml',  'yolov10x.pt',  name='exp_yolo10x_m1', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c3strsp_1.yaml',  'yolov10x.pt',  name='exp_yolo10x_m1', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c3strcp_1.yaml',  'yolov10x.pt',  name='exp_yolo10x_m1', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4str_1.yaml',  'yolov10x.pt',  name='exp_yolo10x_m1', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4str2_1.yaml',  'yolov10x.pt',  name='exp_yolo10x_m1', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4str3_1.yaml',  'yolov10x.pt',  name='exp_yolo10x_m1', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4strcbam_1.yaml',  'yolov10x.pt',  name='exp_yolo10x_m1', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4strcbam2_1.yaml',  'yolov10x.pt',  name='exp_yolo10x_m1', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4strcbam3_1.yaml',  'yolov10x.pt',  name='exp_yolo10x_m1', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c3str2_1.yaml', 'yolov10x.pt',  name='exp_yolo10x_m1', mloss_enlarge=0.3)
    #
    # myolo10('yolov10x-mdetect-psa_c3tr_1_res.yaml',  'yolov10x.pt',  name='exp_yolo10x_m1', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c3str_1_res.yaml',  'yolov10x.pt',  name='exp_yolo10x_m1', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c3strsp_1_res.yaml',  'yolov10x.pt',  name='exp_yolo10x_m1', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c3strcp_1_res.yaml',  'yolov10x.pt',  name='exp_yolo10x_m1', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4str_1_res.yaml',  'yolov10x.pt',  name='exp_yolo10x_m1', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4str2_1_res.yaml',  'yolov10x.pt',  name='exp_yolo10x_m1', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4str3_1_res.yaml',  'yolov10x.pt',  name='exp_yolo10x_m1', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4strcbam_1_res.yaml',  'yolov10x.pt',  name='exp_yolo10x_m1', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4strcbam2_1_res.yaml',  'yolov10x.pt',  name='exp_yolo10x_m1', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4strcbam3_1_res.yaml',  'yolov10x.pt',  name='exp_yolo10x_m1', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c3str2_1_res.yaml', 'yolov10x.pt',  name='exp_yolo10x_m1',mloss_enlarge=0.3 )

    # myolo10('yolov10x-mdetect-psa_c4strn2_1.yaml', 'yolov10x.pt', name='exp_yolo10x_mm', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c3strcpn2_3.yaml', 'yolov10x.pt', name='exp_yolo10x_mm', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4strcbam2n2_3.yaml', 'yolov10x.pt', name='exp_yolo10x_mm', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c3strn2_3_res.yaml', 'yolov10x.pt', name='exp_yolo10x_mm', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c3strcpn2_3_res.yaml', 'yolov10x.pt', name='exp_yolo10x_mm', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4strn2_3_res.yaml', 'yolov10x.pt', name='exp_yolo10x_mm', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4str2n2_3_res.yaml', 'yolov10x.pt', name='exp_yolo10x_mm', mloss_enlarge=0.3)

    # myolo10('yolov10x-mdetect-psa_c3strcp_3.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt', name='exp_yolo10x_mmm',
    #         mloss_enlarge=0.3, retrain=True, fine_tune=True)
    # myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head32.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt', name='exp_yolo10x_mmm',
    #         mloss_enlarge=0.3, retrain=True, fine_tune=True)
    # myolo10('yolov10x-mdetect-psa_c3strcp_3_sep1head100.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt', name='exp_yolo10x_mmm',
    #         mloss_enlarge=0.3, retrain=True, fine_tune=True)
    # myolo10('yolov10x-mdetect-psa_c3strcp_3_sep2head32.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt', name='exp_yolo10x_mmm',
    #         mloss_enlarge=0.3, retrain=True, fine_tune=True)
    # myolo10('yolov10x-mdetect-psa_c3strcp_3_sep2head100.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt', name='exp_yolo10x_mmm',
    #         mloss_enlarge=0.3, retrain=True, fine_tune=True)