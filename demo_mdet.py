import torch
from ultralytics import YOLO
BATCH_SIZE = 32
EPOCHS = 100
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
    model.val(data=DATA, device=DEVICE, **kwargs)

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

def model_predict(weight_path, img_dir, network=YOLO):
    model = network(weight_path, task=TASK)
    model.predict(
        img_dir,
        save=True,
        conf=CONF,
        device=DEVICE,
        imgsz=IMGSZ,
        save_txt=True,
        save_conf=True,
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




    # mayolo('yolov10x-mdetect-plus.yaml', 'yolov10x.pt', name='exp_retrain', mloss_enlarge=0.3)
    # mayolo('mayolovx.yaml', 'runs/mdetect/exp_retrain/weights/best.pt', name='exp_retrain', mloss_enlarge=0.3, retrain=True)
    # mayolo('yolov10x-mdetect-psa_c3strcp_3.yaml', 'yolov10x.pt', name='exp_mayolox', mloss_enlarge=0.3)
    # mayolo('yolov10x-mdetect-psa_c3strcp_3_sep1head32_gatcos_res.yaml', 'runs/mdetect/exp_mayolox3/weights/best.pt', name='exp_retrain', mloss_enlarge=0.3, retrain=True)


    # myolo10('yolov10x-mdetect-psa_c3tr_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3__', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c3str_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3_', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c3strsp_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3_', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c3strcp_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3_', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4str_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3_', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4str2_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3_', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4str3_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3_', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4strcbam_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3_', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4strcbam2_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3_', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4strcbam3_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3_', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c3str2_3.yaml', 'yolov10x.pt',  name='exp_yolo10x_m3_', mloss_enlarge=0.3)

    # myolo10('yolov10x-mdetect-psa_c3trsp_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3_', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c3trcp_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3_', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4trcbam1_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3_', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4trcbam2_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3_', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4trcbam3_3.yaml',  'yolov10x.pt',  name='exp_yolo10x_m3_', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4str2_3_res.yaml', 'yolov10x.pt', name='debug', mloss_enlarge=0.3)
    # myolo9('yolov9m-mdetect.yaml', 'yolov9s.pt', name='debug', mloss_enlarge=0.3)

    #

    # myolo10('yolov10x-mdetect-psa_c4str2n2_3.yaml', 'yolov10x.pt', name='exp_yolo10x_m4_', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4str20_3.yaml', 'yolov10x.pt', name='exp_yolo10x_m4_', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4str21_3.yaml', 'yolov10x.pt', name='exp_yolo10x_m4_', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4str22_3.yaml', 'yolov10x.pt', name='exp_yolo10x_m4_', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c4str23_3.yaml', 'yolov10x.pt', name='exp_yolo10x_m4_', mloss_enlarge=0.3)
    #
    # myolo9('yolov9m-mdetect.yaml', 'yolov9s.pt', name='yolov9n', mloss_enlarge=0.3)

    # myolo10('yolov10x-mdetect-psa_c3trcp_3.yaml', 'runs/mdetect/exp_yolo10x_m34/weights/best.pt', name='exp_yolo10x_exp_', mloss_enlarge=0.3)
    # myolo10('yolov10x-mdetect-psa_c3trcp_3.yaml', 'yolov10x.pt', name='exp_yolo10x_exp_', mloss_enlarge=0.3)


    # myolo10('yolov10x-mdetect-psa_c4str2_3_res_9head.yaml',
    #         weight_path=r'yolov10x.pt',
    #         name='exp_yolo10x_m3res_6_head', mloss_enlarge=0.3
    #         )
    # myolo10('yolov10x-mdetect-psa_c4str2_3_res_8head.yaml',
    #         weight_path=r'yolov10x.pt',
    #         name='exp_yolo10x_m3res_6_head', mloss_enlarge=0.3
    #         )
    # myolo10('yolov10x-mdetect-psa_c4str2_3_res_9head.yaml',
    #         weight_path=r'runs/mdetect/exp_yolo10x_m3res_6/weights/best.pt',
    #         name='exp_yolo10x_m3res_6_head', mloss_enlarge=0.3
    #         )
    # myolo10('yolov10x-mdetect-psa_c4str2_3_res_8head.yaml',
    #         weight_path=r'runs/mdetect/exp_yolo10x_m3res_6/weights/best.pt',
    #         name='exp_yolo10x_m3res_6_head', mloss_enlarge=0.3
    #         )
    # myolo10('yolov10x-mdetect-psa_c4str2_3_res_sep9head.yaml',
    #         weight_path=r'yolov10x.pt',
    #         name='exp_yolo10x_m3res_6_head', mloss_enlarge=0.3
    #         )
    # myolo10('yolov10x-mdetect-psa_c4str2_3_res_sep8head.yaml',
    #         weight_path=r'runs/mdetect/exp_yolo10x_m3res_6/weights/best.pt',
    #         name='exp_yolo10x_m3res_6_head', mloss_enlarge=0.3
    #         )

    # myolo10('yolov10x-mdetect-psa_c4str2_3_res_sep1head100.yaml',
    #         weight_path=r'runs/mdetect/exp_yolo10x_m3res_6/weights/best.pt',
    #         name='exp_yolo10x_m3res_6_head', mloss_enlarge=0.3
    #         )
    # myolo10('yolov10x-mdetect-psa_c4str2_3_res_sep1head32.yaml',
    #         weight_path=r'runs/mdetect/exp_yolo10x_m3res_6/weights/best.pt',
    #         name='exp_yolo10x_m3res_6_head', mloss_enlarge=0.3
    #         )

    # mayolo('yolov10n-mdetect-plus.yaml', 'yolov10n.pt', name='exp__mayolon', mloss_enlarge=0.3)
    # mayolo('yolov10s-mdetect-plus.yaml', 'yolov10s.pt', name='exp__mayolos', mloss_enlarge=0.3)
    # mayolo('yolov10m-mdetect-plus.yaml', 'yolov10m.pt', name='exp__mayolom', mloss_enlarge=0.3)
    # mayolo('yolov10b-mdetect-plus.yaml', 'yolov10b.pt', name='exp__mayolob', mloss_enlarge=0.3)
    # mayolo('yolov10l-mdetect-plus.yaml', 'yolov10l.pt', name='exp__mayolol', mloss_enlarge=0.3)
    # mayolo('yolov10x-mdetect-plus.yaml', 'yolov10x.pt', name='exp__mayolox', mloss_enlarge=0.3)

    # mayolo('mayolovn.yaml', 'runs/mdetect/exp_mayolon/weights/best.pt', name='exp_mayolon', mloss_enlarge=0.3, retrain=True)
    # mayolo('mayolovs.yaml', 'runs/mdetect/exp_mayolos/weights/best.pt', name='exp_mayolos', mloss_enlarge=0.3, retrain=True)
    # mayolo('mayolovm.yaml', 'runs/mdetect/exp_mayolom/weights/best.pt', name='exp_mayolom', mloss_enlarge=0.3, retrain=True)
    # mayolo('mayolovb.yaml', 'runs/mdetect/exp_mayolob/weights/best.pt', name='exp_mayolob', mloss_enlarge=0.3, retrain=True)
    # mayolo('mayolovl.yaml', 'runs/mdetect/exp_mayolol/weights/best.pt', name='exp_mayolol', mloss_enlarge=0.3, retrain=True)
    # mayolo('mayolovx.yaml', 'runs/mdetect/exp_mayolox/weights/best.pt', name='exp_mayolox', mloss_enlarge=0.3, retrain=True)

    # myolo10('yolov10x-mdetect-psa_c3strcp5_3.yaml',
    #         weight_path=r'yolov10x.pt',
    #         name='exp_yolo10x_exp',
    #         mloss_enlarge=0.3
    #         )
    # myolo10('yolov10x-mdetect-psa_c3strss_3.yaml',
    #         weight_path=r'yolov10x.pt',
    #         name='exp_yolo10x_exp',
    #         mloss_enlarge=0.3
    #         )
    # myolo10('yolov10x-mdetect-psa_c4trcbam1_3_res_8head.yaml',
    #         weight_path=r'runs/mdetect/exp_yolo10x_m3res_14/weights/best.pt',
    #         name='exp_yolo10x_exp', mloss_enlarge=0.3
    #         )
    # myolo10('yolov10x-mdetect-psa_c4trcbam1_3_res_sep8head.yaml',
    #         weight_path=r'runs/mdetect/exp_yolo10x_m3res_14/weights/best.pt',
    #         name='exp_yolo10x_exp', mloss_enlarge=0.3,
    #         )

    # model_val(r'runs/mdetect/exp_yolo10x_m34/weights/best.pt')



    # myolo10('yolov10x-mdetect-psa_c3strcp_3_gatcom6.yaml',
    #         'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #         name='exp_yolo10x_ablation_head',
    #         mloss_enlarge=0.3, retrain=True)
    # myolo10('yolov10x-mdetect-psa_c3strcp_3_gatcom6_nosf.yaml',
    #         'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #         name='exp_yolo10x_ablation_head',
    #         mloss_enlarge=0.3, retrain=True)
    # myolo10('yolov10x-mdetect-psa_c3strcp_3_gatcom6_nosf_pure.yaml',
    #         'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #         name='exp_yolo10x_ablation_head',
    #         mloss_enlarge=0.3, retrain=True)
    # myolo10('yolov10x-mdetect-psa_c3strcp_3_gatcom6_pure.yaml',
    #         'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #         name='exp_yolo10x_ablation_head',
    #         mloss_enlarge=0.3, retrain=True)
    # myolo10('yolov10x-mdetect-psa_c3strcp_3_gatcom6_res.yaml',
    #         'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #         name='exp_yolo10x_ablation_head',
    #         mloss_enlarge=0.3, retrain=True)
    # myolo10('yolov10x-mdetect-psa_c3strcp_3_gatcom6_res_nosf.yaml',
    #         'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #         name='exp_yolo10x_ablation_head',
    #         mloss_enlarge=0.3, retrain=True)
    # myolo10('yolov10x-mdetect-psa_c3strcp_3_gatcom6_res_nosf_pure.yaml',
    #         'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #         name='exp_yolo10x_ablation_head',
    #         mloss_enlarge=0.3, retrain=True)
    # myolo10('yolov10x-mdetect-psa_c3strcp_3_gatcom6_res_pure.yaml',
    #         'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #         name='exp_yolo10x_ablation_head',
    #         mloss_enlarge=0.3, retrain=True)
    # 
    # 
    # myolo10('yolov10x-mdetect-psa_c3strcp_3_gatcos.yaml',
    #         'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #         name='exp_yolo10x_ablation_head',
    #         mloss_enlarge=0.3, retrain=True)
    # myolo10('yolov10x-mdetect-psa_c3strcp_3_gatcos_res.yaml',
    #         'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #         name='exp_yolo10x_ablation_head',
    #         mloss_enlarge=0.3, retrain=True)
    # myolo10('yolov10x-mdetect-psa_c3strcp_3_gatcost.yaml',
    #         'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #         name='exp_yolo10x_ablation_head',
    #         mloss_enlarge=0.3, retrain=True)
    # myolo10('yolov10x-mdetect-psa_c3strcp_3_gatcost_res.yaml',
    #         'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #         name='exp_yolo10x_ablation_head',
    #         mloss_enlarge=0.3, retrain=True)
    # myolo10('yolov10x-mdetect-psa_c3strcp_3_gatmlp.yaml',
    #         'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #         name='exp_yolo10x_ablation_head',
    #         mloss_enlarge=0.3, retrain=True)
    # myolo10('yolov10x-mdetect-psa_c3strcp_3_gatmlp_res.yaml',
    #         'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #         name='exp_yolo10x_ablation_head',
    #         mloss_enlarge=0.3, retrain=True)
    # myolo10('yolov10x-mdetect-psa_c3strcp_3_gatmlpr.yaml',
    #         'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #         name='exp_yolo10x_ablation_head',
    #         mloss_enlarge=0.3, retrain=True)
    # myolo10('yolov10x-mdetect-psa_c3strcp_3_gatmlpr_res.yaml',
    #         'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #         name='exp_yolo10x_ablation_head',
    #         mloss_enlarge=0.3, retrain=True)
    # myolo10('yolov10x-mdetect-psa_c3strcp_3_gatmlpt.yaml',
    #         'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #         name='exp_yolo10x_ablation_head',
    #         mloss_enlarge=0.3, retrain=True)
    # myolo10('yolov10x-mdetect-psa_c3strcp_3_gatmlpt_res.yaml',
    #         'runs/mdetect/exp_yolo10x_m34/weights/best.pt',
    #         name='exp_yolo10x_ablation_head',
    #         mloss_enlarge=0.3, retrain=True)


    model_val('runs/mdetect/exp_yolo10x_ablation_head2/weights/best.pt')
    model_val('runs/mdetect/exp_yolo10x_ablation_head3/weights/best.pt')
    model_val('runs/mdetect/exp_yolo10x_ablation_head4/weights/best.pt')
    model_val('runs/mdetect/exp_yolo10x_ablation_head5/weights/best.pt')
    model_val('runs/mdetect/exp_yolo10x_ablation_head6/weights/best.pt')
    model_val('runs/mdetect/exp_yolo10x_ablation_head7/weights/best.pt')
    model_val('runs/mdetect/exp_yolo10x_ablation_head8/weights/best.pt')
    model_val('runs/mdetect/exp_yolo10x_ablation_head9/weights/best.pt')
    model_val('runs/mdetect/exp_yolo10x_ablation_head10/weights/best.pt')
    model_val('runs/mdetect/exp_yolo10x_ablation_head11/weights/best.pt')
    model_val('runs/mdetect/exp_yolo10x_ablation_head12/weights/best.pt')
    model_val('runs/mdetect/exp_yolo10x_ablation_head13/weights/best.pt')
    model_val('runs/mdetect/exp_yolo10x_ablation_head14/weights/best.pt')
    model_val('runs/mdetect/exp_yolo10x_ablation_head15/weights/best.pt')
    model_val('runs/mdetect/exp_yolo10x_ablation_head16/weights/best.pt')
    model_val('runs/mdetect/exp_yolo10x_ablation_head17/weights/best.pt')
    model_val('runs/mdetect/exp_yolo10x_ablation_head18/weights/best.pt')
    model_val('runs/mdetect/exp_yolo10x_ablation_head19/weights/best.pt')