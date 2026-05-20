import os
import sys
import yaml
import torch
from dotenv import load_dotenv
load_dotenv('ultralytics/cfg/.env')
from ultralytics import YOLO

BATCH_SIZE = 32
EPOCHS = 500
IMGSZ = 640
CONF_VAL = 0.001
CONF_PREDICT = 0.25
TASK = 'msegment'
DEVICE = torch.device('cpu')
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
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def tee_log_to_run_dir(trainer):
    save_dir = trainer.save_dir
    log_fp = open(os.path.join(save_dir, 'console.log'), 'w', buffering=1, encoding='utf-8')

    stdout_orig = sys.__stdout__
    stderr_orig = sys.__stderr__

    sys.stdout = Tee(stdout_orig, log_fp)
    sys.stderr = Tee(stderr_orig, log_fp)


def model_train(cfg_path, pretrain_path, network=YOLO, auto_optim=True, retrain=False, **kwargs):
    model = network(cfg_path, task=TASK)
    model.load(pretrain_path)
    model.add_callback("on_train_start", tee_log_to_run_dir)
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
                'freeze_att_head': [f'{freeze_num}.cva.{[freeze_att_num]}' for freeze_att_num in kwargs['freeze_att_nums']] if 'freeze_att_nums' in kwargs else None,
                'freeze_bn':True,
                'box': 0,
                'seg': 0,
                'cls': 0,
                'dfl': 0,
                'mdet': 10,
                'close_mosaic': 30,
            }
        )

    train_params.update(kwargs)
    if "name" not in train_params:
        train_params["name"] = f"{train_params['data'].replace('.yaml', '')}-[{cfg_path.replace('.yaml', '')}]"
    result = model.train(**train_params)
    return result

def model_val(weight_path, weight_name=True, network=YOLO, save_txt=False, **kwargs):
    if weight_name:
        weight_path = os.path.join('runs', TASK, weight_path, 'weights', 'best.pt')

    print(f'val with {weight_path}')
    model = network(weight_path, task=TASK)

    val_params = {
        'device': DEVICE,
        'batch': BATCH_SIZE,
        'conf': CONF_VAL,
        'save_txt':save_txt,
    }
    val_params.update(kwargs)
    result = model.val(**val_params)

    args_path = os.path.join(os.path.dirname(os.path.dirname(weight_path)), 'args.yaml')
    # print('project information:')
    # with open(args_path, 'r') as f:
    #     data = yaml.load(f, Loader=yaml.FullLoader)
    #     print(data)
    # print('============FINISH=============')
    return result

def model_predict(weight_path, img_dir, weight_name=True, network=YOLO, save=True, save_txt=True, stream=True, **kwargs):
    if weight_name:
        weight_path = os.path.join('runs', TASK, weight_path, 'weights', 'best.pt')
    model = network(weight_path, task=TASK)
    predict_params = {
        'device': DEVICE,
        'batch': BATCH_SIZE,
        'conf': CONF_PREDICT,
        'save' : save,
        'save_txt' : save_txt,
        'stream' : stream,
    }
    predict_params.update(kwargs)

    result = model.predict(img_dir, **predict_params,)
    for _ in result: pass

def model_track(weight_path, img_dir, weight_name=True, network=YOLO, single=False, save=True, save_txt=True, stream=True, **kwargs):
    if weight_name:
        weight_path = os.path.join('runs', TASK, weight_path, 'weights', 'best.pt')
    model = network(weight_path, task=TASK)
    predict_params = {
        'device': DEVICE,
        'batch': BATCH_SIZE,
        'conf': CONF_PREDICT,
        'save' : save,
        'save_txt' : save_txt,
        'stream' : stream,
        'tracker' : "botsort.yaml",
        'persist' : True,
    }
    predict_params.update(kwargs)
    if single:
        image_list = os.listdir(img_dir)
        for image_name in image_list:
            image_path = os.path.join(img_dir, image_name)
            result = model.track(image_path, **predict_params, )
    else:
        result = model.track(img_dir, **predict_params,)
        for _ in result: pass

def model_export(weight_path, format='onnx', weight_name=True, network=YOLO, **kwargs):
    if weight_name:
        weight_path = os.path.join('runs', TASK, weight_path, 'weights', 'best.pt')
    model = network(weight_path, task=TASK)
    model.export(format=format, device=DEVICE, **kwargs)

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