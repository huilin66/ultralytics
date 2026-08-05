import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

from ultralytics import YOLO

BATCH_SIZE = 32
EPOCHS = 500
IMGSZ = 640
CONF_VAL = 0.001
CONF_PREDICT = 0.25
TASK = "msegment"
DEVICE = torch.device("cuda:0")
DATA = "billboard_mseg_389.yaml"
FREEZE_NUMS = {
    "yolov8": 22,
    "yolov9e": 42,
    "yolov9": 22,
    "yolov10": 23,
    "yolov11": 23,
    "yolo12": 21,
}


# region meta tools
def read_yaml_safely(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.load(f, Loader=yaml.FullLoader) or {}
    except Exception as e:
        print(f"[WARN] failed to read yaml: {path}, error: {e}")
        return {}


def append_dict_to_csv(row, csv_path):
    """
    Append one dict row to CSV.
    If new keys appear, rewrite CSV with expanded header.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    row = {k: _to_scalar(v) for k, v in row.items()}

    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        old_rows = list(reader)
        old_fields = reader.fieldnames or []

    new_fields = list(old_fields)
    for k in row.keys():
        if k not in new_fields:
            new_fields.append(k)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        writer.writerows(old_rows)
        writer.writerow(row)


def _to_scalar(v):
    """
    Make values CSV-safe.
    """
    if v is None:
        return ""

    if isinstance(v, (str, int, float, bool)):
        return v

    try:
        if hasattr(v, "item"):
            return v.item()
    except Exception:
        pass

    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def collect_val_metrics(result):
    """
    Collect scalar metrics from Ultralytics val result.
    """
    row = {}

    if hasattr(result, "results_dict"):
        row.update(result.results_dict)

    if hasattr(result, "speed") and isinstance(result.speed, dict):
        for k, v in result.speed.items():
            row[f"speed/{k}"] = v

    # Optional common attributes, depending on Ultralytics version/task.
    for attr in ["fitness"]:
        if hasattr(result, attr):
            try:
                row[attr] = getattr(result, attr)
            except Exception:
                pass

    return row


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
    log_fp = open(os.path.join(save_dir, "console.log"), "w", buffering=1, encoding="utf-8")

    stdout_orig = sys.__stdout__
    stderr_orig = sys.__stderr__

    sys.stdout = Tee(stdout_orig, log_fp)
    sys.stderr = Tee(stderr_orig, log_fp)


def model_train(cfg_path, pretrain_path, network=YOLO, auto_optim=True, retrain=False, **kwargs):
    model = network(cfg_path, task=TASK)
    model.load(pretrain_path)
    model.add_callback("on_train_start", tee_log_to_run_dir)
    train_params = {
        "data": DATA,
        "device": DEVICE,
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "val": True,
        "batch": BATCH_SIZE,
        "patience": EPOCHS,
        "plots": False,
    }

    if not auto_optim:
        train_params.update({"optimizer": "AdamW", "lr0": 0.0001})
    if retrain:
        freeze_num = get_freeze_num(cfg_path)
        train_params.update(
            {
                "freeze": freeze_num,
                "freeze_head": [f"{freeze_num}.cv2", f"{freeze_num}.cv3", f"{freeze_num}.cv4", f"{freeze_num}.proto"],
                "freeze_att_head": [
                    f"{freeze_num}.cva.{[freeze_att_num]}" for freeze_att_num in kwargs["freeze_att_nums"]
                ]
                if "freeze_att_nums" in kwargs
                else None,
                "freeze_bn": True,
                "box": 0,
                "seg": 0,
                "cls": 0,
                "dfl": 0,
                "mdet": 10,
                "close_mosaic": 30,
            }
        )

    train_params.update(kwargs)
    if "name" not in train_params or train_params["name"] is None:
        train_params["name"] = f"{train_params['data'].replace('.yaml', '')}-[{cfg_path.replace('.yaml', '')}]"
    result = model.train(**train_params)
    return result


def model_train_resume(
    last_path,
    network=YOLO,
    device=None,
    batch=None,
    workers=None,
    **kwargs,
):
    """Resume an interrupted Ultralytics training run."""

    last_path = Path(last_path)
    if not last_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {last_path}")

    model = network(str(last_path), task=TASK)
    model.add_callback("on_train_start", tee_log_to_run_dir)

    resume_args = {"resume": True}

    if device is not None:
        resume_args["device"] = device
    if batch is not None:
        resume_args["batch"] = batch
    if workers is not None:
        resume_args["workers"] = workers

    resume_args.update(kwargs)

    return model.train(**resume_args)


def model_val(weight_path, weight_name=True, network=YOLO, save_txt=False, **kwargs):
    if isinstance(weight_path, list):
        for w_path in weight_path:
            model_val(w_path, weight_name=weight_name, network=network, save_txt=save_txt, **kwargs)
        return
    if weight_name:
        weight_path = os.path.join("runs", TASK, weight_path, "weights", "best.pt")

    if os.path.exists(weight_path):
        print(f"[VAL] val with {weight_path}")
    else:
        print(f"[ERROR] {weight_path} not exists")
        return
    model = network(weight_path, task=TASK)

    val_params = {
        "device": DEVICE,
        "batch": BATCH_SIZE,
        "conf": CONF_VAL,
        "save_txt": save_txt,
    }
    val_params.update(kwargs)
    result = model.val(**val_params)

    args_path = os.path.join(os.path.dirname(os.path.dirname(weight_path)), "args.yaml")
    print("project information:")
    with open(args_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        print(data)
    print("============FINISH=============")
    return result


def model_val_summary(
    weight_path,
    weight_name=True,
    network=YOLO,
    save_txt=False,
    save_csv=True,
    csv_path=None,
    save_console_log=True,
    **kwargs,
):
    if isinstance(weight_path, list):
        for w_path in weight_path:
            model_val_summary(
                w_path,
                weight_name=weight_name,
                network=network,
                save_txt=save_txt,
                save_csv=save_csv,
                csv_path=csv_path,
                save_console_log=save_console_log,
                **kwargs,
            )
        return
    if weight_name:
        weight_path = os.path.join("runs", TASK, weight_path, "weights", "best.pt")

    weight_path = os.path.normpath(weight_path)
    run_dir = os.path.dirname(os.path.dirname(weight_path))  # runs/TASK/exp_name
    exp_name = os.path.basename(run_dir)

    print(f"val with {weight_path}")

    model = network(weight_path, task=TASK)

    val_params = {
        "device": DEVICE,
        "batch": BATCH_SIZE,
        "conf": CONF_VAL,
        "save_txt": save_txt,
        "imgsz": IMGSZ,
    }
    val_params.update(kwargs)

    # Optional: write val console output to run_dir/val_console.log
    stdout_orig = sys.stdout
    stderr_orig = sys.stderr
    log_fp = None

    if save_console_log:
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "val_console.log")
        log_fp = open(log_path, "a", buffering=1, encoding="utf-8")
        sys.stdout = Tee(stdout_orig, log_fp)
        sys.stderr = Tee(stderr_orig, log_fp)

    try:
        result = model.val(**val_params)

        args_path = os.path.join(run_dir, "args.yaml")
        train_args = read_yaml_safely(args_path)

        print("project information:")
        print(train_args)

        row = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "exp_name": exp_name,
            "weight_path": weight_path,
            "data": val_params.get("data", ""),
            "imgsz": val_params.get("imgsz", ""),
            "batch": val_params.get("batch", ""),
            "device": str(val_params.get("device", "")),
            "conf": val_params.get("conf", ""),
            "iou": val_params.get("iou", ""),
            "split": val_params.get("split", "val"),
            "save_txt": val_params.get("save_txt", ""),
        }

        # Add useful train args from args.yaml
        for k in [
            "model",
            "epochs",
            "patience",
            "optimizer",
            "lr0",
            "lrf",
            "box",
            "cls",
            "dfl",
            "seg",
            "mdet",
            "freeze",
            "name",
        ]:
            if k in train_args:
                row[f"train/{k}"] = train_args[k]

        # Add Ultralytics validation metrics
        row.update(collect_val_metrics(result))

        if save_csv:
            if csv_path is None:
                csv_path = os.path.join(run_dir, "val_results.csv")

            append_dict_to_csv(row, csv_path)
            append_dict_to_csv(row, os.path.join("runs", TASK, "all_val_results.csv"))

            print(f"[CSV] saved val metrics to: {csv_path}")
            print(f"[CSV] appended global metrics to: {os.path.join('runs', TASK, 'all_val_results.csv')}")

        print("============FINISH=============")
        return result

    finally:
        if save_console_log:
            sys.stdout = stdout_orig
            sys.stderr = stderr_orig
            if log_fp is not None:
                log_fp.close()


def model_predict(
    weight_path, img_dir, weight_name=True, network=YOLO, save=True, save_txt=True, stream=True, **kwargs
):
    if weight_name:
        weight_path = os.path.join("runs", TASK, weight_path, "weights", "best.pt")
    model = network(weight_path, task=TASK)
    predict_params = {
        "device": DEVICE,
        "batch": BATCH_SIZE,
        "conf": CONF_PREDICT,
        "save": save,
        "save_txt": save_txt,
        "stream": stream,
    }
    predict_params.update(kwargs)

    result = model.predict(
        img_dir,
        **predict_params,
    )
    for _ in result:
        pass


def model_track(
    weight_path, img_dir, weight_name=True, network=YOLO, single=False, save=True, save_txt=True, stream=True, **kwargs
):
    if weight_name:
        weight_path = os.path.join("runs", TASK, weight_path, "weights", "best.pt")
    model = network(weight_path, task=TASK)
    predict_params = {
        "device": DEVICE,
        "batch": BATCH_SIZE,
        "conf": CONF_PREDICT,
        "save": save,
        "save_txt": save_txt,
        "stream": stream,
        "tracker": "botsort.yaml",
        "persist": True,
    }
    predict_params.update(kwargs)
    if single:
        image_list = os.listdir(img_dir)
        for image_name in image_list:
            image_path = os.path.join(img_dir, image_name)
            result = model.track(
                image_path,
                **predict_params,
            )
    else:
        result = model.track(
            img_dir,
            **predict_params,
        )
        for _ in result:
            pass


def model_export(weight_path, format="onnx", weight_name=True, network=YOLO, **kwargs):
    if weight_name:
        weight_path = os.path.join("runs", TASK, weight_path, "weights", "best.pt")
    model = network(weight_path, task=TASK)
    model.export(format=format, device=DEVICE, **kwargs)


# endregion


# region other tools


def get_freeze_num(cfg_path):
    for k, v in FREEZE_NUMS.items():
        if k in cfg_path:
            return v
    print(f"freeze num error for cfg_path {cfg_path}")
    return None


# endregion


# region run tools


def yolo8(cfg_path, weight_path="yolov8x.pt", auto_optim=True, retrain=False, **kwargs):
    assert "yolov8" in cfg_path or "yolo8" in cfg_path, ValueError(cfg_path, "is not yolov8 config!")
    model_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)


def yolo9(cfg_path, weight_path="yolov9e.pt", auto_optim=True, retrain=False, **kwargs):
    assert "yolov9" in cfg_path or "yolo9" in cfg_path, ValueError(cfg_path, "is not yolov9 config!")
    model_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)


def yolo10(cfg_path, weight_path="yolov10x.pt", auto_optim=True, retrain=False, **kwargs):
    assert "yolov10" in cfg_path or "yolo10" in cfg_path, ValueError(cfg_path, "is not yolov10 config!")
    model_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)


def yolo11(cfg_path, weight_path="yolo11x.pt", auto_optim=True, retrain=False, **kwargs):
    assert "yolov11" in cfg_path or "yolo11" in cfg_path, ValueError(cfg_path, "is not yolov11 config!")
    model_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)


def yolo12(cfg_path, weight_path="yolo12x.pt", auto_optim=True, retrain=False, **kwargs):
    assert "yolov12" in cfg_path or "yolo12" in cfg_path, ValueError(cfg_path, "is not yolov12 config!")
    model_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)


def yolo26(cfg_path, weight_path="yolo26x.pt", auto_optim=True, retrain=False, **kwargs):
    assert "yolov26" in cfg_path or "yolo26" in cfg_path, ValueError(cfg_path, "is not yolov26 config!")
    model_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)


# endregion

if __name__ == "__main__":
    pass
    # yolo8x('yolov8x-mseg.yaml', auto_optim=False, name=f'debug', retrain=True,task='msegment',
    #        weight_path=r'runs/segment/billboard_seg_3895/weights/best.pt')
