import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import torch

from ultralytics import YOLO

BATCH_SIZE = 32
EPOCHS = 500
IMGSZ = 640
CONF = 0.5
TASK = "mdetect"
DEVICE = torch.device("cuda:1")
DATA = "billboard_mdet5_10_c_0806m.yaml"
FREEZE_NUMS = {
    "yolov8": 22,
    "yolov9e": 42,
    "yolov9": 22,
    "yolov10": 23,
    "mayolo": 23,
}
# MLOSS_ENLARGE = 0.3
# region meta tools


def myolo_train_full(
    cfg_path,
    pretrain_path,
    network=YOLO,
    auto_optim=False,
    stage1_epochs=100,
    stage2_epochs=100,
    project=r"runs\mdetect",
    stage1_name="stage1",
    stage2_name="stage2_attribute",
    **kwargs,
):
    model_path_s1 = myolo_train(
        cfg_path,
        pretrain_path,
        network=network,
        auto_optim=auto_optim,
        retrain=False,
        epochs=stage1_epochs,
        project=project,
        name=stage1_name,
        **kwargs,
    )
    model_path_s1 = myolo_train(
        cfg_path,
        model_path_s1,
        network=network,
        auto_optim=auto_optim,
        retrain=True,
        epochs=stage2_epochs,
        project=project,
        name=stage2_name,
        patience=stage2_epochs,
        **kwargs,
    )
    return model_path_s1


def myolo_train(cfg_path, pretrain_path, network=YOLO, auto_optim=False, retrain=False, **kwargs):
    model = network(cfg_path, task=TASK)
    model.load(pretrain_path)

    train_params = {
        "data": DATA,
        "device": DEVICE,
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "val": True,
        "batch": BATCH_SIZE,
        "patience": EPOCHS,
    }

    if not auto_optim:
        train_params.update({"optimizer": "AdamW", "lr0": 0.0001})

    if retrain:
        train_params.update(
            {
                "freeze": get_freeze_num(cfg_path),
                "freeze_head": [".cv2", ".cv3"]
                if "yolov10" not in cfg_path and "mayolo" not in cfg_path
                else [".cv2", ".cv3", ".one2one_cv2", ".one2one_cv3"],
                "freeze_bn": True,
            }
        )
    train_params.update(kwargs)
    model.train(**train_params)
    return model.trainer.best


def model_val(weight_path, network=YOLO, **kwargs):
    model = network(weight_path, task=TASK)
    print(weight_path)
    print(model.info(detailed=False))
    val_params = {
        "data": DATA,
        "device": DEVICE,
    }
    val_params.update(kwargs)
    return model.val(**val_params)


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


def model_predict(weight_path, img_dir, network=YOLO, name=None, visualize=False):
    model = network(weight_path, task=TASK)
    model.predict(
        img_dir,
        save=True,
        conf=CONF,
        device=DEVICE,
        imgsz=IMGSZ,
        save_txt=True,
        save_conf=True,
        name=name,
        visualize=visualize,
    )


def model_export(weight_path, format="onnx", network=YOLO):
    model = network(weight_path, task=TASK)
    model.export(format=format)


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


def myolo8(cfg_path, weight_path="yolov8x.pt", auto_optim=False, **kwargs):
    assert "yolov8" in cfg_path, ValueError(cfg_path, "is not yolov8 config!")
    myolo_train_full(
        cfg_path,
        pretrain_path=weight_path,
        auto_optim=auto_optim,
        stage1_name="myolo8_stage1",
        stage2_name="myolo8_stage2",
        **kwargs,
    )


def myolo9(cfg_path, weight_path="yolov9e.pt", auto_optim=False, **kwargs):
    assert "yolov9" in cfg_path, ValueError(cfg_path, "is not yolov9 config!")
    myolo_train_full(
        cfg_path,
        pretrain_path=weight_path,
        auto_optim=auto_optim,
        stage1_name="myolo9_stage1",
        stage2_name="myolo9_stage2",
        **kwargs,
    )


def myolo10(cfg_path, weight_path="yolov10x.pt", auto_optim=False, **kwargs):
    assert "yolov10" in cfg_path, ValueError(cfg_path, "is not yolov10 config!")
    myolo_train_full(
        cfg_path,
        pretrain_path=weight_path,
        auto_optim=auto_optim,
        stage1_name="myolo10_stage1",
        stage2_name="myolo10_stage2",
        **kwargs,
    )


def mayolo(cfg_path, weight_path="yolov10x.pt", auto_optim=False, **kwargs):
    myolo_train_full(
        cfg_path,
        pretrain_path=weight_path,
        auto_optim=auto_optim,
        stage1_name="mayolo_stage1",
        stage2_name="mayolo_stage2",
        **kwargs,
    )


# endregion

if __name__ == "__main__":
    # test
    mayolo(r"mayolovx.yaml")
