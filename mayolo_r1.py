import csv
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
import torch

from ultralytics import RTDETR, YOLO

BATCH_SIZE = 16
STAGE1_EPOCHS = 100
STAGE2_EPOCHS = 100
# Default epoch budget for direct single-stage calls.
EPOCHS = STAGE1_EPOCHS
IMGSZ = 640
CONF = 0.5
TASK = "mdetect"
DEVICE = torch.device("cuda:0")
DATA = "billboard_mdet5_10_c_0806m.yaml"
FREEZE_NUMS = {
    "yolov8": 22,
    "yolov9e": 42,
    "yolov9": 22,
    "yolov10": 23,
    "yolov11": 23,
    "yolov12": 21,
    "yolov13": 32,
    "yolov26": 23,
    "yolo11": 23,
    "yolo12": 21,
    "yolo13": 32,
    "yolo26": 23,
    "mayolo": 23,
}
# MLOSS_ENLARGE = 0.3
# region meta tools


def _is_rtdetr(network):
    """Return whether the requested network is the RT-DETR wrapper."""
    return network is RTDETR


def _build_model(network, model_path):
    """Build a model while keeping the YOLO and RT-DETR constructor signatures separate."""
    return network(model_path) if _is_rtdetr(network) else network(model_path, task=TASK)


def _rtdetr_attribute_only_params(model):
    """Freeze RT-DETR except for its encoder and decoder attribute heads in stage two."""
    detector = model.model
    layers = detector.model
    return {
        # Freeze backbone, neck and all layers before the RT-DETR decoder head.
        "freeze": list(range(max(len(layers) - 1, 0))),
        # Freeze the complete decoder except enc_attribute_head/dec_attribute_head.
        "freeze_head": [
            ".input_proj",
            ".decoder",
            ".denoising_class_embed",
            ".tgt_embed",
            ".query_pos_head",
            ".enc_output",
            ".enc_score_head",
            ".enc_bbox_head",
            ".dec_score_head",
            ".dec_bbox_head",
        ],
        "freeze_bn": True,
    }


def myolo_train_full(
    cfg_path,
    pretrain_path,
    network=YOLO,
    auto_optim=False,
    stage1_epochs=STAGE1_EPOCHS,
    stage2_epochs=STAGE2_EPOCHS,
    stage1_name="stage1",
    stage2_name="stage2",
    **kwargs,
):
    model_path_s1 = myolo_train(
        cfg_path,
        pretrain_path,
        network=network,
        auto_optim=auto_optim,
        retrain=False,
        epochs=stage1_epochs,
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
        name=stage2_name,
        patience=stage2_epochs,
        **kwargs,
    )
    return model_path_s1


def myolo_train(cfg_path, pretrain_path, network=YOLO, auto_optim=False, retrain=False, **kwargs):
    model = _build_model(network, cfg_path)
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
        if _is_rtdetr(network):
            train_params.update(_rtdetr_attribute_only_params(model))
        else:
            train_params.update(
                {
                    "freeze": get_freeze_num(cfg_path),
                    "freeze_head": [".cv2", ".cv3"]
                    if all(name not in str(cfg_path).lower() for name in ("yolov10", "mayolo", "yolov26", "yolo26"))
                    else [".cv2", ".cv3", ".one2one_cv2", ".one2one_cv3"],
                    "freeze_bn": True,
                }
            )
    train_params.update(kwargs)
    model.train(**train_params)
    return model.trainer.best


def model_val(weight_path, network=YOLO, run_test=False, **kwargs):
    model = _build_model(network, weight_path)
    print(weight_path)
    print(model.info(detailed=False))
    val_params = {
        # "data": DATA,
        "device": DEVICE,
    }
    val_params.update(kwargs)
    if run_test:
        model.val(split="test", **val_params)
    return model.val(**val_params)


def _find_weight_paths(folder, names=("best.pt",)):
    """Return sorted weight files matching ``names`` under any ``weights`` dir in ``folder``."""
    weights = []
    for root, _, files in os.walk(folder):
        if os.path.basename(root) != "weights":
            continue
        weights.extend(os.path.join(root, name) for name in names if name in files)
    return sorted(weights)


MDETECT_METRIC_KEYS = [
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/OA(A)",
    "metrics/f1_macro(A)",
    "metrics/f1_macro_global(A)",
    "metrics/P_macro(A)",
    "metrics/R_macro(A)",
]
MDETECT_METRIC_LABELS = {
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
    "metrics/OA(A)": "OA",
    "metrics/f1_macro(A)": "F1_macro",
    "metrics/f1_macro_global(A)": "F1_macro_global",
    "metrics/P_macro(A)": "P_macro",
    "metrics/R_macro(A)": "R_macro",
}


def _metric_row(metrics):
    """Return the selected metric values of a ``metrics`` object aligned with MDETECT_METRIC_KEYS."""
    results = metrics.results_dict
    return [results.get(k, "") for k in MDETECT_METRIC_KEYS]


def _save_stats_csv(records, splits, save_txt):
    """Write one row per weight, with val and test metrics side by side, to a CSV file."""
    header = ["weight"] + [f"{MDETECT_METRIC_LABELS[k]}_{split}" for split in splits for k in MDETECT_METRIC_KEYS]
    with open(save_txt, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(records)
    print(f"\n=== saved {len(records)} records to {save_txt} ===")


def model_val_dir(
    folder,
    network=YOLO,
    names=("best.pt",),
    run_test=True,
    save_txt=None,
    **kwargs,
):
    """
    Validate every training result (weight file) found under ``folder``.

    For each weight, run the val split (and the test split when ``run_test``) and collect
    the metrics into a CSV file with val and test columns side by side on one row.
    """
    if save_txt is None:
        save_txt = os.path.join(folder, "summary.csv")
    weight_list = _find_weight_paths(folder, names)
    print(f"=== validating {len(weight_list)} weight files under {folder} ===")
    records = []
    splits = ["val"] + (["test"] if run_test else [])
    for weight_path in weight_list:
        print(f"\n=== validating {weight_path} ===")
        model = _build_model(network, weight_path)
        print(model.info(detailed=False))
        val_params = {"device": DEVICE}
        val_params.update(kwargs)
        split_metrics = {}
        for split in splits:
            metrics = model.val(split=split, **val_params)
            split_metrics[split] = _metric_row(metrics)
        row = [weight_path]
        for split in splits:
            row.extend(split_metrics[split])
        records.append(row)
    _save_stats_csv(records, splits, save_txt)


MDET_ABLATION_DIRS = (
    ("E2.1_GIA_position", r"runs/experiments/E2_1_GIA_position"),
    ("E2.2_GCA", r"runs/experiments/E2_2_GCA"),
    ("E2.3_GIA_GCA", r"runs/experiments/E2_3_GIA_GCA"),
    ("E2.4_HO", r"runs/experiments/E2_4_HO"),
)


def model_val_ablation_dirs(
    experiment_dirs=MDET_ABLATION_DIRS,
    network=YOLO,
    names=("best.pt",),
    run_test=True,
    **kwargs,
):
    """Validate the configured mdet ablation directories and write summaries.

    Each experiment keeps its own ``summary.csv`` so that the stage-1/stage-2
    checkpoints and the val/test columns remain easy to trace back to the
    corresponding ablation.  Missing experiment directories are skipped,
    which allows this helper to be called before all jobs finish.
    """
    for label, folder in experiment_dirs:
        if not os.path.isdir(folder):
            print(f"=== skip {label}: directory not found: {folder} ===")
            continue
        print(f"\n=== {label}: validation summary ===")
        model_val_dir(
            folder,
            network=network,
            names=names,
            run_test=run_test,
            save_txt=os.path.join(folder, "summary.csv"),
            **kwargs,
        )


def model_gat_val(weight_path, com_path, network=YOLO):
    model = _build_model(network, weight_path)
    model.model.model[-1].added_gat_head(com_path)
    print(weight_path)
    print(model.info(detailed=False))
    model.val(data=DATA, device=DEVICE)


def model_val_single(weight_path, network=YOLO):
    model = _build_model(network, weight_path)
    model.model.model[-1].use_one2many_head()
    print(weight_path)
    print(model.info(detailed=False))
    model.val(data=DATA, device=DEVICE)


def model_predict(weight_path, img_dir, network=YOLO, name=None, visualize=False):
    model = _build_model(network, weight_path)
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
    model = _build_model(network, weight_path)
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
    scale = weight_path[-4]
    myolo_train_full(
        cfg_path,
        pretrain_path=weight_path,
        auto_optim=auto_optim,
        stage1_name=f"myolo8{scale}_stage1",
        stage2_name=f"myolo8{scale}_stage2",
        **kwargs,
    )


def myolo9(cfg_path, weight_path="yolov9e.pt", auto_optim=False, **kwargs):
    assert "yolov9" in cfg_path, ValueError(cfg_path, "is not yolov9 config!")
    scale = weight_path[-4]
    myolo_train_full(
        cfg_path,
        pretrain_path=weight_path,
        auto_optim=auto_optim,
        stage1_name=f"myolo9{scale}_stage1",
        stage2_name=f"myolo9{scale}_stage2",
        **kwargs,
    )


def myolo10(cfg_path, weight_path="yolov10x.pt", auto_optim=False, **kwargs):
    assert "yolov10" in cfg_path, ValueError(cfg_path, "is not yolov10 config!")
    scale = weight_path[-4]
    myolo_train_full(
        cfg_path,
        pretrain_path=weight_path,
        auto_optim=auto_optim,
        stage1_name=f"myolo10{scale}_stage1",
        stage2_name=f"myolo10{scale}_stage2",
        **kwargs,
    )


def myolo11(cfg_path, weight_path="yolo11x.pt", auto_optim=False, **kwargs):
    assert "yolo11" in cfg_path or "yolov11" in cfg_path, ValueError(cfg_path, "is not yolov11 config!")
    scale = weight_path[-4]
    myolo_train_full(
        cfg_path,
        pretrain_path=weight_path,
        auto_optim=auto_optim,
        stage1_name=f"yolo11{scale}_stage1",
        stage2_name=f"yolo11{scale}_stage2",
        **kwargs,
    )


def myolo12(cfg_path, weight_path="yolo12x.pt", auto_optim=False, **kwargs):
    assert "yolo12" in cfg_path or "yolov12" in cfg_path, ValueError(cfg_path, "is not yolov12 config!")
    scale = weight_path[-4]
    myolo_train_full(
        cfg_path,
        pretrain_path=weight_path,
        auto_optim=auto_optim,
        stage1_name=f"yolo12{scale}_stage1",
        stage2_name=f"yolo12{scale}_stage2",
        **kwargs,
    )


def myolo13(cfg_path, weight_path="yolov13x.pt", auto_optim=False, **kwargs):
    assert "yolo13" in cfg_path or "yolov13" in cfg_path, ValueError(cfg_path, "is not yolov13 config!")
    scale = weight_path[-4]
    myolo_train_full(
        cfg_path,
        pretrain_path=weight_path,
        auto_optim=auto_optim,
        stage1_name=f"yolov13{scale}_stage1",
        stage2_name=f"yolov13{scale}_stage2",
        **kwargs,
    )


def myolo26(cfg_path, weight_path="yolo26x.pt", auto_optim=False, **kwargs):
    """Train a YOLO26 multi-attribute model with the standard 100+100 stage protocol."""
    assert "yolo26" in str(cfg_path).lower() or "yolov26" in str(cfg_path).lower(), ValueError(
        cfg_path, "is not yolov26 config!"
    )
    scale = weight_path[-4]
    myolo_train_full(
        cfg_path,
        pretrain_path=weight_path,
        auto_optim=auto_optim,
        stage1_name=f"yolo26{scale}_stage1",
        stage2_name=f"yolo26{scale}_stage2",
        **kwargs,
    )


def rtdetr(
    cfg_path="ultralytics/cfg/models/rt-detr/rtdetr-l-md.yaml",
    weight_path="rtdetr-l.pt",
    auto_optim=False,
    **kwargs,
):
    """Train an RT-DETR object-level multi-attribute model for 100+100 epochs by default."""
    assert "rtdetr" in str(cfg_path).lower(), ValueError(cfg_path, "is not an RT-DETR config!")
    scale = weight_path[-4]
    return myolo_train_full(
        cfg_path,
        pretrain_path=weight_path,
        network=RTDETR,
        auto_optim=auto_optim,
        stage1_name=f"rtdetr{scale}_stage1",
        stage2_name=f"rtdetr{scale}_stage2",
        **kwargs,
    )


def mayolo(cfg_path, weight_path="yolov10x.pt", auto_optim=False, **kwargs):
    scale = weight_path[-4]
    myolo_train_full(
        cfg_path,
        pretrain_path=weight_path,
        auto_optim=auto_optim,
        stage1_name=f"mayolo{scale}_stage1",
        stage2_name=f"mayolo{scale}_stage2",
        **kwargs,
    )


# endregion

if __name__ == "__main__":
    # test
    # rtdetr(
    #     r"ultralytics/cfg/models/rt-detr/rtdetr-l-md.yaml",
    #     weight_path=r"rtdetr-l.pt",
    #     data=DATA,
    #     stage1_epochs=100,
    #     stage2_epochs=100,
    # )
    # myolo10(r"yolov10x-mdetect.yaml", data="mayolo_v1.yaml")
    # mayolo(r"mayolovx.yaml", data="mayolo_v1.yaml")
    # model_val(r"runs/mdetect/mayolox_stage1/weights/best.pt")
    # model_val(r"runs/mdetect/mayolox_stage2/weights/best.pt")
    # model_val(r"runs/mdetect/myolo10x_stage1/weights/best.pt")
    # model_val(r"runs/mdetect/myolo10x_stage2/weights/best.pt")
    # model_val_dir(r"runs/mdetect")  # validates best.pt & last.pt under every exp dir
    # model_val_dir(r"runs/mdetect", save_txt=r"runs/mdetect/mdetect_stats.csv")  # val+test in one row, CSV
    # model_val_dir(r"runs/experiments/E0_stage1_sweep")
    model_val_ablation_dirs()
