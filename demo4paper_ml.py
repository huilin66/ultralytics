import os
import cv2
import glob
import torch
import numpy as np
from tqdm import tqdm
import cv2  # 【新增】需要用opencv读取图片获取真实宽高
from ultralytics.utils.metrics import box_iou, ap_per_class # <--- 引入这个!
from ultralytics import YOLO

# ================= 配置区域 =================
# 定义类别映射 (根据你的模型输出)
# 0: background
# 1: wall_signboard
# 2: projecting_signboard
# 3-12: attributes
OBJ_IDS = [1, 2]      # 物体类别ID
ATTR_START_ID = 3     # 属性起始ID
ATTR_COUNT = 10       # 属性数量


BATCH_SIZE = 32
EPOCHS = 500
IMGSZ = 640
CONF = 0.001
TASK = 'mdetect'
DEVICE = torch.device('cuda:0')
DATA = "billboard_mdet5_10_c_0806m_ml.yaml"
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
    return model.val(data=DATA, device=DEVICE, **kwargs)

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
    model.val(data=DATA, device=DEVICE, conf=CONF, rect=False)

def model_predict(weight_path, img_dir, network=YOLO, name=None):
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
    kwargs['mloss_mask'] = True
    myolo_train(cfg_path, pretrain_path=weight_path, auto_optim=auto_optim, retrain=retrain, **kwargs)

# endregion

# ==========================================
# 核心修改部分：解析与评估
# ==========================================
def parse_12cls_txt(txt_path, img_shape):
    """ 解析 GT txt，并根据图片尺寸将归一化坐标转换为绝对坐标 """
    if not os.path.exists(txt_path):
        return torch.empty((0, 4)), torch.empty(0), torch.empty((0, 10))
    
    try:
        data = np.loadtxt(txt_path).reshape(-1, 5)
    except Exception:
        return torch.empty((0, 4)), torch.empty(0), torch.empty((0, 10))

    if data.shape[0] == 0:
        return torch.empty((0, 4)), torch.empty(0), torch.empty((0, 10))

    data = torch.tensor(data, dtype=torch.float32)
    h, w = img_shape

    # 筛选物体和属性
    is_obj = torch.isin(data[:, 0], torch.tensor(OBJ_IDS))
    is_attr = (data[:, 0] >= ATTR_START_ID) & (data[:, 0] < (ATTR_START_ID + ATTR_COUNT))
    
    objs = data[is_obj]
    attrs = data[is_attr]
    
    if len(objs) == 0:
        return torch.empty((0, 4)), torch.empty(0), torch.empty((0, 10))

    # 坐标反归一化
    def denormalize_xywh2xyxy(x, w, h):
        y = x.clone()
        y[:, 0] = (x[:, 0] - x[:, 2] / 2) * w
        y[:, 1] = (x[:, 1] - x[:, 3] / 2) * h
        y[:, 2] = (x[:, 0] + x[:, 2] / 2) * w
        y[:, 3] = (x[:, 1] + x[:, 3] / 2) * h
        return y
    
    obj_boxes = denormalize_xywh2xyxy(objs[:, 1:], w, h)
    attr_boxes = denormalize_xywh2xyxy(attrs[:, 1:], w, h)
    
    # 映射物体类别: 1->0, 2->1
    gt_classes = objs[:, 0].long() - 1 
    
    gt_attributes = torch.zeros((len(objs), ATTR_COUNT))
    
    if len(attrs) > 0:
        ious = box_iou(obj_boxes, attr_boxes)
        for i in range(len(objs)):
            match_idx = torch.where(ious[i] > 0.8)[0]
            if len(match_idx) > 0:
                attr_cls = attrs[match_idx, 0].long()
                attr_indices = attr_cls - ATTR_START_ID
                valid_mask = (attr_indices >= 0) & (attr_indices < ATTR_COUNT)
                valid_indices = attr_indices[valid_mask]
                if len(valid_indices) > 0:
                    gt_attributes[i, valid_indices] = 1.0
                
    return obj_boxes, gt_classes, gt_attributes


def bind_predictions(pred_result, device, bind_thres=0.5):
    """
    重组预测结果
    【修改点】: 新增返回 pred_scores (置信度)，并增加索引越界保护
    """
    preds = pred_result.boxes.data # [x1, y1, x2, y2, conf, cls]
    
    # 筛选
    obj_mask = torch.isin(preds[:, 5], torch.tensor(OBJ_IDS, device=device))
    attr_mask = (preds[:, 5] >= ATTR_START_ID) & (preds[:, 5] < (ATTR_START_ID + ATTR_COUNT))
    
    obj_preds = preds[obj_mask]
    attr_preds = preds[attr_mask]
    
    if len(obj_preds) == 0:
        return None, None, None, None
    
    pred_boxes = obj_preds[:, :4]
    pred_scores = obj_preds[:, 4]             # <--- 【新增】获取物体置信度
    pred_classes = obj_preds[:, 5].long() - 1 # 1->0, 2->1
    
    pred_attributes = torch.zeros((len(obj_preds), ATTR_COUNT), device=device)
    
    if len(attr_preds) > 0:
        ious = box_iou(pred_boxes, attr_preds[:, :4])
        for i in range(len(obj_preds)):
            match_idx = torch.where(ious[i] > bind_thres)[0]
            if len(match_idx) > 0:
                matched_attrs = attr_preds[match_idx]
                attr_cls = matched_attrs[:, 5].long()
                attr_conf = matched_attrs[:, 4]
                
                attr_indices = attr_cls - ATTR_START_ID
                
                # <--- 【新增】越界保护 (修复 IndexError)
                valid_mask = (attr_indices >= 0) & (attr_indices < ATTR_COUNT)
                valid_indices = attr_indices[valid_mask]
                valid_confs = attr_conf[valid_mask]
                
                for idx, conf in zip(valid_indices, valid_confs):
                    if pred_attributes[i, idx] < conf:
                        pred_attributes[i, idx] = conf

    return pred_boxes, pred_classes, pred_attributes, pred_scores # <--- 返回 4 个值


def match_predictions_custom(pred_boxes, pred_classes, gt_boxes, gt_classes, iou_thres=0.5):
    """
    手动实现匹配逻辑：判断每个预测框是否命中 GT
    """
    correct = torch.zeros(len(pred_boxes), 1, dtype=torch.bool, device=pred_boxes.device)
    if len(gt_boxes) == 0:
        return correct
        
    iou = box_iou(pred_boxes, gt_boxes)
    correct_iou = iou >= iou_thres
    correct_class = pred_classes[:, None] == gt_classes[None, :]
    x = torch.where(correct_iou & correct_class)
    
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if matches.shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 0].astype(int)] = True
        
    return correct

def match_predictions(pred_classes, true_classes, iou_matrix, iou_thres=0.5):
    """
    手动实现的匹配逻辑，替代无法 import 的官方函数。
    逻辑：在 IoU > 0.5 且类别匹配的候选里，优先选取 IoU 最高的配对。
    
    Args:
        pred_classes: (N_pred, )
        true_classes: (M_gt, )
        iou_matrix: (M_gt, N_pred)  <-- 注意: box_iou(gt, pred) 的输出
    Returns:
        correct: (N_pred, 1) bool tensor
    """
    # box_iou 返回的是 (GT, Pred)，我们转置一下变 (Pred, GT) 方便处理
    iou = iou_matrix.T 
    
    # 初始化 correct 矩阵 (N_pred, 1)
    correct = torch.zeros(pred_classes.shape[0], 1, dtype=torch.bool, device=iou.device)
    
    # 1. 筛选所有候选配对: IoU > 阈值 AND 类别相同
    # x[0] 是 Pred 索引, x[1] 是 GT 索引
    x = torch.where((iou >= iou_thres) & (true_classes[None, :] == pred_classes[:, None]))
    
    if x[0].shape[0]:
        # 拼接成 [Pred_idx, GT_idx, IoU_value] 的列表
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        
        if matches.shape[0] > 1:
            # 2. 排序：按 IoU 降序排列 (这是 mAP 计算的关键，优先把 GT 分给 IoU 更高的框)
            matches = matches[matches[:, 2].argsort()[::-1]]
            
            # 3. 去重 Pred：每个预测框只能匹配 1 个 GT (取 IoU 最高的那个)
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            
            # 4. 去重 GT：每个 GT 只能被 1 个预测框匹配 (已被抢走的 GT 不能再用)
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            
        # 5. 记录匹配成功的 Pred
        correct[matches[:, 0].astype(int)] = True
        
    return correct
def ml2ma_eval(model_path, image_dir, label_dir, conf_thres, iou_thres, bind_thres):
    """
    计算 mAP50 以及 详细的 Per-Attribute OA (Accuracy)
    """
    model = YOLO(model_path)
    
    img_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    if len(img_files) == 0:
        img_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        
    print(f"[INFO] Calculating Stats (Sorted by Conf, Official Matcher)...")
    
    MAP_CONF = 0.001 
    
    stats = [] 
    total_detected_objects = 0
    
    # === 新增：用于统计每个属性的 TP, TN, FP, FN ===
    # 假设有 10 个属性
    attr_stats = {
        'TP': np.zeros(ATTR_COUNT),
        'TN': np.zeros(ATTR_COUNT),
        'FP': np.zeros(ATTR_COUNT),
        'FN': np.zeros(ATTR_COUNT)
    }
    
    for img_path in tqdm(img_files):
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        
        # 1. 推理
        results = model.predict(img_path, conf=MAP_CONF, iou=iou_thres, imgsz=IMGSZ, verbose=False)
        
        # 2. 重组
        p_boxes, p_classes, p_attrs, p_scores = bind_predictions(results[0], model.device, bind_thres)
        
        # 3. GT 解析
        label_name = os.path.basename(img_path).rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(label_dir, label_name)
        gt_boxes, gt_classes, gt_attrs = parse_12cls_txt(label_path, (h, w))
        gt_boxes = gt_boxes.to(model.device)
        gt_classes = gt_classes.to(model.device)
        
        # --- 4. 收集检测统计量 (mAP) ---
        if p_boxes is None:
            if len(gt_boxes) > 0:
                stats.append((torch.zeros(0, 1, dtype=torch.bool), torch.zeros(0), torch.zeros(0), gt_classes.cpu()))
            continue
        
        if len(gt_boxes) == 0:
            correct = torch.zeros(len(p_boxes), 1, dtype=torch.bool)
            stats.append((correct.cpu(), p_scores.cpu(), p_classes.cpu(), torch.zeros(0)))
            continue

        sort_idx = torch.argsort(p_scores, descending=True)
        p_boxes_sorted = p_boxes[sort_idx]
        p_classes_sorted = p_classes[sort_idx]
        p_scores_sorted = p_scores[sort_idx]
        p_attrs_sorted = p_attrs[sort_idx]

        iou = box_iou(gt_boxes, p_boxes_sorted)
        correct = match_predictions(p_classes_sorted, gt_classes, iou)
        stats.append((correct.cpu(), p_scores_sorted.cpu(), p_classes_sorted.cpu(), gt_classes.cpu()))

        # --- 5. 计算属性 OA (统计每个属性的 TP/TN/FP/FN) ---
        high_conf_mask = p_scores_sorted >= conf_thres
        
        if high_conf_mask.sum() > 0:
            p_boxes_high = p_boxes_sorted[high_conf_mask]
            p_classes_high = p_classes_sorted[high_conf_mask]
            p_attrs_high = p_attrs_sorted[high_conf_mask]
            
            iou_matrix = box_iou(gt_boxes, p_boxes_high)
            correct_obj_mask = (gt_classes[:, None] == p_classes_high[None, :]) & (iou_matrix > 0.5)
            
            gt_attrs = gt_attrs.to(model.device)
            for i in range(len(gt_boxes)):
                matched_indices = torch.where(correct_obj_mask[i])[0]
                if len(matched_indices) > 0:
                    best_idx = matched_indices[torch.argmax(iou_matrix[i, matched_indices])]
                    
                    pred_vec = p_attrs_high[best_idx] # [10,]
                    gt_vec = gt_attrs[i]              # [10,]
                    
                    # 二值化
                    pred_bin = (pred_vec > 0.5).float()
                    
                    # === 统计每个属性的详情 ===
                    for k in range(ATTR_COUNT):
                        p = pred_bin[k].item()
                        g = gt_vec[k].item()
                        
                        if p == 1 and g == 1:
                            attr_stats['TP'][k] += 1
                        elif p == 0 and g == 0:
                            attr_stats['TN'][k] += 1
                        elif p == 1 and g == 0:
                            attr_stats['FP'][k] += 1
                        elif p == 0 and g == 1:
                            attr_stats['FN'][k] += 1
                            
                    total_detected_objects += 1

    # 6. 最终计算
    print("\n[INFO] Loop Finished. Computing Metrics...")
    if len(stats) > 0:
        tp, conf, pred_cls, target_cls = [np.concatenate(x, 0) for x in zip(*stats)]
        if tp.ndim == 1: tp = tp[:, None]
        
        results_ap = ap_per_class(tp, conf, pred_cls, target_cls, plot=False, names={0:'Wall', 1:'Proj'})
        ap50_values = results_ap[5][:, 0]
        unique_classes = results_ap[6]
        
        print("\n" + "="*40)
        print("   MANUAL CALCULATION (Official Logic)  ")
        print("="*40)
        
        map_dict = {int(c): ap for c, ap in zip(unique_classes, ap50_values)}
        if 0 in map_dict: print(f"Wall Signboard (Class 1) mAP50      : {map_dict[0]:.4f}")
        if 1 in map_dict: print(f"Projecting Signboard (Class 2) mAP50: {map_dict[1]:.4f}")
        print(f"Mean mAP50                          : {np.mean(ap50_values):.4f}")
        print("="*40)

    # === 输出详细的 Per-Attribute OA ===
    if total_detected_objects > 0:
        print(f"\n================ Attribute OA Breakdown ================")
        print(f"{'Attr_ID':<10} {'Accuracy (OA)':<15} {'TP':<6} {'TN':<6} {'FP':<6} {'FN':<6}")
        print("-" * 60)
        
        oa_per_attr = []
        
        for k in range(ATTR_COUNT):
            tp = attr_stats['TP'][k]
            tn = attr_stats['TN'][k]
            fp = attr_stats['FP'][k]
            fn = attr_stats['FN'][k]
            
            total = tp + tn + fp + fn
            if total > 0:
                acc = (tp + tn) / total
            else:
                acc = 0.0
                
            oa_per_attr.append(acc)
            print(f"{k:<10} {acc:.4f}          {int(tp):<6} {int(tn):<6} {int(fp):<6} {int(fn):<6}")
            
        mean_oa = np.mean(oa_per_attr)
        print("-" * 60)
        print(f"Total Matched Objects   : {total_detected_objects}")
        print(f"Average OA_mAP50        : {mean_oa:.4f}")
        print(f"========================================================\n")
    else:
        print("No objects matched for OA eval.")


if __name__ == '__main__':
    model_val('runs/detect/exp_ml/weights/best.pt')
    # model_val('runs/detect/exp_ml2/weights/best.pt')
    # model_val('runs/detect/exp_ml3/weights/best.pt')
    # model_val('runs/detect/exp_ml4/weights/best.pt')
    # model_val('runs/detect/exp_ml5/weights/best.pt')
    # model_val('runs/detect/exp_ml6/weights/best.pt')
    # model_val('runs/detect/exp_ml7/weights/best.pt')
    # model_val('runs/detect/exp_ml8/weights/best.pt')
    # model_val('runs/detect/exp_ml9/weights/best.pt')
    # model_val('runs/detect/exp_ml10/weights/best.pt')
    # model_val('runs/detect/exp_ml11/weights/best.pt')
    # model_val('runs/detect/exp_ml12/weights/best.pt')
    # model_val('runs/detect/exp_ml13/weights/best.pt')
    # model_val('runs/detect/exp_ml14/weights/best.pt')
    # 替换为你实际的路径
    # 注意：确保 conf_thres 不要太高，以免漏检
    # ml2ma_eval(
    #     model_path='runs/detect/exp_ml/weights/best.pt',
    #     image_dir=r'/nfsv4/23039356r/data/billboard/data0806_m/yolo_rgb_detection5_10_c_ml/val/images',
    #     label_dir=r'/nfsv4/23039356r/data/billboard/data0806_m/yolo_rgb_detection5_10_c_ml/val/labels',
    #     conf_thres=0.001, 
    #     iou_thres=0.7, 
    #     bind_thres=0.9
    # )