import os
import cv2
import glob
import torch
import numpy as np
from tqdm import tqdm
import cv2  # 【新增】需要用opencv读取图片获取真实宽高
from ultralytics.utils.metrics import box_iou, ap_per_class # <--- 引入这个!
from ultralytics import YOLO


BATCH_SIZE = 32
EPOCHS = 300
IMGSZ = 640
CONF = 0.001
TASK = 'detect'
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
import os
import glob
import torch
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

# ================= 核心配置 (根据 demo4paper_ml.py 修正) =================
# 物体类别 ID (背景=0, Wall=1, Projecting=2)
OBJ_CLASSES = [1, 2]  

# 属性起始 ID (属性从3开始)
ATTR_START_ID = 3     

# 属性数量
ATTR_COUNT = 10       

# 属性名称 (请确认顺序是否对应 ID 3-12)
CLASS_NAMES = {
    1: "wall_signboard",
    2: "projecting_signboard",
    3: "disconnected", 4: "frame_corroded", 5: "frame_deformed",
    6: "surface_corroded", 7: "surface_deformed", 8: "surface_fade",
    9: "surface_incomplete", 10: "surface_missing", 11: "surface_peeling", 12: "unauthorized"
}

def parse_gt_info(txt_path, img_shape):
    """ 解析 GT """
    if not os.path.exists(txt_path): return []
    try:
        data = np.loadtxt(txt_path).reshape(-1, 5)
    except:
        return []
    if data.shape[0] == 0: return []

    h, w = img_shape
    # xywh -> xyxy
    boxes = data[:, 1:].copy()
    boxes[:, 0] = (data[:, 1] - data[:, 3] / 2) * w
    boxes[:, 1] = (data[:, 2] - data[:, 4] / 2) * h
    boxes[:, 2] = (data[:, 1] + data[:, 3] / 2) * w
    boxes[:, 3] = (data[:, 2] + data[:, 4] / 2) * h
    
    classes = data[:, 0].astype(int)
    
    # 筛选物体
    obj_indices = [i for i, c in enumerate(classes) if c in OBJ_CLASSES]
    # 筛选属性
    attr_indices = [i for i, c in enumerate(classes) if c >= ATTR_START_ID]

    gt_instances = []
    for idx in obj_indices:
        obj_box = boxes[idx]
        attr_vec = np.zeros(ATTR_COUNT, dtype=int)
        
        # 绑定属性到物体
        if len(attr_indices) > 0:
            attr_boxes = boxes[attr_indices]
            # 计算 IoU
            ious = box_iou(torch.tensor([obj_box]), torch.tensor(attr_boxes)).numpy()[0]
            matched_attrs = np.where(ious > 0.5)[0] # 阈值可调
            
            for ma_idx in matched_attrs:
                real_attr_idx = attr_indices[ma_idx]
                cls_id = classes[real_attr_idx]
                attr_id = cls_id - ATTR_START_ID
                
                if 0 <= attr_id < ATTR_COUNT:
                    attr_vec[attr_id] = 1 

        gt_instances.append({'box': obj_box, 'cls': classes[idx], 'attrs': attr_vec})
        
    return gt_instances

def process_batch(model, img_path, conf_thres=0.25):
    """ 预测并绑定 """
    # verbose=False 关闭 YOLO 默认打印
    results = model.predict(img_path, conf=conf_thres, verbose=False, imgsz=640)
    res = results[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()

    if len(boxes) == 0: return []

    pred_objs = []
    pred_attrs = []
    
    for i, c in enumerate(clss):
        if c in OBJ_CLASSES:
            pred_objs.append({'box': boxes[i], 'cls': c, 'conf': confs[i]})
        elif c >= ATTR_START_ID:
            pred_attrs.append({'box': boxes[i], 'cls': c, 'conf': confs[i]})

    pred_instances = []
    for obj in pred_objs:
        attr_vec = np.zeros(ATTR_COUNT, dtype=int)
        
        if len(pred_attrs) > 0:
            p_attr_boxes = np.array([a['box'] for a in pred_attrs])
            ious = box_iou(torch.tensor([obj['box']]), torch.tensor(p_attr_boxes)).numpy()[0]
            matched_indices = np.where(ious > 0.5)[0]
            for m_idx in matched_indices:
                attr = pred_attrs[m_idx]
                a_id = attr['cls'] - ATTR_START_ID
                if 0 <= a_id < ATTR_COUNT:
                    attr_vec[a_id] = 1 # 预测为有缺陷

        pred_instances.append({
            'box': obj['box'], 
            'cls': obj['cls'], 
            'attrs': attr_vec
        })
    return pred_instances

def evaluate_dataset(model_path, img_dir, label_dir, sample_model=False):
    model = YOLO(model_path)
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    if len(img_files) == 0:
        img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    
    if not sample_model:
        print(f"[INFO] Found {len(img_files)} images.")
        print(f"[INFO] Config: OBJ_CLASSES={OBJ_CLASSES}, ATTR_START={ATTR_START_ID}")
    
    # 混淆矩阵: [Attr_Idx][GT][Pred]
    attr_matrices = np.zeros((ATTR_COUNT, 2, 2), dtype=int)
    obj_stats = {'TP': 0, 'FP': 0, 'FN': 0} 
    
    total_gt_objects = 0

    if not sample_model:
        print("Processing...")
    for i, img_path in enumerate(tqdm(img_files)):
        label_name = os.path.basename(img_path).rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(label_dir, label_name)
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        
        # 1. 获取 GT 和 Pred
        gt_instances = parse_gt_info(label_path, (h, w))
        pred_instances = process_batch(model, img_path, conf_thres=0.001) # 这里的 conf 建议设低一点，交给后面指标分析
        
        total_gt_objects += len(gt_instances)
        
        # [DEBUG] 打印前5张图的信息，检查是否读到了数据
        if not sample_model:
            if i < 5:
                # tqdm.write 可以在进度条中安全打印
                if len(gt_instances) > 0:
                    tqdm.write(f"[DEBUG] Img: {os.path.basename(img_path)} | Found {len(gt_instances)} GT Objects")
                elif os.path.exists(label_path):
                    tqdm.write(f"[DEBUG] Img: {os.path.basename(img_path)} | Label file exists but NO objects parsed! Check IDs.")

        matched_gt_indices = set()
        matched_pred_indices = set()
        
        # 2. 匹配逻辑
        if len(pred_instances) > 0 and len(gt_instances) > 0:
            p_boxes = torch.tensor(np.array([p['box'] for p in pred_instances]))
            g_boxes = torch.tensor(np.array([g['box'] for g in gt_instances]))
            iou_matrix = box_iou(g_boxes, p_boxes).numpy()
            
            candidates = []
            for g_i in range(len(gt_instances)):
                for p_i in range(len(pred_instances)):
                    # IoU > 0.5 且 类别必须一致 (Wall vs Wall)
                    if iou_matrix[g_i, p_i] >= 0.5 and gt_instances[g_i]['cls'] == pred_instances[p_i]['cls']:
                        candidates.append((g_i, p_i, iou_matrix[g_i, p_i]))
            
            candidates.sort(key=lambda x: x[2], reverse=True)
            
            for g_i, p_i, _ in candidates:
                if g_i not in matched_gt_indices and p_i not in matched_pred_indices:
                    matched_gt_indices.add(g_i)
                    matched_pred_indices.add(p_i)
                    obj_stats['TP'] += 1
                    
                    # === 对比属性 ===
                    g_vec = gt_instances[g_i]['attrs']
                    p_vec = pred_instances[p_i]['attrs']
                    
                    for k in range(ATTR_COUNT):
                        gt_val = g_vec[k]
                        pred_val = p_vec[k]
                        attr_matrices[k][gt_val][pred_val] += 1

        # 3. 漏检 (Object FN) -> 属性视为全 0 (TN for 0, FN for 1)
        for g_i in range(len(gt_instances)):
            if g_i not in matched_gt_indices:
                obj_stats['FN'] += 1
                g_vec = gt_instances[g_i]['attrs']
                for k in range(ATTR_COUNT):
                    # GT=g_vec[k], Pred=0
                    attr_matrices[k][g_vec[k]][0] += 1

        # 4. 误检 (Object FP) -> 属性视为全 0 (TN for 0, FP for 1)
        for p_i in range(len(pred_instances)):
            if p_i not in matched_pred_indices:
                obj_stats['FP'] += 1
                p_vec = pred_instances[p_i]['attrs']
                for k in range(ATTR_COUNT):
                    # GT=0, Pred=p_vec[k]
                    attr_matrices[k][0][p_vec[k]] += 1

    if total_gt_objects == 0:
        print("\n[WARNING] Total GT Objects is 0! Please check 'OBJ_CLASSES' and label files.")
        
    return obj_stats, attr_matrices

def print_macro_results(obj_stats, attr_matrices):
    print("\n" + "="*120)
    print(f"{'Class / Attribute':<25} {'Avg_Prec':<10} {'Avg_Recall':<10} {'Avg_F1':<10} | {'P(1)':<8} {'R(1)':<8} {'F1(1)':<8} | {'P(0)':<8} {'R(0)':<8} {'F1(0)':<8}")
    print("-" * 120)
    
    # Object Stats
    otp, ofp, ofn = obj_stats['TP'], obj_stats['FP'], obj_stats['FN']
    op = otp / (otp + ofp + 1e-8)
    ore = otp / (otp + ofn + 1e-8)
    of1 = 2 * op * ore / (op + ore + 1e-8)
    print(f"{'[Objects Global]':<25} {op:.4f}     {ore:.4f}     {of1:.4f}     | -        -        -        | -        -        -")
    print("-" * 120)
    
    all_f1, all_p, all_r = [], [], []

    for k in range(ATTR_COUNT):
        mat = attr_matrices[k]
        
        # Class 1 (Defect)
        tp1 = mat[1][1]
        fp1 = mat[0][1]
        fn1 = mat[1][0]
        
        p1 = tp1 / (tp1 + fp1 + 1e-8)
        r1 = tp1 / (tp1 + fn1 + 1e-8)
        f1_1 = 2 * p1 * r1 / (p1 + r1 + 1e-8)
        
        # Class 0 (Normal)
        # TN(0,0) -> TP for class 0
        # FN(1,0) -> FP for class 0 (GT是1预测0，对Class0来说是误报了"正常") -> 不对，这是漏检缺陷
        # 等等，Macro Avg 对 Class 0 的定义：
        # Class 0 的 TP 是真负 (GT=0, Pred=0)
        # Class 0 的 FP 是假负 (GT=1, Pred=0) -> 预测为0但其实是1
        # Class 0 的 FN 是假正 (GT=0, Pred=1) -> 预测为1但其实是0
        tp0 = mat[0][0] 
        fp0 = mat[1][0] 
        fn0 = mat[0][1]
        
        p0 = tp0 / (tp0 + fp0 + 1e-8)
        r0 = tp0 / (tp0 + fn0 + 1e-8)
        f1_0 = 2 * p0 * r0 / (p0 + r0 + 1e-8)
        
        macro_p = (p1 + p0) / 2
        macro_r = (r1 + r0) / 2
        macro_f1 = (f1_1 + f1_0) / 2
        
        all_f1.append(macro_f1)
        all_p.append(macro_p)
        all_r.append(macro_r)
        
        name = CLASS_NAMES.get(ATTR_START_ID + k, f"Attr_{k}")
        print(f"{name:<25} {macro_p:.4f}     {macro_r:.4f}     {macro_f1:.4f}     | {p1:.4f}   {r1:.4f}   {f1_1:.4f}   | {p0:.4f}   {r0:.4f}   {f1_0:.4f}")
        
    print("-" * 120)
    print(f"{'[MEAN ATTRIBUTE]':<25} {np.mean(all_p):.4f}     {np.mean(all_r):.4f}     {np.mean(all_f1):.4f}")
    print("=" * 120)
def print_final_summary(obj_stats, attr_matrices, result, sample_model=False):
    """
    格式化输出：OA, F1_Macro, F1_Micro, Precision, Recall
    """
    print("\n" + "="*105)
    # 表头
    header = f"{'Class':<22} | {'OA':<10} {'F1_Macro':<10} {'F1_Micro':<10} {'Precision':<10} {'Recall':<10}"
    print(header)
    print("-" * 105)
    
    # ---------------------------
    # 1. Object Detection (Wall/Projecting)
    # ---------------------------
    # 物体检测通常没有 TN (背景无穷大)，所以 OA通常不适用或仅基于 Recall
    # 这里 F1_Macro 和 F1_Micro 对单类物体来说是一样的 (都是基于 TP/FP/FN)
    otp, ofp, ofn = obj_stats['TP'], obj_stats['FP'], obj_stats['FN']
    op = otp / (otp + ofp + 1e-8)
    ore = otp / (otp + ofn + 1e-8)
    of1 = 2 * op * ore / (op + ore + 1e-8)
    
    print(f"{'[Objects Global]':<22} | {'-':<10} {'-':<10} {of1:<10.4f} {op:<10.4f} {ore:<10.4f}")
    print("-" * 105)
    
    # ---------------------------
    # 2. Attributes (Defects)
    # ---------------------------
    # 收集列表用于计算最后的平均值
    list_oa = []
    list_f1_macro = []
    list_p_macro = []
    list_r_macro = []
    
    # 用于计算 Global Micro F1 (累加所有属性的 TP/FP/FN)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    global_conf_mat = np.zeros((2, 2), dtype=int)

    for k in range(ATTR_COUNT):
        mat = attr_matrices[k] # [[TN, FP], [FN, TP]]
        global_conf_mat += mat
        # --- 数据提取 ---
        tn = mat[0][0]
        fp = mat[0][1] # 真实0，预测1
        fn = mat[1][0] # 真实1，预测0
        tp = mat[1][1]
        
        # 累加用于 Global Micro
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # --- 1. OA (Overall Accuracy) ---
        total_samples = tp + tn + fp + fn
        oa = (tp + tn) / (total_samples + 1e-8)
        
        # --- 2. Class 1 (正类/缺陷) 指标 ---
        p1 = tp / (tp + fp + 1e-8)
        r1 = tp / (tp + fn + 1e-8)
        f1_1 = 2 * p1 * r1 / (p1 + r1 + 1e-8)
        
        # --- 3. Class 0 (负类/正常) 指标 ---
        # 负类的 TP 是 TN，负类的 FP 是 FN，负类的 FN 是 FP
        p0 = tn / (tn + fn + 1e-8)
        r0 = tn / (tn + fp + 1e-8)
        f1_0 = 2 * p0 * r0 / (p0 + r0 + 1e-8)
        
        # --- 4. Macro Average (你的核心需求) ---
        p_macro = (p1 + p0) / 2
        r_macro = (r1 + r0) / 2
        f1_macro = (f1_1 + f1_0) / 2
        
        # --- 5. Micro F1 (Per Attribute) ---
        # 对单个二分类属性，Micro F1 等同于正类 F1 (f1_1)
        # 或者是基于样本加权的 Accuracy (如果正负样本极其不平衡，通常用 f1_1 代表该类的“微观表现”)
        f1_micro_attr = f1_1 

        # 存入列表
        list_oa.append(oa)
        list_f1_macro.append(f1_macro)
        list_p_macro.append(p_macro)
        list_r_macro.append(r_macro)
        
        name = CLASS_NAMES.get(ATTR_START_ID + k, f"Attr_{k}")
        
        # 打印单行
        if not sample_model:
            print(f"{name:<22} | {oa:<10.4f} {f1_macro:<10.4f} {f1_micro_attr:<10.4f} {p_macro:<10.4f} {r_macro:<10.4f}")

    print("-" * 105)
    
    # ---------------------------
    # 3. Global Summary (Averages)
    # ---------------------------
    # Mean Metrics
    mean_oa = np.mean(list_oa)
    mean_f1_macro = np.mean(list_f1_macro)
    mean_p_macro = np.mean(list_p_macro)
    mean_r_macro = np.mean(list_r_macro)
    
    # Global Micro F1 (基于所有属性的总 TP/FP/FN)
    # 这是一个非常硬核的指标，反映整体样本级的准度
    # micro_p = total_tp / (total_tp + total_fp + 1e-8)
    # micro_r = total_tp / (total_tp + total_fn + 1e-8)
    # global_f1_micro = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-8)

    global_tp_vec = global_conf_mat.diagonal() 
        
    # 2. 计算 FP (列和 - 对角线)
    # global_fp_vec = [Sum_FN, Sum_FP]
    # 解析: 对 Class 0 (Normal) 来说，FP 是 "真1测0" (即原来的 FN)
    global_fp_vec = global_conf_mat.sum(axis=0) - global_tp_vec
    
    # 3. 计算 FN (行和 - 对角线)
    # global_fn_vec = [Sum_FP, Sum_FN]
    # 解析: 对 Class 0 (Normal) 来说，FN 是 "真0测1" (即原来的 FP)
    global_fn_vec = global_conf_mat.sum(axis=1) - global_tp_vec
    
    # 4. 向量化计算 P, R, F1 (包含 Class 0 和 Class 1 两个值)
    g_precision = global_tp_vec / (global_tp_vec + global_fp_vec + 1e-8)
    g_recall = global_tp_vec / (global_tp_vec + global_fn_vec + 1e-8)
    g_f1 = 2 * g_precision * g_recall / (g_precision + g_recall + 1e-8)
    
    # 5. 取平均值 (这就是 mf1_micro 的逻辑: f1_micro.mean())
    global_f1_micro_consistent = g_f1.mean()
    
    
    print(f"{'[MEAN / GLOBAL]':<22} | {mean_oa:<10.4f} {mean_f1_macro:<10.4f} {global_f1_micro_consistent:<10.4f} {mean_p_macro:<10.4f} {mean_r_macro:<10.4f}")
    print("=" * 105)
    # print("注: ")
    # print("1. Precision/Recall/F1_Macro 为 (Class1 + Class0) / 2 的平均值。")
    # print("2. F1_Micro (单行) 为该属性正类(Class1)的 F1。")
    # print("3. F1_Micro (最后一行) 为基于所有属性总TP/FP/FN计算的全局 Micro-F1。")

    wall_signboard_map50 = result.class_result(0)[2]
    projecting_signboard_map50 = result.class_result(1)[2]
    if not sample_model:
        print(f'mAP50 wall signboard: {wall_signboard_map50}')
        print(f'mAP50 project signboard: {projecting_signboard_map50}')
    print(f'mAP50 Overall: {(wall_signboard_map50 + projecting_signboard_map50) / 2}')


if __name__ == '__main__':
    # myolo10('yolov10x.yaml', weight_path='yolov10x.pt', auto_optim=True, name='debug_yolov10x_ml', save_period=5)

    # myolo8('yolov8n.yaml', weight_path='yolov8n.pt', auto_optim=False)
    # myolo8('yolov8s.yaml', weight_path='yolov8s.pt', auto_optim=False)
    # myolo8('yolov8m.yaml', weight_path='yolov8m.pt', auto_optim=False)
    # myolo8('yolov8l.yaml', weight_path='yolov8l.pt', auto_optim=False)
    # myolo8('yolov8x.yaml', weight_path='yolov8x.pt', auto_optim=False)


    # myolo9('yolov9s.yaml', weight_path='yolov9s.pt', auto_optim=False)
    # myolo9('yolov9m.yaml', weight_path='yolov9m.pt', auto_optim=False)
    # myolo9('yolov9c.yaml', weight_path='yolov9c.pt', auto_optim=False)
    # myolo9('yolov9e.yaml', weight_path='yolov9e.pt', auto_optim=False)


    # myolo10('yolov10n.yaml', weight_path='yolov10n.pt', auto_optim=False)
    # myolo10('yolov10s.yaml', weight_path='yolov10s.pt', auto_optim=False)
    # myolo10('yolov10m.yaml', weight_path='yolov10m.pt', auto_optim=False)
    # myolo10('yolov10b.yaml', weight_path='yolov10b.pt', auto_optim=False)
    # myolo10('yolov10l.yaml', weight_path='yolov10l.pt', auto_optim=False)
    # myolo10('yolov10x.yaml', weight_path='yolov10x.pt', auto_optim=False)

    # 请修改这里的路径
    # MODEL_PATH = 'runs/detect/yolov10x_ml/weights/best.pt'
    IMG_DIR=r'/nfsv4/23039356r/data/billboard/data0806_m/yolo_rgb_detection5_10_c_ml/val/images'
    LABEL_DIR=r'/nfsv4/23039356r/data/billboard/data0806_m/yolo_rgb_detection5_10_c_ml/val/labels'
    
    # model_dir = r'/localnvme/project/ultralytics/runs/detect/debug_yolov10x_ml4/weights'
    # for model_name in os.listdir(model_dir):
    #     model_path = os.path.join(model_dir, model_name)
    #     result = model_val(model_path, verbose=False)
    #     o_stats, a_matrices = evaluate_dataset(model_path, IMG_DIR, LABEL_DIR, sample_model=True)
    #     # print_macro_results(o_stats, a_matrices)
    #     print_final_summary(o_stats, a_matrices, result, sample_model=True)


    model_path = r'/localnvme/project/ultralytics/runs/detect/yolov10x_ml/weights/best.pt'
    result = model_val(model_path, verbose=True)
    o_stats, a_matrices = evaluate_dataset(model_path, IMG_DIR, LABEL_DIR, sample_model=True)
    print_final_summary(o_stats, a_matrices, result, sample_model=False)


    # model_val('runs/detect/exp_ml/weights/best.pt')
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
