import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

class Evaluator:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        # 混淆矩阵: 行是GT，列是Pred
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, gt_mask, pred_mask):
        """
        更新混淆矩阵
        gt_mask: 真实标签 (H, W)，值在 [0, 1] 之间
        pred_mask: 预测标签 (H, W)，值在 [0, 1] 之间
        """
        assert gt_mask.shape == pred_mask.shape, "GT和预测Mask尺寸不匹配"
        
        # 将二维数组展平，寻找对应关系
        # bincount计算每个 (gt_label * num_classes + pred_label) 出现的次数
        # 从而快速生成混淆矩阵
        gt_mask = gt_mask.flatten().astype(np.int32)
        pred_mask = pred_mask.flatten().astype(np.int32)
        
        # 核心技巧：利用label编码计算混淆矩阵
        category = gt_mask * self.num_classes + pred_mask
        count = np.bincount(category, minlength=self.num_classes ** 2)
        
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += confusion_matrix

    def get_scores(self):
        """
        计算并返回所有指标
        """
        # 混淆矩阵: CM[i, j] 表示 GT为i, 预测为j 的像素数量
        # TP (针对每一类): 对角线元素
        tp = np.diag(self.confusion_matrix)
        
        # FP (针对每一类): 列求和 - TP (预测是该类，但实际不是)
        fp = self.confusion_matrix.sum(axis=0) - tp
        
        # FN (针对每一类): 行求和 - TP (实际是该类，但预测不是)
        fn = self.confusion_matrix.sum(axis=1) - tp
        
        # Union (针对每一类): TP + FP + FN
        union = tp + fp + fn
        
        # ---------------------------
        # 1. 计算 IoU 和 mIoU
        # ---------------------------
        # 防止分母为0，加一个极小值 eps
        eps = 1e-6
        iou = tp / (union + eps)
        miou = np.mean(iou)
        
        # ---------------------------
        # 2. 计算 Foreground (Class 1) 的指标
        # ---------------------------
        # 假设 Class 1 是滑坡 (Landslide)
        fg_idx = 1
        
        precision_fg = tp[fg_idx] / (tp[fg_idx] + fp[fg_idx] + eps)
        recall_fg    = tp[fg_idx] / (tp[fg_idx] + fn[fg_idx] + eps)
        f1_fg        = 2 * precision_fg * recall_fg / (precision_fg + recall_fg + eps)

        return {
            "mIoU": miou,
            "IoU_Background": iou[0],
            "IoU_Landslide": iou[1],
            "Precision": precision_fg,
            "Recall": recall_fg,
            "F1_Score": f1_fg
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))


def result_evaluate(gt_dir, pred_dir):
    """
    评估文件夹中的所有Mask
    """
    # 获取文件列表
    # 假设文件名是一一对应的
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.tif', '.tiff'))])
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(('.png', '.tif', '.tiff'))])

    if len(gt_files) == 0:
        print("错误：GT 文件夹为空")
        return

    evaluator = Evaluator(num_classes=2)

    
    # 使用字典加速查找，防止文件名顺序不完全一致
    pred_dict = {f: os.path.join(pred_dir, f) for f in pred_files}

    for gt_file in tqdm(gt_files, desc="Evaluating", unit='img', leave=False):
        if gt_file not in pred_dict:
            print(f"警告: 在预测文件夹中找不到 {gt_file}，已跳过。")
            continue
            
        gt_path = os.path.join(gt_dir, gt_file)
        pred_path = pred_dict[gt_file]
        
        # 读取图像
        # 0: cv2.IMREAD_GRAYSCALE
        gt_img = cv2.imread(gt_path, 0)
        pred_img = cv2.imread(pred_path, 0)
        
        if gt_img is None or pred_img is None:
            print(f"读取错误: {gt_file}")
            continue
            
        # --- 数据预处理 ---
        # 确保尺寸一致
        if gt_img.shape != pred_img.shape:
            pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        # 二值化处理：确保值只有 0 和 1
        # 如果原始图是 0和255，这里会变成 0和1
        gt_img = (gt_img > 0).astype(np.uint8)
        pred_img = (pred_img > 0).astype(np.uint8)
        
        # 更新统计
        evaluator.update(gt_img, pred_img)
        
    # 计算最终结果
    metrics = evaluator.get_scores()
    
    # 定义表头和数据
    headers = [
        "mIoU", "IoU (Landslide)", "IoU (Backgrd)", 
        "Precision (Fg)", "Recall (Fg)", "F1 Score (Fg)"
    ]
    
    # 格式化数值列表
    values = [
        metrics['mIoU'], metrics['IoU_Landslide'], metrics['IoU_Background'],
        metrics['Precision'], metrics['Recall'], metrics['F1_Score']
    ]

    # 设置列宽
    width = 16 
    
    print("\n" + "=" * (width * 6 + 18))
    print("Mars Landslide Segmentation Evaluation")
    print("=" * (width * 6 + 18))
    
    # 打印第一行：表头
    header_str = " | ".join([f"{h:<{width}}" for h in headers])
    print(header_str)
    
    print("-" * (width * 6 + 18))
    
    # 打印第二行：数值
    value_str = " | ".join([f"{v:<{width}.4f}" for v in values])
    print(value_str)
    
    print("=" * (width * 6 + 18))

# ================================
# 使用示例
# ================================
if __name__ == "__main__":
    # 配置路径
    # GT_DIR: 包含真实的Mask (tif 或 png)
    # PRED_DIR: 包含你预测生成的Mask (建议是 png 格式)
    
    GT_DIR = r'/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase/val/masks'     # 验证集真实标签路径
    PRED_DIR = r'/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase/yolo_data/val/results/masks'   # 模型预测结果路径
    
    if os.path.exists(GT_DIR) and os.path.exists(PRED_DIR):
        result_evaluate(GT_DIR, PRED_DIR)
    else:
        print("路径不存在，请修改代码中的 GT_DIR 和 PRED_DIR")