import os
import cv2
import numpy as np
from tqdm import tqdm

def add_label(img, text, color=(255, 255, 255)):
    """在图片上方添加文字标签的辅助函数"""
    h, w = img.shape[:2]
    # 创建一个黑色背景条用于写字
    label_bg = np.zeros((30, w, 3), dtype=np.uint8)
    cv2.putText(label_bg, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, color, 1, cv2.LINE_AA)
    # 将文字条拼接到原图上方
    return cv2.vconcat([label_bg, img])

def to_bgr(img_gray):
    """将灰度图/二值图转换为 BGR 三通道图，方便拼接和显示"""
    if img_gray.ndim == 2:
        return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    return img_gray

def generate_comparison_vis(img_dir, gt_dir, pred_dir, output_dir):
    """
    生成分割结果对比图（四宫格：原图, GT, Pred, Error Map）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 获取文件列表 (以 GT 文件夹为基准)
    # 假设各文件夹下的文件名（不含后缀）是一一对应的
    gt_files = [f for f in os.listdir(gt_dir) if f.endswith(('.png', '.tif', '.tiff', '.jpg'))]
    gt_files.sort()
    
    if not gt_files:
        print("错误：GT 文件夹为空。")
        return


    # 建立文件名映射，处理不同后缀名的情况 (例如 GT是tif, Pred是png)
    def get_file_map(directory):
        mapping = {}
        for f in os.listdir(directory):
            name, ext = os.path.splitext(f)
            if ext.lower() in ['.png', '.jpg', '.tif', '.tiff']:
                mapping[name] = os.path.join(directory, f)
        return mapping

    img_map = get_file_map(img_dir)
    pred_map = get_file_map(pred_dir)

    for gt_file in tqdm(gt_files, desc="mask compare", unit='img', leave=False):
        base_name = os.path.splitext(gt_file)[0]
        
        # 检查对应的文件是否存在
        if base_name not in img_map or base_name not in pred_map:
            # print(f"警告：找不到 {base_name} 对应的原图或预测图，跳过。")
            continue
            
        gt_path = os.path.join(gt_dir, gt_file)
        img_path = img_map[base_name]
        pred_path = pred_map[base_name]
        save_path = os.path.join(output_dir, base_name + "_vis.jpg") # 保存为JPG以减小体积

        # ===========================
        # 1. 读取图像
        # ===========================
        # 原图 (尝试读取为彩色，如果不行就灰度)
        orig_img = cv2.imread(img_path)
        if orig_img is None:
             orig_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # GT 和 Pred 读取为灰度
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        assert gt_img.shape == pred_img.shape, "GT 和 Pred 尺寸不一致"
        h, w = gt_img.shape

        # ===========================
        # 2. 数据预处理 (二值化)
        # ===========================
        # 将图像转换为严格的 0 和 1，方便计算
        gt_bin = gt_img.astype(np.uint8)
        pred_bin = pred_img.astype(np.uint8)

        # ===========================
        # 3. 生成误差图 (Error Map) - 核心逻辑
        # ===========================
        # 使用巧妙的算术运算组合状态：
        # status = 2 * GT + Pred
        # 结果含义:
        # 0 (GT=0, Pred=0) -> TN (真负, 背景)
        # 1 (GT=0, Pred=1) -> FP (假正, 误报) -> 红色
        # 2 (GT=1, Pred=0) -> FN (假负, 漏报) -> 蓝色
        # 3 (GT=1, Pred=1) -> TP (真正, 正确) -> 白色
        status_map = 2 * gt_bin + pred_bin
        
        # 创建彩色误差图画布 (BGR)
        error_vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 填充颜色 (OpenCV 使用 BGR 顺序)
        error_vis[status_map == 1] = [0, 0, 255]    # Red: FP (误报)
        error_vis[status_map == 2] = [255, 0, 0]    # Blue: FN (漏报)
        error_vis[status_map == 3] = [255, 255, 255] # White: TP (正确)

        # ===========================
        # 4. 拼接四宫格
        # ===========================
        # 将所有图像统一转换为 BGR 三通道以便拼接
        orig_bgr = to_bgr(orig_img)
        # 将GT和Pred拉伸到0-255以便显示，并转为彩色
        gt_bgr = to_bgr(gt_bin * 255)
        pred_bgr = to_bgr(pred_bin * 255)

        # 添加标签
        orig_labeled = add_label(orig_bgr, "Original Image")
        gt_labeled = add_label(gt_bgr, "Ground Truth (Target)")
        pred_labeled = add_label(pred_bgr, "Prediction (Model)")
        # 为误差图添加详细图例说明
        error_labeled = add_label(error_vis, "Error: White=TP, Red=FP, Blue=FN", color=(200, 200, 0))

        # 拼合图像
        # 上排：原图 + GT
        top_row = cv2.hconcat([orig_labeled, error_labeled])
        # 下排：预测 + 误差图
        bottom_row = cv2.hconcat([gt_labeled, pred_labeled])
        # 整体组合
        final_grid = cv2.vconcat([top_row, bottom_row])

        # ===========================
        # 5. 保存
        # ===========================
        # 如果图像太小（比如128x128），拼出来可能看不清文字，可以放大保存
        # if h < 256:
        #     final_grid = cv2.resize(final_grid, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
            
        cv2.imwrite(save_path, final_grid)

    print(f"compare {len(gt_files)} result into: {output_dir}")

# ================================
# 使用示例
# ================================
if __name__ == "__main__":
    IMG_DIR = r'/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase/yolo_data/val/vis_result'
    GT_DIR = r'/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase/val/masks'
    PRED_DIR = r'/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase/yolo_data/val/results/masks'
    OUTPUT_VIS_DIR = r'/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase/yolo_data/val/results/masks_compare'

    if os.path.exists(GT_DIR) and os.path.exists(PRED_DIR):
        generate_comparison_vis(IMG_DIR, GT_DIR, PRED_DIR, OUTPUT_VIS_DIR)
    else:
        print("路径配置错误，请检查 GT_DIR 和 PRED_DIR")