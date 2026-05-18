import os
import numpy as np
import tifffile as tiff
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mutual_info_score
from skimage.feature import canny
from skimage.morphology import dilation, disk

def normalize(img):
    """将图像归一化到 0-255"""
    img = np.nan_to_num(img)
    if img.max() - img.min() == 0:
        return img
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

def get_mask_edges(mask):
    """提取Mask的轮廓边缘"""
    # 膨胀一下mask然后减去原mask得到边界，或者直接用Canny
    # 这里用Canny提取二值图的边缘
    edges = cv2.Canny(mask.astype(np.uint8) * 255, 100, 200)
    return edges > 0

def calculate_edge_overlap(band_img, mask_edges):
    """
    计算波段边缘与Mask边缘的重合度
    """
    # 1. 对波段进行 Canny 边缘检测
    # 由于不同波段动态范围不同，自适应计算阈值
    v = np.median(band_img)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    band_edges = cv2.Canny(band_img, lower, upper)
    band_edges_bool = band_edges > 0
    
    # 2. 为了容忍轻微的像素偏移，对Mask边缘进行微小的膨胀 (1像素半径)
    # 意味着如果波段边缘落在Mask轮廓附近1像素内，也算匹配
    mask_edges_dilated = dilation(mask_edges, disk(1))
    
    # 3. 计算重合 (Intersection)
    # 只关心 Mask 边缘存在的地方，波段是否也检测到了边缘
    intersection = np.logical_and(band_edges_bool, mask_edges_dilated)
    
    # 4. 召回率风格的指标：Mask的边缘有多少被波段检测到了？
    if np.sum(mask_edges) == 0:
        return 0.0
    
    score = np.sum(intersection) / np.sum(mask_edges)
    return score

def analyze_band_importance(img_dir, mask_dir):
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    
    # 定义波段名称 (根据你的数据集描述)
    band_names = [
        "Band 1: Thermal Inertia",
        "Band 2: Slope",
        "Band 3: DEM",
        "Band 4: Grayscale",
        "Band 5: Red",
        "Band 6: Green",
        "Band 7: Blue"
    ]
    
    results = {name: {'edge_score': [], 'correlation': [], 'mi': []} for name in band_names}
    
    print(f"正在分析 {len(img_files)} 张图像...")
    
    for img_file in tqdm(img_files[:100]): # 为了演示速度，这里只取前100张，实际使用建议跑全量
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file)
        
        if not os.path.exists(mask_path): continue
        
        # 读取数据
        try:
            img_data = tiff.imread(img_path)
            if img_data.shape[0] == 7: img_data = np.transpose(img_data, (1, 2, 0)) # (H,W,C)
            
            mask_data = tiff.imread(mask_path)
            if mask_data.ndim == 3: mask_data = mask_data[:,:,0] # 确保是2D
            mask_data = (mask_data > 0).astype(np.uint8) # 二值化 0, 1
            
            # 如果mask是空的（没有滑坡），跳过边缘分析
            has_landslide = np.sum(mask_data) > 0
            mask_edges = get_mask_edges(mask_data) if has_landslide else None
            
            mask_flat = mask_data.flatten()
            
            for i, name in enumerate(band_names):
                band = img_data[:, :, i]
                band_norm = normalize(band) # 归一化用于边缘检测
                band_flat = band.flatten()
                
                # Metric 1: 边缘重合度 (Edge Overlap)
                # 只有当图片里有滑坡时才计算
                if has_landslide:
                    edge_score = calculate_edge_overlap(band_norm, mask_edges)
                    results[name]['edge_score'].append(edge_score)
                
                # Metric 2: 相关性 (Correlation)
                # 计算像素值与标签(0/1)的相关系数
                corr = np.corrcoef(band_flat, mask_flat)[0, 1]
                if not np.isnan(corr):
                    results[name]['correlation'].append(abs(corr)) # 取绝对值，关注相关程度
                
                # Metric 3: 互信息 (Mutual Information) - 简化版（基于直方图太慢，这里用score）
                # 为了速度，对数据进行分箱 (Binning)
                # mi = mutual_info_score(mask_flat, np.digitize(band_flat, bins=10))
                # results[name]['mi'].append(mi)

        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue

    # --- 汇总结果并可视化 ---
    summary = []
    for name in band_names:
        avg_edge = np.mean(results[name]['edge_score']) if results[name]['edge_score'] else 0
        avg_corr = np.mean(results[name]['correlation']) if results[name]['correlation'] else 0
        summary.append({
            "Band": name,
            "Edge Alignment Score": avg_edge,
            "Pixel Correlation": avg_corr
        })
    
    df = pd.DataFrame(summary)
    
    # 绘图
    plt.figure(figsize=(12, 6))
    
    # 图1: 边缘重合度
    plt.subplot(1, 2, 1)
    sns.barplot(data=df, y="Band", x="Edge Alignment Score", palette="viridis")
    plt.title("Method 1: Edge Alignment (Boundary Match)")
    plt.xlabel("Average Overlap Score")
    
    # 图2: 像素相关性
    plt.subplot(1, 2, 2)
    sns.barplot(data=df, y="Band", x="Pixel Correlation", palette="magma")
    plt.title("Method 2: Pixel Intensity Correlation")
    plt.xlabel("Absolute Correlation Coefficient")
    
    plt.tight_layout()
    plt.show()
    
    print("\n分析结果数值:")
    print(df)
    
    # 找出最佳波段
    best_edge = df.loc[df['Edge Alignment Score'].idxmax()]
    print(f"\n结论:\n边界最清晰的波段是: {best_edge['Band']}")

# ============================
# 运行配置
# ============================
if __name__ == "__main__":
    # 修改为你的路径
    dataset_root = r"/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase"
    img_dir = os.path.join(dataset_root, "train/images")
    mask_dir = os.path.join(dataset_root, "train/masks")
    
    analyze_band_importance(img_dir, mask_dir)