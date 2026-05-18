import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
import tifffile as tiff

def txt_to_mask(img_dir, txt_dir, output_dir):
    """
    将 YOLO 格式的 txt 标签/结果转换为二值化 Mask 图片
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图片文件
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.tif', '.tiff'))]
    
    # print(f"找到 {len(img_files)} 张图片，开始转换...")

    for img_file in tqdm(img_files, desc="yolo2mask", unit="img", leave=False):
        # 构造路径
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(img_dir, img_file)
        txt_path = os.path.join(txt_dir, base_name + ".txt")
        save_path = os.path.join(output_dir, base_name + ".tif")

        # 1. 读取原图以获取尺寸 (H, W)
        # 如果为了速度不想读图，且所有图尺寸固定(如128x128)，可以直接硬编码 h=128, w=128
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]

        # 2. 创建空白 Mask
        mask = np.zeros((h, w), dtype=np.uint8)

        # 3. 读取 TXT 并绘制
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3: continue
                
                # parts[0] 是 class_id，如果需要区分类别可以用它
                # class_id = int(parts[0])
                
                # 解析坐标并反归一化
                coords = [float(x) for x in parts[1:]]
                poly_points = []
                
                for i in range(0, len(coords), 2):
                    px = int(coords[i] * w)
                    py = int(coords[i+1] * h)
                    poly_points.append([px, py])
                
                # 转换为 cv2 格式
                pts = np.array(poly_points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                # 填充多边形 (255 表示前景)
                cv2.fillPoly(mask, [pts], 1)
        
        # 4. 保存结果
        tiff.imwrite(save_path, mask)

    print(f"convert {len(img_files)} masks to: {output_dir}")

# 使用示例
if __name__ == "__main__":
    # 图片文件夹 (用来获取宽高的)
    IMAGES_DIR = r'/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase/yolo_data/val/images'
    # 推理生成的 labels 文件夹 (里面是 .txt)
    LABELS_TXT_DIR = r'/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase/yolo_data/val/results/labels'
    # 输出 Mask 文件夹
    OUTPUT_MASK_DIR = r'/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase/yolo_data/val/results/masks'

    txt_to_mask(IMAGES_DIR, LABELS_TXT_DIR, OUTPUT_MASK_DIR)