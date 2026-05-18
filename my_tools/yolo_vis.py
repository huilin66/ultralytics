import os
import cv2
import numpy as np
import random
from tqdm import tqdm

def visualize_and_save_yolo_seg(img_dir, label_dir, output_dir, class_names=None, alpha=0.5):
    """
    批量可视化 YOLO 分割标签并保存到新文件夹。
    
    参数:
        img_dir: 图片文件夹路径
        label_dir: YOLO txt 标签文件夹路径
        output_dir: 结果保存路径
        class_names: 类别名称列表 (可选，用于显示文本)
        alpha: 掩码透明度 (0-1)，0为全透明，1为不透明
    """
    
    # 1. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 2. 获取图片列表
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]
    
    if not img_files:
        print("未找到图片文件！")
        return

    # 3. 生成颜色表 (为每个类别生成固定颜色)
    # 种子固定，保证每次运行颜色一致
    np.random.seed(42)
    colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(80)] 

    print(f"开始处理 {len(img_files)} 张图片...")

    for img_file in tqdm(img_files, desc="Visualizing"):
        # 路径准备
        img_path = os.path.join(img_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)
        save_path = os.path.join(output_dir, img_file)

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # 如果没有标签文件，直接保存原图（或者你可以选择跳过）
        if not os.path.exists(label_path):
            cv2.imwrite(save_path, img)
            continue

        # 创建用于绘制半透明填充的副本
        overlay = img.copy()
        
        # 读取标签内容
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # 遍历每一个实例
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3: 
                continue

            # 解析 Class ID
            class_id = int(parts[0])
            color = colors[class_id % len(colors)] # 获取对应颜色
            
            # 解析坐标 (YOLO格式: x1 y1 x2 y2 ...)
            # 坐标是归一化的 (0-1)
            normalized_coords = [float(x) for x in parts[1:]]
            
            # 将坐标反归一化为像素坐标
            poly_points = []
            for i in range(0, len(normalized_coords), 2):
                x = int(normalized_coords[i] * w)
                y = int(normalized_coords[i+1] * h)
                poly_points.append([x, y])
            
            # 转换为 numpy 数组，形状必须是 (N, 1, 2)
            pts = np.array(poly_points, np.int32)
            pts = pts.reshape((-1, 1, 2))

            # 1. 绘制实心多边形 (画在 overlay 上)
            cv2.fillPoly(overlay, [pts], color)
            
            # 2. 绘制多边形轮廓 (画在原图 img 上，这样轮廓清晰)
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

            # 3. 添加类别标签文本
            if len(poly_points) > 0:
                # 找最上面的点作为标签位置
                label_pos = poly_points[0]
                label_text = f"ID: {class_id}"
                if class_names and class_id < len(class_names):
                    label_text = class_names[class_id]
                
                # 绘制文字背景
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, 
                              (label_pos[0], label_pos[1] - text_h - 5), 
                              (label_pos[0] + text_w, label_pos[1]), 
                              color, -1)
                # 绘制文字
                cv2.putText(img, label_text, (label_pos[0], label_pos[1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 混合原图和填充层 (实现半透明效果)
        # result = alpha * overlay + (1 - alpha) * img
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # 保存结果
        cv2.imwrite(save_path, img)

    print(f"\n处理完成！所有结果已保存至: {output_dir}")

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 配置你的路径
    dataset_root =  r"/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase/yolo_data/val"
    
    images_path = os.path.join(dataset_root, "images")  # 图片文件夹
    labels_path = os.path.join(dataset_root, "labels")  # 标签文件夹
    output_viz_path = os.path.join(dataset_root, "vis_result") # 结果保存位置
    
    # 类别名称 (可选，用于显示名字而不是 ID)
    my_classes = ["landslide"] 

    visualize_and_save_yolo_seg(
        img_dir=images_path, 
        label_dir=labels_path, 
        output_dir=output_viz_path, 
        class_names=my_classes,
        alpha=0.4  # 透明度设置 (0.4 代表 40% 的颜色填充)
    )