import os
import numpy as np
import tifffile as tiff
import cv2
from tqdm import tqdm

def normalize_to_uint8(data):
    """将数据归一化到 0-255 并转为 uint8"""
    data = data.astype(float)
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val > 0:
        data = (data - min_val) / (max_val - min_val) * 255.0
    else:
        data = np.zeros_like(data)
    return data.astype(np.uint8)

def convert_mars_dataset(img_dir, mask_dir, img_rgb_dir, label_dir, class_txt_path):
    """
    1. 读取img_dir中的多波段tif，提取RGB保存到img_rgb_dir
    2. 统计mask类别
    3. 将mask转换为YOLO segmentation格式保存到label_dir
    """
    
    # 1. 创建输出文件夹
    os.makedirs(img_rgb_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True) if mask_dir is not None else None

    # 2. 读取类别文件
    # 假设 class.txt 内容为一行一个类别，例如: "landslide"
    # YOLO 格式中，第一行对应 class_id 0，第二行对应 class_id 1
    class_names = []
    if os.path.exists(class_txt_path):
        with open(class_txt_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
    else:
        print(f"[警告] {class_txt_path} 不存在，默认假设只有一类目标 (class_id=0)")
        class_names = ["landslide"]

    # 获取文件列表
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.tif', '.tiff'))]
    print(f"找到 {len(img_files)} 个图像文件。")

    # 用于统计 mask 中的所有唯一像素值
    global_unique_vals = set()

    for img_file in tqdm(img_files, desc="Processing"):
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file) if mask_dir is not None else None

        # ==========================================
        # 任务 1: 提取 RGB 波段
        # ==========================================
        try:
            # 读取多波段图像
            # MMLS 数据集描述：7个波段。
            # 波段 5, 6, 7 是 RGB (索引 4, 5, 6)
            img_data = tiff.imread(img_path)

            # 处理维度顺序: 有些库读出来是 (Bands, H, W)，有些是 (H, W, Bands)
            # 目标是 (H, W, Bands)
            if img_data.ndim == 3 and img_data.shape[0] == 7:
                # 如果是 (7, 128, 128) -> 转置为 (128, 128, 7)
                img_data = np.transpose(img_data, (1, 2, 0))
            
            h, w, c = img_data.shape

            if c >= 7:
                # 提取 RGB 通道 (索引 4, 5, 6)
                # 注意：OpenCV 保存图片时默认是 BGR 顺序
                # 原始数据: Band 5(R), Band 6(G), Band 7(B)
                r = img_data[:, :, 4]
                g = img_data[:, :, 5]
                b = img_data[:, :, 6]

                # 合并为 BGR (这是 OpenCV 的标准顺序)
                img_bgr = cv2.merge([b, g, r])
                
                # 归一化并转为 uint8 以便保存为 jpg
                img_bgr = normalize_to_uint8(img_bgr)
                
                # 保存
                cv2.imwrite(os.path.join(img_rgb_dir, base_name + ".jpg"), img_bgr)
            else:
                print(f"[错误] {img_file} 波段数不足 7，跳过。")
                continue

        except Exception as e:
            print(f"[错误] 读取图像 {img_file} 失败: {e}")
            continue

        # ==========================================
        # 任务 2 & 3: 处理 Mask 和 生成 YOLO 标签
        # ==========================================
        if mask_path is not None and os.path.exists(mask_path):
            try:
                mask_data = tiff.imread(mask_path)
                
                # 确保 mask 是 2D (H, W)
                if mask_data.ndim == 3:
                    mask_data = mask_data.squeeze() # 去掉多余维度

                # 统计类别
                unique_vals = np.unique(mask_data)
                for v in unique_vals:
                    global_unique_vals.add(v)

                # 准备写入 YOLO txt
                yolo_lines = []
                
                # MMLS 是二分类：0=背景，1=滑坡 (通常情况)
                # 我们只提取前景目标的轮廓
                # 假设 mask 中 > 0 的像素都是滑坡
                
                # 如果是多类别，这里需要循环处理每个 class_id
                # 这里针对 binary segmentation (只处理前景)
                target_pixel_value = 1 
                
                # 如果 mask 里全是 0 (无滑坡)，则生成空 txt
                if target_pixel_value in unique_vals:
                    # 创建二值图
                    binary_mask = np.uint8(mask_data == target_pixel_value) * 255
                    
                    # 查找轮廓
                    # RETR_EXTERNAL 只找外轮廓，RETR_TREE 找所有轮廓
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for cnt in contours:
                        if cv2.contourArea(cnt) < 10: # 过滤极小的噪点
                            continue
                        
                        # 轮廓点归一化 (0-1)
                        # cnt 形状是 (N, 1, 2) -> (x, y)
                        cnt = cnt.reshape(-1, 2)
                        
                        # YOLO Segment 格式: class_id x1 y1 x2 y2 ...
                        # 坐标必须归一化到 0-1 之间
                        normalized_coords = []
                        for point in cnt:
                            x_norm = point[0] / w
                            y_norm = point[1] / h
                            normalized_coords.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])
                        
                        # 对应 class.txt 中的第 0 类
                        class_id = 0 
                        line = f"{class_id} " + " ".join(normalized_coords)
                        yolo_lines.append(line)

                # 保存 txt
                txt_path = os.path.join(label_dir, base_name + ".txt")
                with open(txt_path, 'w') as f_out:
                    f_out.write("\n".join(yolo_lines))

            except Exception as e:
                print(f"[错误] 处理 Mask {mask_path} 失败: {e}")
        else:
            # 如果是测试集可能没有 mask，但这部分逻辑主要针对训练集
            pass

    # 打印类别统计结果
    print("-" * 30)
    print("处理完成！")
    print(f"Mask 中发现的所有像素值类别: {sorted(list(global_unique_vals))}")
    print(f"通常 0 为背景，其他值为目标类别。")
    print("-" * 30)

# ==========================================
# 配置路径并运行
# ==========================================
if __name__ == "__main__":
    # 请修改为你的实际路径
    dataset_root = r"/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase"
    
    # img_dir = os.path.join(dataset_root, "val/images")   # 输入：原始TIF图像文件夹
    # mask_dir = os.path.join(dataset_root, "val/masks")   # 输入：原始TIF掩码文件夹
    
    # img_rgb_dir = os.path.join(dataset_root, "yolo_data/val/images") # 输出：JPG图像
    # label_dir = os.path.join(dataset_root, "yolo_data/val/labels")   # 输出：YOLO txt标签

    img_dir = os.path.join(dataset_root, "test/images")   # 输入：原始TIF图像文件夹  
    img_rgb_dir = os.path.join(dataset_root, "yolo_data/test/images") # 输出：JPG图像
    mask_dir, label_dir = None, None
    class_txt = "class.txt"  # 你的类别文件
    
    # 简单的创建个 class.txt 示例（如果没有的话）
    if not os.path.exists(class_txt):
        with open(class_txt, 'w') as f:
            f.write("landslide\n")

    convert_mars_dataset(img_dir, mask_dir, img_rgb_dir, label_dir, class_txt)