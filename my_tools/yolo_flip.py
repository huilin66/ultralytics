import cv2
import os
from tqdm import tqdm
def augment_yolo_horizontal_flip(img_path, label_path, save_img_dir, save_label_dir):
    """
    读取一张图片及其标签，进行水平翻转，并保存。
    """
    # 1. 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return
    
    # 2. 获取图像尺寸 (虽然YOLO是归一化的，但在某些转换中可能需要，这里主要是为了保存)
    h, w, _ = img.shape
    
    # 3. 水平翻转图像 (flipCode=1 表示水平翻转)
    flipped_img = cv2.flip(img, 1)
    
    # 4. 处理标签文件
    new_labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            
            # # cls_id = int(parts[0])
            # # x_center = float(parts[-4])
            # # y_center = float(parts[-3])
            # # width = float(parts[-2])
            # # height = float(parts[-1])
            # #
            # # --- 核心数学逻辑 ---
            # # 水平翻转后，新的中心点 x = 1 - 原中心点 x
            # new_x_center = 1.0 - x_center

            parts[-4] = str(1.0-float(parts[-4]))
            # 格式化保留6位小数
            # new_line = f"{cls_id} {new_x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            new_line = ' '.join(parts)
            new_labels.append(new_line)
    
    # 5. 保存结果
    # 构造新的文件名 (例如在原文件名前加 'flip_')
    base_name = os.path.basename(img_path)
    name_only, ext = os.path.splitext(base_name)
    
    new_img_name = f"{name_only}_flip{ext}"
    new_label_name = f"{name_only}_flip.txt"
    
    cv2.imwrite(os.path.join(save_img_dir, new_img_name), flipped_img)
    
    if new_labels:
        with open(os.path.join(save_label_dir, new_label_name), 'w') as f:
            f.write('\n'.join(new_labels))

# --- 使用示例 ---
if __name__ == "__main__":
    # 配置路径
    # INPUT_IMAGES_DIR = r"/nfsv4/23039356r/data/billboard/data0806_m/yolo_rgb_detection5_10_c/images"
    # INPUT_LABELS_DIR = r"/nfsv4/23039356r/data/billboard/data0806_m/yolo_rgb_detection5_10_c/labels"
    # OUTPUT_IMAGES_DIR = r"/nfsv4/23039356r/data/billboard/data0806_m/yolo_rgb_detection5_10_c/images_flip"
    # OUTPUT_LABELS_DIR = r"/nfsv4/23039356r/data/billboard/data0806_m/yolo_rgb_detection5_10_c/labels_flip"
    INPUT_IMAGES_DIR = r"/nfsv4/23039356r/data/billboard/data0806_m/yolo_rgb_detection5_10_c_ml/images"
    INPUT_LABELS_DIR = r"/nfsv4/23039356r/data/billboard/data0806_m/yolo_rgb_detection5_10_c_ml/labels"
    OUTPUT_IMAGES_DIR = r"/nfsv4/23039356r/data/billboard/data0806_m/yolo_rgb_detection5_10_c_ml/images_flip"
    OUTPUT_LABELS_DIR = r"/nfsv4/23039356r/data/billboard/data0806_m/yolo_rgb_detection5_10_c_ml/labels_flip"

    # 创建输出目录
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

    # 遍历文件夹进行处理
    image_files = [f for f in os.listdir(INPUT_IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"开始处理 {len(image_files)} 张图片...")
    
    for img_file in tqdm(image_files):
        label_file = os.path.splitext(img_file)[0] + ".txt"
        
        src_img_path = os.path.join(INPUT_IMAGES_DIR, img_file)
        src_label_path = os.path.join(INPUT_LABELS_DIR, label_file)
        
        augment_yolo_horizontal_flip(src_img_path, src_label_path, OUTPUT_IMAGES_DIR, OUTPUT_LABELS_DIR)
        
    print("处理完成！")