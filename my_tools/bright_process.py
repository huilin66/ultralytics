from PIL import Image, ImageEnhance
from tqdm import tqdm
import glob
import os

def increase_brightness_pil(image_path, output_path, factor=1.2):
    """
    将图像亮度提高20%
    factor: 亮度系数，1.0为原图，1.2为提高20%
    """
    # 打开图像
    img = Image.open(image_path)

    # 创建亮度增强器
    enhancer = ImageEnhance.Brightness(img)

    # 提高亮度
    brightened_img = enhancer.enhance(factor)

    # 保存图像
    brightened_img.save(output_path)
    return brightened_img


def batch_increase_brightness_pil(image_dir, output_dir, factor=1.2):
    """
    批量处理目录中的图像，提高亮度
    image_dir: 包含图像的目录
    output_dir: 输出目录，用于保存处理后的图像
    factor: 亮度系数，1.0为原图，1.2为提高20%
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    img_list =glob.glob(os.path.join(image_dir,"*.png"))
    # 遍历目录中的所有图像
    for image_path in tqdm(img_list):
        # 构建输出路径
        output_path = os.path.join(output_dir, os.path.basename(image_path))

        # 提高亮度
        increase_brightness_pil(image_path, output_path, factor)


if __name__ == '__main__':
    pass
    import os
    src_dir = r'/nfsv4/23039356r/data/billboard/data0806_m/yolo_rgb_detection5_10_c'
    brightness_list = [
        # 0.5,
        # 0.75,
        1.25,
        # 1.5,
        # 2.0,
    ]
    for brightness in brightness_list:
        dst_dir = os.path.join(f'{src_dir}_b{int(brightness*100)}', 'images')
        batch_increase_brightness_pil(os.path.join(src_dir, 'images'), dst_dir, brightness)
