import os
import json
import zipfile
import cv2
import numpy as np
from pycocotools import mask as mask_util
from tqdm import tqdm


def txt_to_coco_json(txt_folder, output_json_path, ref_json=None, conf_thresh=0.5):
    """
    函数1: 解析 YOLO seg 结果 txt 文件夹，转换为 COCO 格式 JSON

    参数:
    - txt_folder: 存放 YOLO 预测结果 .txt 的文件夹 (通常是 runs/segment/predict/labels)
    - image_folder: 原始图片文件夹 (我们需要读取图片尺寸来反归一化坐标)
    - output_json_path: 输出 json 的路径
    - category_mapping: 字典 {yolo_class_id: coco_category_id}，默认为 {0: 1}
    """
    with open(ref_json) as f:
        ref_data = json.load(f)

    name2item_dict = {item['file_name']: item for item in ref_data['images']}

    results = []

    txt_list = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]


    count = 0
    for txt_name in tqdm(txt_list):
        image_name = txt_name.replace('.txt', '.jpg')
        image_item = name2item_dict[image_name]
        h, w = image_item['height'], image_item['width']
        image_id = image_item['id']


        with open(os.path.join(txt_folder, txt_name), 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            cls_id = int(parts[0])

            score = float(parts[-1])
            # if score < conf_thresh: continue

            coords = list(map(float, parts[1:-1]))  # 中间全是坐标


            # 4. 坐标反归一化 & 构建多边形
            # coords 是 [x1, y1, x2, y2, ...] (0-1 float)
            poly_points = []
            flat_poly_for_coco = []
            for i in range(0, len(coords), 2):
                px = round(coords[i] * w, 2)
                py = round(coords[i + 1] * h, 2)
                poly_points.append([px, py])
                flat_poly_for_coco.extend([px, py])

            poly_points = np.array(poly_points)

            # 6. 生成 BBox [x, y, w, h]
            x_min = np.min(poly_points[:, 0])
            y_min = np.min(poly_points[:, 1])
            x_max = np.max(poly_points[:, 0])
            y_max = np.max(poly_points[:, 1])
            bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]

            count += 1
            results.append({
                'id': count,
                "image_id": image_id,
                'score': score,
                'iscrowd': 0,
                'area': float(bbox[2] * bbox[3]),
                "category_id": cls_id+1,
                "bbox": bbox,
                "segmentation": [flat_poly_for_coco]
            })

    # ref_data['annotations'] = results
    # with open(output_json_path, 'w') as f:
    #     json.dump(ref_data, f)
    with open(output_json_path, 'w') as f:
            json.dump(results, f)
    print(f"✅ JSON 生成完毕: {output_json_path} (共 {len(results)} 个目标)")


def compress_to_zip(json_path, zip_path):
    """
    函数2: 将指定的 JSON 文件压缩为 ZIP 包
    """
    print(f"正在压缩 {json_path} 到 {zip_path} ...")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # arcname 确保 zip 包内没有多余的文件夹路径
        filename = os.path.basename(json_path)
        zf.write(json_path, arcname=filename)

    print(f"✅ 压缩完成，请提交: {zip_path}")


# ================= 使用示例 =================
if __name__ == "__main__":
    pred_name = 'a100_yolov8x-seg2'

    REF_JSON = r'/data/huilin//scrinvme/huilin/bdd/cp_data/rip_seg/val_no_annotations.json'
    MY_TXT_FOLDER = rf"/data/huilin//scrinvme/huilin/bdd/cp_data/rip_seg/val_images_infer/{pred_name}/labels"
    # MY_IMAGE_FOLDER = r"/scrinvme/huilin/bdd/cp_data/rip_seg/val_images"
    FINAL_JSON = f"submission/{pred_name}/predictions_both.json"
    FINAL_ZIP = f"submission//{pred_name}/submission.zip"

    os.makedirs(os.path.dirname(FINAL_JSON), exist_ok=True)
    if os.path.exists(MY_TXT_FOLDER):
        txt_to_coco_json(MY_TXT_FOLDER, FINAL_JSON, REF_JSON)
        compress_to_zip(FINAL_JSON, FINAL_ZIP)
    else:
        print(f"错误: 找不到 txt 文件夹 {MY_TXT_FOLDER}")