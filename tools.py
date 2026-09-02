import os
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm

def convert_xml_to_yolo(xml_path, classes):
    """单个 XML 转 YOLO 格式"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    yolo_anns = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)

        xml_box = obj.find('bndbox')
        xmin = float(xml_box.find('xmin').text)
        ymin = float(xml_box.find('ymin').text)
        xmax = float(xml_box.find('xmax').text)
        ymax = float(xml_box.find('ymax').text)

        # YOLO 坐标归一化
        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        yolo_anns.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    return yolo_anns

def process_dataset(xml_dir, img_dir, dst_image_dir, dst_label_dir, base_name_list, classes):
    """处理一个数据集分区（trainval / test）"""
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG']

    for xml_file in tqdm(xml_files, desc="处理中"):
        base = os.path.splitext(xml_file)[0]
        base_name_list.append(base)

        # 转换标注
        xml_path = os.path.join(xml_dir, xml_file)
        anns = convert_xml_to_yolo(xml_path, classes)
        with open(f"{dst_label_dir}/{base}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(anns))

        # 复制图片
        for ext in img_exts:
            img_src = os.path.join(img_dir, base + ext)
            if os.path.exists(img_src):
                img_dst = f"{dst_image_dir}/{base}{ext}"
                shutil.copy(img_src, img_dst)
                break

def main():
    # ===================== 路径定义 =====================
    ROOT = "/root/huilin/data/EL2021"
    TRAINVAL_XML = os.path.join(ROOT, "trainval/Annotations")
    TRAINVAL_IMG = os.path.join(ROOT, "trainval/JPEGImages")
    TEST_XML = os.path.join(ROOT, "test/Annotations")
    TEST_IMG = os.path.join(ROOT, "test/JPEGImages")

    DST_TRAINVAL = os.path.join(ROOT, "trainval")
    DST_TRAINVAL_IMG = os.path.join(DST_TRAINVAL, "images")
    DST_TRAINVAL_LABEL = os.path.join(DST_TRAINVAL, "labels")
    DST_TEST = os.path.join(ROOT, "test")
    DST_TEST_IMG = os.path.join(DST_TEST, "images")
    DST_TEST_LABEL = os.path.join(DST_TEST, "labels")

    # 创建输出文件夹
    os.makedirs(DST_TRAINVAL_IMG, exist_ok=True)
    os.makedirs(DST_TEST_IMG, exist_ok=True)
    os.makedirs(DST_TEST_LABEL, exist_ok=True)
    os.makedirs(DST_TRAINVAL_LABEL, exist_ok=True)

    # 1. 提取所有类别（从 trainval + test）
    print("[1/4] 提取所有类别...")
    all_classes = []
    all_xml = []
    for d in [TRAINVAL_XML, TEST_XML]:
        all_xml += [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.xml')]
    
    for xml_path in all_xml:
        tree = ET.parse(xml_path)
        for obj in tree.getroot().iter('object'):
            cls = obj.find('name').text
            if cls not in all_classes:
                all_classes.append(cls)
    
    with open(os.path.join(DST_TRAINVAL, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(all_classes))
    print(f"类别数：{len(all_classes)}")

    # 2. 处理 trainval
    print("\n[2/4] 处理 trainval 数据集...")
    trainval_list = []
    process_dataset(TRAINVAL_XML, TRAINVAL_IMG, DST_TRAINVAL_IMG, DST_TRAINVAL_LABEL, trainval_list, all_classes)

    # 3. 处理 test
    print("\n[3/4] 处理 test 数据集...")
    test_list = []
    process_dataset(TEST_XML, TEST_IMG, DST_TEST_IMG, DST_TEST_LABEL, test_list, all_classes)

    # 4. 生成划分文件
    print("\n[4/4] 生成 trainval.txt / test.txt...")
    with open(os.path.join(DST_TRAINVAL, "trainval.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join([f"{DST_TRAINVAL_IMG}/{name}.jpg" for name in trainval_list]))
    with open(os.path.join(DST_TEST, "test.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join([f"{DST_TEST_IMG}/{name}.jpg" for name in test_list]))

    print(f"\n✅ 全部完成！")
    print(f"📁 {os.path.join(ROOT, 'images')} / {os.path.join(ROOT, 'labels')} / {os.path.join(ROOT, 'classes.txt')} / {os.path.join(ROOT, 'trainval.txt')} / {os.path.join(ROOT, 'test.txt')}")
    print(f"📁 {os.path.join(ROOT, 'labels')} / classes.txt  生成后所有标注")
    print(f"📄 {os.path.join(ROOT, 'classes.txt')}  类别文件")
    print(f"📄 {os.path.join(ROOT, 'trainval.txt')}  训练集列表")
    print(f"📄 {os.path.join(ROOT, 'test.txt')}     测试集列表")

if __name__ == "__main__":
    main()