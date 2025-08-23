import os
import json
import shutil

from PIL import Image
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class COCOeval_diy(COCOeval):

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            category_dim = 1+int(ap)
            if s.shape[category_dim] > 1:
                iStr += ", per category = {}"
                mean_axis = (0, )
                if ap==1:
                    mean_axis = (0, 1)
                per_category_mean_s = np.mean(s, axis=mean_axis).flatten()
                with np.printoptions(precision=3, suppress=True, sign=" ", floatmode="fixed"):
                    print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s, per_category_mean_s))
            else:
                print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()


def select_val(input_dir, val_txt='val.txt'):
    pass
    val_txt_path = os.path.join(input_dir, val_txt)
    image_dir = os.path.join(input_dir, 'images')
    label_dir = os.path.join(input_dir, 'labels')
    val_dir = os.path.join(input_dir, 'val')
    val_image_dir = os.path.join(val_dir, 'images')
    val_label_dir = os.path.join(val_dir, 'labels')
    if os.path.exists(val_label_dir):
        shutil.rmtree(val_label_dir)
    if os.path.exists(val_image_dir):
        shutil.rmtree(val_image_dir)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    df = pd.read_csv(val_txt_path, header=None, index_col=False, names=['image_name'])
    img_list = df['image_name'].to_list()
    for image_path in tqdm(img_list):
        image_name = Path(image_path).name
        txt_name = Path(image_name).stem + '.txt'
        input_image_path = os.path.join(image_dir, image_name)
        ouput_image_path = os.path.join(val_image_dir, image_name)
        input_label_path = os.path.join(label_dir, txt_name)
        output_label_path = os.path.join(val_label_dir, txt_name)
        shutil.copy(input_image_path, ouput_image_path)
        shutil.copy(input_label_path, output_label_path)

def poly2xywh(mask):
    mask = np.array([mask[::2], mask[1::2]])
    x_min,y_min = np.min(mask, axis=1)
    x_max,y_max = np.max(mask, axis=1)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    width = x_max - x_min
    height = y_max - y_min
    return [x_center, y_center, width, height]

def yolo_to_coco(yolo_dir, img_dir, output_file, categories):
    """
    将YOLO格式的标注转换为COCO格式。

    :param yolo_dir: 包含YOLO标注txt文件的目录路径。
    :param img_dir: 包含对应图像的目录路径。
    :param output_file: 输出的COCO格式JSON文件路径。
    :param categories: 类别列表，如[{"id": 0, "name": "cat"}, {"id": 1, "name": "dog"}]
    """
    images = []
    annotations = []
    image_id = 0
    annotation_id = 0

    image_list = os.listdir(img_dir)
    image_stem_list = [Path(image_name).stem for image_name in image_list]
    image2stem_dict = dict(zip(image_stem_list, image_list))

    for txt_file in os.listdir(yolo_dir):
        if not txt_file.endswith('.txt'):
            continue

        image_id = Path(txt_file).stem
        img_file = image2stem_dict[image_id]
        # 获取对应的图像文件名
        img_path = os.path.join(img_dir, img_file)

        if not os.path.exists(img_path):
            print(f"找不到对应的图像文件: {img_file}")
            continue

        # 获取图像尺寸
        with Image.open(img_path) as img:
            width, height = img.size

        # 图像信息
        image_info = {
            "id": image_id,
            "file_name": img_file,
            "width": width,
            "height": height
        }
        images.append(image_info)

        # 解析YOLO标注
        with open(os.path.join(yolo_dir, txt_file), 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            category_id = int(parts[0])+1
            polygons = list(map(float, parts[1:]))
            xywh = poly2xywh(polygons)
            x_center, y_center, bbox_width, bbox_height = xywh

            # 计算边界框的绝对坐标
            abs_x = x_center * width
            abs_y = y_center * height
            abs_w = bbox_width * width
            abs_h = bbox_height * height

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [abs_x - abs_w / 2, abs_y - abs_h / 2, abs_w, abs_h],
                "area": abs_w * abs_h,
                "iscrowd": 0
            }
            annotations.append(annotation)
            annotation_id += 1


    coco_format_json = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_file, 'w') as f:
        json.dump(coco_format_json, f)
    print(f'save to {output_file}')

def coco_val(annFile, resFile):
    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(resFile)

    # 先评估整体性能 (area='all')
    cocoEval = COCOeval_diy(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == '__main__':
    # 示例调用
    categories = [
        {"id": 1, "name": "wall frame"},
        {"id": 2, "name": "wall display"},
        {"id": 3, "name": "projecting frame"},
        {"id": 4, "name": "projecting display"},
        {"id": 5, "name": "hanging frame"},
        {"id": 6, "name": "hanging display"},
        {"id": 7, "name": "other"},
    ]  # 根据实际情况修改

    data_dir = r'/localnvme/data/billboard/fused_data/data3899_seg_c5_0818'

    # select_val(data_dir, val_txt='val.txt')
    select_val(data_dir, val_txt='val_80p.txt')
    yolo_to_coco(
        os.path.join(data_dir, 'val', 'labels'),
        os.path.join(data_dir, 'val', 'images'),
        os.path.join(data_dir, 'val', "coco_annotations.json"),
        categories)

    coco_val(
        os.path.join(data_dir, 'val', 'coco_annotations.json'),
        r'/localnvme/project/ultralytics/runs/segment/val174/predictions.json',
    )


