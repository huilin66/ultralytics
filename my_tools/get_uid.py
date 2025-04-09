import os

import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

BATCH_SIZE = 32
EPOCHS = 500
IMGSZ = 640
CONF = 0.5
TASK = 'detect'

import cv2


def resize_image(image, target_size=640):
    """将图像调整为target_size x target_size"""
    h, w = image.shape[:2]

    # 计算缩放比例
    scale = target_size / max(h, w)

    # 等比例缩放
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))

    # 填充到正方形
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # 添加黑色边框
    resized = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    return resized


class LightweightTracker:
    def __init__(self, model_path, iou_threshold=CONF, max_age=5, img_size=IMGSZ):
        """
        轻量级跟踪器 - 仅依赖YOLOv8

        参数:
            model_path: YOLOv8模型路径
            iou_threshold: IOU匹配阈值
            max_age: 目标最大丢失帧数
        """
        self.model = YOLO(model_path, task=TASK)
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = defaultdict(dict)
        self.next_id = 0
        self.frame_count = 0
        self.img_size = img_size

    def _calculate_iou(self, box1, box2):
        """计算两个bbox的IOU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter_area / (box1_area + box2_area - inter_area)

    def update(self, frame):
        """处理新的一帧"""
        self.frame_count += 1
        active_ids = []

        # 使用YOLOv8内置跟踪功能
        results = self.model.track(
            frame,
            persist=True,  # 保持跟踪状态
            verbose=False,
            tracker="botsort.yaml"  # YOLOv8内置的跟踪配置
        )

        # 获取检测结果
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []
        class_ids = results[0].boxes.cls.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        # 如果没有得到跟踪ID，使用自定义IOU匹配
        if len(track_ids) == 0:
            current_detections = [{'box': box, 'class': cls, 'score': score}
                                  for box, cls, score in zip(boxes, class_ids, scores)]

            # 匹配现有轨迹
            matched_pairs = []
            for track_id, track_info in list(self.tracks.items()):
                if self.frame_count - track_info['last_seen'] > self.max_age:
                    del self.tracks[track_id]
                    continue

                best_iou = 0
                best_idx = -1
                for idx, det in enumerate(current_detections):
                    if 'matched' in det:
                        continue
                    iou = self._calculate_iou(track_info['box'], det['box'])
                    if iou > self.iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_idx = idx

                if best_idx != -1:
                    matched_pairs.append((track_id, best_idx))
                    current_detections[best_idx]['matched'] = True

            # 更新匹配的轨迹
            for track_id, det_idx in matched_pairs:
                det = current_detections[det_idx]
                self.tracks[track_id] = {
                    'box': det['box'],
                    'class': det['class'],
                    'score': det['score'],
                    'last_seen': self.frame_count
                }
                active_ids.append(track_id)

            # 为新检测创建轨迹
            for idx, det in enumerate(current_detections):
                if 'matched' not in det:
                    self.tracks[self.next_id] = {
                        'box': det['box'],
                        'class': det['class'],
                        'score': det['score'],
                        'last_seen': self.frame_count
                    }
                    active_ids.append(self.next_id)
                    self.next_id += 1
        else:
            # 使用YOLOv8提供的跟踪ID
            for box, track_id, cls, score in zip(boxes, track_ids, class_ids, scores):
                self.tracks[track_id] = {
                    'box': box,
                    'class': cls,
                    'score': score,
                    'last_seen': self.frame_count
                }
                active_ids.append(track_id)

        return active_ids

    def visualize(self, frame, active_ids):
        """可视化跟踪结果"""
        vis_frame = frame.copy()
        for track_id in active_ids:
            if track_id in self.tracks:
                info = self.tracks[track_id]
                box = info['box']
                class_id = int(info['class'])
                score = info['score']

                # 绘制边界框
                cv2.rectangle(vis_frame,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              (0, 255, 0), 2)

                # 显示ID和类别
                label = f"ID:{track_id} {self.model.names[class_id]} {score:.2f}"
                cv2.putText(vis_frame, label,
                            (int(box[0]), int(box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return vis_frame


import cv2
import numpy as np
import onnxruntime as ort
from collections import defaultdict


class ONNXObjectTracker:
    def __init__(self, model_path, img_size=640, iou_thresh=0.5, max_age=5):
        """
        独立的目标跟踪器 - 仅需ONNX模型

        参数:
            onnx_path: 导出的YOLO ONNX模型路径
            img_size: 输入图像尺寸
            iou_thresh: IOU匹配阈值
            max_age: 目标最大丢失帧数
        """
        # 初始化ONNX运行时
        self.session = ort.InferenceSession(model_path)
        self.img_size = img_size
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks = defaultdict(dict)
        self.next_id = 0
        self.frame_count = 0

        # 获取输入输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def preprocess(self, image):
        """图像预处理"""
        # 保持宽高比的resize
        h, w = image.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(image, (new_w, new_h))

        # 添加padding
        top = (self.img_size - new_h) // 2
        bottom = self.img_size - new_h - top
        left = (self.img_size - new_w) // 2
        right = self.img_size - new_w - left

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # 转换为模型需要的格式
        blob = cv2.dnn.blobFromImage(
            padded, 1 / 255.0, (self.img_size, self.img_size),
            swapRB=True, crop=False
        )

        return blob, (scale, (left, top))

    def postprocess(self, outputs, scale, padding):
        """后处理检测结果"""
        boxes, scores, class_ids = [], [], []
        x_pad, y_pad = padding

        # 假设输出是[yolo_output]或[boxes, scores, class_ids]
        if len(outputs) == 1:  # YOLO格式输出
            predictions = np.squeeze(outputs[0])
            # 这里需要根据你的ONNX输出格式解析
            # 示例解析(需要根据实际模型调整):
            for pred in predictions:
                if pred[4] > 0.5:  # 置信度阈值
                    x, y, w, h = pred[:4]
                    x = (x - x_pad) / scale
                    y = (y - y_pad) / scale
                    w = w / scale
                    h = h / scale
                    boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
                    scores.append(pred[4])
                    class_ids.append(np.argmax(pred[5:]))
        else:  # 分离的输出
            boxes, scores, class_ids = outputs

        return boxes, scores, class_ids

    def update(self, frame):
        """更新跟踪状态"""
        self.frame_count += 1
        active_ids = []

        # 预处理
        blob, (scale, padding) = self.preprocess(frame)

        # ONNX推理
        outputs = self.session.run(
            self.output_names,
            {self.input_name: blob}
        )

        # 后处理
        boxes, scores, class_ids = self.postprocess(outputs, scale, padding)

        # 简单的IOU跟踪器
        current_detections = [
            {'box': box, 'score': score, 'class': class_id}
            for box, score, class_id in zip(boxes, scores, class_ids)
        ]

        # 匹配现有轨迹
        matched_pairs = []
        for track_id, track_info in list(self.tracks.items()):
            if self.frame_count - track_info['last_seen'] > self.max_age:
                del self.tracks[track_id]
                continue

            best_iou = 0
            best_idx = -1
            for idx, det in enumerate(current_detections):
                if 'matched' in det:
                    continue
                iou = self._calculate_iou(track_info['box'], det['box'])
                if iou > self.iou_thresh and iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx != -1:
                matched_pairs.append((track_id, best_idx))
                current_detections[best_idx]['matched'] = True

        # 更新匹配的轨迹
        for track_id, det_idx in matched_pairs:
            det = current_detections[det_idx]
            self.tracks[track_id] = {
                'box': det['box'],
                'class': det['class'],
                'score': det['score'],
                'last_seen': self.frame_count
            }
            active_ids.append(track_id)

        # 为新检测创建轨迹
        for idx, det in enumerate(current_detections):
            if 'matched' not in det:
                self.tracks[self.next_id] = {
                    'box': det['box'],
                    'class': det['class'],
                    'score': det['score'],
                    'last_seen': self.frame_count
                }
                active_ids.append(self.next_id)
                self.next_id += 1

        return active_ids

    def _calculate_iou(self, box1, box2):
        """计算IOU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter / (area1 + area2 - inter + 1e-6)

    def visualize(self, frame, active_ids):
        """可视化结果"""
        vis = frame.copy()
        for track_id in active_ids:
            if track_id in self.tracks:
                info = self.tracks[track_id]
                box = info['box']
                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis, f"ID:{track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
        return vis



# 使用示例
if __name__ == "__main__":
    model_path = r'/runs/detect/train14/weights/best.pt'
    onnx_path = r'/runs/detect/train14/weights/best.onnx'
    img_dir = r'/nfsv4/23039356r/data/billboard/data0806_m/demo'
    save_dir = r'/nfsv4/23039356r/data/billboard/data0806_m/demo_result'
    # 初始化跟踪器
    # tracker = LightweightTracker(model_path=model_path)
    tracker = ONNXObjectTracker(model_path=onnx_path)

    file_list = sorted(os.listdir(img_dir))
    # 模拟输入帧
    frames = [cv2.imread(os.path.join(img_dir, img_name)) for img_name in file_list]

    for i, frame in enumerate(frames):
        # 更新跟踪器
        active_ids = tracker.update(frame)

        # 可视化
        vis_frame = tracker.visualize(frame, active_ids)
        cv2.imwrite(os.path.join(save_dir, file_list[i]), vis_frame)
        # 显示结果
        # cv2.imshow("Tracking", vis_frame)
        # cv2.waitKey(500)

        print(f"Frame {i + 1} Active IDs: {active_ids}")

    cv2.destroyAllWindows()