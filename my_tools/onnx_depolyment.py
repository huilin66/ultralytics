# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
import os.path

import cv2
import numpy as np
import time
from typing import List, Optional, Union
from collections import defaultdict
import onnxruntime as ort

CLASSES = [
    'background',
    'wall signboard',
    'projecting signboard',
]
ATTRIBUTES = [
    'surface_missing',
    'surface_incomplete',
    'surface_corroded',
    'frame_corroded',
    'surface_peeling',
    'surface_fade',
    'surface_deformed',
    'frame_deformed',
    'disconnected',
    'added_billboard'
]
LEVELS = [
    'no',
    'yes',
]
# CLASSES = [
#     'background',
#     'wall frame',
#     'wall display',
#     'projecting frame',
#     'projecting display',
#     'hanging frame',
#     'hanging display',
#     'other'
# ]
# ATTRIBUTES = [
#     'deformation',
#     'broken',
#     'abandonment',
#     'corrosion'
# ]
# LEVELS = [
#     'no',
#     'medium',
#     'high'
# ]


# region utils

def crop_mask(masks, boxes):
    """
    Crop masks to bounding boxes.

    Args:
        masks (np.ndarray): [n, h, w] array of masks.
        boxes (np.ndarray): [n, 4] array of bbox coordinates in relative point form.

    Returns:
        (np.ndarray): Cropped masks.
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, axis=1)  # x1 shape(n,1,1)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    Rescale bounding boxes from img1_shape to img0_shape.

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (np.ndarray): The bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2).
        img0_shape (tuple): The shape of the target image, in the format of (height, width).
        ratio_pad (tuple): A tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
        xywh (bool): The box format is xywh or not.

    Returns:
        (np.ndarray): The scaled bounding boxes, in the format of (x1, y1, x2, y2).
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes = boxes.copy()  # Create a copy to avoid modifying the original array
    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


def clip_boxes(boxes, shape):
    """
    Clip bounding boxes to image shape (height, width).

    Args:
        boxes (np.ndarray): Bounding boxes to clip, in (x1, y1, x2, y2) format.
        shape (tuple): Image shape (height, width).

    Returns:
        (np.ndarray): Clipped bounding boxes.
    """
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes


def non_max_suppression_with_attributes(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        na=0,
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
        in_place=True,
        rotated=False,
        end2end=False,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes using NumPy.

    Args:
        prediction (np.ndarray): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
        conf_thres (float): Confidence threshold
        iou_thres (float): IoU threshold for NMS
        classes (List[int]): Filter by class
        agnostic (bool): Class-agnostic NMS
        multi_label (bool): Allow multiple labels per box
        labels (List[List[Union[int, float, np.ndarray]]]): A priori labels
        max_det (int): Maximum number of detections
        nc (int): Number of classes
        max_time_img (float): Max time per image
        max_nms (int): Maximum boxes into NMS
        max_wh (int): Maximum box width and height
        in_place (bool): Modify prediction in place
        rotated (bool): Use rotated boxes
        end2end (bool): Model doesn't require NMS

    Returns:
        List[np.ndarray]: List of detections per image
    """
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]  # select only inference output

    if classes is not None:
        classes = np.array(classes)

    if prediction.shape[-1] == 6+na or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[np.isin(pred[:, 5:6], classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4 - na  # number of masks
    ai = 4 + nc # attribute start index
    mi = 4 + nc + na  # mask start index
    xc = np.amax(prediction[:, 4:mi], axis=1) > conf_thres

    # Settings
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box

    prediction = np.transpose(prediction, (0, 2, 1))  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = np.concatenate((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), axis=-1)

    t = time.time()
    output = [np.zeros((0, 6 + nm  + na))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = np.zeros((len(lb), nc + nm + 4))
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[np.arange(len(lb)), lb[:, 0].astype(int) + 4] = 1.0  # cls
            x = np.concatenate((x, v), axis=0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box = x[:, :4]  # Slice first 4 columns
        cls = x[:, 4:4 + nc]  # Slice next nc columns
        attribute = x[:, 4 + nc:4 + nc + na]  # Slice next na columns
        mask = x[:, 4 + nc + na:]  # Slice remaining columns (nm)

        if multi_label:
            i, j = np.where(cls > conf_thres)
            x = np.concatenate((box[i], x[i, 4 + j, None], j[:, None].astype(float), mask[i]), axis=1)
        else:  # best class only
            conf = np.max(cls, axis=1, keepdims=True)
            j = np.argmax(cls, axis=1, keepdims=True)
            x = np.concatenate((box, conf, j.astype(float), attribute, mask), axis=1)
            x = x[conf.flatten() > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[np.isin(x[:, 5:6], classes).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[np.argsort(-x[:, 4])[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores

        if rotated:
            boxes = np.concatenate((x[:, :2] + c, x[:, 2:4], x[:, -1:]), axis=-1)  # xywhr
            i = nms_rotated_np(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            i = nms_np(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f"WARNING: NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output


def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def nms_np(boxes, scores, iou_threshold):
    """Pure NumPy NMS for axis-aligned boxes."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep)


def nms_rotated_np(boxes, scores, iou_threshold):
    """Pure NumPy NMS for rotated boxes (placeholder - implement as needed)."""
    # This is a placeholder - actual rotated NMS implementation would be more complex
    return np.arange(len(boxes))  # dummy implementation
# endregion

class YOLOBaseDeployer:
    def __init__(self, onnx_model, save_dir, conf=0.25, iou=0.7, imgsz=640, classes=CLASSES):
        self.session = ort.InferenceSession(
            onnx_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.save_dir = save_dir
        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        self.classes = classes
        self.conf = conf
        self.iou = iou
        self.nc = len(classes)
        _, _, self.input_width, self.input_height = self.session.get_inputs()[0].shape
        self.color_palette = [
            (255, 42, 4),
            (79, 68, 255),
            (255, 0, 189),
            (255, 180, 0),
            (186, 0, 221),
            (0, 192, 38),
            (255, 36, 125),
            (104, 0, 123),
            (108, 27, 255),
            (47, 109, 252),
            (104, 31, 17),
        ]
    def __call__(self, input_path):
        """
        Run inference on the input image using the ONNX model.

        Args:
            img (np.ndarray): The original input image in BGR format.

        Returns:
            (List[Results]): Processed detection results after post-processing, containing bounding boxes and
                segmentation masks.
        """
        if os.path.isdir(input_path):
            file_list = [os.path.join(input_path, file_name) for file_name in sorted(os.listdir(input_path))]
            results = []
            for idx, file_path in enumerate(file_list):
                print(f'processing {idx}/{len(file_list)}:')
                img_data, pad = self.preprocess(file_path, self.imgsz)
                outs = self.session.run(None, {self.session.get_inputs()[0].name: img_data})
                result = self.postprocess(self.img, img_data, outs, file_path)
                results.append(result)
            results.append(result)
        else:
            img_data, pad = self.preprocess(input_path, self.imgsz)
            outs = self.session.run(None, {self.session.get_inputs()[0].name: img_data})
            results = self.postprocess(self.img, img_data, outs, input_path)
        return results

    def letterbox(self, img, new_shape=(640, 640)):
        """
        Resize and pad image while maintaining aspect ratio.

        Args:
            img (np.ndarray): Input image in BGR format.
            new_shape (Tuple[int, int]): Target shape as (height, width).

        Returns:
            (np.ndarray): Resized and padded image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, (top, left)

    def preprocess(self, img_path, new_shape):
        """
        Preprocess the input image before feeding it into the model.

        Args:
            img (np.ndarray): The input image in BGR format.
            new_shape (Tuple[int, int]): The target shape for resizing as (height, width).

        Returns:
            (np.ndarray): Preprocessed image ready for model inference, with shape (1, 3, height, width) and normalized.
        """
        self.img = cv2.imread(img_path)
        self.img_height, self.img_width = self.img.shape[:2]

        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        img, pad = self.letterbox(img, new_shape)
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data, pad

    def postprocess(self, img, prep_img, outs, img_path):
        raise NotImplementedError("This deployer doesn't support postprocess")

    def draw_result(self, img, boxs, scores, class_ids, attributes, masks, img_path, alpha=0.5) -> None:
        raise NotImplementedError("This deployer doesn't support draw_result")

class YOLOMDetDeployer(YOLOBaseDeployer):
    def __init__(self, onnx_model, save_dir, conf=0.4, iou=0.1, imgsz=640, classes=CLASSES, attributes=ATTRIBUTES, levels=LEVELS):
        """
        Initialize the instance segmentation model using an ONNX model.

        Args:
            onnx_model (str): Path to the ONNX model file.
            conf (float): Confidence threshold for filtering detections.
            iou (float): IoU threshold for non-maximum suppression.
            imgsz (int | Tuple[int, int]): Input image size of the model. Can be an integer for square input or a tuple
                for rectangular input.
        """
        self.session = ort.InferenceSession(
            onnx_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            # providers=["CPUExecutionProvider"],
        )
        self.save_dir = save_dir
        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        self.classes = classes
        self.attributes = attributes
        self.levels = levels
        self.conf = conf
        self.iou = iou
        self.nc = len(classes)
        self.na = len(attributes)
        self.nl = len(levels)
        _, _, self.input_width, self.input_height = self.session.get_inputs()[0].shape
        self.color_palette = [
            (255, 42, 4),
            (79, 68, 255),
            (255, 0, 189),
            (255, 180, 0),
            (186, 0, 221),
            (0, 192, 38),
            (255, 36, 125),
            (104, 0, 123),
            (108, 27, 255),
            (47, 109, 252),
            (104, 31, 17),
        ]


    def postprocess(self, img, prep_img, outs, img_path):
        cat_feature, preds = outs[0], outs[1:4]
        preds = non_max_suppression_with_attributes(cat_feature, self.conf, self.iou, nc=self.nc, na=self.na)

        results = []
        for i, pred in enumerate(preds):
            pred[:, :4] = scale_boxes(prep_img.shape[2:], pred[:, :4], img.shape)
            boxs = pred[:, :4]
            scores = pred[:, 4:5]
            class_ids = pred[:, 5:6]
            attributes = pred[:,  6:6+self.na]
            attributes = np.floor(attributes * (self.nl)).astype(np.int64)
            attributes = np.clip(attributes, 0, self.nl-1)  # Ensure no value exceeds 2


            vis_img = self.draw_result(img, boxs, scores, class_ids, attributes, img_path)


            object_results = []
            for i in range(len(boxs)):
                box, score, class_id, attribute = boxs[i], scores[i], class_ids[i],  attributes[i]
                box = np.array(box, dtype=np.int32)
                class_name = self.classes[int(class_id)]
                attribute_result = [f'{self.attributes[idx_att]} : {self.levels[int(level_att)]}' for idx_att, level_att
                                    in enumerate(attribute)]
                object_result = (box, score, class_name, attribute_result)
                object_results.append(object_result)

            return object_results

    def draw_result(self, img, boxs, scores, class_ids, attributes, img_path, alpha=0.5, uids=None) -> None:
        colors = np.array([self.color_palette[int(i)] for i in class_ids]) / 255.0


        for idx, (box, score, class_id, attribute, color) in enumerate(zip(boxs, scores, class_ids, attributes, colors)):
            uid = uids[idx] if uids is not None else None
            class_id = int(class_id)
            score = float(score)
            # Extract the coordinates of the bounding box
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Retrieve the color for the class ID
            color = self.color_palette[class_id]

            # Draw the bounding box on the image
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Create the label text with class name and score
            label = f"{self.classes[class_id]}: {score:.2f}" if uid is None else f"{self.classes[class_id]}: {score:.2f} | uid:{uid}"

            # Calculate the dimensions of the label text
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Calculate the position of the label text
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

            # Draw a filled rectangle as the background for the label text
            cv2.rectangle(
                img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
            )

            # Draw the label text on the image
            cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Draw attibutes:
            p1, p2 = (int(box[0]), int(box[1])), (int(box[0])+15*len(attribute), int(box[1])+15*len(attribute))
            cv2.rectangle(img, p1, p2, (100, 100, 100))
            overlay = img.copy()
            cv2.rectangle(overlay, p1, p2, color, -1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            for idx, att in enumerate(attribute):
                att_name = self.attributes[idx]
                att_level = self.levels[int(att)]
                att_label = f'{att_name}:{att_level}'
                pos = [x1, y1 + 15 * (idx + 1) - 10]
                cv2.putText(img, att_label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


        cv2.imwrite(os.path.join(self.save_dir, os.path.basename(img_path)), img)
        return img

class YOLOMSegDeployer(YOLOBaseDeployer):
    """
    YOLOv8 segmentation model for performing instance segmentation using ONNX Runtime.

    This class implements a YOLOv8 instance segmentation model using ONNX Runtime for inference. It handles
    preprocessing of input images, running inference with the ONNX model, and postprocessing the results to
    generate bounding boxes and segmentation masks.

    Attributes:
        session (ort.InferenceSession): ONNX Runtime inference session for model execution.
        imgsz (Tuple[int, int]): Input image size as (height, width) for the model.
        classes (Dict): Dictionary mapping class indices to class names from the dataset.
        conf (float): Confidence threshold for filtering detections.
        iou (float): IoU threshold used by non-maximum suppression.
    """

    def __init__(self, onnx_model, save_dir, conf=0.25, iou=0.7, imgsz=640, classes=CLASSES, attributes=ATTRIBUTES, levels=LEVELS):
        """
        Initialize the instance segmentation model using an ONNX model.

        Args:
            onnx_model (str): Path to the ONNX model file.
            conf (float): Confidence threshold for filtering detections.
            iou (float): IoU threshold for non-maximum suppression.
            imgsz (int | Tuple[int, int]): Input image size of the model. Can be an integer for square input or a tuple
                for rectangular input.
        """
        self.session = ort.InferenceSession(
            onnx_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            # providers=["CPUExecutionProvider"],
        )
        self.save_dir = save_dir
        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        self.classes = classes
        self.attributes = attributes
        self.levels = levels
        self.conf = conf
        self.iou = iou
        self.nc = len(classes)
        self.na = len(attributes)
        self.nl = len(levels)
        _, _, self.input_width, self.input_height = self.session.get_inputs()[0].shape
        self.color_palette = [
            (255, 42, 4),
            (79, 68, 255),
            (255, 0, 189),
            (255, 180, 0),
            (186, 0, 221),
            (0, 192, 38),
            (255, 36, 125),
            (104, 0, 123),
            (108, 27, 255),
            (47, 109, 252),
            (104, 31, 17),
        ]


    def postprocess(self, img, prep_img, outs, img_path):
        """
        Post-process model predictions to extract meaningful results.

        Args:
            img (np.ndarray): The original input image.
            prep_img (np.ndarray): The preprocessed image used for inference.
            outs (List): Model outputs containing predictions and prototype masks.

        Returns:
            (List[Results]): Processed detection results containing bounding boxes and segmentation masks.
        """
        '''
        x:
            1 * 48 * 8400;
            1 * 76 * 80 * 80;
            1 * 76 * 40 * 40;
            1 * 76 * 20 * 20;
        mc: 
            1 * 32 * 8400;
        p:
            1 * 32 * 160
        '''

        cat_feature, preds, mask_coefficients,protos = outs[0], outs[1:4], outs[4], outs[5]
        preds = non_max_suppression_with_attributes(cat_feature, self.conf, self.iou, nc=self.nc, na=self.na)

        results = []
        for i, pred in enumerate(preds):
            pred[:, :4] = scale_boxes(prep_img.shape[2:], pred[:, :4], img.shape)
            boxs = pred[:, :4]
            scores = pred[:, 4:5]
            class_ids = pred[:, 5:6]
            attributes = pred[:,  6:6+self.na]
            attributes = np.floor(attributes * (self.nl)).astype(np.int64)
            attributes = np.clip(attributes, 0, self.nl-1)  # Ensure no value exceeds 2
            masks = self.process_mask(protos[i], pred[:, 6+self.na:], pred[:, :4], img.shape[:2], upsample=True)
            if masks is not None:
                keep = masks.sum((-2, -1)) > 0  # only keep predictions with masks
                boxs, scores, class_ids, attributes, masks = boxs[keep], scores[keep], class_ids[keep], attributes[keep], masks[keep]

            vis_img = self.draw_result(img, boxs, scores, class_ids, attributes, masks, img_path)

            masks = masks.astype(np.uint8) * 255
            object_results = []
            for i in range(len(boxs)):
                box, score, class_id, mask, attribute = boxs[i], scores[i], class_ids[i], masks[i], attributes[i]
                box = np.array(box, dtype=np.int32)
                class_name = self.classes[int(class_id)]
                attribute_result = [f'{self.attributes[idx_att]} : {self.levels[int(level_att)]}' for idx_att, level_att
                                    in enumerate(attribute)]
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                mask_contour_list = [cnt.squeeze().tolist() for cnt in contours if len(cnt) >= 3]
                object_result = (box, score, class_name, mask_contour_list, attribute_result)
                object_results.append(object_result)

            return object_results

    def draw_result(self, img, boxs, scores, class_ids, attributes, masks, img_path, alpha=0.5) -> None:
        colors = np.array([self.color_palette[int(i)] for i in class_ids]) / 255.0

        masks_color = masks[..., None] * (colors[:, None, None, :] * alpha)

        inv_alpha_masks = np.cumprod(1 - masks[..., None] * alpha, axis=0)

        mcs = np.max(masks_color, axis=0)

        img = img * inv_alpha_masks[-1] + mcs*255  # Use last cumulative product

        img = (np.clip(img, 0, 255)).astype(np.uint8)


        for box, score, class_id, attribute, color in zip(boxs, scores, class_ids, attributes, colors):
            class_id = int(class_id)
            score = float(score)
            # Extract the coordinates of the bounding box
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Retrieve the color for the class ID
            color = self.color_palette[class_id]

            # Draw the bounding box on the image
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Create the label text with class name and score
            label = f"{self.classes[class_id]}: {score:.2f}"

            # Calculate the dimensions of the label text
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Calculate the position of the label text
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

            # Draw a filled rectangle as the background for the label text
            cv2.rectangle(
                img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
            )

            # Draw the label text on the image
            cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Draw attibutes:
            p1, p2 = (int(box[0]), int(box[1])), (int(box[0])+15*len(attribute), int(box[1])+15*len(attribute))
            cv2.rectangle(img, p1, p2, (100, 100, 100))
            overlay = img.copy()
            cv2.rectangle(overlay, p1, p2, color, -1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            for idx, att in enumerate(attribute):
                att_name = self.attributes[idx]
                att_level = self.levels[int(att)]
                att_label = f'{att_name}:{att_level}'
                pos = [x1, y1 + 15 * (idx + 1) - 10]
                cv2.putText(img, att_label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


        cv2.imwrite(os.path.join(self.save_dir, os.path.basename(img_path)), img)
        return img

    def process_mask(self, protos, masks_in, bboxes, shape, upsample=False):
        """
        Apply masks to bounding boxes using the output of the mask head.

        Args:
            protos (np.ndarray): A array of shape [mask_dim, mask_h, mask_w]
            masks_in (np.ndarray): A array of shape [n, mask_dim]
            bboxes (np.ndarray): A array of shape [n, 4]
            shape (tuple): Input image shape (h, w)
            upsample (bool): Whether to upsample the mask

        Returns:
            np.ndarray: Binary mask array of shape [n, h, w]
        """
        c, mh, mw = protos.shape  # CHW
        ih, iw = shape

        # Matrix multiplication equivalent
        masks = np.matmul(masks_in, protos.reshape(c, -1)).reshape(-1, mh, mw)

        # Scale bboxes to mask dimensions
        width_ratio = mw / iw
        height_ratio = mh / ih

        downsampled_bboxes = bboxes.copy()
        downsampled_bboxes[:, [0, 2]] *= width_ratio  # x1, x2
        downsampled_bboxes[:, [1, 3]] *= height_ratio  # y1, y2

        masks = crop_mask(masks, downsampled_bboxes)

        if upsample:
            # Resize each mask individually
            resized_masks = np.zeros((masks.shape[0], ih, iw))
            for i, mask in enumerate(masks):
                resized_masks[i] = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_LINEAR)
            masks = resized_masks

        return masks > 0.0  # Binary mask



class YOLOMDetTrackDeployer(YOLOMDetDeployer):
    def __init__(self, onnx_model, save_dir,max_age):
        super(YOLOMDetTrackDeployer, self).__init__(onnx_model, save_dir)
        self.max_age = max_age
        self.tracks = defaultdict(dict)
        self.next_id = 0
        self.frame_count = 0

    def postprocess(self, img, prep_img, outs, img_path):
        cat_feature, preds = outs[0], outs[1:4]
        preds = non_max_suppression_with_attributes(cat_feature, self.conf, self.iou, nc=self.nc, na=self.na)

        results = []
        for i, pred in enumerate(preds):
            pred[:, :4] = scale_boxes(prep_img.shape[2:], pred[:, :4], img.shape)
            boxs = pred[:, :4]
            scores = pred[:, 4:5]
            class_ids = pred[:, 5:6]
            attributes = pred[:,  6:6+self.na]
            attributes = np.floor(attributes * (self.nl)).astype(np.int64)
            attributes = np.clip(attributes, 0, self.nl-1)  # Ensure no value exceeds 2

            object_results = []
            for i in range(len(boxs)):
                box, score, class_id, attribute = boxs[i], scores[i], class_ids[i],  attributes[i]
                box = np.array(box, dtype=np.int32)
                class_name = self.classes[int(class_id)]
                attribute_result = [f'{self.attributes[idx_att]} : {self.levels[int(level_att)]}' for idx_att, level_att
                                    in enumerate(attribute)]
                object_result = (box, score, class_name, attribute_result)
                object_results.append(object_result)

            uids = self.track_update(object_results)
            print(uids)
            vis_img = self.draw_result(img, boxs, scores, class_ids, attributes, img_path, uids=uids)

            return object_results

    def track_update(self, frame_result):
        """更新跟踪状态并返回每个box的UID"""
        self.frame_count += 1
        uid_to_box = {}  # 最终输出：{UID: box}的字典
        # 1. 准备当前帧检测数据
        current_detections = [
            {'box': record[0], 'score': record[1], 'class': record[2], 'attribute': record[3]}
            for record in frame_result
        ]
        # 2. 初始化未匹配的检测索引
        unmatched_det_indices = set(range(len(current_detections)))

        # 3. 优先匹配现有轨迹
        for track_id, track_info in list(self.tracks.items()):
            # 清理过期轨迹（超过max_age帧未更新）
            if self.frame_count - track_info['last_seen'] > self.max_age:
                del self.tracks[track_id]
                continue

            # 寻找最佳IOU匹配
            best_iou, best_idx = 0, -1
            for idx in list(unmatched_det_indices):
                iou = self._calculate_iou(
                    track_info['box'],
                    current_detections[idx]['box']
                )
                if iou > self.iou and iou > best_iou:
                    best_iou, best_idx = iou, idx

            # 更新匹配成功的轨迹
            if best_idx != -1:
                self.tracks[track_id] = {
                    'box': current_detections[best_idx]['box'],
                    'last_seen': self.frame_count
                }
                uid_to_box[track_id] = current_detections[best_idx]['box']
                unmatched_det_indices.remove(best_idx)

        # 4. 关键步骤：为未匹配的检测创建新轨迹
        for idx in unmatched_det_indices:
            new_id = self.next_id
            self.tracks[new_id] = {
                'box': current_detections[idx]['box'],
                'last_seen': self.frame_count
            }
            uid_to_box[new_id] = current_detections[idx]['box']
            self.next_id += 1  # 确保ID唯一递增

        # 5. 按原始检测顺序返回UID列表
        uids = []
        for det in current_detections:
            for uid, box in uid_to_box.items():
                if np.array_equal(box, det['box']):
                    uids.append(uid)
                    break
        return uids
        # return [
        #     next((uid for uid, box in uid_to_box.items() if np.array_equal(box, det['box'])) for det in current_detections
        # ]

        active_ids = []
        current_detections = []
        result_uids = []
        for record in result:
            current_det = {'box': record[0], 'score': record[1], 'class': record[2], 'attribute': record[3], }
            current_detections.append(current_det)
            # 匹配现有轨迹
            matched_pairs = []
            used_detections = set()  # 改用集合记录已匹配的检测索引
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
                    if iou > self.iou and iou > best_iou:
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

            # 6. 返回格式: 每个box对应的UID列表（与输入frame_result顺序一致）

            for det in current_detections:
                # 查找这个box对应的UID
                found_uid = None
                for uid, box in self.tracks.items():
                    if np.array_equal(box['box'], det['box']):
                        found_uid = uid
                        break
                result_uids.append(found_uid)
        return result_uids

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
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, default=r'/nfsv4/23039356r/repository/ultralytics/runs/msegment/debug108/weights/best.onnx', help="Path to ONNX model")
    parser.add_argument("--model", type=str, default=r'/nfsv4/23039356r/repository/ultralytics/runs/mdetect/exp_yolo10x_head1231_20/weights/best.onnx', help="Path to ONNX model")
    # parser.add_argument("--source", type=str, default=r'/nfsv4/23039356r/data/billboard/bd_data/selected_sample/images', help="Path to input image")
    parser.add_argument("--source", type=str, default=r'/nfsv4/23039356r/data/billboard/data0806_m/demo2', help="Path to input image")
    # parser.add_argument("--save_dir", type=str, default=r'/nfsv4/23039356r/data/billboard/bd_data/selected_sample/results', help="Confidence threshold")
    parser.add_argument("--save_dir", type=str,default=r'/nfsv4/23039356r/data/billboard/data0806_m/demo_result2',help="Confidence threshold")
    args = parser.parse_args()

    # model = YOLOMSegDeployer(args.model, args.save_dir, args.conf, args.iou)
    # model = YOLOMDetDeployer(args.model, args.save_dir, args.conf, args.iou)
    model = YOLOMDetTrackDeployer(args.model, args.save_dir, 5)
    results = model(args.source)
    print(results)

