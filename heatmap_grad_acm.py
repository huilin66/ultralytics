import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import inspect
import torch, yaml, cv2, os, shutil, sys
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy, non_max_suppression
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients


# Stable palette copied from the project's visualization convention.  The
# source palette is defined in OpenCV's BGR order; CAM images in this script
# are RGB arrays, so the palette is reversed once here instead of importing
# the external visualization package.
_CV2_COLORS_BGR = (
    (255, 42, 4),
    (235, 219, 11),
    (243, 243, 243),
    (183, 223, 0),
    (104, 31, 17),
    (221, 111, 255),
    (79, 68, 255),
    (0, 237, 204),
    (68, 243, 0),
    (255, 0, 189),
    (255, 180, 0),
    (186, 0, 221),
    (255, 255, 0),
    (0, 192, 38),
    (179, 255, 1),
    (255, 36, 125),
    (104, 0, 123),
    (108, 27, 255),
    (47, 109, 252),
    (11, 255, 162),
)
_HEATMAP_COLORS_RGB = tuple(tuple(reversed(color)) for color in _CV2_COLORS_BGR)
_DARK_LABEL_TEXT_RGB = (17, 31, 104)


def _heatmap_line_width(image):
    """Use the same image-size-dependent line width as the project renderer."""
    return max(round(sum(image.shape) / 2 * 0.003), 2)


def _heatmap_text_color(color):
    """Select readable label text for the RGB version of the stable palette."""
    dark_colors_rgb = {
        tuple(reversed(color))
        for color in (
            (235, 219, 11),
            (243, 243, 243),
            (183, 223, 0),
            (221, 111, 255),
            (0, 237, 204),
            (68, 243, 0),
            (255, 255, 0),
            (179, 255, 1),
            (11, 255, 162),
        )
    }
    return _DARK_LABEL_TEXT_RGB if tuple(color) in dark_colors_rgb else (255, 255, 255)


def _draw_heatmap_label(image, x, y, text, color, line_width):
    """Draw a filled, boundary-aware top-left label on an RGB CAM image."""
    if not text:
        return

    height, width = image.shape[:2]
    font_thickness = max(line_width - 1, 1)
    font_scale = line_width / 3
    (text_width, text_height), baseline = cv2.getTextSize(
        str(text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )
    pad = 2
    rect_width = text_width + pad * 2
    rect_height = text_height + baseline + pad * 2
    anchor_x = max(0, min(int(round(x)), max(width - 1, 0)))
    anchor_y = max(0, min(int(round(y)), max(height - 1, 0)))

    # Prefer the label above the box, and move it inside/below the box when
    # the box touches the upper image boundary.
    outside = anchor_y >= rect_height
    if outside and anchor_y - rect_height < 0:
        outside = False
    elif not outside and anchor_y + rect_height >= height:
        outside = True

    left = max(0, min(anchor_x, max(width - rect_width, 0)))
    if outside:
        top = max(0, anchor_y - rect_height)
        bottom = min(height - 1, anchor_y)
        text_baseline = max(top + text_height + pad, bottom - baseline - pad)
    else:
        top = min(max(anchor_y, 0), max(height - rect_height, 0))
        bottom = min(height - 1, top + rect_height)
        text_baseline = min(bottom - baseline - pad, top + text_height + pad)
    right = min(width - 1, left + rect_width)

    color = tuple(int(value) for value in color)
    cv2.rectangle(image, (left, top), (right, bottom), color, -1, cv2.LINE_AA)
    cv2.putText(
        image,
        str(text),
        (left + pad, int(text_baseline)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        _heatmap_text_color(color),
        thickness=font_thickness,
        lineType=cv2.LINE_AA,
    )


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()
  
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        post_result, pre_post_boxes, post_boxes = self.post_process(model_output[0])
        return [[post_result, pre_post_boxes]]

    def release(self):
        for handle in self.handles:
            handle.remove()

class yolov8_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio
    
    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result)

class yolov8_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_box, renormalize):
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        model = attempt_load_weights(weight, device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()
        
        target = yolov8_target(backward_type, conf_threshold, ratio)
        target_layers = [model.model[l] for l in layer]
        method_class = eval(method)
        method_parameters = inspect.signature(method_class).parameters
        method_kwargs = {}
        if 'use_cuda' in method_parameters:
            method_kwargs['use_cuda'] = device.type == 'cuda'
        method = method_class(model, target_layers, **method_kwargs)
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)
        
        colors = np.asarray(
            [_HEATMAP_COLORS_RGB[index % len(_HEATMAP_COLORS_RGB)] for index in range(len(model_names))],
            dtype=np.uint8,
        )
        self.__dict__.update(locals())
    
    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65)[0]
        return result

    def draw_detections(self, box, color, name, img):
        values = np.asarray(box, dtype=np.float32).reshape(-1)[:4]
        if values.size != 4 or not np.isfinite(values).all():
            return img

        height, width = img.shape[:2]
        xmin, ymin, xmax, ymax = values.tolist()
        xmin, xmax = sorted((xmin, xmax))
        ymin, ymax = sorted((ymin, ymax))
        xmin = int(np.clip(round(xmin), 0, max(width - 1, 0)))
        ymin = int(np.clip(round(ymin), 0, max(height - 1, 0)))
        xmax = int(np.clip(round(xmax), 0, max(width - 1, 0)))
        ymax = int(np.clip(round(ymax), 0, max(height - 1, 0)))
        if xmax <= xmin or ymax <= ymin:
            return img

        line_width = _heatmap_line_width(img)
        draw_color = tuple(int(value) for value in color)
        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            draw_color,
            thickness=line_width,
            lineType=cv2.LINE_AA,
        )
        _draw_heatmap_label(img, xmin, ymin, name, draw_color, line_width)
        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1] 
        inside every bounding boxes, and zero outside of the bounding boxes. """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            if not np.isfinite((x1, y1, x2, y2)).all():
                continue
            x1, x2 = sorted((int(np.floor(x1)), int(np.ceil(x2))))
            y1, y2 = sorted((int(np.floor(y1)), int(np.ceil(y2))))
            x1 = max(0, min(x1, grayscale_cam.shape[1]))
            y1 = max(0, min(y1, grayscale_cam.shape[0]))
            x2 = max(0, min(x2, grayscale_cam.shape[1]))
            y2 = max(0, min(y2, grayscale_cam.shape[0]))
            if x2 <= x1 or y2 <= y1:
                continue
            region = grayscale_cam[y1:y2, x1:x2].copy()
            if region.size == 0:
                continue
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(region)
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized
    
    def process(self, img_path, save_path):
        # img process
        img = cv2.imread(img_path)
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
        
        try:
            grayscale_cam = self.method(tensor, [self.target])
        except AttributeError as e:
            return
        
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        
        pred = self.model(tensor)[0]
        pred = self.post_process(pred)
        if self.renormalize:
            cam_image = self.renormalize_cam_in_bounding_boxes(pred[:, :4].cpu().detach().numpy().astype(np.int32), img, grayscale_cam)
        if self.show_box:
            for data in pred:
                data = data.cpu().detach().numpy()
                cam_image = self.draw_detections(data[:4], self.colors[int(data[4:].argmax())], f'{self.model_names[int(data[4:].argmax())]} {float(data[4:].max()):.2f}', cam_image)
        
        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path)
    
    def __call__(self, img_path, save_path, grad_name):
        # remove dir if exist
        # if os.path.exists(save_path):
        #     shutil.rmtree(save_path)
        # make dir if not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(img_path):
            for img_path_ in os.listdir(img_path):
                name = img_path_.rsplit('.')[0]
                end_name = img_path_.rsplit('.')[-1]
                self.process(f'{img_path}/{img_path_}', f'{save_path}/{name}_{grad_name}.{end_name}')
        else:
            self.process(img_path, f'{save_path}/result_{grad_name}.png')
        
def get_params():
    # 绘制热力图方法列表
    grad_list = [
        'GradCAM',
        'GradCAMPlusPlus',
        'XGradCAM',
        'EigenCAM',
        'HiResCAM',
        'LayerCAM',
        'RandomCAM',
        'EigenGradCAM'
    ]
    # 自定义需要绘制热力图的层索引，可以用列表绘制不同层的热力图,如[10, 12, 14, 16, 18]，将多层的话会将结果进行汇总到一张图上
    layers = [10, 12, 14, 16, 18]
    for grad_name in grad_list:
        params = {
            'weight': r'/scrinvme/huilin/tp/FLIR1444_img_mayolo', # 训练好的权重路径
            'device': 'cuda:0',  # cpu或者cuda:0
            'method': grad_name, # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
            'layer': layers,  # 计算梯度的层, 指定层的索引
            'backward_type': 'class', # class, box, all
            'conf_threshold': 0.2, # 置信度阈值默认0.2, 根据情况调节
            'ratio': 0.02, # 建议0.02-0.1，取前多少数据，默认是0.02，只取置信度排序后的前百分之2的目标进行计算热力图。
            'show_box': False,  #是否显示检测框
            'renormalize': True   #是否优化热力图显示结果
        }
        yield params

if __name__ == '__main__':
    for each in get_params():
        model = yolov8_heatmap(**each)
        # model第一个参数：单张图片路径或者图片文件夹路径; 第二个参数：保存路径; 第三个参数：绘制热力图方法
        # model(r'images/00052.jpg', 'result', each['method'])
        model(
        r'/scrinvme/huilin/tp/FLIR1444_img_mayolo/FLIR1444_img', 
        r'/scrinvme/huilin/tp/FLIR1444_img_mayolo_grad_acm', 
        each['method'])
