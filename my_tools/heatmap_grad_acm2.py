import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import DetectionModel as Model
from ultralytics.utils.torch_utils import intersect_dicts
# from ultralytics.yolo.data.augment import LetterBox
from ultralytics.utils.ops import xywh2xyxy
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
# from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from ultralytics import YOLO

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
        if isinstance(output, tuple) or isinstance(output, list):
            activation = output[0]
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if isinstance(output, tuple) or isinstance(output, list):
            output = output[0]

        if not hasattr(output, "requires_grad"):
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        return model_output

    def release(self):
        for handle in self.handles:
            handle.remove()

def compute_iou(box1, box2):
    # box1: [x1, y1, x2, y2] (Target from Model A)
    # box2: [x1, y1, x2, y2] (Candidate from Model B)
    # 计算交集
    xx1 = np.maximum(box1[0], box2[0])
    yy1 = np.maximum(box1[1], box2[1])
    xx2 = np.minimum(box1[2], box2[2])
    yy2 = np.minimum(box1[3], box2[3])
    
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
    inter = w * h
    
    # 计算并集
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter + 1e-6
    
    return inter / union
from ultralytics.nn.tasks import attempt_load_weights
class yolov8_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold_max, conf_threshold_min, ratio):
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32

        model = attempt_load_weights(weight, device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()
        model.model[23].use_one2many_head()
        print(f'Transferred {len(csd)}/{len(model.state_dict())} items')

        # target_layers = [eval(model.model.model[layer])]
        # method = eval(method)
        target_layers = [model.model[layer]]
        method = eval(method)
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.uint8)
        self.__dict__.update(locals())

    def post_process(self, result):
        result = result[0]
        if result.shape[-1] == 16 and result.shape[0] == 300:
            scores = result[:, 4]
            boxes = result[:, :4]
            sorted_scores, indices = torch.sort(scores, descending=True)

            sorted_boxes = boxes[indices]
            sorted_result = result[indices]

            post_results = sorted_result[:, 4:5]

            pre_post_boxes = sorted_boxes

            post_boxes = xywh2xyxy(sorted_boxes).cpu().detach().numpy()

            return post_results, pre_post_boxes, post_boxes
        else:
            result = result.transpose(0, 1)
            boxes = result[:, :4]
            class_scores = result[:, 4:-10]

            scores, labels = torch.max(class_scores, dim=1)

            sorted_scores, indices = torch.sort(scores, descending=True)
            post_results = sorted_scores.unsqueeze(-1)

            sorted_boxes = boxes[indices]
            pre_post_boxes = sorted_boxes
            post_boxes = xywh2xyxy(sorted_boxes).cpu().detach().numpy()
            return post_results, pre_post_boxes, post_boxes

        # return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
        #     indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
                    lineType=cv2.LINE_AA)
        return img

    def __call__(self, img_path, save_path, target_box=None):
        # remove dir if exist
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        # make dir if not exist
        os.makedirs(save_path, exist_ok=True)

        # img process
        img = cv2.imread(img_path)
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        tensor.requires_grad = True

        # init ActivationsAndGradients
        grads = ActivationsAndGradients(self.model, self.target_layers, reshape_transform=None)

        # get ActivationsAndResult
        result = grads(tensor)
        activations = grads.activations[0].cpu().detach().numpy()


        if target_box is not None:
            result = result[0]
            print(f"🔍 [Force Mode] 正在寻找与目标框 {target_box} 最匹配的潜在预测...")
            
            # 1. 获取模型 B 的原始输出
            # result[0] 是 [300, 16] 或 [16, 6300]，先统一转成 [N, 16] 格式
            raw_output = result[0]
            if raw_output.shape[0] != 300 and raw_output.shape[-1] != 16:
                 raw_output = raw_output.transpose(0, 1) # 转成 [N, C]
            
            # 2. 解析坐标 (xywh -> xyxy)
            # 注意：raw_output 里的坐标通常是归一化的或者是基于当前输入尺寸(640)的
            # 这里假设是基于 640x640 的像素坐标 (YOLOv8/v10通常如此)
            pred_boxes = raw_output[:, :4] # [N, 4] xywh
            pred_boxes_xyxy = xywh2xyxy(pred_boxes).cpu().detach().numpy()
            
            # 3. 遍历所有候选框，找 IoU 最大的那个
            best_iou = -1
            best_idx = -1
            
            for idx, pred_box in enumerate(pred_boxes_xyxy):
                iou = compute_iou(target_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            print(f"✅ 找到最佳匹配: Index {best_idx}, IoU: {best_iou:.4f}")
            
            # 即使 IoU 很低（说明模型B完全没预测对位置），我们也强制看这个 Index
            # 因为我们想知道“在这个位置，模型B最关注什么特征”
            
            # 4. 构造一个伪造的 post_result 列表，只包含这一个最佳匹配
            # 我们直接复用下面的循环逻辑，只循环这一次
            target_indices = [best_idx]

        else:
            # 原来的逻辑：自动后处理
            post_result, pre_post_boxes, post_boxes = self.post_process(result[0])
            target_indices = range(int(post_result.size(0) * self.ratio))


            # --- 添加调试代码 ---
            print(f"Total detections: {len(post_result)}")
            if len(post_result) > 0:
                print(f"Max score: {float(post_result.max())}")
            else:
                print("No detections found in post_process")
            # ------------------
        
        raw_output_for_grad = result[0] if result[0].shape[-1] == 16 else result[0].transpose(0, 1)
        for i in target_indices:
            if target_box is None:
                if float(post_result[i].max()) > self.conf_threshold_max:
                    continue
                if float(post_result[i].max()) < self.conf_threshold_min:
                    break

            self.model.zero_grad()
            # get max probability for this prediction
            if self.backward_type == 'class' or self.backward_type == 'all':
                if target_box is None:
                    score = post_result[i].max()
                else:
                    score = raw_output_for_grad[i, 4]
                    # print(f'index {i} class score: {score:.4f}')
                score.backward(retain_graph=True)

            if self.backward_type == 'box' or self.backward_type == 'all':
                for j in range(4):
                    score = pre_post_boxes[i, j]
                    score.backward(retain_graph=True)

            # process heatmap
            if self.backward_type == 'class':
                gradients = grads.gradients[0]
            elif self.backward_type == 'box':
                gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3]
            else:
                gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3] + \
                            grads.gradients[4]
            b, k, u, v = gradients.size()
            weights = self.method.get_cam_weights(self.method, None, None, None, activations,
                                                  gradients.detach().numpy())
            weights = weights.reshape((b, k, 1, 1))
            saliency_map = np.sum(weights * activations, axis=1)
            saliency_map = np.squeeze(np.maximum(saliency_map, 0))
            saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)))
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            if (saliency_map_max - saliency_map_min) == 0:
                continue
            saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)

            # add heatmap and box to image
            cam_image = show_cam_on_image(img.copy(), saliency_map, use_rgb=True)
            if target_box is None:
                print(post_boxes[i])
            # cam_image = self.draw_detections(post_boxes[i], self.colors[int(post_result[i, :].argmax())],
            #                                  f'{self.model_names[int(post_result[i, :].argmax())]} {float(post_result[i].max()):.2f}',
            #                                  cam_image)
            cam_image = Image.fromarray(cam_image)
            cam_image.save(f'{save_path}/{i}.png')
        grads.release()

def get_params():
    params = {
        'device': 'cuda:0',
        'method': 'GradCAM',  # GradCAMPlusPlus, GradCAM, XGradCAM
        'layer': 10,
        'backward_type': 'class',  # class, box, all
        'conf_threshold_max': 0.65,  # 0.6
        'conf_threshold_min': 0.63,  # 0.6
        'ratio': 1.0  # 0.02-0.1
    }
    return params


if __name__ == '__main__':
    img_path = r'/scrinvme/huilin/tp/FLIR1444_img.png'
    weight = r'/localnvme/project/ultralytics/runs/exp_results/exp_mayolox_/weights/best.pt'
    save_dir = r'/scrinvme/huilin/tp/FLIR1444_img_mayolo_grad_acm2'
    target_box = [64.289, -4.7371, 116.63, 454.71]
    weight = r'/localnvme/project/ultralytics/runs/exp_results/exp_yolo10x/weights/best.pt'
    save_dir = r'/scrinvme/huilin/tp/FLIR1444_img_yolo10_grad_acm2'
    params = get_params()
    params['weight'] = weight
    model = yolov8_heatmap(**params)
    model(img_path, save_dir, target_box=target_box)