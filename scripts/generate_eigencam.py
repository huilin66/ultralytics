"""Reproduce the historical eight-method CAM visualization used for E3.

The visualization deliberately follows the 2026-09-22 reference procedure:
one-to-many inference, attribute-aware NMS, conf=0.5, NMS IoU=0.7, and CAM
target layer 22.  The heatmap is generated from the raw class score associated
with the NMS detection, while the displayed boxes come from the same NMS call.
The final annotation uses a local copy of the project's
``yolo_data_manager.vis.renderer`` OpenCV drawing rules so the CAM overlay
does not introduce a second visualization style or a runtime dependency on
that repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pytorch_grad_cam import (  # noqa: E402
    EigenCAM,
    EigenGradCAM,
    GradCAM,
    GradCAMPlusPlus,
    HiResCAM,
    LayerCAM,
    RandomCAM,
    XGradCAM,
)
from ultralytics.nn.tasks import attempt_load_weights  # noqa: E402
from ultralytics.utils.ops import non_max_suppression_with_attributes, xywh2xyxy  # noqa: E402


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CAM_METHODS = {
    "GradCAM": GradCAM,
    "GradCAMPlusPlus": GradCAMPlusPlus,
    "XGradCAM": XGradCAM,
    "EigenCAM": EigenCAM,
    "HiResCAM": HiResCAM,
    "LayerCAM": LayerCAM,
    "RandomCAM": RandomCAM,
    "EigenGradCAM": EigenGradCAM,
}
DEFAULT_METHODS = tuple(CAM_METHODS)

# Copied from yolo_data_manager.vis.renderer.  These values are OpenCV BGR
# colors, matching the palette used by the normal prediction visualizer.
CV2_COLORS = (
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
YDM_DARK_COLORS = {
    (235, 219, 11),
    (243, 243, 243),
    (183, 223, 0),
    (221, 111, 255),
    (0, 237, 204),
    (68, 243, 0),
    (255, 255, 0),
    (179, 255, 1),
    (11, 255, 162),
}


def normalize_device(device: str) -> str:
    value = str(device).strip()
    return f"cuda:{value}" if value.isdigit() else value


def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    stride: int = 32,
) -> np.ndarray:
    """Match the reference script's 640-pixel letterbox preprocessing."""
    shape = image.shape[:2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    return cv2.copyMakeBorder(
        image,
        int(round(dh - 0.1)),
        int(round(dh + 0.1)),
        int(round(dw - 0.1)),
        int(round(dw + 0.1)),
        cv2.BORDER_CONSTANT,
        value=color,
    )


def overlay_cam(rgb_float: np.ndarray, cam: np.ndarray) -> np.ndarray:
    heat = cv2.applyColorMap(np.uint8(np.clip(cam, 0, 1) * 255), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.uint8(np.clip(0.50 * rgb_float + 0.50 * heat, 0, 1) * 255)


def get_names(model) -> dict[int, str]:
    names = getattr(model, "names", {})
    if isinstance(names, (list, tuple)):
        return {index: str(name) for index, name in enumerate(names)}
    return {int(index): str(name) for index, name in names.items()}


def get_attribute_spec(model, head) -> tuple[dict[str, list[str]], int, int, bool]:
    """Return attribute metadata in the format used by the project renderer.

    A small fallback keeps CAM visualization usable for older checkpoints
    that contain the attribute head but not the attribute-name mapping.
    """
    model_core = getattr(model, "model", model)
    attribute_names = getattr(model, "attribute_names", None)
    if attribute_names is None:
        attribute_names = getattr(model_core, "attribute_names", None)

    attribute_count = int(getattr(head, "na", 0) or 0)
    attribute_levels = int(getattr(head, "nal", 2) or 2)
    if not isinstance(attribute_names, dict) or len(attribute_names) != attribute_count:
        level_names = ["No risk", "High risk"] if attribute_levels == 2 else [str(i) for i in range(attribute_levels)]
        attribute_names = {
            f"attribute_{index}": list(level_names) for index in range(attribute_count)
        }
    else:
        # Existing checkpoints may store the two levels as False/True, no/yes,
        # or 0/1.  Keep the prediction index unchanged but use the paper-facing
        # labels in the visualization.
        normalized = {}
        for name, levels in attribute_names.items():
            normalized[str(name)] = [
                normalize_attribute_level(level, index)
                for index, level in enumerate(levels)
            ]
        attribute_names = normalized
    multiclass = bool(getattr(head, "multiclass_attributes", False))
    return attribute_names, attribute_count, attribute_levels, multiclass


def normalize_attribute_level(level, index: int) -> str:
    """Map legacy boolean levels to the displayed risk terminology."""
    if isinstance(level, (bool, np.bool_)):
        return "High risk" if bool(level) else "No risk"
    text = str(level).strip()
    lowered = text.casefold()
    if lowered in {"true", "yes", "1", "high risk", "high-risk"}:
        return "High risk"
    if lowered in {"false", "no", "0", "no risk", "no-risk"}:
        return "No risk"
    # Some old checkpoints store numeric levels as integers.  Only the first
    # two binary levels are relabeled; other multi-level names are preserved.
    if index == 0 and lowered == "0":
        return "No risk"
    if index == 1 and lowered == "1":
        return "High risk"
    return text


def ydm_line_width(image: np.ndarray) -> int:
    """Use yolo_data_manager's image-size-dependent line width."""
    return max(round(sum(image.shape) / 2 * 0.003), 2)


def ydm_text_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Use yolo_data_manager's readable text color for a label background."""
    return (104, 31, 17) if tuple(color) in YDM_DARK_COLORS else (255, 255, 255)


def ydm_draw_label(
    image: np.ndarray,
    x: float,
    y: float,
    text: str,
    color: tuple[int, int, int],
    line_width: int,
) -> dict[str, int | float | bool] | None:
    """Draw the yolo_data_manager class label and return its layout bounds."""
    if not text:
        return None
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

    cv2.rectangle(image, (left, top), (right, bottom), color, -1, cv2.LINE_AA)
    cv2.putText(
        image,
        str(text),
        (left + pad, int(text_baseline)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        ydm_text_color(color),
        thickness=font_thickness,
        lineType=cv2.LINE_AA,
    )
    return {
        "left": left,
        "bottom": bottom,
        "outside": outside,
        "font_scale": font_scale,
        "font_thickness": font_thickness,
        "line_height": max(text_height + baseline, 1),
    }


def ydm_is_negative_attribute(value) -> bool:
    """Identify the non-risk level for attribute-text coloring."""
    if value is False or value is None:
        return True
    return str(value).strip().casefold() in {
        "false",
        "no",
        "no risk",
        "no-risk",
        "0",
    }


def ydm_draw_attributes(
    image: np.ndarray,
    attributes: list[tuple[str, object]],
    label_info: dict[str, int | float | bool] | None,
) -> None:
    """Draw the yolo_data_manager attribute panel without cross-box changes."""
    if not attributes or label_info is None:
        return
    height, width = image.shape[:2]
    font_scale = float(label_info["font_scale"])
    font_thickness = int(label_info["font_thickness"])
    line_height = max(int(float(label_info["line_height"]) * 0.85), 12)
    texts = [f"{name}-{value}" for name, value in attributes]
    sizes = [
        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        for text in texts
    ]
    max_text_width = max(size[0][0] for size in sizes)
    text_height = max(size[0][1] for size in sizes)
    baseline = max(size[1] for size in sizes)
    x = max(0, min(int(label_info["left"]) + 2, max(width - max_text_width - 4, 0)))
    start_y = int(label_info["bottom"]) + line_height
    top = start_y - text_height - 2
    bottom = start_y + line_height * (len(texts) - 1) + baseline + 2
    if bottom >= height:
        shift = bottom - height + 1
        start_y -= shift
        top -= shift
        bottom -= shift
    if top < 0:
        start_y -= top
        bottom -= top
        top = 0
    right = min(width - 1, x + max_text_width + 5)
    bottom = min(height - 1, max(bottom, top))

    overlay = image.copy()
    cv2.rectangle(overlay, (x, top), (right, bottom), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)
    for index, ((_, value), text) in enumerate(zip(attributes, texts)):
        text_y = min(max(start_y + line_height * index, top + text_height), height - 1)
        text_color = (0, 0, 0) if ydm_is_negative_attribute(value) else (255, 0, 0)
        cv2.putText(
            image,
            text,
            (x + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )


def decode_attributes(
    attributes: torch.Tensor,
    attribute_names: dict[str, list[str]],
    attribute_count: int,
    attribute_levels: int,
    multiclass: bool,
) -> list[list[tuple[str, object]]]:
    """Decode NMS attribute channels using the same levels as predictions."""
    if attribute_count == 0 or attributes.numel() == 0:
        return [[] for _ in range(len(attributes))]
    data = attributes.detach()
    names = list(attribute_names.items())
    if multiclass:
        expected = attribute_count * attribute_levels
        if data.shape[-1] != expected:
            raise RuntimeError(
                f"Expected {expected} attribute channels, got {data.shape[-1]}"
            )
        indices = data.reshape(-1, attribute_count, attribute_levels).argmax(dim=-1)
        max_levels = torch.tensor(
            [len(levels) for _, levels in names], device=indices.device, dtype=indices.dtype
        )
        indices = torch.minimum(indices, (max_levels - 1).clamp_min(0))
    else:
        indices = torch.floor(data * attribute_levels).long().clamp(0, attribute_levels - 1)

    decoded: list[list[tuple[str, object]]] = []
    for row in indices.detach().cpu().tolist():
        decoded.append(
            [
                (name, levels[min(int(level), len(levels) - 1)])
                for (name, levels), level in zip(names, row)
                if levels
            ]
        )
    return decoded


def draw_detections(
    rgb: np.ndarray,
    detections: torch.Tensor,
    model,
) -> np.ndarray:
    """Draw CAM detections with yolo_data_manager's OpenCV renderer."""
    if detections.numel() == 0:
        return rgb

    model_core = getattr(model, "model", model)
    model_container = getattr(model_core, "model", model_core)
    head = model_container[-1]
    attribute_names, attribute_count, attribute_levels, multiclass = get_attribute_spec(model, head)
    attribute_channels = int(getattr(head, "attribute_channels", attribute_count))

    # NMS coordinates are in the letterboxed 640x640 image space.  Clamp only
    # the displayed copy so labels and rectangles cannot leave the CAM image.
    rendered = detections.detach().clone().float()
    height, width = rgb.shape[:2]
    rendered[:, [0, 2]].clamp_(0, width - 1)
    rendered[:, [1, 3]].clamp_(0, height - 1)
    boxes = rendered[:, :6]
    attributes = rendered[:, 6 : 6 + attribute_channels]
    if attributes.shape[1] != attribute_channels:
        raise RuntimeError(
            f"NMS returned {attributes.shape[1]} attribute channels, "
            f"but the model head declares {attribute_channels}"
        )

    output = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    decoded_attributes = decode_attributes(
        attributes,
        attribute_names,
        attribute_count,
        attribute_levels,
        multiclass,
    )
    names = get_names(model)
    line_width = ydm_line_width(output)
    for detection, decoded in zip(rendered, decoded_attributes):
        values = detection[:4].detach().cpu().numpy().astype(np.float32)
        x1, y1, x2, y2 = [int(round(float(value))) for value in values]
        x1, x2 = sorted((max(0, min(x1, width - 1)), max(0, min(x2, width - 1))))
        y1, y2 = sorted((max(0, min(y1, height - 1)), max(0, min(y2, height - 1))))
        if x2 <= x1 or y2 <= y1:
            continue

        confidence = float(detection[4].detach().cpu().item())
        class_id = int(detection[5].detach().cpu().item())
        color = CV2_COLORS[class_id % len(CV2_COLORS)]
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            thickness=line_width,
            lineType=cv2.LINE_AA,
        )
        label = f"{names.get(class_id, class_id)} {confidence:.2f}"
        label_info = ydm_draw_label(output, x1, y1, label, color, line_width)
        ydm_draw_attributes(output, decoded, label_info)
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


def box_iou_one(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])
    intersection = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    area1 = (box[2] - box[0]).clamp_min(0) * (box[3] - box[1]).clamp_min(0)
    area2 = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    return intersection / (area1 + area2 - intersection).clamp_min(1e-9)


class RawModel(torch.nn.Module):
    """Expose raw model output to pytorch-grad-cam."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        output = self.base(x)
        return output[0] if isinstance(output, tuple) else output


class RawDetectionTarget:
    """Target one raw class score corresponding to a post-NMS detection."""

    def __init__(self, class_index: int, raw_index: int):
        self.class_index = class_index
        self.raw_index = raw_index

    def __call__(self, model_output):
        # pytorch-grad-cam removes the batch dimension before calling targets.
        if model_output.ndim == 3:
            model_output = model_output[0]
        return model_output[4 + self.class_index, self.raw_index]


def prepare_detection(model, tensor, conf: float, iou: float):
    """Run reference NMS and map its first detection to a raw prediction."""
    with torch.no_grad():
        output = model(tensor)
        prediction = output[0] if isinstance(output, tuple) else output
        head = model.model[-1]
        detections = non_max_suppression_with_attributes(
            prediction.detach().clone(),
            conf_thres=conf,
            iou_thres=iou,
            max_det=300,
            nc=int(head.nc),
            na=int(head.attribute_channels),
            in_place=True,
        )[0]

    if len(detections) == 0:
        raise RuntimeError("No NMS detections")

    target_detection = detections[0]
    target_class = int(float(target_detection[5]))
    raw_boxes = xywh2xyxy(prediction[0, :4, :].transpose(0, 1))
    raw_classes = prediction[0, 4 : 4 + int(head.nc), :].transpose(0, 1)
    candidates = torch.where(raw_classes.argmax(1) == target_class)[0]
    if len(candidates) == 0:
        raise RuntimeError(f"No raw prediction matched class {target_class}")
    overlaps = box_iou_one(target_detection[:4], raw_boxes[candidates])
    raw_index = int(candidates[overlaps.argmax()].item())
    return detections, target_class, raw_index, float(target_detection[4]), float(overlaps.max())


def collect_images(source: Path, image_names: list[str] | None) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Image file or directory not found: {source}")
    if image_names:
        images = [source / name for name in image_names]
        missing = [str(path) for path in images if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing requested images: " + ", ".join(missing))
        return images
    images = sorted(
        path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No supported images found: {source}")
    return images


def run_one(
    model,
    model_name: str,
    image: Path,
    output_root: Path,
    device: str,
    conf: float,
    iou: float,
    layer_index: int,
    method_names: list[str],
) -> dict[str, Path]:
    bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image)
    bgr = letterbox(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_float = rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb_float.transpose(2, 0, 1)).unsqueeze(0).to(device)

    detections, class_index, raw_index, target_conf, match_iou = prepare_detection(model, tensor, conf, iou)
    wrapper = RawModel(model).to(device)
    target_layer = model.model[layer_index]
    model_output = output_root / model_name
    model_output.mkdir(parents=True, exist_ok=True)

    print(
        model_name,
        image.name,
        "nms=",
        len(detections),
        "target_conf=",
        f"{target_conf:.6f}",
        "target_cls=",
        class_index,
        "raw_nms_iou=",
        f"{match_iou:.6f}",
        flush=True,
    )

    outputs: dict[str, Path] = {}
    for method_name in method_names:
        method_class = CAM_METHODS[method_name]
        target = RawDetectionTarget(class_index, raw_index)
        cam = method_class(wrapper, [target_layer])
        try:
            grayscale = cam(input_tensor=tensor, targets=[target], aug_smooth=False, eigen_smooth=False)[0]
        finally:
            cam.activations_and_grads.release()
        grayscale = cv2.resize(grayscale, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        result = draw_detections(
            overlay_cam(rgb_float, grayscale),
            detections,
            model,
        )
        destination = model_output / f"{image.stem}_{method_name}.png"
        if not cv2.imwrite(str(destination), cv2.cvtColor(result, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Failed to write {destination}")
        outputs[method_name] = destination
        print("  ", method_name, destination, flush=True)
    return outputs


def make_side_by_side(images: list[Path], method_names: list[str], output_root: Path) -> None:
    side_output = output_root / "side_by_side"
    side_output.mkdir(parents=True, exist_ok=True)
    for image in images:
        for method_name in method_names:
            left = cv2.imread(
                str(output_root / "YOLOv10x" / f"{image.stem}_{method_name}.png"), cv2.IMREAD_COLOR
            )
            right = cv2.imread(
                str(output_root / "MAYOLOx" / f"{image.stem}_{method_name}.png"), cv2.IMREAD_COLOR
            )
            if left is None or right is None:
                raise RuntimeError(f"Cannot read CAM pair for {image.name} ({method_name})")
            if left.shape[0] != right.shape[0]:
                right = cv2.resize(right, (right.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
            destination = side_output / f"{image.stem}_{method_name}.png"
            if not cv2.imwrite(str(destination), np.concatenate((left, right), axis=1)):
                raise OSError(f"Failed to write {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mayolo-weight", required=True, type=Path)
    parser.add_argument("--yolov10-weight", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--image-names", nargs="+", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--layers", type=int, nargs="+", dest="legacy_layers", default=None)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--methods", nargs="+", choices=tuple(CAM_METHODS), default=list(DEFAULT_METHODS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = normalize_device(args.device)
    if args.legacy_layers is not None:
        if len(args.legacy_layers) != 1:
            raise ValueError("The historical CAM procedure uses exactly one layer: 22")
        args.layer = args.legacy_layers[0]
    if args.layer < 0:
        raise ValueError("--layer must be non-negative")
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        raise ValueError("--conf and --iou must be in [0, 1]")
    for path, label in (
        (args.mayolo_weight, "MAYOLO weight"),
        (args.yolov10_weight, "YOLOv10 weight"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    torch.manual_seed(0)
    np.random.seed(0)
    images = collect_images(args.images, args.image_names)
    args.output.mkdir(parents=True, exist_ok=True)
    for model_name, weight in (
        ("YOLOv10x", args.yolov10_weight),
        ("MAYOLOx", args.mayolo_weight),
    ):
        model = attempt_load_weights(str(weight), device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        if hasattr(model.model[-1], "use_one2many_head"):
            model.model[-1].use_one2many_head()
        for image in images:
            run_one(
                model,
                model_name,
                image,
                args.output,
                device,
                args.conf,
                args.iou,
                args.layer,
                args.methods,
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    make_side_by_side(images, args.methods, args.output)

    metadata = {
        "procedure": "historical_2026-09-22_cam",
        "renderer": "local port of yolo_data_manager.vis.renderer (OpenCV)",
        "methods": list(args.methods),
        "mayolo_weight": str(args.mayolo_weight),
        "yolov10_weight": str(args.yolov10_weight),
        "images": [str(path) for path in images],
        "device": device,
        "head_mode": "one2many",
        "input_size": [640, 640],
        "cam_layer": args.layer,
        "confidence": args.conf,
        "nms_iou": args.iou,
        "target": "raw class score matched to the first attribute-aware NMS detection",
        "overlay": "0.5 original image + 0.5 JET CAM",
    }
    (args.output / "cam_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"DONE {args.output}", flush=True)


if __name__ == "__main__":
    main()
