"""Reproduce the historical eight-method CAM visualization used for E3.

The visualization deliberately follows the 2026-09-22 reference procedure:
one-to-many inference, attribute-aware NMS, conf=0.5, NMS IoU=0.7, and CAM
target layer 22.  The heatmap is generated from the raw class score associated
with the NMS detection, while the displayed boxes come from the same NMS call.
The final annotation is rendered through the local ``MdetResults`` renderer,
which follows the project's ``yolo_data_manager`` visualization convention.
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
from ultralytics.engine.results import MdetResults  # noqa: E402
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
    """Return the attribute metadata required by ``MdetResults.plot``.

    The result renderer is the same renderer used by the project prediction
    path.  A small fallback keeps CAM visualization usable for older
    checkpoints that contain the head but not the attribute-name mapping.
    """
    model_core = getattr(model, "model", model)
    attribute_names = getattr(model, "attribute_names", None)
    if attribute_names is None:
        attribute_names = getattr(model_core, "attribute_names", None)

    attribute_count = int(getattr(head, "na", 0) or 0)
    attribute_levels = int(getattr(head, "nal", 2) or 2)
    if not isinstance(attribute_names, dict) or len(attribute_names) != attribute_count:
        level_names = ["no", "yes"] if attribute_levels == 2 else [str(i) for i in range(attribute_levels)]
        attribute_names = {
            f"attribute_{index}": list(level_names) for index in range(attribute_count)
        }
    multiclass = bool(getattr(head, "multiclass_attributes", False))
    return attribute_names, attribute_count, attribute_levels, multiclass


def draw_detections(
    rgb: np.ndarray,
    detections: torch.Tensor,
    model,
) -> np.ndarray:
    """Draw CAM detections with the project/yolo_data_manager-style renderer.

    This deliberately uses ``MdetResults.plot`` instead of a second custom
    label implementation.  Consequently class colors, attribute colors,
    attribute text format, top-left placement, and image-boundary handling
    stay identical to normal prediction visualization.  The CAM target is
    represented by the heatmap itself; boxes are rendered exactly as ordinary
    predictions rather than with an extra target-only color.
    """
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

    cam_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    results = MdetResults(
        cam_bgr,
        path="cam",
        names=get_names(model),
        boxes=boxes,
        attributes=attributes,
        attribute_names=attribute_names,
        nc=int(getattr(head, "nc", len(get_names(model)))),
        na=attribute_count,
        nal=attribute_levels,
        risk_enlarge=1,
        multiclass_attributes=multiclass,
        attribute_channels=attribute_channels,
    )
    annotated_bgr = results.plot(
        img=cam_bgr,
        conf=True,
        labels=True,
        boxes=True,
        attributes=True,
        masks=False,
        probs=False,
        filter_no=True,
    )
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


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
        "renderer": "ultralytics.engine.results.MdetResults.plot (yolo_data_manager-compatible)",
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
