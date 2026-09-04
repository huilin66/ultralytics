"""Validation with explicit multi-label NMS and metric-only GT expansion."""

import torch

from ultralytics.data import build_dataloader
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import colorstr, ops

from .dataset import MultiLabelYOLODataset


class MultiLabelDetectionValidator(DetectionValidator):
    """Reuse native class-wise metrics after expanding n-hot GTs in memory."""

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build the custom validation dataset."""
        if self.args.classes is not None:
            raise NotImplementedError("classes filtering is not supported in multi-label mode")
        if self.args.single_cls:
            raise NotImplementedError("single_cls is not supported in multi-label mode")
        if self.args.save_hybrid:
            raise NotImplementedError("save_hybrid is not supported in multi-label mode")
        stride = int(self.stride.max()) if isinstance(self.stride, torch.Tensor) else int(self.stride)
        return MultiLabelYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=self.args.rect,
            cache=self.args.cache or None,
            single_cls=False,
            stride=stride,
            pad=0.5,
            prefix=colorstr("val: "),
            task="detect",
            classes=None,
            data=self.data,
            fraction=1.0,
        )

    def get_dataloader(self, dataset_path, batch_size):
        """Build a dataloader whose collate function includes ``cls_nhot``."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)

    def preprocess(self, batch):
        """Move images, physical boxes, transport IDs, and n-hot labels to the device."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for key in ["batch_idx", "cls", "bboxes", "cls_nhot"]:
            batch[key] = batch[key].to(self.device)
        return batch

    def postprocess(self, preds):
        """Keep every class above threshold for a candidate box."""
        return ops.non_max_suppression(
            preds,
            conf_thres=self.args.conf,
            iou_thres=self.args.iou,
            labels=self.lb,
            nc=self.nc,
            multi_label=True,
            classes=self.args.classes,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )

    @staticmethod
    def expand_targets(bboxes, cls_nhot):
        """Expand physical boxes into class-wise metric targets in memory only."""
        if cls_nhot.ndim != 2:
            raise ValueError(f"cls_nhot must be [N, nc], got {tuple(cls_nhot.shape)}")
        rows, classes = torch.where(cls_nhot > 0)
        return bboxes[rows], classes.long(), rows

    def _prepare_batch(self, si, batch):
        """Prepare class-wise metric GTs from one image's physical n-hot objects."""
        idx = batch["batch_idx"] == si
        physical_bbox = batch["bboxes"][idx]
        physical_nhot = batch["cls_nhot"][idx]
        bbox, cls, _ = self.expand_targets(physical_bbox, physical_nhot)
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def plot_val_samples(self, batch, ni):
        """Disable the native plot because its scalar class column is transport-only."""
        return None
