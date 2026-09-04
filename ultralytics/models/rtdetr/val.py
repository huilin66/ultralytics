# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

import torch

from ultralytics.data import YOLODataset, YOLOMDETDataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.data.dataset import DATASET_CACHE_VERSION
from ultralytics.data.utils import (
    check_det_dataset,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
)
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.mdetect.val import MDetectionValidator
from ultralytics.utils import colorstr, ops

__all__ = ("RTDETRValidator", "RTDETRMDetectionValidator")


class RTDETRDataset(YOLODataset):
    """
    Real-Time DEtection and TRacking (RT-DETR) dataset class extending the base YOLODataset class.

    This specialized dataset class is designed for use with the RT-DETR object detection model and is optimized for
    real-time detection and tracking tasks.
    """

    def __init__(self, *args, data=None, task=None, **kwargs):
        """Initialize the RTDETRDataset class by inheriting from the YOLODataset class."""
        # RT-DETR normally receives ``task='detect'`` from its model wrapper. An attribute dataset is identified from
        # the data YAML, so it must explicitly use the mdet parser/formatter to keep one attribute row per object.
        task = task or ("mdetect" if data and data.get("na") else "detect")
        self.use_mdetect = task in {"mdetect", "msegment"}
        super().__init__(*args, data=data, task=task, **kwargs)

    def get_labels(self):
        """Use the attribute-aware cache reader only for mdet datasets."""
        if not self.use_mdetect:
            return YOLODataset.get_labels(self)

        # RT-DETR and YOLO detection normally share the same cache filename. If a vanilla cache is present, rebuild
        # it with the mdet verifier before delegating to the normal cache reader.
        self.label_files = img2label_paths(self.im_files)
        self.check_files()
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache = load_dataset_cache_file(cache_path)
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.label_files + self.im_files)
            needs_mdet_cache = any("mdet_attributes" not in label for label in cache.get("labels", []))
        except (FileNotFoundError, AssertionError, AttributeError):
            needs_mdet_cache = False
        if needs_mdet_cache:
            self.cache_labels(cache_path)
        return YOLOMDETDataset.get_labels(self)

    def cache_labels(self, path=Path("./labels.cache")):
        """Use the attribute-aware label verifier only for mdet datasets."""
        return (
            YOLOMDETDataset.cache_labels(self, path)
            if self.use_mdetect
            else YOLODataset.cache_labels(self, path)
        )

    def update_labels_info(self, label):
        """Keep attributes aligned with boxes through the RT-DETR transform pipeline."""
        return (
            YOLOMDETDataset.update_labels_info(self, label)
            if self.use_mdetect
            else super().update_labels_info(label)
        )

    @staticmethod
    def collate_fn(batch):
        """Collate both vanilla RT-DETR samples and object-level mdet samples."""
        return (
            YOLOMDETDataset.collate_fn(batch)
            if "mdet_attributes" in batch[0]
            else YOLODataset.collate_fn(batch)
        )

    def load_image(self, i, rect_mode=False):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        return super().load_image(i=i, rect_mode=rect_mode)

    def build_transforms(self, hyp=None):
        """
        Build transformation pipeline for the dataset.

        Args:
            hyp (Dict, optional): Hyperparameters for transformations.

        Returns:
            (Compose): Composition of transformation functions.
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            # transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, scale_fill=True)])
            transforms = Compose([])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_mdetect=self.use_mdetect,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms


class RTDETRValidator(DetectionValidator):
    """
    RTDETRValidator extends the DetectionValidator class to provide validation capabilities specifically tailored for
    the RT-DETR (Real-Time DETR) object detection model.

    The class allows building of an RTDETR-specific dataset for validation, applies Non-maximum suppression for
    post-processing, and updates evaluation metrics accordingly.

    Examples:
        >>> from ultralytics.models.rtdetr import RTDETRValidator
        >>> args = dict(model="rtdetr-l.pt", data="coco8.yaml")
        >>> validator = RTDETRValidator(args=args)
        >>> validator()

    Note:
        For further details on the attributes and methods, refer to the parent DetectionValidator class.
    """

    def __call__(self, trainer=None, model=None):
        """Dispatch to the object-level attribute validator when the dataset declares attributes."""
        data = trainer.data if trainer is not None else None
        if data is None and isinstance(getattr(self.args, "data", None), (str, Path)):
            try:
                data = check_det_dataset(self.args.data, autodownload=False)
            except Exception:
                data = None
        if isinstance(data, dict) and data.get("na"):
            validator = RTDETRMDetectionValidator(
                self.dataloader,
                save_dir=self.save_dir,
                pbar=self.pbar,
                args=self.args,
                _callbacks=self.callbacks,
            )
            return validator(trainer=trainer, model=model)
        return super().__call__(trainer=trainer, model=model)

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build an RTDETR Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (RTDETRDataset): Dataset configured for RT-DETR validation.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # no augmentation
            hyp=self.args,
            rect=False,  # no rect
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )

    def postprocess(self, preds):
        """
        Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (List | Tuple | torch.Tensor): Raw predictions from the model.

        Returns:
            (List[torch.Tensor]): List of processed predictions for each image in batch.
        """
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
            preds = [preds, None]

        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        bboxes *= self.args.imgsz
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1)  # (300, )
            # Do not need threshold for evaluation as only got 300 boxes here
            # idx = score > self.args.conf
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)  # filter
            # Sort by confidence to correctly get internal metrics
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred  # [idx]

        return outputs

    def _prepare_batch(self, si, batch):
        """
        Prepares a batch for validation by applying necessary transformations.

        Args:
            si (int): Batch index.
            batch (Dict): Batch data containing images and annotations.

        Returns:
            (Dict): Prepared batch with transformed annotations.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox)  # target boxes
            bbox[..., [0, 2]] *= ori_shape[1]  # native-space pred
            bbox[..., [1, 3]] *= ori_shape[0]  # native-space pred
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """
        Prepares predictions by scaling bounding boxes to original image dimensions.

        Args:
            pred (torch.Tensor): Raw predictions.
            pbatch (Dict): Prepared batch information.

        Returns:
            (torch.Tensor): Predictions scaled to original image dimensions.
        """
        predn = pred.clone()
        predn[..., [0, 2]] *= pbatch["ori_shape"][1] / self.args.imgsz  # native-space pred
        predn[..., [1, 3]] *= pbatch["ori_shape"][0] / self.args.imgsz  # native-space pred
        return predn.float()


class RTDETRMDetectionValidator(MDetectionValidator):
    """Validator for RT-DETR with one object query carrying a box, class, and attribute vector."""

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build the RT-DETR dataset while forcing the mdet label parser."""
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
            task="mdetect",
        )

    def postprocess(self, preds):
        """Convert RT-DETR query predictions into the mdet validator row format."""
        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]

        raw = preds[0]
        bs, _, nd = raw.shape
        expected = 4 + self.nc + self.attribute_channels
        if nd != expected:
            raise ValueError(f"Expected RT-DETR output width {expected}, got {nd}")
        bboxes, scores, attributes = raw.split((4, self.nc, self.attribute_channels), dim=-1)
        bboxes = bboxes * self.args.imgsz
        outputs = []
        for bbox, score, attribute in zip(bboxes, scores, attributes):
            bbox = ops.xywh2xyxy(bbox)
            confidence, cls = score.max(-1)
            order = confidence.argsort(descending=True)[: self.args.max_det]
            outputs.append(
                torch.cat(
                    (bbox[order], confidence[order, None], cls[order, None], attribute[order]),
                    dim=-1,
                )
            )
        return outputs

    def _prepare_batch(self, si, batch):
        """Prepare RT-DETR labels and retain the attributes aligned with each ground-truth box."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        mdet_attributes = batch["mdet_attributes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox)
            bbox[..., [0, 2]] *= ori_shape[1]
            bbox[..., [1, 3]] *= ori_shape[0]
        return {
            "cls": cls,
            "bbox": bbox,
            "mdet_attributes": mdet_attributes,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
        }

    def _prepare_pred(self, pred, pbatch):
        """Scale RT-DETR predictions from the model image to native image coordinates."""
        predn = pred.clone()
        predn[..., [0, 2]] *= pbatch["ori_shape"][1] / self.args.imgsz
        predn[..., [1, 3]] *= pbatch["ori_shape"][0] / self.args.imgsz
        return predn.float()
