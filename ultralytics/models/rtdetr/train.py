# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

import torch

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.torch_utils import de_parallel

from .val import RTDETRDataset, RTDETRMDetectionValidator, RTDETRValidator


class RTDETRTrainer(DetectionTrainer):
    """
    Trainer class for the RT-DETR model developed by Baidu for real-time object detection.

    This class extends the DetectionTrainer class for YOLO to adapt to the specific features and architecture of RT-DETR.
    The model leverages Vision Transformers and has capabilities like IoU-aware query selection and adaptable inference
    speed.

    Attributes:
        loss_names (Tuple[str]): Names of the loss components used for training.
        data (Dict): Dataset configuration containing class count and other parameters.
        args (Dict): Training arguments and hyperparameters.
        save_dir (Path): Directory to save training results.
        test_loader (DataLoader): DataLoader for validation/testing data.

    Notes:
        - F.grid_sample used in RT-DETR does not support the `deterministic=True` argument.
        - AMP training can lead to NaN outputs and may produce errors during bipartite graph matching.

    Examples:
        >>> from ultralytics.models.rtdetr.train import RTDETRTrainer
        >>> args = dict(model="rtdetr-l.yaml", data="coco8.yaml", imgsz=640, epochs=3)
        >>> trainer = RTDETRTrainer(overrides=args)
        >>> trainer.train()
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Initialize and return an RT-DETR model for object detection tasks.

        Args:
            cfg (Dict, optional): Model configuration.
            weights (str, optional): Path to pre-trained model weights.
            verbose (bool): Verbose logging if True.

        Returns:
            (RTDETRDetectionModel): Initialized model.
        """
        model = RTDETRDetectionModel(
            cfg,
            nc=self.data["nc"],
            na=self.data.get("na"),
            nal=self.data.get("nal"),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build and return an RT-DETR dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): Dataset mode, either 'train' or 'val'.
            batch (int, optional): Batch size for rectangle training.

        Returns:
            (RTDETRDataset): Dataset object for the specific mode.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_validator(self):
        """Returns a DetectionValidator suitable for RT-DETR model validation."""
        has_attributes = bool(self.data.get("na"))
        self.loss_names = ("giou_loss", "cls_loss", "l1_loss", "attribute_loss") if has_attributes else (
            "giou_loss", "cls_loss", "l1_loss"
        )
        validator = RTDETRMDetectionValidator if has_attributes else RTDETRValidator
        return validator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def set_model_attributes(self):
        """Attach dataset metadata to the RT-DETR model and its object-level attribute head."""
        super().set_model_attributes()
        model = de_parallel(self.model)
        head = model.model[-1]
        expected_channels = int(self.data.get("na", 0) or 0) * int(self.data.get("nal", 1) or 1)
        actual_channels = int(getattr(head, "attribute_channels", 0) or 0)
        if expected_channels and actual_channels != expected_channels:
            raise ValueError(
                f"RT-DETR mdet dataset requires {expected_channels} attribute channels, but the model head has "
                f"{actual_channels}. Use an attribute-aware RT-DETR YAML such as rtdetr-l-md.yaml."
            )
        model.na = int(getattr(head, "na", self.data.get("na", 0)) or 0)
        model.nal = int(getattr(head, "nal", self.data.get("nal", 1)) or 1)
        model.attribute_channels = int(getattr(head, "attribute_channels", model.na * model.nal))
        model.multiclass_attributes = bool(
            getattr(head, "multiclass_attributes", model.na > 0 and model.nal > 1)
        )
        model.attribute_names = self.data.get("attributes", {})

    def preprocess_batch(self, batch):
        """
        Preprocess a batch of images by scaling and converting to float format.

        Args:
            batch (Dict): Dictionary containing a batch of images, bboxes, and labels.

        Returns:
            (Dict): Preprocessed batch with ground truth bounding boxes and classes separated by batch index.
        """
        batch = super().preprocess_batch(batch)
        bs = len(batch["img"])
        batch_idx = batch["batch_idx"]
        gt_bbox, gt_class = [], []
        for i in range(bs):
            gt_bbox.append(batch["bboxes"][batch_idx == i].to(batch_idx.device))
            gt_class.append(batch["cls"][batch_idx == i].to(device=batch_idx.device, dtype=torch.long))
        return batch
