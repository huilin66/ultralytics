"""Training integration for true multi-label YOLO detection."""

from copy import copy

from ultralytics.data import build_dataloader
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import de_parallel

from .dataset import MultiLabelYOLODataset
from .model import MultiLabelDetectionModel
from .validator import MultiLabelDetectionValidator


class MultiLabelDetectionTrainer(DetectionTrainer):
    """DetectionTrainer using physical boxes plus n-hot class vectors."""

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build the multi-label dataset without scalar-class filtering."""
        if self.args.classes is not None:
            raise NotImplementedError("classes filtering is not supported in multi-label mode")
        if self.args.single_cls:
            raise NotImplementedError("single_cls is not supported in multi-label mode")
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return MultiLabelYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task="detect",
            classes=None,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build the native architecture with the multi-label criterion wrapper."""
        model = MultiLabelDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights is not None:
            model.load(weights)
        return model

    def get_validator(self):
        """Return the validator that expands GT labels only at metric time."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return MultiLabelDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def set_model_attributes(self):
        """Attach dataset metadata without treating transport IDs as classes."""
        super().set_model_attributes()
        self.model.multilabel = True
        self.model.multilabel_nc = self.data["nc"]
        self.model.multilabel_names = self.data["names"]

    def plot_training_labels(self):
        """Log real class counts; do not plot surrogate transport IDs as classes."""
        counts = self.train_loader.dataset.get_class_counts()
        LOGGER.info("Multi-label physical-object class counts: %s", counts.tolist())

    def plot_training_samples(self, batch, ni):
        """Disable the native scalar-class plot, which cannot display n-hot labels safely."""
        return None

    def auto_batch(self):
        """Estimate batch size from physical boxes, not label cardinality."""
        train_dataset = self.build_dataset(self.trainset, mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
        return BaseTrainer.auto_batch(self, max_num_obj)
