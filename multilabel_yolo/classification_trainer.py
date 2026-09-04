"""Training integration for image-level multi-label classification."""

from copy import copy

import torch

from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import is_parallel, torch_distributed_zero_first

from .classification_dataset import (
    MultiLabelClassificationDataset,
    build_multilabel_classification_dataloader,
    load_multilabel_classification_data,
)
from .classification_model import MultiLabelClassificationModel
from .classification_validator import MultiLabelClassificationValidator


class MultiLabelClassificationTrainer(ClassificationTrainer):
    """Train a YOLO classifier with one independent binary target per class."""

    def get_dataset(self):
        """Load the explicit n-hot dataset schema instead of ImageFolder."""
        if self.args.single_cls:
            raise ValueError("single_cls is not supported for image-level multi-label classification")
        if getattr(self.args, "classes", None) is not None:
            raise ValueError("classes filtering is not supported for image-level multi-label classification")
        self.data = load_multilabel_classification_data(self.args.data)
        return self.data["train"], self.data["val"]

    def set_model_attributes(self):
        """Attach class names and an explicit multi-label marker to the model."""
        self.model.names = self.data["names"]
        self.model.nc = self.data["nc"]
        self.model.multilabel = True
        self.model.multilabel_threshold = self.data["threshold"]

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build the native YOLO classification architecture with a custom criterion."""
        model = MultiLabelClassificationModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        for module in model.modules():
            if not self.args.pretrained and hasattr(module, "reset_parameters"):
                module.reset_parameters()
            if isinstance(module, torch.nn.Dropout) and self.args.dropout:
                module.p = self.args.dropout
        for parameter in model.parameters():
            parameter.requires_grad = True
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build an image-level dataset without class-directory assumptions."""
        return MultiLabelClassificationDataset(
            root=img_path,
            args=self.args,
            nc=self.data["nc"],
            label_root=self.data["labels"].get(mode),
            augment=mode == "train",
            prefix=mode,
            names=self.data["names"],
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Build a dataloader carrying n-hot targets and attach val transforms."""
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode)
        loader = build_multilabel_classification_dataloader(
            dataset, batch_size, self.args.workers, shuffle=mode == "train", rank=rank
        )
        if mode != "train":
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms
        return loader

    def preprocess_batch(self, batch):
        """Move images and n-hot targets to the training device."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["cls_nhot"] = batch["cls_nhot"].to(self.device, non_blocking=True).float()
        batch["cls"] = batch["cls_nhot"]  # BaseTrainer uses this only for batch-size logging.
        return batch

    def get_validator(self):
        """Return the matching multi-label validator."""
        self.loss_names = ["loss"]
        return MultiLabelClassificationValidator(
            self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_labels(self):
        """Log image-level positive counts instead of drawing scalar class boxes."""
        counts = self.train_loader.dataset.class_counts()
        LOGGER.info("Image-level multi-label positive counts: %s", counts.tolist())

    def plot_training_samples(self, batch, ni):
        """Avoid the native scalar-label visualization for n-hot targets."""
        return None


__all__ = ("MultiLabelClassificationTrainer",)
