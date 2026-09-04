"""Validation for image-level multi-label classification."""

from __future__ import annotations

import torch

from ultralytics.data.build import build_dataloader
from ultralytics.models.yolo.classify.val import ClassificationValidator
from ultralytics.utils import LOGGER

from .classification_dataset import MultiLabelClassificationDataset, load_multilabel_classification_data
from .classification_loss import MultiLabelClassificationLoss
from .classification_metrics import MultiLabelClassificationMetrics


class MultiLabelClassificationValidator(ClassificationValidator):
    """Evaluate independent class probabilities instead of top-1 predictions."""

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "classify"
        self.scores = []
        self.targets = []
        self.metrics = MultiLabelClassificationMetrics()

    def get_desc(self):
        """Return the aggregate multi-label validation columns."""
        header = ("classes", *[key.rsplit("/", 1)[-1] for key in self.metrics.keys])
        return ("%22s" + "%11s" * len(self.metrics.keys)) % header

    def _load_data_if_needed(self):
        """Reload the custom YAML after BaseValidator's generic dataset check."""
        if not isinstance(self.data, dict) or not isinstance(self.data.get("labels"), dict):
            self.data = load_multilabel_classification_data(self.args.data)
        return self.data

    def init_metrics(self, model):
        """Initialize names, threshold, and score/target buffers."""
        self.model = model
        data = self._load_data_if_needed()
        self.names = data["names"]
        self.nc = data["nc"]
        self.threshold = float(data["threshold"])
        # BaseValidator uses conf=0.001 by default for detection-style AP.
        # Treat a non-default explicit conf as the requested classification
        # threshold, while otherwise honoring the YAML value.
        configured_threshold = getattr(self.args, "conf", None)
        if configured_threshold not in (None, 0.001):
            self.threshold = float(configured_threshold)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"Classification threshold must be in [0, 1], got {self.threshold}")
        self.scores = []
        self.targets = []
        self.metrics = MultiLabelClassificationMetrics(self.names, threshold=self.threshold)

    def preprocess(self, batch):
        """Move images and n-hot labels to the validation device."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["cls_nhot"] = batch["cls_nhot"].to(self.device, non_blocking=True).float()
        batch["cls"] = batch["cls_nhot"]
        return batch

    def postprocess(self, preds):
        """Convert raw classifier logits to independent sigmoid probabilities."""
        if isinstance(preds, (tuple, list)) and len(preds) >= 2:
            logits = MultiLabelClassificationLoss._logits(preds)
            return logits.float().sigmoid()
        backend_model = getattr(self.model, "model", self.model)
        head = getattr(backend_model, "model", None)
        if (
            getattr(backend_model, "multilabel", False)
            and head is not None
            and getattr(head[-1], "export", False)
        ):
            return torch.as_tensor(preds).float().clamp_(0.0, 1.0)
        logits = MultiLabelClassificationLoss._logits(preds)
        return logits.float().sigmoid()

    def update_metrics(self, preds, batch):
        """Accumulate probabilities and n-hot targets for aggregate metrics."""
        self.scores.append(preds.detach().cpu())
        self.targets.append(batch["cls_nhot"].detach().cpu())

    def get_stats(self):
        """Compute multi-label metrics from all validation images."""
        self.metrics.process(self.targets, self.scores)
        return self.metrics.results_dict

    def finalize_metrics(self, *args, **kwargs):
        """Attach runtime metadata and report per-class support when useful."""
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir
        LOGGER.info("Multi-label threshold: %.3f", self.threshold)

    def print_results(self):
        """Print metrics with a header that matches the numeric columns."""
        values = [self.metrics.results_dict[key] for key in self.metrics.keys]
        LOGGER.info(("%22s" + "%11.3g" * len(values)) % ("all", *values))

    def build_dataset(self, img_path):
        """Create the custom validation dataset."""
        data = self._load_data_if_needed()
        split = getattr(self.args, "split", "val")
        return MultiLabelClassificationDataset(
            root=img_path,
            args=self.args,
            nc=data["nc"],
            label_root=data["labels"].get(split),
            augment=False,
            prefix=split,
            names=data["names"],
        )

    def get_dataloader(self, dataset_path, batch_size):
        """Build validation data from the custom YAML rather than ImageFolder."""
        data = load_multilabel_classification_data(self.args.data)
        self.data = data
        split = getattr(self.args, "split", "val")
        dataset = self.build_dataset(data[split])
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)

    def plot_val_samples(self, batch, ni):
        """Disable scalar-class plotting, which cannot represent n-hot labels."""
        return None

    def plot_predictions(self, batch, preds, ni):
        """Disable the native top-1 prediction plot."""
        return None


__all__ = ("MultiLabelClassificationValidator",)
