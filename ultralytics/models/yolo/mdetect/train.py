# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import random
from copy import copy

import numpy as np
import torch
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset, build_yolo_mdet_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import MDetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first


class MDetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model='yolov8n.pt', data='coco8.yaml', epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_mdet_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(self.args.imgsz * 0.5, self.args.imgsz * 1.5 + self.stride)
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def _setup_train(self, world_size):
        """Set up training, then compute attribute weights once labels are available."""
        super()._setup_train(world_size)
        self.set_class_weights()

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        self.model.na = self.data["na"] # attach number of attributes to model
        self.model.nal = self.data['nal']
        self.model.attribute_names = self.data["attributes"] # attach attribute names to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_attribute_class_counts(self):
        """Return per-attribute, per-level frequencies from the training labels."""
        na = int(self.data["na"])
        nal = int(self.data["nal"])
        counts = np.zeros((na, nal), dtype=np.float64)

        for label in self.train_loader.dataset.labels:
            attributes = label.get("mdet_attributes")
            if attributes is None:
                continue
            attributes = np.asarray(attributes, dtype=np.float64)
            if attributes.size == 0:
                continue
            if attributes.ndim == 1:
                attributes = attributes.reshape(1, -1)

            for attribute_index in range(min(na, attributes.shape[1])):
                values = attributes[:, attribute_index]
                valid = (
                    np.isfinite(values)
                    & (values >= 0)
                    & (values < nal)
                    & (values == np.floor(values))
                )
                if valid.any():
                    counts[attribute_index] += np.bincount(
                        values[valid].astype(np.int64), minlength=nal
                    )[:nal]
        return counts

    def compute_attribute_class_weights(self, class_counts):
        """Convert attribute frequencies to normalized inverse-frequency weights."""
        power = float(getattr(self.args, "att_pw", 0.0))
        if not 0.0 <= power <= 1.0:
            raise ValueError(f"att_pw must be in the range [0, 1], got {power}")

        active = class_counts > 0
        safe_counts = np.where(active, class_counts, 1.0)
        weights = np.power(1.0 / safe_counts, power)
        weights[~active] = 0.0

        # Each attribute has its own softmax. Normalize within each attribute so
        # a rare attribute does not change the global mdet loss scale.
        for attribute_index in range(weights.shape[0]):
            active_weights = active[attribute_index]
            if active_weights.any():
                weights[attribute_index] /= weights[attribute_index, active_weights].mean()
        return weights.astype(np.float32)

    def set_class_weights(self):
        """Compute and attach per-attribute class weights for the softmax mdet loss."""
        model = de_parallel(self.model)
        power = float(getattr(self.args, "att_pw", 0.0))
        if not 0.0 <= power <= 1.0:
            raise ValueError(f"att_pw must be in the range [0, 1], got {power}")

        # Clear weights from a resumed checkpoint when weighting is disabled or
        # when the model uses the legacy single-logit attribute layout.
        model.attribute_class_weights = None
        model.criterion = None
        if getattr(self, "ema", None) is not None:
            self.ema.ema.attribute_class_weights = None

        # nal=1 is not a classification problem and follows the legacy path.
        if power == 0.0 or int(self.data["nal"]) <= 1:
            return

        class_counts = self.get_attribute_class_counts()
        if not class_counts.any():
            LOGGER.warning("WARNING ⚠️ No valid attribute labels found; attribute class weighting is disabled")
            return

        weights = self.compute_attribute_class_weights(class_counts)
        model.attribute_class_weights = torch.from_numpy(weights).to(self.device)
        model.criterion = None
        if getattr(self, "ema", None) is not None:
            self.ema.ema.attribute_class_weights = model.attribute_class_weights.detach().clone()
        LOGGER.info(f"Attribute class weights: {weights.round(3).tolist()}")

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = MDetectionModel(
            cfg,
            nc=self.data["nc"],
            na=self.data.get("na"),
            nal=self.data.get("nal"),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "att_loss"
        return yolo.mdetect.MDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)
