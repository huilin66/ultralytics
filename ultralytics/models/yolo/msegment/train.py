# Ultralytics YOLO 🚀, AGPL-3.0 license
from copy import copy

from ultralytics.nn.tasks import MSegmentationModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics.models import yolo
from ultralytics.models.yolo.mdetect import MDetectionTrainer


class MSegmentationTrainer(MDetectionTrainer):
    """
    A class extending the MDetectionTrainer class for training based on a msegmentation model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model='yolov8n.pt', data='coco8.yaml', epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a MSegmentationTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "msegment"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO SegmentationModel model."""
        model = MSegmentationModel(cfg, ch=3, nc=self.data["nc"], na=self.data["na"], nal=self.data["nal"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "att_loss"
        return yolo.msegment.MSegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, segment=True, on_plot=self.on_plot)  # save results.png