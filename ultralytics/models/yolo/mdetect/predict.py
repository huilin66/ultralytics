# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import MdetResults
from ultralytics.utils import ops


def _get_mdetect_head(model):
    """Return the final mdet head from an AutoBackend or a raw detection model."""
    model_core = getattr(model, "model", model)
    model_container = getattr(model_core, "model", model_core)
    return model_container[-1]


class MDetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model='yolov8n.pt', source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Post-processes predictions and returns a list of Results objects."""
        head = _get_mdetect_head(self.model)
        preds = ops.non_max_suppression_with_attributes(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            nc=head.nc,
            na=head.attribute_channels,
        )
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        return self.construct_results(preds, img, orig_imgs, **kwargs)

    def construct_results(self, preds, img, orig_imgs):
        """
        Construct a list of Results objects from model predictions.

        Args:
            preds (List[torch.Tensor]): List of predicted bounding boxes and scores for each image.
            img (torch.Tensor): Batch of preprocessed images used for inference.
            orig_imgs (List[np.ndarray]): List of original images before preprocessing.

        Returns:
            (List[Results]): List of Results objects containing detection information for each image.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]
    def construct_result(self, pred, img, orig_img, img_path):
        """
        Construct a single Results object from one image prediction.

        Args:
            pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
            img (torch.Tensor): Preprocessed image tensor used for inference.
            orig_img (np.ndarray): Original image before preprocessing.
            img_path (str): Path to the original image file.

        Returns:
            (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        attributes = pred[:, 6:]
        head = _get_mdetect_head(self.model)
        model_core = getattr(self.model, "model", self.model)
        attribute_names = model_core.attribute_names
        return MdetResults(
            orig_img,
            path=img_path,
            names=self.model.names,
            boxes=pred[:, :6],
            attributes=attributes,
            attribute_names=attribute_names,
            nc=head.nc,
            na=head.na,
            nal=head.nal,
            risk_enlarge=self.args.risk_enlarge,
            multiclass_attributes=head.multiclass_attributes,
            attribute_channels=head.attribute_channels,
        )
