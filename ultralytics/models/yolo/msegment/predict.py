# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo.mdetect.predict import MDetectionPredictor
from ultralytics.engine.results import MdetResults
from ultralytics.utils import DEFAULT_CFG, ops


class MSegmentationPredictor(MDetectionPredictor):
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
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the SegmentationPredictor with the provided configuration, overrides, and callbacks."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "msegment"

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        protos = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
        return super().postprocess(preds[0], img, orig_imgs, protos=protos)

    def construct_results(self, preds, img, orig_imgs, protos):
        """
        Construct a list of result objects from the predictions.

        Args:
            preds (List[torch.Tensor]): List of predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_imgs (List[np.ndarray]): List of original images before preprocessing.
            protos (List[torch.Tensor]): List of prototype masks.

        Returns:
            (List[Results]): List of result objects containing the original images, image paths, class names,
                bounding boxes, and masks.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path, proto)
            for pred, orig_img, img_path, proto in zip(preds, orig_imgs, self.batch[0], protos)
        ]

    def construct_result(self, pred, img, orig_img, img_path, proto):
        """
        Construct a single result object from the prediction.

        Args:
            pred (np.ndarray): The predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_img (np.ndarray): The original image before preprocessing.
            img_path (str): The path to the original image.
            proto (torch.Tensor): The prototype masks.

        Returns:
            (Results): Result object containing the original image, image path, class names, bounding boxes, and masks.
        """
        nc, na = self.model.model.nc, self.model.model.na
        attributes = pred[:, 6:6+na]
        attribute_names = self.model.model.attribute_names
        if not len(pred):  # save empty boxes
            masks = None
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto, pred[:, 6+na:], pred[:, :4], orig_img.shape[:2])  # HWC
        else:
            masks = ops.process_mask(proto, pred[:, 6+na:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.sum((-2, -1)) > 0  # only keep predictions with masks
            pred, masks, attributes = pred[keep], masks[keep], attributes[keep]
        return MdetResults(
            orig_img,
            path=img_path,
            names=self.model.names,
            boxes=pred[:, :6],
            masks=masks,
            attributes=attributes,
            attribute_names=attribute_names,
            nc=self.model.model.nc,
            na=self.model.model.na,
            nal=self.model.model.nal,
        )
