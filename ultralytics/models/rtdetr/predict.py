# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import MdetResults, Results
from ultralytics.utils import ops


def _get_rtdetr_head(model):
    """Return the RT-DETR head from either a raw model or an AutoBackend wrapper."""
    model_core = getattr(model, "model", model)
    model_container = getattr(model_core, "model", model_core)
    return model_container[-1]


class RTDETRPredictor(BasePredictor):
    """
    RT-DETR (Real-Time Detection Transformer) Predictor extending the BasePredictor class for making predictions.

    This class leverages Vision Transformers to provide real-time object detection while maintaining high accuracy.
    It supports key features like efficient hybrid encoding and IoU-aware query selection.

    Attributes:
        imgsz (int): Image size for inference (must be square and scale-filled).
        args (dict): Argument overrides for the predictor.
        model (torch.nn.Module): The loaded RT-DETR model.
        batch (List): Current batch of processed inputs.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.rtdetr import RTDETRPredictor
        >>> args = dict(model="rtdetr-l.pt", source=ASSETS)
        >>> predictor = RTDETRPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def postprocess(self, preds, img, orig_imgs):
        """
        Postprocess the raw predictions from the model to generate bounding boxes and confidence scores.

        The method filters detections based on confidence and class if specified in `self.args`. It converts
        model predictions to Results objects containing properly scaled bounding boxes.

        Args:
            preds (List | Tuple): List of [predictions, extra] from the model, where predictions contain
                bounding boxes and scores.
            img (torch.Tensor): Processed input images with shape (N, 3, H, W).
            orig_imgs (List | torch.Tensor): Original, unprocessed images.

        Returns:
            (List[Results]): A list of Results objects containing the post-processed bounding boxes, confidence scores,
                and class labels.
        """
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
            preds = [preds, None]

        head = _get_rtdetr_head(self.model)
        attribute_channels = int(getattr(head, "attribute_channels", 0))
        if attribute_channels:
            bboxes, scores, attributes = preds[0].split(
                (4, head.nc, attribute_channels), dim=-1
            )
        else:
            nd = preds[0].shape[-1]
            bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
            attributes = None

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        model_core = getattr(self.model, "model", self.model)
        attribute_names = getattr(model_core, "attribute_names", None)
        if attribute_channels and not attribute_names:
            attribute_names = {
                str(i): [str(level) for level in range(head.nal)] for i in range(head.na)
            }

        attribute_batches = attributes if attributes is not None else [None] * len(bboxes)
        for bbox, score, attribute, orig_img, img_path in zip(
            bboxes, scores, attribute_batches, orig_imgs, self.batch[0]
        ):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            max_score, cls = score.max(-1, keepdim=True)  # (300, 1)
            idx = max_score.squeeze(-1) > self.args.conf  # (300, )
            if self.args.classes is not None:
                idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx
            pred = torch.cat([bbox, max_score, cls], dim=-1)[idx]  # filter
            attributes_pred = attribute[idx] if attribute is not None else None
            oh, ow = orig_img.shape[:2]
            pred[..., [0, 2]] *= ow  # scale x coordinates to original width
            pred[..., [1, 3]] *= oh  # scale y coordinates to original height
            if attributes_pred is None:
                results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
            else:
                results.append(
                    MdetResults(
                        orig_img,
                        path=img_path,
                        names=self.model.names,
                        boxes=pred,
                        attributes=attributes_pred,
                        attribute_names=attribute_names,
                        nc=head.nc,
                        na=head.na,
                        nal=head.nal,
                        risk_enlarge=self.args.risk_enlarge,
                        multiclass_attributes=head.multiclass_attributes,
                        attribute_channels=attribute_channels,
                    )
                )
        return results

    def pre_transform(self, im):
        """
        Pre-transforms the input images before feeding them into the model for inference. The input images are
        letterboxed to ensure a square aspect ratio and scale-filled. The size must be square(640) and scale_filled.

        Args:
            im (list[np.ndarray] |torch.Tensor): Input images of shape (N,3,h,w) for tensor, [(h,w,3) x N] for list.

        Returns:
            (list): List of pre-transformed images ready for model inference.
        """
        letterbox = LetterBox(self.imgsz, auto=False, scale_fill=True)
        return [letterbox(image=x) for x in im]
