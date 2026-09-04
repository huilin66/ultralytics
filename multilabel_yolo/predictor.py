"""Qualitative prediction path retaining multiple labels per candidate box."""

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import ops

from .inference import prepare_prediction_for_multilabel_nms


class MultiLabelDetectionPredictor(DetectionPredictor):
    """Detection predictor with explicit multi-label NMS."""

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Keep all class scores above the confidence threshold."""
        preds = prepare_prediction_for_multilabel_nms(preds, self.model)
        preds = ops.non_max_suppression(
            preds,
            conf_thres=self.args.conf,
            iou_thres=self.args.iou,
            classes=self.args.classes,
            agnostic=self.args.agnostic_nms,
            multi_label=True,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            end2end=False,
            rotated=self.args.task == "obb",
        )
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        return self.construct_results(preds, img, orig_imgs, **kwargs)
