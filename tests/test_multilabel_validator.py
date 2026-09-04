"""Tests for metric-only GT expansion and multi-label NMS."""

import torch

from multilabel_yolo.validator import MultiLabelDetectionValidator
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.utils.metrics import box_iou


def test_metric_expansion_does_not_change_physical_boxes():
    """A physical box is expanded only in the validator's local metric view."""
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
    nhot = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0]])
    expanded_boxes, classes, rows = MultiLabelDetectionValidator.expand_targets(boxes, nhot)
    assert boxes.shape == (1, 4)
    assert expanded_boxes.shape == (2, 4)
    assert torch.equal(classes, torch.tensor([1, 3]))
    assert torch.equal(rows, torch.tensor([0, 0]))
    assert torch.equal(expanded_boxes[0], expanded_boxes[1])


def test_multilabel_nms_keeps_two_classes_for_one_candidate():
    """Explicit multi-label NMS returns both class rows for the same candidate."""
    prediction = torch.tensor([[[0.5], [0.5], [0.4], [0.4], [0.1], [0.9], [0.2], [0.8], [0.1], [0.1]]])
    detections = ops.non_max_suppression(
        prediction, conf_thres=0.5, iou_thres=0.5, multi_label=True, nc=5, max_det=10
    )[0]
    assert detections.shape[0] == 2
    assert set(detections[:, 5].long().tolist()) == {1, 3}


def test_metric_matching_counts_each_positive_label_and_rejects_extra_label(tmp_path):
    """The class-wise metric view yields two TPs and an extra-label FP."""
    validator = MultiLabelDetectionValidator(
        save_dir=tmp_path,
        args=get_cfg(DEFAULT_CFG, overrides={"task": "detect", "plots": False}),
    )
    gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    gt_classes = torch.tensor([1, 3])
    pred_boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]]
    )
    pred_classes = torch.tensor([1, 3, 4])
    correct, _, _ = validator.match_predictions(
        pred_boxes, gt_boxes, pred_classes, gt_classes, box_iou(gt_boxes, pred_boxes)
    )
    assert correct[:, 0].tolist() == [True, True, False]
