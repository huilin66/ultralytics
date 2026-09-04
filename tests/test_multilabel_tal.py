"""Tests for multi-label task-aligned assignment."""

import torch

from multilabel_yolo.tal import MultiLabelTaskAlignedAssigner
from ultralytics.utils.tal import TaskAlignedAssigner


def test_multilabel_target_has_all_positive_classes_and_safe_transport_id():
    """A transport ID outside nc must not be used to index prediction scores."""
    nc = 5
    assigner = MultiLabelTaskAlignedAssigner(topk=2, num_classes=nc, alpha=0.5, beta=6.0)
    pd_scores = torch.full((1, 4, nc), 0.5)
    pd_bboxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0]] * 4])
    anc_points = torch.tensor([[0.5, 0.5], [0.25, 0.25], [0.75, 0.75], [1.5, 1.5]])
    gt_labels = torch.tensor([[[99.0]]])
    gt_bboxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
    gt_nhot = torch.tensor([[[0.0, 1.0, 0.0, 1.0, 0.0]]])
    mask_gt = torch.ones((1, 1, 1), dtype=torch.bool)

    _, _, target_scores, fg_mask, _ = assigner(
        pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, gt_nhot, mask_gt
    )
    positive = fg_mask[0]
    assert positive.any()
    assert (target_scores[0, positive, 1] > 0).all()
    assert (target_scores[0, positive, 3] > 0).all()
    assert (target_scores[0, positive][:, [0, 2, 4]] == 0).all()


def test_single_label_alignment_matches_native_tal():
    """For one positive class, the n-hot score reduces to native TAL."""
    nc = 5
    multi = MultiLabelTaskAlignedAssigner(topk=2, num_classes=nc, alpha=0.5, beta=6.0)
    native = TaskAlignedAssigner(topk=2, num_classes=nc, alpha=0.5, beta=6.0)
    multi.bs = native.bs = 1
    multi.n_max_boxes = native.n_max_boxes = 1
    pd_scores = torch.tensor([[[0.2, 0.8, 0.3, 0.4, 0.1]] * 3])
    pd_bboxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0]] * 3])
    gt_bboxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
    mask_gt = torch.ones((1, 1, 3), dtype=torch.bool)
    gt_label = torch.tensor([[[1.0]]])
    gt_nhot = torch.tensor([[[0.0, 1.0, 0.0, 0.0, 0.0]]])

    multi_metric, multi_iou = multi.get_box_metrics(pd_scores, pd_bboxes, gt_bboxes, gt_nhot, mask_gt)
    native_metric, native_iou = native.get_box_metrics(pd_scores, pd_bboxes, gt_label, gt_bboxes, mask_gt)
    assert torch.allclose(multi_metric, native_metric)
    assert torch.allclose(multi_iou, native_iou)
