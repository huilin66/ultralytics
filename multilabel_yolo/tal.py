"""Task-aligned assignment for multi-label physical objects."""

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.tal import TaskAlignedAssigner


class MultiLabelTaskAlignedAssigner(TaskAlignedAssigner):
    """Task-aligned assigner whose classification target is an n-hot vector."""

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, gt_labels_nhot, mask_gt):
        """Assign anchors using the mean score of the GT's positive classes."""
        if gt_labels_nhot.ndim != 3 or gt_labels_nhot.shape[-1] != self.num_classes:
            raise ValueError(
                "gt_labels_nhot must have shape [batch, max_gt, nc], "
                f"got {tuple(gt_labels_nhot.shape)} for nc={self.num_classes}"
            )
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx, device=device),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0], dtype=torch.bool),
                torch.zeros_like(pd_scores[..., 0], dtype=torch.long),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, gt_labels_nhot, mask_gt)
        except torch.OutOfMemoryError:
            LOGGER.warning("WARNING: CUDA OutOfMemoryError in MultiLabelTaskAlignedAssigner, using CPU")
            cpu_tensors = [
                tensor.cpu()
                for tensor in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, gt_labels_nhot, mask_gt)
            ]
            result = self._forward(*cpu_tensors)
            return tuple(tensor.to(device) for tensor in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, gt_labels_nhot, mask_gt):
        """Run assignment after the no-GT and OOM handling in :meth:`forward`."""
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_bboxes, gt_labels_nhot, anc_points, mask_gt
        )
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, gt_labels_nhot, target_gt_idx, fg_mask
        )

        # Keep the native Ultralytics alignment-quality normalization.  The
        # classification component itself is cardinality-invariant because
        # get_box_metrics uses a positive-label mean.
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_bboxes, gt_labels_nhot, anc_points, mask_gt):
        """Return candidate, top-k, and final foreground masks."""
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_bboxes, gt_labels_nhot, mask_in_gts * mask_gt
        )
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_bboxes, gt_labels_nhot, mask_gt):
        """Compute TAL metrics from the mean score over each GT's positive labels."""
        n_anchors = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros(
            (self.bs, self.n_max_boxes, n_anchors), dtype=pd_bboxes.dtype, device=pd_bboxes.device
        )

        nhot = gt_labels_nhot.to(dtype=pd_scores.dtype)
        label_count = nhot.sum(-1, keepdim=True).clamp_min(1.0)
        # [batch, anchors, nc] x [batch, gt, nc] -> [batch, gt, anchors]
        bbox_scores = torch.einsum("bac,bgc->bga", pd_scores, nhot) / label_count
        bbox_scores = bbox_scores.masked_fill(~mask_gt, 0)

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, n_anchors, -1)[mask_gt]
        if pd_boxes.numel():
            overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.clamp_min(0).pow(self.alpha) * overlaps.clamp_min(0).pow(self.beta)
        return align_metric, overlaps

    def get_targets(self, gt_labels, gt_bboxes, gt_labels_nhot, target_gt_idx, fg_mask):
        """Gather one box and its full n-hot vector for every assigned anchor."""
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        flat_target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[flat_target_gt_idx]
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[flat_target_gt_idx]
        target_scores = gt_labels_nhot.view(-1, self.num_classes)[flat_target_gt_idx]
        target_scores = target_scores * fg_mask.unsqueeze(-1).to(target_scores.dtype)
        return target_labels, target_bboxes, target_scores
