"""YOLO detection loss with n-hot classification targets."""

import torch
import torch.nn as nn

from ultralytics.utils.loss import v8DetectionLoss

from .tal import MultiLabelTaskAlignedAssigner


class MultiLabelDetectionLoss(v8DetectionLoss):
    """Detection loss retaining one localization target per physical object."""

    def __init__(self, model, tal_topk=10):
        super().__init__(model, tal_topk=tal_topk)
        # The repository's native loss has experiment-specific class-weight
        # behavior.  This independent path deliberately uses plain BCE as
        # required by the multi-label formulation.
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = MultiLabelTaskAlignedAssigner(
            topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0
        ).to(self.device)
        # YOLOv10's one-to-one branch uses the same loss formulation with a
        # single aligned candidate per object.  Keep a separate assigner so
        # the one-to-many and one-to-one branches do not share mutable TAL
        # state (``bs`` and ``n_max_boxes`` are updated on every call).
        self.one2one_assigner = MultiLabelTaskAlignedAssigner(
            topk=1, num_classes=self.nc, alpha=0.5, beta=6.0
        ).to(self.device)

    @staticmethod
    def normalize_box_scores(target_scores):
        """Normalize positive class weights so box/DFL weight is per object."""
        label_count = (target_scores > 0).sum(-1, keepdim=True).clamp_min(1)
        return target_scores / label_count

    def _validate_nhot(self, batch):
        """Validate the physical-box/n-hot alignment before target construction."""
        if "cls_nhot" not in batch:
            raise KeyError("Multi-label batches must contain 'cls_nhot'")
        cls_nhot = batch["cls_nhot"]
        bboxes = batch["bboxes"]
        if cls_nhot.ndim != 2 or cls_nhot.shape[-1] != self.nc:
            raise ValueError(f"cls_nhot must have shape [N, {self.nc}], got {tuple(cls_nhot.shape)}")
        if cls_nhot.shape[0] != bboxes.shape[0]:
            raise ValueError(
                f"Physical bbox count ({bboxes.shape[0]}) and cls_nhot count ({cls_nhot.shape[0]}) differ"
            )
        if cls_nhot.numel():
            if not torch.all((cls_nhot == 0) | (cls_nhot == 1)):
                raise ValueError("Raw cls_nhot targets must be binary")
            if not torch.all(cls_nhot.sum(1) >= 1):
                raise ValueError("Every physical object must have at least one positive class")
        return cls_nhot

    @staticmethod
    def _unwrap_predictions(preds):
        """Unwrap the training payload returned by a YOLO detection head."""
        return preds[1] if isinstance(preds, tuple) else preds

    def _branch_loss(self, preds, batch, assigner):
        """Calculate one branch of box, independent per-class BCE, and DFL losses."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [feature.view(feats[0].shape[0], self.no, -1) for feature in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = self._make_anchors(feats, dtype)

        cls_nhot = self._validate_nhot(batch).to(self.device, dtype=dtype)
        targets = torch.cat(
            (
                batch["batch_idx"].view(-1, 1).to(self.device, dtype=dtype),
                batch["cls"].view(-1, 1).to(self.device, dtype=dtype),  # transport ID, never a class index
                batch["bboxes"].to(self.device, dtype=dtype),
                cls_nhot,
            ),
            1,
        )
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_transport_ids, gt_bboxes, gt_labels_nhot = targets.split((1, 4, self.nc), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ = assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_transport_ids,
            gt_bboxes,
            gt_labels_nhot,
            mask_gt,
        )

        target_scores = target_scores.to(dtype)
        target_scores_sum = target_scores.sum().clamp_min(1.0)
        loss[1] = self.bce(pred_scores, target_scores).sum() / target_scores_sum

        if fg_mask.any():
            target_bboxes /= stride_tensor
            target_scores_box = self.normalize_box_scores(target_scores)
            target_scores_box_sum = target_scores_box.sum().clamp_min(1.0)
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores_box,
                target_scores_box_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss.sum() * batch_size, loss.detach()

    def __call__(self, preds, batch):
        """Calculate losses for a standard head or both YOLOv10 E2E branches."""
        preds = self._unwrap_predictions(preds)
        if isinstance(preds, dict) and {"one2many", "one2one"}.issubset(preds):
            loss_one2many = self._branch_loss(preds["one2many"], batch, self.assigner)
            loss_one2one = self._branch_loss(preds["one2one"], batch, self.one2one_assigner)
            return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]
        return self._branch_loss(preds, batch, self.assigner)

    def _make_anchors(self, feats, dtype):
        """Build anchors while keeping the native loss imports local to this extension."""
        from ultralytics.utils.tal import make_anchors

        return make_anchors(feats, self.stride, 0.5)
