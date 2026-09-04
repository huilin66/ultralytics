"""Synthetic tests for the multi-label loss."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from multilabel_yolo.loss import MultiLabelDetectionLoss


class _Head(nn.Module):
    """Minimal detection head metadata required by v8DetectionLoss."""

    def __init__(self):
        super().__init__()
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.nc = 5
        self.reg_max = 4


class _Model(nn.Module):
    """Minimal model exposing the native loss API."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))
        self.model = nn.ModuleList([nn.Identity(), _Head()])
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)


def test_forward_backward_with_single_multi_and_background_objects():
    """The loss handles one-label, two-label, and empty images together."""
    model = _Model()
    criterion = MultiLabelDetectionLoss(model, tal_topk=5)
    no = criterion.no
    features = [
        torch.randn(3, no, 8, 8, requires_grad=True),
        torch.randn(3, no, 4, 4, requires_grad=True),
        torch.randn(3, no, 2, 2, requires_grad=True),
    ]
    batch = {
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [99.0]]),  # second value is deliberately not a real class ID
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25], [0.4, 0.4, 0.2, 0.2]]),
        "cls_nhot": torch.tensor([[0, 1, 0, 0, 0], [0, 1, 0, 1, 0]], dtype=torch.float32),
    }
    total, items = criterion(features, batch)
    assert torch.isfinite(total)
    assert torch.isfinite(items).all()
    total.backward()
    assert any(feature.grad is not None and torch.isfinite(feature.grad).all() for feature in features)


def test_box_weight_is_cardinality_invariant():
    """The localization target sum is unchanged when one object gains labels."""
    single = torch.tensor([[[0.0, 0.8, 0.0, 0.0, 0.0]]])
    triple = torch.tensor([[[0.0, 0.8, 0.0, 0.8, 0.8]]])
    single_norm = MultiLabelDetectionLoss.normalize_box_scores(single)
    triple_norm = MultiLabelDetectionLoss.normalize_box_scores(triple)
    assert torch.allclose(single_norm.sum(-1), triple_norm.sum(-1))
    assert torch.allclose(triple_norm[0, 0, [1, 3, 4]], torch.full((3,), 0.8 / 3))
