"""Semantic COCO-to-multimodal weight transfer without modifying Ultralytics checkpoint loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ultralytics.models.yolo.model import YOLO
from ultralytics.utils import LOGGER


@dataclass
class PretrainedTransferReport:
    """Summary of a semantic pretrained-weight transfer."""

    copied_tensors: int = 0
    transformed_tensors: int = 0
    skipped_tensors: int = 0
    initialized_layers: list[int] = field(default_factory=list)
    mappings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a concise transfer summary suitable for the training log."""
        return (
            f"copied={self.copied_tensors}, transformed={self.transformed_tensors}, "
            f"initialized={len(self.initialized_layers)}, skipped={self.skipped_tensors}"
        )


def load_coco_pretrained(
    target: nn.Module | Any,
    weights: str | Path | nn.Module | Any,
    plan: dict[str, Any] | None = None,
    verbose: bool = True,
) -> PretrainedTransferReport:
    """Transfer a YOLOv8 COCO checkpoint according to a model YAML semantic mapping.

    Args:
        target (nn.Module | Any): A ``MultiModalYOLO`` facade or its underlying task model.
        weights (str | Path | nn.Module | Any): YOLOv8 checkpoint, facade, or task model used as the source.
        plan (dict[str, Any] | None): Optional mapping. Defaults to ``target.yaml['multimodal']['pretrained']``.
        verbose (bool): Whether to log the resulting transfer report.

    Returns:
        (PretrainedTransferReport): Counts of copied, transformed, initialized, and skipped tensors.
    """
    target_model = _unwrap_model(target)
    plan = plan or getattr(target_model, "yaml", {}).get("multimodal", {}).get("pretrained")
    if not isinstance(plan, dict):
        raise ValueError("Multi-modal COCO transfer requires a multimodal.pretrained YAML mapping.")
    if plan.get("source") != "yolov8":
        raise ValueError(f"Only a yolov8 source is supported, got {plan.get('source')!r}.")

    source_model = _load_source_model(weights)
    source_layers = _model_layers(source_model, "source")
    target_layers = _model_layers(target_model, "target")
    report = PretrainedTransferReport()

    copies = plan.get("copies", [])
    if not isinstance(copies, list) or not copies:
        raise ValueError("multimodal.pretrained.copies must contain at least one semantic layer mapping.")
    for group_index, mapping in enumerate(copies):
        if not isinstance(mapping, dict):
            raise TypeError("Each multimodal.pretrained.copies entry must be a mapping.")
        source_indices = _expand_layer_ranges(mapping.get("source"), "source")
        target_indices = _expand_layer_ranges(mapping.get("target"), "target")
        if len(source_indices) != len(target_indices):
            raise ValueError(f"Copy mapping {group_index} has different source and target layer counts.")
        input_init = mapping.get("input_init", "copy")
        if input_init not in {"copy", "mean", "tile_scaled"}:
            raise ValueError(f"Unsupported input_init {input_init!r} in copy mapping {group_index}.")

        for offset, (source_index, target_index) in enumerate(zip(source_indices, target_indices)):
            source_layer = _get_layer(source_layers, source_index, "source")
            target_layer = _get_layer(target_layers, target_index, "target")
            _copy_layer_state(
                source_layer,
                target_layer,
                input_init=input_init if offset == 0 else "copy",
                report=report,
            )
            report.mappings.append(f"{source_index}->{target_index}")

    for target_index in _layer_indices(plan.get("identity_projections", []), "identity_projections"):
        _initialize_projection(
            _get_layer(target_layers, target_index, "target"), target_index, average=False, report=report
        )
    for target_index in _layer_indices(plan.get("fusion_projections", []), "fusion_projections"):
        _initialize_projection(
            _get_layer(target_layers, target_index, "target"), target_index, average=True, report=report
        )

    if verbose:
        LOGGER.info(f"Multi-modal COCO transfer: {report.summary()}")
    return report


def _unwrap_model(model: nn.Module | Any) -> nn.Module:
    """Return a task model from a facade or pass through a task model."""
    if isinstance(model, nn.Module):
        return model
    if isinstance(getattr(model, "model", None), nn.Module):
        return model.model
    raise TypeError(f"Expected an Ultralytics model or facade, got {type(model).__name__}.")


def _load_source_model(weights: str | Path | nn.Module | Any) -> nn.Module:
    """Load a YOLO source checkpoint or unwrap an already-built source model."""
    if isinstance(weights, (str, Path)):
        return YOLO(weights).model
    return _unwrap_model(weights)


def _model_layers(model: nn.Module, role: str) -> nn.Sequential:
    """Return the top-level sequential layer graph used by YOLO YAMLs."""
    layers = getattr(model, "model", None)
    if not isinstance(layers, nn.Sequential):
        raise TypeError(f"The {role} model does not expose a YOLO nn.Sequential graph.")
    return layers


def _expand_layer_ranges(value: Any, role: str) -> list[int]:
    """Expand one inclusive range or a list of inclusive ranges from YAML metadata."""
    if not isinstance(value, list) or not value:
        raise TypeError(f"{role} layer mapping must be a non-empty list.")
    if all(isinstance(item, int) for item in value):
        if len(value) == 1:
            return value
        if len(value) == 2:
            start, end = value
            if start > end:
                raise ValueError(f"{role} layer range must be ascending, got {value}.")
            return list(range(start, end + 1))
        raise ValueError(f"{role} layer mapping must be one index, one range, or a list of ranges.")
    ranges_are_valid = all(
        isinstance(item, list) and len(item) == 2 and all(isinstance(index, int) for index in item) for item in value
    )
    if not ranges_are_valid:
        raise TypeError(f"{role} layer mapping must contain inclusive integer ranges.")
    indices = []
    for start, end in value:
        if start > end:
            raise ValueError(f"{role} layer range must be ascending, got {[start, end]}.")
        indices.extend(range(start, end + 1))
    return indices


def _layer_indices(value: Any, name: str) -> list[int]:
    """Validate a simple YAML list of top-level target layer indices."""
    if not isinstance(value, list) or any(not isinstance(index, int) or index < 0 for index in value):
        raise TypeError(f"multimodal.pretrained.{name} must be a list of non-negative integer layer indices.")
    return value


def _get_layer(layers: nn.Sequential, index: int, role: str) -> nn.Module:
    """Return one top-level layer while producing a descriptive mapping error."""
    if not isinstance(index, int) or not 0 <= index < len(layers):
        raise IndexError(f"{role} layer {index!r} is outside the model graph of length {len(layers)}.")
    return layers[index]


def _copy_layer_state(
    source: nn.Module,
    target: nn.Module,
    *,
    input_init: str,
    report: PretrainedTransferReport,
) -> None:
    """Copy matching module state and adapt only a mapped first convolution when needed."""
    source_state = source.state_dict()
    target_state = target.state_dict()
    updates = {}
    for name, target_tensor in target_state.items():
        source_tensor = source_state.get(name)
        if source_tensor is not None and source_tensor.shape == target_tensor.shape:
            updates[name] = source_tensor
            report.copied_tensors += 1
        elif name == "conv.weight" and source_tensor is not None:
            adapted = _adapt_input_weight(source_tensor, target_tensor, input_init)
            if adapted is not None:
                updates[name] = adapted
                report.transformed_tensors += 1
            else:
                report.skipped_tensors += 1
        else:
            report.skipped_tensors += 1
    target.load_state_dict(updates, strict=False)


def _adapt_input_weight(source: torch.Tensor, target: torch.Tensor, strategy: str) -> torch.Tensor | None:
    """Adapt a 2D convolution's input-channel dimension without changing output or kernel geometry."""
    if (
        source.ndim != 4
        or target.ndim != 4
        or source.shape[0] != target.shape[0]
        or source.shape[2:] != target.shape[2:]
    ):
        return None
    source = source.to(device=target.device, dtype=target.dtype)
    target_channels = target.shape[1]
    if strategy == "copy":
        adapted = torch.zeros_like(target)
        adapted[:, : min(source.shape[1], target_channels)] = source[:, : min(source.shape[1], target_channels)]
        return adapted
    if strategy == "mean":
        return source.mean(dim=1, keepdim=True).expand(-1, target_channels, -1, -1).contiguous()
    if strategy == "tile_scaled":
        repeats = (target_channels + source.shape[1] - 1) // source.shape[1]
        return source.repeat(1, repeats, 1, 1)[:, :target_channels] * (source.shape[1] / target_channels)
    return None


def _initialize_projection(
    layer: nn.Module,
    layer_index: int,
    *,
    average: bool,
    report: PretrainedTransferReport,
) -> None:
    """Initialize a new 1x1 Conv projection as identity or a block-wise feature average."""
    conv = getattr(layer, "conv", None)
    if not isinstance(conv, nn.Conv2d) or conv.kernel_size != (1, 1) or conv.groups != 1:
        raise TypeError(f"Projection layer {layer_index} must be an Ultralytics 1x1 Conv with groups=1.")
    if average and conv.in_channels % conv.out_channels:
        raise ValueError(
            f"Fusion projection layer {layer_index} has incompatible {conv.in_channels}->{conv.out_channels} channels."
        )
    if not average and conv.in_channels != conv.out_channels:
        raise ValueError(
            "Identity projection layer "
            f"{layer_index} must preserve channels, got {conv.in_channels}->{conv.out_channels}."
        )

    branches = conv.in_channels // conv.out_channels if average else 1
    with torch.no_grad():
        conv.weight.zero_()
        for channel in range(conv.out_channels):
            for branch in range(branches):
                conv.weight[channel, branch * conv.out_channels + channel, 0, 0] = 1.0 / branches
        if conv.bias is not None:
            conv.bias.zero_()
        batch_norm = getattr(layer, "bn", None)
        if isinstance(batch_norm, nn.BatchNorm2d):
            batch_norm.weight.fill_(1)
            batch_norm.bias.zero_()
            batch_norm.running_mean.zero_()
            batch_norm.running_var.fill_(1)
            batch_norm.num_batches_tracked.zero_()
    report.initialized_layers.append(layer_index)
