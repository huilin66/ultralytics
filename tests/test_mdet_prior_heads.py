"""Tests for fixed co-occurrence prior attribute heads."""

from pathlib import Path

import torch
from torch import nn

from ultralytics.nn.modules.head import (
    CoOccurrencePriorBias,
    CoOccurrencePriorChannelAttention,
    CoOccurrencePriorMixtureHead,
    CoOccurrencePriorSpatialAttention,
    CoOccurrenceTextureAttention,
    MDetect,
)
from scripts.train_mdet_experiments import _materialize_config


def test_prior_heads_preserve_initial_attribute_predictions():
    torch.manual_seed(0)
    features = torch.randn(2, 32, 8, 8)
    output_layer = nn.Conv2d(32, 20, 1)
    baseline = output_layer(features)
    heads = (
        CoOccurrencePriorBias(10, 2),
        CoOccurrencePriorChannelAttention(32, 10, 2),
        CoOccurrencePriorSpatialAttention(32, 10, 2),
        CoOccurrencePriorMixtureHead(32, 10, 2),
        CoOccurrenceTextureAttention(32, 10, 2),
    )

    for head in heads:
        output = head(baseline) if isinstance(head, CoOccurrencePriorBias) else head(
            features, baseline, output_layer
        )
        torch.testing.assert_close(output, baseline, rtol=1e-5, atol=1e-5)


def test_prior_heads_keep_multiscale_mdetect_output_shape():
    torch.manual_seed(0)
    features = [torch.randn(2, 32, 8, 8), torch.randn(2, 64, 4, 4), torch.randn(2, 128, 2, 2)]
    for token in (
        "com_prior_bias",
        "com_prior_channel",
        "com_prior_spatial",
        "com_prior_moe",
        "com_prior_texture",
        "com_prior_channel_conditional",
    ):
        head = MDetect(nc=2, na=10, nal=2, params=[False, None, token, False, None], ch=[32, 64, 128])
        outputs = head([feature.clone() for feature in features])
        assert [tuple(output.shape) for output in outputs] == [
            (2, 86, 8, 8),
            (2, 86, 4, 4),
            (2, 86, 2, 2),
        ]


def test_prior_stage2_materializes_head_and_matrix_mode(tmp_path):
    config = tmp_path / "prior.yaml"
    matrix = tmp_path / "co_occurrence_matrix_train.csv"
    config.write_text(
        "head: [v10MDetect, [False, None, 'com_prior_channel', False, "
        "/nfsv4/data/co_occurrence_matrix_train.csv]]\n",
        encoding="utf-8",
    )
    matrix.write_text("placeholder", encoding="utf-8")

    resolved = _materialize_config(
        str(config),
        str(matrix),
        str(tmp_path / "generated"),
        prior_type="spatial",
        prior_conditional=True,
    )
    generated = Path(resolved).read_text(encoding="utf-8")
    assert "com_prior_spatial_conditional" in generated
    assert matrix.resolve().as_posix() in generated
