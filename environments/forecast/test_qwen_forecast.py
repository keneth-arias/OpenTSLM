# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT

import pytest
import torch
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

from opentslm.forecast import QwenForecaster


def tiny_model():
    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["full_attention"],
    )
    return QwenForecaster(Qwen3_5TextModel(config), channels=2, horizon=6, targets=2)


def test_masked_values_and_weight_updates():
    torch.manual_seed(27)
    model = tiny_model()
    history = torch.randn(2, 24, 2)
    mask = torch.ones_like(history, dtype=torch.bool)
    mask[:, :, 1] = False
    forecast = model(history, mask)
    history[:, :, 1] = torch.nan
    assert torch.equal(forecast, model(history, mask))
    assert forecast.shape == (2, 6, 2)
    assert forecast.isfinite().all()
    decoder_weight = model.decoder.layers[0].mlp.down_proj.weight
    before = [p.detach().clone() for p in (decoder_weight, model.head.weight)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    forecast.square().mean().backward()
    optimizer.step()
    for original, updated in zip(before, (decoder_weight, model.head.weight), strict=True):
        assert not torch.equal(original, updated)


def test_invalid_history_rejected():
    model = tiny_model()
    with pytest.raises(ValueError, match="patch"):
        model(torch.zeros(1, 23, 2), torch.ones(1, 23, 2, dtype=torch.bool))


def test_future_changes_loss_but_not_prediction():
    from opentslm.forecast import smoke_step

    history = torch.zeros(1, 24, 2)
    mask = torch.ones_like(history, dtype=torch.bool)
    results = []
    for target in (0.0, 100.0):
        torch.manual_seed(27)
        model = tiny_model()
        results.append(smoke_step(model, history, mask, torch.full((1, 6, 2), target)))
    assert results[0]["prediction_before_step"] == results[1]["prediction_before_step"]
    assert results[0]["mse"] != results[1]["mse"]
