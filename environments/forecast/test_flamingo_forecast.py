# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from opentslm import flamingo_forecast as ff
from opentslm.forecast import task_loss
from opentslm.forecast_checkpoint import sha256
from opentslm.forecast_inference import forecast
from opentslm.forecast_inputs import verify_flamingo_binding


class ToyBackbone(nn.Module):
    """Condition four features on a raw history, with Gemma's projection names."""

    hidden_size = 4

    def __init__(self):
        super().__init__()
        self.flamingo = nn.Module()
        self.flamingo.lang_encoder = nn.Module()
        core = self.flamingo.lang_encoder.language_model = nn.Module()
        layer = nn.Module()
        core.layers = nn.ModuleList([layer])
        layer.decoder_layer = nn.Module()
        attn = layer.decoder_layer.self_attn = nn.Module()
        attn.q_proj = nn.Linear(4, 4)
        attn.v_proj = nn.Linear(4, 4)

    def forecast_features(self, history, observed):
        self.seen = history
        attn = self.flamingo.lang_encoder.language_model.layers[0].decoder_layer.self_attn
        x = history.masked_fill(~observed, 0)[:, -4:]
        return attn.q_proj(x) + attn.v_proj(x)


def test_raw_history_readout_and_direct_losses():
    model = ff.FlamingoForecaster(ToyBackbone())
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
    raw = torch.tensor([[5.0] * 24, [8.0, 12.0] * 12]).unsqueeze(-1)
    predicted, center, scale = forecast(model, raw, SimpleNamespace(backbone="flamingo", precision="fp32"))
    torch.testing.assert_close(model.backbone.seen, raw.squeeze(-1))
    assert predicted.shape == (2, 6, 1)
    assert center.flatten().tolist() == [5.0, 10.0]
    assert scale.flatten().tolist() == [1.0, 2.0]
    targets = torch.tensor([[7.0] * 6, [14.0] * 6]).unsqueeze(-1)
    assert task_loss("direct", predicted, targets, center, scale).item() == 4.0
    assert task_loss("direct", predicted, targets, center, scale, direct_objective="physical-mae").item() == 3.0


def test_lora_allowlist_gradient_and_generation_restore():
    torch.manual_seed(27)
    model = ff.FlamingoForecaster(ToyBackbone())
    backbone = model.backbone
    history = torch.randn(2, 24, 1)
    observed = torch.ones_like(history, dtype=torch.bool)
    observed[:, -3] = False
    original = model(history, observed).detach()
    names = model.add_lora()
    assert len(names) == 2
    assert all("lora_" in n or n.startswith("head.") for n, p in model.named_parameters() if p.requires_grad)
    model(history, observed).sum().backward()
    for name in names:
        param = model.decoder.get_parameter(name)
        assert param.grad.abs().sum() > 0
        with torch.no_grad():
            param.fill_(0.1)
    active = model(history, observed).detach()
    assert not torch.allclose(original, active)
    with model.generation_mode() as base:
        assert base is backbone
        torch.testing.assert_close(model(history, observed), original)
    torch.testing.assert_close(model(history, observed), active)
    changed = history.masked_fill(~observed, float("nan"))
    torch.testing.assert_close(model(changed, observed), active)


def test_role_binding_rejects_swaps_and_qwen_snapshot(tmp_path):
    path = tmp_path / "binding.json"
    original = "1d76cc3cfd4c15f9c749372611e5ca6e0b6f910b1fcf0551ce9157e9bfef29c1"
    fit, inner = "1" * 64, "2" * 64
    spec = dict(
        schema="airaware-flamingo-binding-v1",
        model=ff.MODEL_ID,
        revision=ff.MODEL_REVISION,
        weights_manifest_sha256=ff.WEIGHTS_MANIFEST_SHA256,
        cohort_weights_manifest_sha256=original,
        roles={"fit": fit, "inner_validation": inner},
    )
    path.write_text(json.dumps(spec))
    assert verify_flamingo_binding(path, sha256(path), ff.WEIGHTS_MANIFEST_SHA256, fit, inner) == original
    for weights, first, second in ((ff.WEIGHTS_MANIFEST_SHA256, inner, fit), (original, fit, inner)):
        with pytest.raises(ValueError, match="binding"):
            verify_flamingo_binding(path, sha256(path), weights, first, second)
