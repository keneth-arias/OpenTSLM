# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
"""Exercise selected forecast restore on a synthetic Gemma 3 in the Flamingo runtime."""

import json

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from opentslm import flamingo_forecast as ff
from opentslm.forecast_checkpoint import publish, finish_direct, sha256
from opentslm.forecast_inference import state_options
from .test_per_series import model as model  # shared tiny, random Gemma fixture


def test_selected_forecast_roundtrip(model, tmp_path, monkeypatch):
    raw = torch.arange(48, dtype=torch.float32).reshape(2, 24, 1)
    observed = torch.ones_like(raw, dtype=torch.bool)
    weights = tmp_path / "weights"
    model.save_pretrained(weights)
    wrapped = ff.FlamingoForecaster(model)
    adapters = wrapped.add_lora()
    expected_names = [
        f"layers.{i}.decoder_layer.self_attn.{name}.lora_B.default.weight"
        for i in (0, 1)
        for name in ("q_proj", "v_proj")
    ]
    assert sorted(adapters) == sorted(expected_names)
    with torch.no_grad():
        for name in adapters:
            wrapped.decoder.get_parameter(name).fill_(0.125)
    expected = wrapped.eval()(raw, observed).detach()
    assert expected.shape == (2, 6, 1) and expected.isfinite().all()
    state = {"model": wrapped.adapter_state(get_model_state_dict(wrapped, options=state_options(True)))}
    assert set(state["model"]) == {key for key, p in wrapped.named_parameters() if p.requires_grad}
    identity = dict(
        model=ff.MODEL_REVISION,
        backbone="flamingo",
        weights=ff.WEIGHTS_MANIFEST_SHA256,
        precision="fp32",
        head="direct",
        lora=True,
        forecast_schema=ff.forecast_schema(),
    )
    stage = tmp_path / "staging"
    dcp.save(state, checkpoint_id=stage / "dcp", no_dist=True)
    checkpoint, _ = publish(tmp_path, stage, 500, identity)
    finish_direct(tmp_path, identity, 500, [dict(step=500, outcome="completed", best_step=500)])
    receipt = tmp_path / "direct-result.json"
    # Only the signed production weight-pin check is replaced for this tiny CPU fixture.
    monkeypatch.setattr(ff, "verify_base", lambda path: None)
    restored = ff.restore_selected(weights, checkpoint, receipt, expected_receipt_sha256=sha256(receipt))
    torch.testing.assert_close(restored(raw, observed), expected)
    text = torch.tensor([[3, 4]])
    with restored.generation_mode() as base:
        after = base.flamingo.lang_encoder(text, use_cache=False).logits
    plain = ff.FlamingoForecaster.from_pretrained(checkpoint_path=weights, local_files_only=True)
    with torch.no_grad():
        before = plain.backbone.flamingo.lang_encoder(text, use_cache=False).logits
    torch.testing.assert_close(after, before)
    torch.testing.assert_close(restored(raw, observed), expected)
    wrong = json.loads(receipt.read_bytes())
    wrong["best_step"] = 1000
    receipt.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="selected generation"):
        ff.restore_selected(weights, checkpoint, receipt, expected_receipt_sha256=sha256(receipt))
