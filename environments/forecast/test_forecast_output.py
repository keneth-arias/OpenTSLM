# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp
from opentslm.forecast import QwenForecaster
from opentslm.forecast_inference import (
    add_lora,
    forecast_batches,
    load_checkpoint_model,
    state_options,
    write_predictions,
)
from test_qwen_forecast import tiny_model
from torch.distributed.checkpoint.state_dict import get_model_state_dict


def build(lora):
    torch.manual_seed(9)
    model = QwenForecaster(tiny_model().decoder)
    if lora:
        add_lora(model)
    return model


@pytest.mark.parametrize("lora", [False, True])
def test_restored_rows_match_before_save_and_exclude_last_batch_padding(tmp_path, lora):
    model = build(lora)
    args = SimpleNamespace(batch=4, precision="fp32")
    history = torch.arange(5 * 24, dtype=torch.float32).reshape(5, 24, 1)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.005)
    model.eval()
    expected = torch.cat([p for _, p in forecast_batches(model, history, args, 0, 1, "cpu")])
    dcp.save({"model": get_model_state_dict(model, options=state_options(lora))}, checkpoint_id=tmp_path / "dcp")
    restored = build(lora)
    load_checkpoint_model(restored, tmp_path, lora)
    shards = []
    for rank in range(2):
        batches = list(forecast_batches(restored.eval(), history, args, rank, 2, "cpu"))
        shards.append(
            (np.concatenate([i.numpy() for i, _ in batches]), np.concatenate([p.numpy() for _, p in batches]))
        )
    ids = np.array(["e", "b", "a", "c", "d"])
    output = tmp_path / "predictions.npz"
    write_predictions(output, ids, shards)
    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["window_id"], ids)
        np.testing.assert_array_equal(data["forecast_hour"], np.arange(1, 7))
        np.testing.assert_allclose(data["predicted"], expected.squeeze(-1).numpy(), rtol=1e-5)
    with pytest.raises(FileExistsError):
        write_predictions(output, ids, shards)
    with pytest.raises(ValueError):
        write_predictions(tmp_path / "bad.npz", ids, shards[:1])


@pytest.mark.parametrize("joint", [None, SimpleNamespace(channels=("pm25", "pm10"), view="both")])
def test_strict_history_archive_excludes_future_and_unrecognized_fields(joint):
    from opentslm.forecast_inference import read_history

    shape = (2, 24, 2) if joint else (2, 24)
    data = {"window_id": np.array(["b", "a"]), "history": np.ones(shape)}
    if joint:
        data["channels"] = np.array(joint.channels)
    ids, _ = read_history(data, joint, require_history_only=True)
    np.testing.assert_array_equal(ids, data["window_id"])
    for field in ("targets", "future_max"):
        with pytest.raises(ValueError, match="history-only"):
            read_history({**data, field: np.zeros(2)}, joint, require_history_only=True)
    read_history({**data, "targets": np.ones((2, 6))}, joint)
