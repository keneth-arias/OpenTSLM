# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from opentslm.forecast import QwenForecaster
from opentslm.forecast_inference import JointInputs, forecast, read_history
from test_qwen_forecast import tiny_model


@pytest.mark.parametrize("view", ["first", "both"])
def test_joint_mask_and_original_units(tmp_path, view):
    path = tmp_path / "scaler.json"
    path.write_text(
        json.dumps(dict(channels=["pm25", "pm10"], mean=[12, 40], scale=[3, 8], unit="ug/m3", version="fixture"))
    )
    joint = JointInputs.load(path, view)
    model = QwenForecaster(tiny_model().decoder, channels=2, targets=2).eval()
    ids = np.array(["a", "b"])
    x = torch.arange(96, dtype=torch.float32).reshape(2, 24, 2).numpy()
    data = dict(window_id=ids, history=x.copy(), channels=np.array(joint.channels))
    args = SimpleNamespace(precision="fp32", joint=joint)

    def run():
        _, history = read_history(data, joint)
        prediction, center, scale = forecast(model, history, args)
        return prediction * scale + center

    before = run()
    assert before.shape == (2, 6, 2)
    data["history"][:, :, 1] = 1000 if view == "both" else np.nan
    after = run()
    assert torch.equal(before, after) == (view == "first")
    after.square().mean().backward()
    assert model.head.weight.grad.isfinite().all()
    data["channels"] = data["channels"][::-1]
    with pytest.raises(ValueError):
        run()


def test_restored_joint_outputs_include_both_channels_and_final_example(tmp_path):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    from opentslm.forecast_inference import forecast_batches, load_checkpoint_model, state_options, write_predictions

    scaler = tmp_path / "scaler.json"
    scaler.write_text(json.dumps(dict(channels=["x", "y"], mean=[0, 3], scale=[1, 9], version="v1")))
    joint = JointInputs.load(scaler, "first")
    model = tiny_model().eval()
    args = SimpleNamespace(batch=2, precision="fp32", joint=joint)
    x = torch.randn(3, 24, 2)
    ids = np.array(["c", "b", "a"])

    def predictions(current):
        batches = list(forecast_batches(current, x, args, 0, 1, "cpu"))
        return np.concatenate([indices for indices, _ in batches]), np.concatenate([v for _, v in batches])

    expected = predictions(model)[1]
    dcp.save({"model": get_model_state_dict(model, options=state_options(False))}, checkpoint_id=tmp_path / "dcp")
    loaded = tiny_model().eval()
    load_checkpoint_model(loaded, tmp_path, False)
    x[:, :, 1] = float("nan")
    output = tmp_path / "predictions.npz"
    write_predictions(output, ids, [predictions(loaded)], "abc", joint)
    with np.load(output) as data:
        np.testing.assert_array_equal(expected, data["predicted"])
        np.testing.assert_array_equal(data["channels"], joint.channels)
        assert data["scaler_sha256"].item() == joint.digest
