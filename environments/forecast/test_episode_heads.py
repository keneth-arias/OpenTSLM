# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from opentslm.forecast import EpisodeForecaster, QuantileForecaster, task_loss
from opentslm.forecast_inference import forecast, forecast_batches, read_task_data, write_predictions
from test_qwen_forecast import tiny_model


@pytest.mark.parametrize("head", ["quantile", "episode"])
def test_new_heads_train_and_export_without_future_observations(head, tmp_path):
    torch.manual_seed(8)
    model_type = QuantileForecaster if head == "quantile" else EpisodeForecaster
    model = model_type(tiny_model().decoder)
    x = torch.arange(72.).reshape(3, 24, 1)
    context = torch.tensor([20., 40., 60.]) if head == "episode" else None
    args = SimpleNamespace(head=head, precision="fp32", batch=2, joint=None)
    result, center, scale = forecast(model, x, args, context)
    if head == "episode":
        assert result.shape == (3,)
        shifted, _, _ = forecast(model, x*3+19, args, context*3+19)
        torch.testing.assert_close(result, shifted)
        observed = torch.tensor([0., 1., 0.])
    else:
        assert result.shape == (3, 6, 3)
        assert torch.all(result.diff(dim=-1) >= 0)
        observed = torch.ones((3, 6, 1))
    loss = task_loss(head, result, observed, center, scale)
    loss.backward()
    assert model.head.weight.grad.abs().sum() > 0
    data = [(i.numpy(), p.numpy()) for i,p in forecast_batches(model, x, args, 0, 1, "cpu", context)]
    ids = np.array(["b", "c", "a"])
    path = tmp_path/"forecast.npz"
    write_predictions(path, ids, data, head=head, input_sha256="a"*64, checkpoint_sha256="c"*64,
                      thresholds_sha256="f"*64 if head=="episode" else None)
    with np.load(path) as out:
        np.testing.assert_array_equal(out["window_id"],ids)
        key = "episode_logit" if head == "episode" else "quantiles"
        assert out[key].shape == tuple(result.shape)
        if head == "episode":
            np.testing.assert_allclose(out[key],result.detach().numpy(), atol=1e-5)
        else:
            np.testing.assert_allclose(out[key],(result*scale+center).detach().numpy(), atol=1e-5)


def test_episode_training_filters_only_ineligible_labels(tmp_path):
    p = tmp_path / "train.npz"
    np.savez(p, window_id=["a","b","c"], history=np.ones((3,24)), event=[1,-1,0],
             threshold_ug_m3=[9., 10., 11.], thresholds_sha256="a"*64)
    ids, history, y, limits = read_task_data(p,None,"episode","a"*64)
    assert ids.tolist() == ["a","c"]
    assert tuple(history.shape) == (2,24,1)
    assert y.tolist() == [1.,0.]
    assert limits.tolist() == [9.,11.]
    with pytest.raises(ValueError,match="threshold"):
        read_task_data(p,None,"episode","e"*64)


def test_known_pinball_loss():
    q = torch.tensor([[[0., 1., 2.]]])
    y = torch.tensor([[[1.]]])
    assert task_loss("quantile",q,y,torch.tensor(0.),torch.tensor(1.)).item() == pytest.approx(0.2/3)


@pytest.mark.parametrize("head,model_cls", [("episode", EpisodeForecaster), ("quantile", QuantileForecaster)])
def test_lora_checkpoint_restores_task_projection(tmp_path, head, model_cls):
    import torch.distributed.checkpoint as dcp
    from opentslm.forecast_inference import add_lora, load_checkpoint_model, state_options
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    def build():
        torch.manual_seed(27)
        model = model_cls(tiny_model().decoder)
        add_lora(model)
        return model.eval()

    model = build()
    x = torch.arange(48.).reshape(2,24,1)
    args = SimpleNamespace(head=head, precision="fp32")
    context = torch.tensor([10.,55.]) if head == "episode" else None
    output, center, scale = forecast(model, x, args, context)
    y = torch.tensor([0.,1.]) if head=="episode" else torch.ones(2,6,1)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss = task_loss(head,output,y,center,scale)
    loss.backward()
    optimizer.step()
    expected = forecast(model,x,args,context)[0].detach()
    assert not torch.equal(output.detach(),expected)
    dcp.save({"model":get_model_state_dict(model,options=state_options(True))}, checkpoint_id=tmp_path/"dcp")
    restored=build()
    load_checkpoint_model(restored,tmp_path,True)
    torch.testing.assert_close(forecast(restored,x,args,context)[0],expected)
