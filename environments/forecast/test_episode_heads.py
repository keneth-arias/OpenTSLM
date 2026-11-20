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
    x = torch.arange(72.0).reshape(3, 24, 1)
    context = torch.tensor([20.0, 40.0, 60.0]) if head == "episode" else None
    args = SimpleNamespace(head=head, precision="fp32", batch=2, joint=None)
    result, center, scale = forecast(model, x, args, context)
    if head == "episode":
        assert result.shape == (3,)
        shifted, _, _ = forecast(model, x * 3 + 19, args, context * 3 + 19)
        torch.testing.assert_close(result, shifted)
        observed = torch.tensor([0.0, 1.0, 0.0])
    else:
        assert result.shape == (3, 6, 3)
        assert torch.all(result.diff(dim=-1) >= 0)
        observed = torch.ones((3, 6, 1))
    loss = task_loss(head, result, observed, center, scale)
    loss.backward()
    assert model.head.weight.grad.abs().sum() > 0
    data = [(i.numpy(), p.numpy()) for i, p in forecast_batches(model, x, args, 0, 1, "cpu", context)]
    ids = np.array(["b", "c", "a"])
    path = tmp_path / "forecast.npz"
    write_predictions(
        path,
        ids,
        data,
        head=head,
        input_sha256="a" * 64,
        input_manifest_sha256="1" * 64,
        checkpoint_sha256="c" * 64,
        thresholds_sha256="f" * 64 if head == "episode" else None,
    )
    with np.load(path) as out:
        np.testing.assert_array_equal(out["window_id"], ids)
        assert out["input_manifest_sha256"].item() == "1" * 64
        assert out["model_revision"].item()
        key = "episode_logit" if head == "episode" else "quantiles"
        assert out[key].shape == tuple(result.shape)
        if head == "episode":
            np.testing.assert_allclose(out[key], result.detach().numpy(), atol=1e-5)
        else:
            np.testing.assert_allclose(out[key], (result * scale + center).detach().numpy(), atol=1e-5)


def test_episode_training_filters_only_ineligible_labels(tmp_path):
    p = tmp_path / "train.npz"
    np.savez(
        p,
        window_id=["a", "b", "c"],
        station_key=["s", "t", "u"],
        history=np.ones((3, 24)),
        event=[1, -1, 0],
        threshold_ug_m3=[9.0, 10.0, 11.0],
        thresholds_sha256="a" * 64,
    )
    ids, history, y, limits = read_task_data(p, None, "episode", "a" * 64)
    assert ids.tolist() == ["a", "c"]
    assert tuple(history.shape) == (2, 24, 1)
    assert y.tolist() == [1.0, 0.0]
    assert limits.tolist() == [9.0, 11.0]
    with pytest.raises(ValueError, match="threshold"):
        read_task_data(p, None, "episode", "e" * 64)


def test_known_pinball_loss():
    q = torch.tensor([[[0.0, 1.0, 2.0]]])
    y = torch.tensor([[[1.0]]])
    assert task_loss("quantile", q, y, torch.tensor(0.0), torch.tensor(1.0)).item() == pytest.approx(0.2 / 3)


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
    x = torch.arange(48.0).reshape(2, 24, 1)
    args = SimpleNamespace(head=head, precision="fp32")
    context = torch.tensor([10.0, 55.0]) if head == "episode" else None
    output, center, scale = forecast(model, x, args, context)
    y = torch.tensor([0.0, 1.0]) if head == "episode" else torch.ones(2, 6, 1)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss = task_loss(head, output, y, center, scale)
    loss.backward()
    optimizer.step()
    expected = forecast(model, x, args, context)[0].detach()
    assert not torch.equal(output.detach(), expected)
    dcp.save({"model": get_model_state_dict(model, options=state_options(True))}, checkpoint_id=tmp_path / "dcp")
    restored = build()
    load_checkpoint_model(restored, tmp_path, True)
    torch.testing.assert_close(forecast(restored, x, args, context)[0], expected)


def test_candidate_manifest_binds_role_units_archive_and_station_thresholds(tmp_path):
    import json
    from opentslm.forecast_inference import verify_candidate_input
    from opentslm.forecast_checkpoint import sha256

    archive = tmp_path / "data.npz"
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"unit": "ug/m3", "stations": {"s1": 10, "s2": 20}}))
    digest = sha256(thresholds)

    def save(values):
        np.savez(
            archive,
            window_id=["a", "b"],
            station_key=["s1", "s2"],
            history=np.ones((2, 24)),
            threshold_ug_m3=values,
            thresholds_sha256=digest,
        )

    save([10, 20])
    manifest = tmp_path / "manifest.json"
    spec = {
        "schema": "forecast-input-v1",
        "unit": "ug/m3",
        "phase": "infer",
        "head": "episode",
        "files": {"data.npz": sha256(archive), "thresholds.json": digest},
        "thresholds_sha256": digest,
    }
    manifest.write_text(json.dumps(spec))
    verify_candidate_input(archive, manifest, sha256(manifest), "infer", "episode", digest)
    with pytest.raises(ValueError, match="phase"):
        verify_candidate_input(archive, manifest, sha256(manifest), "fit", "episode", digest)
    save([20, 10])
    with pytest.raises(ValueError, match="archive"):
        verify_candidate_input(archive, manifest, sha256(manifest), "infer", "episode", digest)
    spec["files"]["data.npz"] = sha256(archive)
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="threshold"):
        verify_candidate_input(archive, manifest, sha256(manifest), "infer", "episode", digest)


def test_nearly_flat_episode_history_keeps_context_bounded():
    class ContextOnly(torch.nn.Module):
        def forward(self, normalized, observed, context):
            return context

    args = SimpleNamespace(head="episode", precision="fp32")
    history = torch.full((2, 24, 1), 30.0)
    history[1, -1, 0] += 0.001
    logits, _, _ = forecast(ContextOnly(), history, args, torch.tensor([55.0, 55.0]))
    torch.testing.assert_close(logits, torch.tensor([25.0, 100.0]))


def test_checkpoint_task_identity_rejects_wrong_transform_levels_head_or_thresholds():
    from copy import deepcopy
    from opentslm.forecast import MODEL_REVISION
    from opentslm.forecast_inference import checkpoint_head, task_schema

    legacy = {"model": MODEL_REVISION, "head": "direct", "precision": "fp32", "lora": True}
    assert checkpoint_head(legacy, None, None, None) == "direct"
    for head in ("episode", "quantile"):
        spec = task_schema(head, "a" * 64)
        record = {**legacy, "head": head, "task_schema": spec}
        assert checkpoint_head(record, spec, head, None) == head
        for field, value in (
            ("transform", "other"),
            ("version", 888),
            ("thresholds_sha256", "f" * 64),
            ("levels", [0.9, 0.5, 0.1]),
        ):
            bad = deepcopy(spec)
            bad[field] = value
            with pytest.raises(ValueError):
                checkpoint_head(record, bad, head, None)
        with pytest.raises(ValueError):
            checkpoint_head(record, None, None, None)
        with pytest.raises(ValueError):
            checkpoint_head(record, spec, "direct", None)


def test_resume_recognizes_only_the_known_legacy_trainer(tmp_path):
    from opentslm.forecast_checkpoint import publish
    from runpy import run_path
    from pathlib import Path

    train = run_path(str(Path(__file__).resolve().parents[2] / "scripts/train_qwen_forecast.py"))
    recorded = {"head": "direct", "seed": 27, "trainer": train["LEGACY_TRAINER_SHA256"]}
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "model.bin").write_bytes(b"model")
    destination, _ = publish(tmp_path, stage, 12, recorded)
    current = {**recorded, "trainer": "current"}
    assert train["resume_generation"](tmp_path, current, False) == ((12, destination), recorded["trainer"])
    assert train["resume_generation"](tmp_path, {**current, "seed": 28}, False)[0] is None
    assert train["resume_generation"](tmp_path, current, True)[0] is None


def test_interval_validation_includes_last_partial_batch(monkeypatch):
    from pathlib import Path
    from runpy import run_path

    trainer = run_path(str(Path(__file__).resolve().parents[2] / "scripts/train_qwen_forecast.py"))

    class Fixed(torch.nn.Module):
        def forward(self, x, observed):
            return torch.tensor([0.0, 1.0, 2.0]).expand(len(x), 6, 3)

    monkeypatch.setattr(torch.distributed, "all_reduce", lambda values: None)
    args = SimpleNamespace(head="quantile", precision="fp32", batch=2, joint=None)
    # Flat zero histories leave these three quantiles in their original units.
    data = (["a", "b", "c"], torch.zeros(3, 24, 1), torch.tensor([0.0, 1.0, 4.0])[:, None, None].expand(3, 6, 1), None)
    result = trainer["validation"](Fixed(), data, args, 0, 1, "cpu")
    assert result["coverage_by_horizon"] == pytest.approx([2 / 3] * 6)
    assert result["width_by_horizon"] == [2.0] * 6 and result["crossed_by_horizon"] == [0] * 6


@pytest.mark.parametrize("count,expected", [(5, 2), (1, 1), (4, 1)])
def test_matched_point_uses_final_partial_batch_without_changing_legacy(count, expected):
    from pathlib import Path
    from runpy import run_path

    train = run_path(str(Path(__file__).resolve().parents[2] / "scripts/train_qwen_forecast.py"))
    assert train["epoch_batches"](count, 4, include_partial=True) == expected
    assert train["epoch_batches"](count, 4, include_partial=False) == count // 4


@pytest.mark.parametrize("head", ["direct", "quantile", "episode", "persistence"])
def test_previous_unpinned_checkpoint_resumes_only_with_same_recipe(tmp_path, head):
    from runpy import run_path
    from pathlib import Path
    from opentslm.forecast_checkpoint import publish
    from opentslm.forecast_inference import task_schema

    train = run_path(str(Path(__file__).resolve().parents[2] / "scripts/train_qwen_forecast.py"))
    prior = {"head": head, "trainer": train["PREVIOUS_TRAINER_SHA256"], "seed": 27}
    if head in ("episode", "quantile"):
        prior["task_schema"] = task_schema(head, "a" * 64)
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "weights").write_bytes(b"fixture")
    final, _ = publish(tmp_path, stage, 4, prior)
    now = {**prior, "trainer": "current"}
    resume = train["resume_generation"]
    assert resume(tmp_path, now, False) == ((4, final), prior["trainer"])
    assert resume(tmp_path, {**now, "matched_point": "complete_epoch_v1"}, False)[0] is None
    assert resume(tmp_path, {**now, "seed": 28}, False)[0] is None
    assert resume(tmp_path, now, True)[0] is None


def test_direct_and_quantile_cli_visit_identical_window_ids(monkeypatch, tmp_path):
    from runpy import run_path
    from pathlib import Path
    import sys

    train = run_path(str(Path(__file__).resolve().parents[2] / "scripts/train_qwen_forecast.py"))
    ids = np.array(["e", "b", "a", "c", "d"])
    base = [
        "train",
        "--mode",
        "single",
        "--batch",
        "4",
        "--data",
        str(tmp_path / "data.npz"),
        "--weights",
        str(tmp_path / "weights"),
        "--run-dir",
        str(tmp_path / "run"),
        "--stop-step",
        "2",
        "--schedule-steps",
        "2",
    ]
    schedules = {}
    for head in ("direct", "quantile", "legacy"):
        opt = ["--head", "quantile"] if head == "quantile" else ["--matched-point"] if head == "direct" else []
        monkeypatch.setattr(sys, "argv", base + opt)
        args = train["arguments"]()
        schedules[head] = [[ids[idx.numpy()].tolist() for idx in train["epoch_order"](ids, args, e).split(args.batch)] for e in (0, 1)]
        if head != "legacy":
            assert args.lora and args.validate_at == "final"
    assert schedules["direct"] == schedules["quantile"]
    for epoch, legacy in zip(schedules["direct"], schedules["legacy"], strict=True):
        assert [len(batch) for batch in epoch] == [4, 1]
        assert sorted(epoch[0] + epoch[1]) == sorted(ids) and epoch[:1] == legacy
