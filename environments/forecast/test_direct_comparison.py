# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from opentslm.forecast import task_loss
from opentslm.forecast_checkpoint import finish_direct, publish, selected_generation
from opentslm.forecast_direct import evaluate_direct, selection
from opentslm.forecast_inputs import direct_cohorts


def make_cohorts(root, weights, fit_count=32, validation_count=3):
    """Only synthetic measurements, with non-overlapping time intervals."""
    pins = []
    for phase, count, offset in (
        ("fit", fit_count, 0),
        ("inner_validation", validation_count, fit_count + 40),
    ):
        path = root / phase
        path.mkdir(parents=True)
        ids = np.array([f"{phase}-{i:06d}" for i in range(count)])
        identity = sha256("".join(i + "\n" for i in ids).encode()).hexdigest()
        if phase == "fit":
            fit_ids = identity
        history = np.tile(np.arange(24, dtype=np.float32), (count, 1))
        rows = dict(
            window_id=ids,
            station_key=np.repeat("synthetic", count),
            origin_hour=np.arange(count, dtype=np.int64) + offset,
            history=history,
            targets=np.tile(np.arange(20, 26, dtype=np.float32), (count, 1)),
        )
        if phase == "fit":
            rows["graph_sample"] = np.arange(count) < validation_count
        np.savez(path / "data.npz", **rows)
        (path / "thresholds.json").write_text(
            json.dumps(
                dict(
                    method="linear_p95_unique_fit_history_station_hours",
                    fit_ids_sha256=fit_ids,
                    unit="ug/m3",
                    stations={"synthetic": 23.0},
                )
            )
        )
        spec = dict(
            schema="forecast-direct-cohort-v1",
            unit="ug/m3",
            head="direct",
            phase=phase,
            windows=count,
            weights_manifest_sha256=weights,
            protocol_sha256="c" * 64,
            source_manifest_sha256="d" * 64,
            ids_sha256=identity,
            files={k: sha256((path / k).read_bytes()).hexdigest() for k in ("data.npz", "thresholds.json")},
        )
        manifest = path / "manifest.json"
        manifest.write_text(json.dumps(spec))
        pins.extend((manifest, sha256(manifest.read_bytes()).hexdigest()))
    return pins


def test_pair_requires_separately_pinned_roles(tmp_path):
    pins = make_cohorts(tmp_path, "a" * 64)
    fit, inner = direct_cohorts(*pins, "a" * 64)
    assert fit["graph_sample"].sum() == len(inner["window_id"])
    with pytest.raises(ValueError, match="role, base or schema"):
        direct_cohorts(*pins[2:], *pins[:2], "a" * 64)
    (pins[0].parent / "thresholds.json").write_text("{}")
    with pytest.raises(ValueError, match="Corrupt"):
        direct_cohorts(*pins, "a" * 64)


def test_physical_loss_weights_pairs_and_gradients():
    p = torch.zeros((2, 6, 1), requires_grad=True)
    center, scale = torch.tensor([1.0, 5.0]).view(2, 1, 1), torch.tensor([2.0, 4.0]).view(2, 1, 1)
    y = center.expand_as(p) + scale
    assert task_loss("direct", p, y, center, scale).item() == 1.0
    loss = task_loss("direct", p, y, center, scale, direct_objective="physical-mae")
    assert loss.item() == 3.0
    loss.backward()
    torch.testing.assert_close(p.grad, -scale.expand_as(p) / 12)


def test_metrics_use_pair_counts_including_partial_batch_and_empty_high_strata(tmp_path):
    _, inner = direct_cohorts(*make_cohorts(tmp_path, "a" * 64), "a" * 64)

    class Constant(torch.nn.Module):
        def forward(self, x, mask):
            return x.new_zeros((len(x), 6, 1))

    model = Constant()
    args = SimpleNamespace(batch=2, head="direct", precision="fp32", direct_objective="physical-mae")
    inner["history"][:] = [[1] * 24, [3] * 24, [7] * 24]
    metrics = evaluate_direct(model, inner, args, "cpu")
    e = inner["history"][:, -1:] - inner["targets"]
    assert metrics["pairs"] == 18 and metrics["high_pairs"] == 9 and metrics["high_station_hours"] == 5
    assert metrics["high_blocks_168h"] == 1 and model.training
    assert metrics["objective"] == pytest.approx(np.abs(e).mean())
    assert metrics["mae_by_horizon"] == pytest.approx(np.abs(e).mean(axis=0))
    assert metrics["high_mae"] == pytest.approx(np.abs(e[:, 3:]).mean())
    inner["threshold"][:] = 100
    assert evaluate_direct(model, inner, args, "cpu")["high_mae"] is None


def test_resume_repairs_best_and_publishes_one_terminal_result(tmp_path):
    identity = {"direct_objective": "physical-mae", "seed": 27}
    history = []
    for step, mae in enumerate([0.0, 1.10, 1.08, 1.08, 1.07, 1.07, 1.065]):
        update = step * 500
        history.append(selection(history, update, mae, 5000))
        partial = tmp_path / ".saving"
        partial.mkdir()
        (partial / "training.pt.json").write_text(json.dumps(dict(evaluations=history)))
        publish(tmp_path, partial, update, identity)
    assert history[0]["best_step"] is None
    assert history[-1]["best_step"] == 1000 and history[-1]["step"] == 3000
    assert history[-1]["patience"] == 4 and history[-1]["outcome"] == "early-stopped"
    (tmp_path / "best").write_text("step-00000500")
    selected_generation(tmp_path, identity, history[-1]["best_step"])
    first = finish_direct(tmp_path, identity, 3000, history)
    assert (tmp_path / "best").read_text().strip() == "step-00001000"
    assert finish_direct(tmp_path, identity, 3000, history) == first
    with pytest.raises(ValueError, match="another run"):
        finish_direct(tmp_path, dict(identity, seed=28), 3000, history)


def test_rotation_keeps_selected_checkpoint_for_fallback(tmp_path):
    from opentslm.forecast_checkpoint import recover, rotate_direct

    identity = {"direct_objective": "physical-mae"}
    for step, best in [(500, 500), (1000, 500), (1500, 1500)]:
        partial = tmp_path / ".saving"
        partial.mkdir()
        metadata = {"evaluations": [{"step": step, "best_step": best}]}
        (partial / "training.pt.json").write_text(json.dumps(metadata))
        final, _ = publish(tmp_path, partial, step, identity)
        rotate_direct(tmp_path, identity, final)
    (final / "training.pt.json").write_text("damaged")
    assert recover(tmp_path, identity)[0] == 1000
    assert selected_generation(tmp_path, identity, 500).is_dir()


def test_previous_unpinned_trainer_can_resume(tmp_path):
    import runpy
    from pathlib import Path

    script = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/train_qwen_forecast.py"))
    previous = {
        "head": "direct",
        "trainer": "e4a32f79f1533d2583c71f3b74619bd36202b532197f62bcf74f95ef0339e920",
    }
    partial = tmp_path / ".saving"
    partial.mkdir()
    (partial / "state").write_bytes(b"state")
    final, _ = publish(tmp_path, partial, 2, previous)
    current = dict(previous, trainer="new")
    assert script["resume_generation"](tmp_path, current, False) == ((2, final), previous["trainer"])
    assert script["resume_generation"](tmp_path, current, True)[0] is None


def test_existing_snapshot_links_and_strict_direct_weights(tmp_path):
    from opentslm.forecast_inputs import verify_weight_snapshot

    weights = tmp_path / "weights"
    weights.mkdir()
    blob = tmp_path / "blob"
    blob.write_bytes(b"fixed")
    (weights / "model").symlink_to(blob)
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(sha256(blob.read_bytes()).hexdigest() + "  weights/model\n")
    digest = sha256(sums.read_bytes()).hexdigest()
    verify_weight_snapshot(weights, digest)
    with pytest.raises(ValueError, match="snapshot manifest"):
        verify_weight_snapshot(weights, digest, strict=True)


@pytest.mark.parametrize("rotate_before_crash", [True, False])
def test_damaged_shared_best_replays_past_abandoned_future(tmp_path, rotate_before_crash):
    from opentslm.forecast_checkpoint import reconcile_direct_resume, recover, rotate_direct

    identity = {"direct_objective": "normalized-mse"}

    def checkpoint(step, best):
        partial = tmp_path / ".pending"
        partial.mkdir()
        metadata = {"evaluations": [{"step": step, "best_step": best}]}
        (partial / "training.pt.json").write_text(json.dumps(metadata))
        return publish(tmp_path, partial, step, identity)[0]

    for step, best in [(500, 500), (1000, 1000), (1500, 1000)]:
        final = checkpoint(step, best)
        if step != 1500 or rotate_before_crash:
            rotate_direct(tmp_path, identity, final)
    (tmp_path / "step-00001000" / "training.pt.json").write_text("damaged")
    step, _ = recover(tmp_path, identity)
    assert step == 500
    reconcile_direct_resume(tmp_path, identity, step)
    assert (tmp_path / "latest").read_text().strip() == "step-00000500"
    for step, best in [(1000, 1000), (1500, 1000)]:
        rotate_direct(tmp_path, identity, checkpoint(step, best))
    assert recover(tmp_path, identity)[0] == 1500
    (tmp_path / "direct-result.json").write_text(
        json.dumps({"step": 1500, "identity": identity, "latest_manifest_sha256": "bad"})
    )
    with pytest.raises(ValueError, match="published terminal checkpoint"):
        reconcile_direct_resume(tmp_path, identity, 1000)


def test_allocator_budget_is_opt_in(monkeypatch):
    from unittest.mock import Mock
    from opentslm.forecast_inference import configure_device

    monkeypatch.setattr(torch.cuda, "set_device", Mock())
    fraction = Mock()
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", fraction)
    monkeypatch.delenv("FORECAST_GPU_LIMIT_BYTES", raising=False)
    configure_device(0)
    fraction.assert_not_called()
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda rank: (3 * 2**30, 4 * 2**30))
    monkeypatch.setenv("FORECAST_GPU_LIMIT_BYTES", str(2**30))
    configure_device(0)
    fraction.assert_called_once_with(0.25, 0)
