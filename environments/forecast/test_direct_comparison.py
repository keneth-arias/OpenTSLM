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
