# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from opentslm import qwen_text, qwen_text_parity as parity


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"cases": [["es", "Hola", "hola"]]}))
    history = tmp_path / "history.npz"
    np.savez(history, window_id=["fixture"], history=np.zeros((1, 24), dtype="float32"))
    monkeypatch.setattr(qwen_text, "FIXTURE_SHA256", parity.digest(fixture.read_bytes()))
    monkeypatch.setattr(parity, "HISTORY_SHA256", parity.digest(history.read_bytes()))
    receipt = {
        "base_revision": qwen_text.BASE_REVISION, "base_manifest_sha256": qwen_text.BASE_MANIFEST_SHA256,
        "fixture_sha256": qwen_text.FIXTURE_SHA256, "tokenizer_manifest_sha256": qwen_text.TOKENIZER_MANIFEST_SHA256,
        "probe_sha256": parity.digest(Path(qwen_text.__file__).read_bytes()), "precision": "float32",
        "torch": "test", "transformers": "test", "cuda": "test", "gpu": "synthetic", "device": "cuda:0",
        "cases": [{"case": "es", "prompt_token_ids": [1], "reference_ids": [2],
                   "greedy_ids": [3], "reference_token_nll": [1.0]}],
    }
    base, repeat, after = [tmp_path / name for name in ("base", "repeat", "after")]
    for path, nll in ((base, 1), (repeat, 1.125), (after, 1.125)):
        receipt["cases"][0]["reference_token_nll"] = [nll]
        path.write_text(json.dumps(receipt))
    checkpoint = "a" * 64
    outputs = [tmp_path / name for name in ("first.npz", "second.npz")]
    for path in outputs:
        np.savez(path, checkpoint_sha256=checkpoint, window_id=["fixture"],
                 forecast_hour=np.arange(1, 7), predicted=np.ones((1, 6), dtype="float32"))
    return Namespace(fixture=fixture, history=history, base=base, repeat=repeat, after=after,
                     approved_baselines=[parity.digest(p.read_bytes()) for p in (base, repeat)],
                     checkpoint_sha256=checkpoint, forecasts=outputs)


def test_full_parity(inputs):
    result = parity.qualify(inputs)
    assert result["matching_horizons"] == 6
    assert result["repeat_max_abs_nll"] == result["restored_max_abs_nll"] == .125


@pytest.mark.parametrize(("field", "value"), [
    ("reference_token_nll", [1.12501]), ("reference_token_nll", [float("nan")]),
    ("reference_token_nll", []), ("reference_ids", [9]), ("greedy_ids", [False]), ("case", "wrong"),
])
def test_language_drift(inputs, field, value):
    receipt = json.loads(inputs.after.read_bytes())
    receipt["cases"][0][field] = value
    inputs.after.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        parity.qualify(inputs)


@pytest.mark.parametrize(("field", "value"), [
    ("checkpoint_sha256", "b" * 64), ("window_id", ["elsewhere"]),
    ("forecast_hour", np.arange(6)), ("predicted", np.zeros((1, 6), dtype="float32")),
    ("predicted", np.full((1, 6), np.nan, dtype="float32")),
])
def test_wrong_restore(inputs, field, value):
    path = inputs.forecasts[1]
    with np.load(path, allow_pickle=False) as archive:
        changed = dict(archive)
    changed[field] = value
    np.savez(path, **changed)
    with pytest.raises(ValueError):
        parity.qualify(inputs)


def test_changed_approval(inputs):
    inputs.repeat.write_bytes(inputs.after.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="approval"):
        parity.qualify(inputs)


def test_changed_runtime(inputs):
    receipt = json.loads(inputs.after.read_bytes())
    receipt["gpu"] = "other"
    inputs.after.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="runtime"):
        parity.qualify(inputs)


def test_two_paths_to_one_output(inputs):
    inputs.forecasts[1].unlink()
    inputs.forecasts[1].hardlink_to(inputs.forecasts[0])
    with pytest.raises(ValueError, match="separate runs"):
        parity.qualify(inputs)
