# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT

import json
from copy import deepcopy
from uuid import uuid4

import pytest

from opentslm import flamingo_qualification as probe
from opentslm.forecast_checkpoint import sha256


def case_rows():
    return [{"case": row[0], "prompt_token_ids": [4, 5], "reference_ids": [6],
             "greedy_ids": [7], "reference_token_nll": [1.0]} for row in probe.CASES]


@pytest.fixture
def runs(tmp_path):
    runtime = {"device": "cuda:0", "gpu": "fixture", "cuda": "test", "torch": "test",
               "transformers": "test", "peft": "test", "image_digest": "sha256:" + "b" * 64}
    result = []
    for index in range(4):
        rows = case_rows()
        receipt = {"protocol": probe.protocol(), "runtime": runtime, "process": str(uuid4()),
                   "before": rows}
        if index == 1:
            rows[0]["reference_token_nll"] = [1.125]
        if index >= 2:
            receipt.update(selected_receipt_sha256="a" * 64, selected_manifest_sha256="c" * 64,
                           forecast=[0.0] * 6, after=deepcopy(rows))
            receipt["after"][0]["reference_token_nll"] = [1.125]
        path = tmp_path / str(index)
        path.write_text(json.dumps(receipt))
        result.append(path)
    return result


def qualify(runs):
    return probe.compare(runs[:2], [sha256(p) for p in runs[:2]], runs[2:], "a" * 64, "c" * 64)


def test_repeat_tolerance_and_separate_restores(runs):
    assert qualify(runs)["repeat_max_abs_nll"] == 0.125


@pytest.mark.parametrize("change", ["nll", "tokens", "runtime", "forecast", "short",
                                    "nan", "wrong_selection", "duplicate", "cpu", "missing_probe"])
def test_rejects_drift_or_incomplete_evidence(runs, change):
    last = json.loads(runs[-1].read_bytes())
    if change == "nll":
        last["before"][-1]["reference_token_nll"] = [1.126]
    elif change == "tokens":
        last["after"][0]["greedy_ids"] = [8]
    elif change == "runtime":
        last["runtime"]["image_digest"] = "sha256:" + "f" * 64
    elif change == "forecast":
        last["forecast"][0] = 0.01
    elif change == "short":
        last["forecast"].pop()
    elif change == "nan":
        last["forecast"][0] = float("nan")
    elif change == "wrong_selection":
        last["selected_manifest_sha256"] = "b" * 64
    elif change == "duplicate":
        last["process"] = json.loads(runs[2].read_bytes())["process"]
    elif change == "cpu":
        last["runtime"]["device"] = "cpu"
    else:
        last["after"].pop()
    runs[-1].write_text(json.dumps(last))
    with pytest.raises(ValueError):
        qualify(runs)


def test_preapproved_baseline_bytes(runs):
    digests = [sha256(p) for p in runs[:2]]
    runs[1].write_bytes(runs[1].read_bytes() + b" ")
    with pytest.raises(ValueError, match="approval"):
        probe.compare(runs[:2], digests, runs[2:], "a" * 64, "c" * 64)


def test_freeze_emits_approval_evidence_before_fit(runs, capsys):
    probe.main(["freeze", "--baselines", *map(str, runs[:2])])
    summary = json.loads(capsys.readouterr().out)
    assert summary["baseline_receipts_sha256"] == [sha256(p) for p in runs[:2]]
    assert summary["repeat_max_abs_nll"] == 0.125
    assert "restored_receipts_sha256" not in summary


def test_cpu_baselines_cannot_qualify_a_managed_gpu(runs):
    for path in runs[:2]:
        receipt = json.loads(path.read_bytes())
        receipt["runtime"]["device"] = "cpu"
        path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="runtime"):
        probe.compare(runs[:2], None)
