# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
"""Compare two frozen base receipts with text and forecast receipts from a restore."""

import argparse
import hashlib
import io
import json
import math
import re
from pathlib import Path

import numpy as np

from . import qwen_text

HISTORY_SHA256 = "62e44272eef33aa33c82b189edaee17c4b0e5d78435a0a5f81849bcc5e264e2a"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_cases(receipt, names):
    cases = receipt["cases"]
    if [row["case"] for row in cases] != names:
        raise ValueError("Incomplete or reordered language fixture")
    tokens, losses = [], []
    for row in cases:
        ids = [row[key] for key in ("prompt_token_ids", "reference_ids", "greedy_ids")]
        if any(not isinstance(seq, list) or not seq or
               any(type(i) is not int or i < 0 for i in seq) for seq in ids):
            raise ValueError("Invalid token sequence")
        nll = row["reference_token_nll"]
        if len(nll) != len(ids[1]) or any(type(x) not in (float, int) or
                                           not math.isfinite(x) or x < 0 for x in nll):
            raise ValueError("Invalid per-token reference losses")
        tokens.append(ids)
        losses.extend(nll)
    return tokens, np.asarray(losses, dtype=np.float64)


def compare_text(receipts, names):
    expected = {
        "fixture_sha256": qwen_text.FIXTURE_SHA256,
        "base_revision": qwen_text.BASE_REVISION,
        "base_manifest_sha256": qwen_text.BASE_MANIFEST_SHA256,
        "tokenizer_manifest_sha256": qwen_text.TOKENIZER_MANIFEST_SHA256,
        "probe_sha256": digest(Path(qwen_text.__file__).read_bytes()),
        "precision": "float32",
    }
    baseline = receipts[0]
    for key, value in expected.items():
        if baseline.get(key) != value:
            raise ValueError(f"Base receipt differs from the release at {key}")
    for key in ("torch", "transformers", "gpu", "device", "cuda"):
        if not isinstance(baseline.get(key), str) or not baseline[key]:
            raise ValueError(f"Missing runtime identity at {key}")
    metadata = {k: v for k, v in baseline.items() if k != "cases"}
    tokens, losses = read_cases(baseline, names)
    deltas = []
    for receipt in receipts[1:]:
        if {k: v for k, v in receipt.items() if k != "cases"} != metadata:
            raise ValueError("Language runtime or inputs changed")
        candidate_tokens, candidate_losses = read_cases(receipt, names)
        if candidate_tokens != tokens:
            raise ValueError("Language token IDs changed")
        deltas.append(float(np.max(np.abs(candidate_losses - losses))))
    tolerance, observed = deltas
    if observed > tolerance:
        raise ValueError("Restored language likelihoods exceed the frozen repeat tolerance")
    return {"cases": len(tokens), "reference_tokens": len(losses),
            "repeat_max_abs_nll": tolerance, "restored_max_abs_nll": observed}


def compare_forecasts(outputs, window, checkpoint):
    """Check actual direct-inference payloads from two separate clean restores."""
    if not re.fullmatch(r"[a-f0-9]{64}", checkpoint):
        raise ValueError("Expected a checkpoint manifest SHA-256")
    first = None
    for raw in outputs:
        with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
            if sorted(saved.files) != ["checkpoint_sha256", "forecast_hour", "predicted", "window_id"]:
                raise ValueError("Expected a single direct forecast")
            if saved["checkpoint_sha256"].shape != () or saved["checkpoint_sha256"].item() != checkpoint:
                raise ValueError("Prediction checkpoint differs from the selected manifest")
            if not np.array_equal(saved["window_id"], window) or not np.array_equal(saved["forecast_hour"], np.arange(1, 7)):
                raise ValueError("Prediction window or horizon mismatch")
            value = saved["predicted"]
            if value.shape != (1, 6) or value.dtype != np.dtype("float32") or not np.isfinite(value).all():
                raise ValueError("Expected six finite FP32 predictions")
            if first is not None and not np.array_equal(value, first):
                raise ValueError("Forecast changed after restore")
            first = value


def qualify(args):
    artifacts = [args.base, args.repeat, args.after, *args.forecasts]
    if len({(p.stat().st_dev, p.stat().st_ino) for p in artifacts}) != len(artifacts):
        raise ValueError("Provide receipt and prediction files from separate runs")
    base, repeat, after, *predictions = [p.read_bytes() for p in artifacts]
    if [digest(base), digest(repeat)] != args.approved_baselines:
        raise ValueError("Baseline receipt differs from pre-fit approval")
    fixture = args.fixture.read_bytes()
    history = args.history.read_bytes()
    if digest(fixture) != qwen_text.FIXTURE_SHA256 or digest(history) != HISTORY_SHA256:
        raise ValueError("Language fixture or forecast history differs from the frozen inputs")
    names = [row[0] for row in json.loads(fixture)["cases"]]
    result = compare_text([json.loads(raw) for raw in (base, repeat, after)], names)
    with np.load(io.BytesIO(history), allow_pickle=False) as inputs:
        compare_forecasts(predictions, inputs["window_id"], args.checkpoint_sha256)
    return {**result, "checkpoint_sha256": args.checkpoint_sha256, "matching_horizons": 6,
            "baseline_receipts_sha256": args.approved_baselines, "restored_text_sha256": digest(after),
            "forecast_files_sha256": [digest(raw) for raw in predictions]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "history", "base", "repeat", "after"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--approved-baselines", nargs=2, required=True, metavar=("BASE_SHA", "REPEAT_SHA"))
    parser.add_argument("--forecasts", type=Path, nargs=2, required=True, metavar=("FIRST", "SECOND"))
    parser.add_argument("--checkpoint-sha256", required=True)
    print(json.dumps(qualify(parser.parse_args()), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
