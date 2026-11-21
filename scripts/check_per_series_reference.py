# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
"""Verify the staged Gemma checkpoint against the pinned AWS implementation on a Nebius GPU.

Provide the reference scripts/utils.py at AWS example commit 7967032 and the staged
weights/ directory with its parent SHA256SUMS. Each implementation loads sequentially.
"""

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path

import torch
import transformers

from opentslm.flamingo_forecast import MODEL_REVISION, WEIGHTS_MANIFEST_SHA256
from opentslm.forecast_inputs import verify_weight_snapshot
from opentslm.model.llm.per_series import PerSeriesFlamingo, SeriesInput

REFERENCE_SHA256 = "c19db31d2ce6e84a4ce02c9a0fdfe0869c53775f77b04dd4be5c893afd8335e2"
QUESTIONS = [
    ("Which sensor spiked?", [("A", [1, 1, None, 8, 1]), ("B", [1, 1, 1])]),
    ("Describe the trend.", [("PM2.5", [0, 2, 3, 6, 7, 10])]),
]


def reference_module(path):
    # Compile the exact verified bytes. The training environment stages this source.
    data = path.read_bytes()
    from hashlib import sha256

    if sha256(data).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Unexpected AWS reference source digest")
    spec = importlib.util.spec_from_file_location("opentslm_arfbench_reference", path)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    exec(compile(data, str(path), "exec"), reference.__dict__)
    return reference


@torch.inference_mode()
def traces(model, series_type):
    results = []
    for question, rows in QUESTIONS:
        series = [series_type(values=values, name=name) for name, values in rows]
        prompt, values, observed = model.prepare(series, question)
        ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        language = model.flamingo.lang_encoder
        with model.conditioned_on(values, observed):
            latents = language.language_model.layers[0].vis_x.detach().cpu()
            logits = language(ids, use_cache=False, logits_to_keep=1).logits.detach().cpu()
            generated = language.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=8,
                do_sample=False,
                eos_token_id=model.tokenizer.eos_token_id,
            )
        results.append(
            dict(
                prompt=prompt,
                ids=ids.cpu(),
                values=values,
                observed=observed,
                latents=latents,
                logits=logits,
                generated=generated[:, ids.shape[1] :].cpu(),
            )
        )
    return results


def compare(expected, actual):
    evidence = []
    for ref, result in zip(expected, actual, strict=True):
        if ref["prompt"] != result["prompt"]:
            raise ValueError("Prompt differs from reference")
        for key in ("ids", "values", "observed", "generated"):
            torch.testing.assert_close(result[key], ref[key], rtol=0, atol=0)
        case = {}
        for key in ("latents", "logits"):
            torch.testing.assert_close(result[key], ref[key], rtol=1e-5, atol=1e-5)
            case[key + "_max_abs_error"] = float((ref[key] - result[key]).abs().max())
        case["generated_ids"] = result["generated"][0].tolist()
        evidence.append(case)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if transformers.__version__ != "4.57.6" or not torch.cuda.is_available():
        raise ValueError("Run the pinned Transformers 4.57.6 environment on a managed GPU worker")
    verify_weight_snapshot(args.weights, WEIGHTS_MANIFEST_SHA256, strict=True)
    reference = reference_module(args.reference)
    baseline = reference.OpenTSLMFlamingo.from_file(args.weights, device="cuda", dtype=torch.float32)
    expected = traces(baseline, reference.SeriesInput)
    del baseline
    gc.collect()
    torch.cuda.empty_cache()
    model = PerSeriesFlamingo.from_file(args.weights, device="cuda", dtype=torch.float32)
    evidence = compare(expected, traces(model, SeriesInput))
    report = dict(
        protocol="per-series-reference-parity-v1",
        revision=MODEL_REVISION,
        reference_sha256=REFERENCE_SHA256,
        weights_manifest_sha256=WEIGHTS_MANIFEST_SHA256,
        torch=torch.__version__,
        transformers=transformers.__version__,
        device=torch.cuda.get_device_name(),
        dtype="float32",
        atol=1e-5,
        rtol=1e-5,
        cases=evidence,
    )
    with args.receipt.open("x") as out:
        json.dump(report, out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
