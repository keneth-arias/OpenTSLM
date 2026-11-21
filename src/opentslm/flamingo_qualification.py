# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
"""Freeze bilingual text and time-series QA references and qualify a selected Gemma restore."""

import argparse
import hashlib
import json
import os
import re
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace
from uuid import UUID, uuid4

import numpy as np
import torch

from . import flamingo_forecast
from .forecast_checkpoint import sha256
from .qwen_text_parity import read_cases

# The checkpoint specifies plain_text. A QA case is formatted by the backend
# (including its observed-value statistics and media marker). Text has no media.
CASES = (
    ("es-summary", "Resume en una frase: Ana midió el aire al amanecer y anotó el resultado.",
     "Ana registró una medición del aire al amanecer.", ()),
    ("en-summary", "Summarize in one sentence: Lee opened the window. Then Lee wrote down the time.",
     "Lee recorded when the window was opened.", ()),
    ("es-extract", "Extrae solo la hora: La reunión comienza a las ocho.", "ocho", ()),
    ("en-extract", "Extract only the color: The blue notebook is on the desk.", "blue", ()),
    ("es-instruction", "Escribe los días lunes y martes en orden inverso.", "martes, lunes", ()),
    ("en-instruction", "Write the words red and gold in reverse order.", "gold, red", ()),
    ("es-conversation", "Hola, ¿cómo te va hoy?", "Me va bien, gracias.", ()),
    ("en-conversation", "How was your walk today?", "It was pleasant.", ()),
    ("es-series", "¿La serie sube o baja?", "Sube.", (("A", (2, 4, None, 8, 10, 12)),)),
    ("en-series", "Which series ends higher, A or B?", "B.",
     (("A", (8, 6, 4, 2)), ("B", (2, 4, 6, 8)))),
)
FORECAST_HISTORY = tuple(range(1, 25))
MAX_NEW_TOKENS = 12
PROCESS_ID = str(uuid4())


def protocol():
    fixture = json.dumps([CASES, FORECAST_HISTORY, MAX_NEW_TOKENS], separators=(",", ":")).encode()
    return {
        "schema": "flamingo-language-qa-parity-v1",
        "base_revision": flamingo_forecast.MODEL_REVISION,
        "base_manifest_sha256": flamingo_forecast.WEIGHTS_MANIFEST_SHA256,
        "fixture_sha256": hashlib.sha256(fixture).hexdigest(),
        "probe_sha256": sha256(Path(__file__)),
        "precision": "float32",
    }


@torch.inference_mode()
def run_case(backbone, case):
    from .model.llm.per_series import SeriesInput

    name, question, reference, series = case
    context = nullcontext()
    prompt = question
    if series:
        items = [SeriesInput(values, name=label) for label, values in series]
        prompt, values, observed = backbone.prepare(items, question)
        context = backbone.conditioned_on(values, observed)
    tokenizer = backbone.tokenizer
    ids = tokenizer(prompt)["input_ids"]
    # Match PerSeriesFlamingo.loss: the training continuation has a leading
    # space at this boundary and includes EOS in its scored reference.
    answer = tokenizer(" " + reference, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    prefix = torch.tensor([ids], device=backbone.device)
    full = torch.tensor([ids + answer], device=backbone.device)
    lang = backbone.flamingo.lang_encoder
    with context:
        output = lang.generate(
            prefix, attention_mask=torch.ones_like(prefix), do_sample=False, num_beams=1,
            eos_token_id=tokenizer.eos_token_id, max_new_tokens=MAX_NEW_TOKENS,
        )
        logits = lang(full, attention_mask=torch.ones_like(full), use_cache=False,
                      logits_to_keep=len(answer) + 1).logits[0, :-1]
        loss = torch.nn.functional.cross_entropy(logits.float(), full[0, len(ids):], reduction="none")
    return {"case": name, "prompt_token_ids": ids, "reference_ids": answer,
            "greedy_ids": output[0, len(ids):].tolist(), "reference_token_nll": loss.tolist()}


def measurements(backbone):
    cases = [run_case(backbone, row) for row in CASES]
    read_cases({"cases": cases}, [row[0] for row in CASES])
    return cases


@torch.inference_mode()
def exercise_restore(model):
    from .forecast_inference import forecast

    with model.generation_mode() as base:
        before = measurements(base)
    device = next(model.parameters()).device
    history = torch.tensor(FORECAST_HISTORY, dtype=torch.float32, device=device).reshape(1, 24, 1)
    args = SimpleNamespace(backbone="flamingo", precision="fp32", head="direct")
    pred, center, scale = forecast(model, history, args)
    physical = pred * scale + center
    if physical.shape != (1, 6, 1) or physical.dtype != torch.float32 or not physical.isfinite().all():
        raise ValueError("Expected six finite FP32 concentrations")
    with model.generation_mode() as base:
        after = measurements(base)
    return {"before": before, "forecast": physical[0, :, 0].tolist(), "after": after}


def collect(args):
    from .forecast_inference import configure_device
    from .model.llm.per_series import PerSeriesFlamingo

    device = torch.device("cuda:0")
    configure_device(device)
    torch.manual_seed(27)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_digest):
        raise ValueError("Expected the digest of the deployed image")
    receipt = {
        "protocol": protocol(), "process": PROCESS_ID,
        "runtime": {"device": str(device), "gpu": torch.cuda.get_device_name(device),
                    "cuda": torch.version.cuda, "torch": str(torch.__version__),
                    "transformers": version("transformers"), "peft": version("peft"),
                    "image_digest": args.image_digest},
    }
    if args.command == "base":
        flamingo_forecast.verify_base(args.weights)
        base = PerSeriesFlamingo.from_pretrained(args.weights, device=str(device), dtype=torch.float32)
        receipt["before"] = measurements(base)
    else:
        model = flamingo_forecast.restore_selected(
            args.weights, args.checkpoint, args.selected_receipt,
            expected_receipt_sha256=args.selected_receipt_sha256, device=device,
        )
        receipt.update(selected_receipt_sha256=args.selected_receipt_sha256,
                       selected_manifest_sha256=sha256(args.checkpoint / "manifest.json"))
        receipt.update(exercise_restore(model))
    with NamedTemporaryFile(mode="w", dir=args.output.parent) as tmp:
        json.dump(receipt, tmp, sort_keys=True, allow_nan=False)
        tmp.flush()
        os.link(tmp.name, args.output)
    return {"receipt_sha256": sha256(args.output)}


def require_same_runtime(receipts):
    runtime = receipts[0]["runtime"]
    if (any(not isinstance(runtime.get(k), str) or not runtime[k]
            for k in ("torch", "transformers", "peft", "gpu", "cuda", "image_digest"))
            or runtime.get("device") != "cuda:0"
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", runtime["image_digest"])):
        raise ValueError("Incomplete target runtime identity")
    if len({UUID(r["process"]) for r in receipts}) != len(receipts):
        raise ValueError("Each receipt requires a fresh process")
    expected = protocol()
    for receipt in receipts:
        if receipt["protocol"] != expected or receipt["runtime"] != runtime:
            raise ValueError("Language protocol or runtime changed")


def reference_delta(rows, names, baseline_tokens, baseline_losses):
    tokens, losses = read_cases({"cases": rows}, names)
    if tokens != baseline_tokens:
        raise ValueError("Prompt, reference or greedy token IDs changed")
    return float(np.max(np.abs(losses - baseline_losses)))


def compare(baselines, approved, restored=(), selected_receipt=None, selected_manifest=None):
    if len(baselines) != 2 or len(restored) not in (0, 2):
        raise ValueError("Require two baseline receipts and a pair of selected restores")
    raw = [p.read_bytes() for p in (*baselines, *restored)]
    digests = [hashlib.sha256(b).hexdigest() for b in raw]
    if approved is not None and digests[:2] != approved:
        raise ValueError("Baseline receipt differs from pre-fit approval")
    receipts = [json.loads(b) for b in raw]
    base, repeat, *runs = receipts
    fields = {"protocol", "runtime", "process", "before"}
    extra = {"after", "selected_receipt_sha256", "selected_manifest_sha256", "forecast"}
    if any(set(r) != fields for r in (base, repeat)) or any(set(r) != fields | extra for r in runs):
        raise ValueError("Unexpected fields for this receipt")
    require_same_runtime(receipts)
    names = [case[0] for case in CASES]
    tokens, losses = read_cases({"cases": base["before"]}, names)
    tolerance = reference_delta(repeat["before"], names, tokens, losses)
    result = {"baseline_receipts_sha256": digests[:2], "repeat_max_abs_nll": tolerance,
              "cases": len(names), "reference_tokens": len(losses)}
    if not runs:
        return result
    if not all(isinstance(s, str) and re.fullmatch("[0-9a-f]{64}", s)
               for s in (selected_receipt, selected_manifest)):
        raise ValueError("Expected approved selection SHA-256 values")
    for run in runs:
        if (run["selected_receipt_sha256"] != selected_receipt
                or run["selected_manifest_sha256"] != selected_manifest):
            raise ValueError("Restored selection differs from the approved checkpoint")
        values = np.asarray(run["forecast"])
        if values.shape != (6,) or values.dtype.kind != "f" or not np.isfinite(values).all():
            raise ValueError("Invalid forecast")
    if runs[0]["forecast"] != runs[1]["forecast"]:
        raise ValueError("Forecast changed across fresh restores")
    observed = max(reference_delta(run[stage], names, tokens, losses)
                   for run in runs for stage in ("before", "after"))
    if observed > tolerance:
        raise ValueError("Reference likelihood changed beyond measured repeat tolerance")
    return {**result, "restored_receipts_sha256": digests[2:], "selected_receipt_sha256": selected_receipt,
            "selected_manifest_sha256": selected_manifest, "restored_max_abs_nll": observed,
            "matching_horizons": 6}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="command", required=True)
    for name in ("base", "restore"):
        sub = modes.add_parser(name, help="Collect one receipt in a fresh managed GPU process")
        sub.add_argument("--weights", type=Path, required=True)
        sub.add_argument("--image-digest", required=True)
        sub.add_argument("--output", type=Path, required=True)
        if name == "restore":
            sub.add_argument("--checkpoint", type=Path, required=True)
            sub.add_argument("--selected-receipt", type=Path, required=True)
            sub.add_argument("--selected-receipt-sha256", required=True)
    for name in ("freeze", "check"):
        sub = modes.add_parser(name, help="Compare previously collected receipts on CPU")
        sub.add_argument("--baselines", type=Path, nargs=2, required=True)
        if name == "check":
            sub.add_argument("--approved-baselines", nargs=2, required=True)
            sub.add_argument("--restored", type=Path, nargs=2, required=True)
            sub.add_argument("--selected-receipt-sha256", required=True)
            sub.add_argument("--selected-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = compare(args.baselines, None)
    elif args.command == "check":
        result = compare(args.baselines, args.approved_baselines, args.restored,
                         args.selected_receipt_sha256, args.selected_manifest_sha256)
    else:
        result = collect(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
