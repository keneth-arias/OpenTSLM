# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
"""Offline text generation and reference scoring on the frozen Qwen base."""

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

from .forecast_checkpoint import sha256

BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
BASE_MANIFEST_SHA256 = "1d76cc3cfd4c15f9c749372611e5ca6e0b6f910b1fcf0551ce9157e9bfef29c1"
FIXTURE_SHA256 = "6a0b61962bcb6de384da9acab313ca61c10ec0c1f47b4a65e98676660e9b6981"
TOKENIZER_FILES = {
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
}
TOKENIZER_MANIFEST_SHA256 = hashlib.sha256("".join(
    f"{digest}  {name}\n" for name, digest in sorted(TOKENIZER_FILES.items())
).encode("ascii")).hexdigest()


def verify_inputs(weights, expected_base_digest, tokenizer, fixture):
    from .forecast_inputs import verify_weight_snapshot

    if expected_base_digest != BASE_MANIFEST_SHA256:
        raise ValueError("Expected the pinned Qwen base manifest")
    verify_weight_snapshot(weights, expected_base_digest, strict=True)
    if tokenizer.is_symlink() or {path.name for path in tokenizer.iterdir()} != TOKENIZER_FILES.keys():
        raise ValueError("Tokenizer directory must contain exactly the pinned files")
    for name, digest in TOKENIZER_FILES.items():
        path = tokenizer / name
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Tokenizer checksum mismatch for {name}")
    fixture_bytes = fixture.read_bytes()
    if hashlib.sha256(fixture_bytes).hexdigest() != FIXTURE_SHA256:
        raise ValueError("Language fixture checksum mismatch")
    return json.loads(fixture_bytes)


def run_case(model, tokenizer, name, prompt, reference, max_new_tokens, device):
    """Decode one greedy continuation and score the fixed reference token by token."""
    import torch

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    answer = tokenizer.encode(reference, add_special_tokens=False)
    if not ids or not answer:
        raise ValueError(f"Case {name} requires nonempty prompt and reference token IDs")
    with torch.inference_mode():
        prefix = torch.tensor([ids], device=device)
        generated = model.generate(
            input_ids=prefix,
            attention_mask=torch.ones_like(prefix),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )[0, len(ids):]
        target = torch.tensor([ids + answer], device=device)
        logits = model(
            input_ids=target, attention_mask=torch.ones_like(target), use_cache=False,
        ).logits[0, len(ids) - 1:-1]
        nll = torch.nn.functional.cross_entropy(
            logits.float(), target[0, len(ids):], reduction="none",
        )
        if not torch.isfinite(nll).all():
            raise ValueError(f"Case {name} produced nonfinite reference token loss")
    return {
        "case": name,
        "prompt_token_ids": ids,
        "reference_ids": answer,
        "greedy_ids": generated.tolist(),
        "reference_token_nll": nll.tolist(),
    }


def run_fixture(weights, tokenizer_path, fixture, device):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    torch.manual_seed(27)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False,
    )
    model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        weights, local_files_only=True, dtype=torch.float32, output_loading_info=True,
    )
    if info["missing_keys"] or info["mismatched_keys"]:
        raise ValueError("Incomplete language weights")
    model.eval().to(device)
    return {
        "precision": "float32",
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "device": str(device),
        "cases": [
            run_case(model, tokenizer, name, prompt, reference, fixture["max_new_tokens"], device)
            for name, prompt, reference in fixture["cases"]
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Path to the base weights directory")
    parser.add_argument("--base-manifest-sha256", required=True, help="Digest used by forecast mode")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Path to the pinned tokenizer directory")
    parser.add_argument("--fixture", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="New private JSON receipt path")
    mode.add_argument("--verify-only", action="store_true", help="Verify input files without loading the model")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    args = parser.parse_args(argv)

    fixture = verify_inputs(args.base, args.base_manifest_sha256, args.tokenizer, args.fixture)
    receipt = {
        "base_revision": BASE_REVISION,
        "base_manifest_sha256": BASE_MANIFEST_SHA256,
        "fixture_sha256": FIXTURE_SHA256,
        "probe_sha256": sha256(Path(__file__)),
        "tokenizer_manifest_sha256": TOKENIZER_MANIFEST_SHA256,
    }
    if args.verify_only:
        print(json.dumps(receipt, sort_keys=True))
        return

    if args.output.exists():
        raise FileExistsError(args.output)
    receipt.update(run_fixture(args.base, args.tokenizer, fixture, args.device))
    payload = json.dumps(receipt, sort_keys=True, allow_nan=False)
    with open(
        args.output, "x", opener=lambda path, flags: os.open(path, flags, 0o600),
    ) as stream:
        stream.write(payload + "\n")


if __name__ == "__main__":
    main()
