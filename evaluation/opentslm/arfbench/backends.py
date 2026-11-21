# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT

"""Checkpoint adapters for prompt-only generation and answer-token training."""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from opentslm.model.llm.per_series import PerSeriesFlamingo, SeriesInput, _scale, _statistic
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

DEFAULTS = {
    "legacy": ("OpenTSLM/llama-3.2-1b-tsqa-flamingo", "923332423e1b892d78d4dc8e5854d424b6fa9a92"),
    "per-series": ("opentime/tslm-nov26", "5e030b001fb8d4be6d10ebb3a8b9bc3e6ef80983"),
}
BACKBONE = "meta-llama/Llama-3.2-1B"
BACKBONE_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"


def snapshot(spec, revision, patterns):
    from huggingface_hub import snapshot_download

    path = Path(spec)
    if path.exists():
        if revision:
            raise ValueError("omit the Hub revision for a local checkpoint")
        files = [path] if path.is_file() else sorted({f for p in patterns for f in path.glob(p)})
        hashes = {}
        for f in files:
            with f.open("rb") as stream:
                hashes[f.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        return path, {"path": str(path.resolve()), "sha256": hashes}
    path = Path(snapshot_download(spec, revision=revision, allow_patterns=patterns))
    return path, {"repo": spec, "revision": path.name}


def legacy_batch(series, question):
    rows, descriptions = [], []
    for item in series:
        scaled, _, mean, std = _scale(item.values, item.observed)
        rows.append(scaled)
        descriptions.append(
            f"This is time series channel {item.name}, with mean {_statistic(mean)} and std {_statistic(std)}."
        )
    return extend_time_series_to_match_patch_size_and_aggregate(
        [dict(pre_prompt="", time_series=rows, time_series_text=descriptions, post_prompt=question)]
    )


class Legacy(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def trained_modules(self):
        core = self.model.model
        return (
            core.vision_encoder,
            core.perceiver,
            core.lang_encoder.gated_cross_attn_layers,
            core.lang_encoder.get_input_embeddings(),
        )

    def generate(self, series, question, *, max_new_tokens):
        return self.model.generate(
            legacy_batch(series, question), max_new_tokens=max_new_tokens, do_sample=False, num_beams=1
        )[0]

    def loss(self, series, question, answer, reduction="mean"):
        ids, images, _, _ = self.model.pad_and_apply_batch(legacy_batch(series, question), include_labels=True)
        tokenizer = self.model.text_tokenizer
        target = tokenizer(" " + answer, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
        labels = ids.new_tensor(target)[None, :]
        full = torch.cat((ids, labels), dim=1)
        logits = self.model.model(vision_x=images, lang_x=full, attention_mask=torch.ones_like(full)).logits
        return nn.functional.cross_entropy(logits[0, -len(target) - 1 : -1].float(), labels[0], reduction=reduction)

    def save_pretrained(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.model.store_to_file(directory / "model_checkpoint.pt")


def load(args):
    backend = args.backend
    default, pinned = DEFAULTS[backend]
    spec = args.checkpoint or default
    revision = args.revision or (pinned if spec == default else None)
    patterns = ["*.json", "*.model", "model.safetensors"] if backend == "per-series" else ["model_checkpoint.pt"]
    path, provenance = snapshot(spec, revision, patterns)
    if backend == "per-series":
        net = PerSeriesFlamingo.from_pretrained(path, device=args.device, dtype=torch.bfloat16)
        settings = dict(
            backbone=json.loads((path / "config.json").read_text()).get("source"),
            prompt="per-series marker, name + observed mean/std + missingness, end-of-chunk; question",
            preprocessing="observed-only population normalization; explicit availability and patch mask",
        )
    else:
        from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo

        base = args.backbone or BACKBONE
        base_rev = args.backbone_revision or (BACKBONE_REVISION if base == BACKBONE else None)
        base_path, source = snapshot(base, base_rev, ["*.json", "*.model", "*.safetensors"])
        net = Legacy(
            OpenTSLMFlamingo(
                device=args.device, llm_id=str(base_path), cross_attn_every_n_layers=1, gradient_checkpointing=False
            )
        )
        file = path / "model_checkpoint.pt" if path.is_dir() else path
        net.model.load_from_file(str(file), verify_weights=True)
        counts = {
            name: len(list(module.parameters()))
            for name, module in zip(("encoder", "perceiver", "cross_attention", "embeddings"), net.trained_modules())
        }
        net.to(args.device)
        settings = dict(
            backbone=source,
            restored_parameter_counts=counts,
            prompt="legacy pooled media, markers + name + observed mean/std, end-of-chunk; question",
            preprocessing="observed-only population normalization then gap and right-padding zero fill",
        )
    return net, dict(backend=backend, checkpoint=provenance, **settings)


def inputs(manifest, row):
    return [
        SeriesInput(values=trace["values"], name=trace["name"], observed=trace.get("observed"))
        for key in row["series"]
        for trace in [manifest["traces"][key]]
    ]
