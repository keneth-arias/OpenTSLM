# SPDX-FileCopyrightText: Amazon.com, Inc. or its affiliates
# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT-0 AND MIT
"""Per-series Gemma 3 checkpoint backend adapted from the AWS OpenTSLM ARFBench example.

The Flamingo layers in this file (PerceiverResampler, MaskedCrossAttention,
GatedCrossAttentionBlock, FlamingoLayer, FlamingoLMMixin and add_flamingo_layers) are adapted
from OpenFlamingo (https://github.com/mlfoundations/open_flamingo), which adapts
flamingo-pytorch (https://github.com/lucidrains/flamingo-pytorch). Both are under the MIT
License:

    OpenFlamingo: Copyright (c) 2023 Anas Awadalla, Irena Gao, Joshua Gardner, Jack Hessel,
    Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Jenia Jitsev,
    Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, Ludwig Schmidt.

    flamingo-pytorch: Copyright (c) 2022 Phil Wang

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

MEDIA_TOKEN = "<image>"  # tokenizer special token, not a credential  # nosec B105
END_OF_CHUNK_TOKEN = "<|endofchunk|>"  # tokenizer special token, not a credential  # nosec B105
FORECAST_PROMPT = "Forecast the PM2.5 concentration in micrograms per cubic meter for each of the next six hours."
WEIGHTS_FILE = "model.safetensors"
# The other files of a model directory, which save() writes back unchanged.
CONFIG_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


@dataclass(frozen=True)
class SeriesInput:
    """One time series: its values (NaN where missing), an optional text before its statistics,
    and its name in the prompt (its position, from 1, without one)."""

    values: Sequence[float | None]
    text: str | None = None
    name: str | None = None
    observed: Sequence[bool] | None = None


# ---- The model


class PatchTokenizer(nn.Module):
    """Embed patches of one series; each timestep also has a flag that tells if it is observed."""

    def __init__(self, output_dim: int, patch_size: int, max_patches: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.patch_embedding = nn.Conv1d(2, output_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_patches, output_dim))
        self.input_norm = nn.LayerNorm(output_dim)

    def forward(self, values: Tensor, observed: Tensor) -> Tensor:
        inputs = torch.stack((values.masked_fill(~observed, 0.0), observed.to(values.dtype)), 1)
        patches = self.patch_embedding(inputs).transpose(1, 2)
        return self.input_norm(patches + self.position_embedding[:, : patches.shape[1]])


class PerceiverAttention(nn.Module):
    """Let the latent vectors attend to the media embeddings and to themselves."""

    def __init__(self, dim: int, dim_head: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        self.media_norm = nn.LayerNorm(dim)
        self.latent_norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * heads * 2, bias=False)
        self.to_out = nn.Linear(dim_head * heads, dim, bias=False)

    def forward(self, media: Tensor, latents: Tensor, media_mask: Tensor) -> Tensor:
        media = self.media_norm(media)
        normalized_latents = self.latent_norm(latents)
        query = _split_heads(self.to_q(normalized_latents), self.heads)
        key, value = self.to_kv(torch.cat((media, normalized_latents), dim=-2)).chunk(2, dim=-1)
        key, value = _split_heads(key, self.heads), _split_heads(value, self.heads)
        similarity = torch.einsum("bthid,bthjd->bthij", query * self.scale, key)
        latent_mask = torch.ones(*latents.shape[:-1], dtype=torch.bool, device=latents.device)
        key_mask = torch.cat((media_mask, latent_mask), dim=-1)
        similarity = similarity.masked_fill(~key_mask[:, :, None, None, :], torch.finfo(similarity.dtype).min)
        output = torch.einsum("bthij,bthjd->bthid", similarity.softmax(dim=-1), value)
        batch, media_count, _, tokens, _ = output.shape
        output = output.permute(0, 1, 3, 2, 4).reshape(batch, media_count, tokens, -1)
        return self.to_out(output)


def _split_heads(values: Tensor, heads: int) -> Tensor:
    batch, media_count, tokens, inner_dim = values.shape
    values = values.reshape(batch, media_count, tokens, heads, inner_dim // heads)
    return values.permute(0, 1, 3, 2, 4)


class PerceiverFeedForward(nn.Module):
    def __init__(self, dim: int, mult: int) -> None:
        super().__init__()
        self.layers = _feedforward(dim, mult)

    def forward(self, values: Tensor) -> Tensor:
        return self.layers(values)


def _feedforward(dim: int, mult: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult, bias=False),
        nn.GELU(),
        nn.Linear(dim * mult, dim, bias=False),
    )


class PerceiverResampler(nn.Module):
    """Compress the patch embeddings of each series into a fixed number of latent vectors."""

    def __init__(self, dim, dim_head, heads, mult, depth, num_latents, max_media) -> None:
        super().__init__()
        self.media_position = nn.Parameter(torch.zeros(max_media, 1, dim))
        self.latents = nn.Parameter(torch.zeros(num_latents, dim))
        self.attention_layers = nn.ModuleList([PerceiverAttention(dim, dim_head, heads) for _ in range(depth)])
        self.feedforward_layers = nn.ModuleList([PerceiverFeedForward(dim, mult) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, media: Tensor, media_mask: Tensor) -> Tensor:
        """Turn media ``[media, patches, dim]`` into latents ``[1, media, latents, dim]``."""
        if media.ndim == 3:
            media, media_mask = media.unsqueeze(0), media_mask.unsqueeze(0)
        batch, media_count = media.shape[:2]
        media = media + self.media_position[:media_count]
        latents = self.latents.expand(batch, media_count, -1, -1)
        for attention, feedforward in zip(self.attention_layers, self.feedforward_layers):
            latents = attention(media, latents, media_mask) + latents
            latents = feedforward(latents) + latents
        return self.norm(latents)


class MaskedCrossAttention(nn.Module):
    """Let each text token attend to the latents of the most recent media token before it."""

    def __init__(self, dim: int, dim_visual: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_kv = nn.Linear(dim_visual, dim_head * heads * 2, bias=False)
        self.to_out = nn.Linear(dim_head * heads, dim, bias=False)

    def forward(self, x: Tensor, media: Tensor, media_locations: Tensor) -> Tensor:
        query = self.to_q(self.norm(x)) * self.scale
        key, value = self.to_kv(media).chunk(2, dim=-1)
        text_time = media_locations.cumsum(dim=-1)
        if int(text_time.max()) > media.shape[1]:
            raise ValueError("the text has more media tokens than there are media items")
        if media.shape[1] == 1:
            # Single-channel batches share a slot index, but each row has distinct latents.
            q, k, v = (
                tensor.unflatten(-1, (self.heads, -1)).transpose(1, 2) for tensor in (query, key[:, 0], value[:, 0])
            )
            attended = ((q @ k.transpose(-1, -2)).softmax(-1) @ v).transpose(1, 2).flatten(2)
            return self.to_out(attended.masked_fill(text_time.unsqueeze(-1) == 0, 0))
        # The tokens after one media token read only its latents. Scoring each run of tokens
        # against those latents alone, not against all media, keeps the memory small.
        rows = []
        for index in range(len(x)):
            items, counts = torch.unique_consecutive(text_time[index], return_counts=True)
            parts = []
            for item, segment in zip(items.tolist(), query[index].split(counts.tolist())):
                if item == 0:  # a text token before the first media token attends to nothing
                    parts.append(torch.zeros_like(segment))
                    continue
                q, k, v = (
                    tensor.unflatten(-1, (self.heads, -1)).transpose(0, 1)
                    for tensor in (segment, key[index, item - 1], value[index, item - 1])
                )
                attended = (q @ k.transpose(-1, -2)).softmax(dim=-1) @ v
                parts.append(attended.transpose(0, 1).flatten(1))
            rows.append(torch.cat(parts))
        return self.to_out(torch.stack(rows))


class GatedCrossAttentionBlock(nn.Module):
    """Cross-attention and a feed-forward block, each scaled by a learned tanh gate."""

    def __init__(self, dim: int, dim_visual: int) -> None:
        super().__init__()
        self.attn = MaskedCrossAttention(dim=dim, dim_visual=dim_visual)
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))
        self.ff = _feedforward(dim, 4)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x: Tensor, media: Tensor, media_locations: Tensor) -> Tensor:
        x = self.attn(x, media, media_locations) * self.attn_gate.tanh() + x
        return self.ff(x) * self.ff_gate.tanh() + x


class FlamingoLayer(nn.Module):
    """A Gemma 3 decoder layer, with a gated cross-attention block before it."""

    def __init__(self, gated_cross_attn_layer: nn.Module | None, decoder_layer: Any) -> None:
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x: Tensor | None = None
        self.media_locations: Tensor | None = None

    @property
    def attention_type(self) -> str:
        # Gemma 3 reads this attribute of each decoder layer to pick its attention mask.
        return self.decoder_layer.attention_type

    def forward(self, lang_x: Tensor, attention_mask: Any = None, **kwargs: Any) -> Any:
        if self.gated_cross_attn_layer is not None and self.vis_x is not None:
            lang_x = self.gated_cross_attn_layer(lang_x, self.vis_x, self.media_locations)
        return self.decoder_layer(lang_x, attention_mask=attention_mask, **kwargs)


class FlamingoLMMixin(nn.Module):
    """Gated cross-attention layers for a Gemma 3 vision-language model.

    In a cached generation step, the model gets only the newest token, without the media
    tokens. That token attends to the last media item, as the end of the prompt does.
    """

    def forward(
        self,
        input_ids: Tensor,
        *,
        logits_to_keep: Any = 0,  # generate() passes it only if forward() names it
        cache_position: Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        layers = self.language_model.layers
        if layers[0].vis_x is None:
            media_locations = None
        elif cache_position is not None and bool(cache_position[0] > 0):
            media_locations = torch.zeros_like(input_ids)
            media_locations[:, 0] = layers[0].vis_x.shape[1]
        else:
            media_locations = (input_ids == self.media_token_id).long()
            if not (media_locations.sum(dim=-1) == layers[0].vis_x.shape[1]).all():
                raise ValueError("the number of tokenized media markers must match the number of series")
        for layer in layers:
            layer.media_locations = media_locations
        return super().forward(
            input_ids=input_ids,
            logits_to_keep=logits_to_keep,
            cache_position=cache_position,
            **kwargs,
        )


def add_flamingo_layers(language_model: Any, media_token_id: int, vis_dim: int, every: int) -> None:
    """Give every ``every``-th decoder layer of a Gemma 3 vision-language model a gated
    cross-attention block, in place."""
    language_model.__class__ = type(type(language_model).__name__, (FlamingoLMMixin, type(language_model)), {})
    language_model.media_token_id = media_token_id
    hidden_size = language_model.config.text_config.hidden_size
    decoder_layers = language_model.language_model.layers
    # None stands for a layer without cross-attention; ModuleList keeps it at run time.
    language_model.gated_cross_attn_layers = nn.ModuleList(
        [
            GatedCrossAttentionBlock(hidden_size, vis_dim) if (index + 1) % every == 0 else None
            for index in range(len(decoder_layers))
        ]
    )
    language_model.language_model.layers = nn.ModuleList(
        FlamingoLayer(block, layer) for block, layer in zip(language_model.gated_cross_attn_layers, decoder_layers)
    )


class PerSeriesFlamingo(nn.Module):
    """A Gemma 3 language model that reads time series through cross-attention."""

    def __init__(self, config: dict[str, Any], tokenizer: Any, language_model: Any) -> None:
        super().__init__()
        encoder, resampler = config["encoder_config"], config["resampler_config"]
        self.patch_size = int(encoder["patch_size"])
        self.max_values = int(encoder["max_patches"]) * self.patch_size
        self.max_media = int(resampler["max_media"])
        self.compute_dtype = torch.float32  # from_file sets the dtype it loads in
        self.hidden_size = language_model.config.text_config.hidden_size
        self.tokenizer = tokenizer
        self.files: dict[str, bytes] = {}  # the config files, which save() writes back
        language_model.resize_token_embeddings(len(tokenizer))
        add_flamingo_layers(
            language_model,
            media_token_id=tokenizer.convert_tokens_to_ids(MEDIA_TOKEN),
            vis_dim=int(encoder["output_dim"]),
            every=int(config["cross_attention_every_n_layers"]),
        )
        # The weights are named flamingo.vision_encoder, flamingo.perceiver, flamingo.lang_encoder.
        self.flamingo = nn.Module()
        self.flamingo.vision_encoder = PatchTokenizer(
            int(encoder["output_dim"]), self.patch_size, int(encoder["max_patches"])
        )
        self.flamingo.perceiver = PerceiverResampler(
            dim=int(resampler["attention"]["dim"]),
            dim_head=int(resampler["attention"]["dim_head"]),
            heads=int(resampler["attention"]["heads"]),
            mult=int(resampler["feedforward"]["mult"]),
            depth=int(resampler["depth"]),
            num_latents=int(resampler["num_latents"]),
            max_media=self.max_media,
        )
        self.flamingo.lang_encoder = language_model

    @classmethod
    def from_pretrained(
        cls,
        spec: str | Path,
        *,
        revision: str | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        cache_dir: str | None = None,
    ) -> PerSeriesFlamingo:
        """Load a local snapshot or Hub repository at the requested revision."""
        from huggingface_hub import snapshot_download

        if Path(spec).is_dir():
            if revision is not None:
                raise ValueError(
                    "revision applies only to a Hub repository; verify local snapshot provenance separately"
                )
            root = Path(spec)
        else:
            root = Path(
                snapshot_download(
                    str(spec).removeprefix("hf://"),
                    revision=revision,
                    cache_dir=cache_dir,
                    allow_patterns=[*CONFIG_FILES, WEIGHTS_FILE],
                )
            )
        return cls.from_file(root, device=device, dtype=dtype)

    @classmethod
    def from_file(cls, path: str | Path, device: str, dtype: torch.dtype) -> PerSeriesFlamingo:
        """Load a model directory: config.json, generation_config.json, the tokenizer files and
        model.safetensors, as in the model's Hugging Face repository.

        Load in float32, then keep the series encoder, the perceiver, the cross-attention layers and the token embeddings in float32 and store the
        rest of the language model in ``dtype``, as the checkpoint's training and its weights file do.
        With a ``dtype`` other than float32 the forward passes run under autocast to ``dtype``."""
        root = Path(path)
        config = json.loads((root / "config.json").read_text())
        if config.get("language_model_config", {}).get("model_type") != "gemma3" or config.get("attend_all_channels"):
            raise ValueError("expected a per-series Gemma 3 Flamingo checkpoint")
        if config.get("prompt_style", "plain_text") != "plain_text":
            raise ValueError("unsupported per-series prompt style")
        tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)  # nosec B615
        tokenizer.add_special_tokens({"additional_special_tokens": [END_OF_CHUNK_TOKEN, MEDIA_TOKEN]})
        if tokenizer.pad_token_id is None:
            # Tokenizer padding token, not a credential.
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})  # nosec B105
        weights = load_file(root / WEIGHTS_FILE)
        with safe_open(root / WEIGHTS_FILE, framework="pt") as file:
            # save() stores each tied weight once; the metadata names its other names.
            aliases = json.loads((file.metadata() or {}).get("aliases", "{}"))
        weights.update({alias: weights[name] for alias, name in aliases.items()})

        language_config = AutoConfig.for_model(**copy.deepcopy(config["language_model_config"]))
        # transformers builds each part in the dtype that its own config names.
        for part in [
            language_config,
            *map(language_config.__getattribute__, language_config.sub_configs),
        ]:
            if part is not None:
                part.dtype = torch.float32
        with _default_dtype(torch.float32), torch.device(device):
            language_model = AutoModelForCausalLM.from_config(
                language_config,
                attn_implementation=config.get("attention_implementation", "eager"),
            )
            # Local directory.
            language_model.generation_config = GenerationConfig.from_pretrained(  # nosec B615
                root, local_files_only=True
            )
            model = cls(config, tokenizer, language_model)
        model.load_state_dict(weights, strict=True)
        kept = {id(p) for module in model.trained_modules() for p in module.parameters()}
        for parameter in model.parameters():
            if id(parameter) not in kept:
                parameter.data = parameter.data.to(dtype)
        model.compute_dtype = dtype
        model.files = {name: (root / name).read_bytes() for name in CONFIG_FILES if (root / name).is_file()}
        return model.eval()

    def trained_modules(self) -> list[nn.Module]:
        """Give the modules that the checkpoint's training trains, and keeps in float32."""
        lang = self.flamingo.lang_encoder
        # Gemma 3 ties the output layer to the token embeddings.
        return [
            self.flamingo.vision_encoder,
            self.flamingo.perceiver,
            lang.gated_cross_attn_layers,
            lang.get_input_embeddings(),
        ]

    def save_pretrained(self, directory: str | Path) -> None:
        """Write the model directory; each tied weight is stored once."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        for name, contents in self.files.items():
            (root / name).write_bytes(contents)
        stored, aliases, seen = {}, {}, {}
        for name, tensor in self.state_dict().items():
            place = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tensor.shape)
            if place in seen:
                aliases[name] = seen[place]
            else:
                seen[place] = name
                stored[name] = tensor.contiguous()
        save_file(stored, root / WEIGHTS_FILE, metadata={"aliases": json.dumps(aliases)})

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.inference_mode()
    def generate(
        self,
        series: list[SeriesInput],
        question: str,
        *,
        context: Sequence[str] = (),
        prefix: str = "",
        max_new_tokens: int = 256,
    ) -> str:
        """Answer a question about the series, with greedy decoding. Each ``context`` line goes
        between the series and the question."""
        prompt, values, observed = self.prepare(series, question, context)
        if prefix:
            prompt = f"{prefix} {prompt}"
        tokens = self.tokenizer([prompt], return_tensors="pt")
        input_ids = tokens["input_ids"].to(self.device)
        with self.conditioned_on(values, observed):
            generated = self.flamingo.lang_encoder.generate(
                input_ids,
                attention_mask=tokens["attention_mask"].to(self.device),
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        answer = generated[0, input_ids.shape[1] :]
        return self.tokenizer.decode(answer, skip_special_tokens=True).strip()

    def eval_prompt(self, prompt, max_new_tokens: int = 256) -> str:
        """Answer a raw OpenTSLM FullPrompt with its original series order and text."""
        data = prompt.to_dict()
        series = [
            SeriesInput(values, text=text)
            for values, text in zip(data["time_series"], data["time_series_text"], strict=True)
        ]
        return self.generate(series, data["post_prompt"], prefix=data["pre_prompt"], max_new_tokens=max_new_tokens)

    def generate_batch(self, examples, *, max_new_tokens: int = 256) -> list[str]:
        """Generate one answer per (series, question), supporting different channel counts."""
        return [self.generate(series, question, max_new_tokens=max_new_tokens) for series, question in examples]

    def forecast_features(self, history: Tensor, observed: Tensor) -> Tensor:
        """Return the final prompt token's hidden state for each single-channel raw history."""
        if history.ndim != 2 or not history.shape[0] or observed.shape != history.shape or observed.dtype != torch.bool:
            raise ValueError("forecast histories require [batch, time] values and matching boolean masks")
        raw, flags = history.detach().cpu().tolist(), observed.detach().cpu().tolist()
        prepared = [
            self.prepare([SeriesInput(values, name="PM2.5", observed=mask)], FORECAST_PROMPT)
            for values, mask in zip(raw, flags, strict=True)
        ]
        prompts, values, masks = zip(*prepared, strict=True)
        tokens = self.tokenizer(list(prompts), padding=True, return_tensors="pt").to(self.device)
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        language = self.flamingo.lang_encoder
        locations = input_ids.eq(language.media_token_id)
        if not (locations.sum(-1) == 1).all():
            raise ValueError("expected exactly one media marker per forecast prompt")
        with self.conditioned_on(torch.stack(values), torch.stack(masks)):
            for layer in language.language_model.layers:
                layer.media_locations = locations
            # Skip the vocabulary projection and storage of intermediate layer states.
            positions = (attention_mask.cumsum(-1) - 1).clamp_min(0)
            hidden = language.language_model(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=positions, use_cache=False
            ).last_hidden_state
            index = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
            last = index.masked_fill(~attention_mask.bool(), -1).amax(dim=-1)
            return hidden[torch.arange(len(input_ids), device=self.device), last]

    def loss(self, series: list[SeriesInput], question: str, answer: str, reduction: str = "mean") -> Tensor:
        """Give the cross-entropy of the answer tokens and the end-of-sequence token, which
        follow the prompt as in training: their mean, or one per token with ``"none"``."""
        prompt, values, observed = self.prepare(series, question)
        # As the training collator: a space before the answer unless the prompt ends in one.
        target = answer if prompt.endswith((" ", "\n")) else f" {answer}"
        prompt_ids = self.tokenizer(prompt)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        target_ids = [*target_ids, self.tokenizer.eos_token_id]
        input_ids = torch.tensor([prompt_ids + target_ids], device=self.device)
        with self.conditioned_on(values, observed):
            # Only the logits that predict the target tokens: the full vocabulary is large.
            logits = self.flamingo.lang_encoder(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                logits_to_keep=len(target_ids) + 1,
            ).logits
        labels = torch.tensor(target_ids, device=logits.device)
        return nn.functional.cross_entropy(logits[0, :-1].float(), labels, reduction=reduction)

    def prepare(
        self, series: list[SeriesInput], question: str, context: Sequence[str] = ()
    ) -> tuple[str, Tensor, Tensor]:
        """Give the prompt and the padded series ``[series, time]`` with their observed flags,
        in the layout that the model trained on.

        Raises:
            ValueError: If there are more series than the model takes, a series is longer than
                the encoder takes, or a series has no observed or an infinite value.
        """
        if not series:
            raise ValueError("provide at least one series")
        if len(series) > self.max_media:
            raise ValueError(f"the model takes at most {self.max_media} series")
        words, rows = [], []
        for position, item in enumerate(series):
            if len(item.values) > self.max_values:
                raise ValueError(
                    f"series {position} has {len(item.values)} values, but the model takes at most {self.max_values}"
                )
            try:
                scaled, observed, mean, std = _scale(item.values, item.observed)
            except ValueError as error:
                raise ValueError(f"series {item.name or position + 1}: {error}") from error
            sentence = (
                f"This is time series channel {item.name or position + 1}, with mean "
                f"{_statistic(mean)} and std {_statistic(std)}."
            )
            if not observed.all():
                sentence += " Some values of this series are missing."
            words += [
                MEDIA_TOKEN,
                *([item.text] if item.text else []),
                sentence,
                END_OF_CHUNK_TOKEN,
            ]
            rows.append((scaled, observed))
        words += [*context, question]
        length = max(len(scaled) for scaled, _ in rows)
        length += -length % self.patch_size
        values = torch.zeros((len(rows), length), dtype=torch.float32)
        observed = torch.zeros((len(rows), length), dtype=torch.bool)
        for index, (scaled, mask) in enumerate(rows):
            values[index, : len(scaled)] = torch.from_numpy(scaled)
            observed[index, : len(mask)] = torch.from_numpy(mask)
        return " ".join(words), values, observed

    @contextmanager
    def conditioned_on(self, values: Tensor, observed: Tensor):
        """Let the language model attend to the encoded series while the block runs, under
        autocast to the model's compute dtype."""
        dtype = self.compute_dtype
        autocast = torch.autocast(self.device.type, dtype, enabled=dtype != torch.float32)
        values, observed = values.to(self.device, torch.float32), observed.to(self.device)
        layers = self.flamingo.lang_encoder.language_model.layers
        try:
            with autocast:
                if values.ndim == 2:
                    values, observed = values.unsqueeze(0), observed.unsqueeze(0)
                batch, count, steps = values.shape
                encoded = self.flamingo.vision_encoder(values.flatten(0, 1), observed.flatten(0, 1))
                encoded = encoded.reshape(batch, count, steps // self.patch_size, -1)
                patch_mask = observed.unflatten(-1, (-1, self.patch_size)).any(dim=-1)
                latents = self.flamingo.perceiver(encoded, patch_mask)
                for layer in layers:
                    layer.vis_x = latents
                yield
        finally:
            for layer in layers:
                layer.vis_x = layer.media_locations = None


def _scale(
    values: Sequence[float | None], mask: Sequence[bool] | None = None
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Scale the observed values to a mean of 0 and a standard deviation of 1, 0 where missing,
    as in training. Dividing by the largest magnitude first keeps the squares finite."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("time series must be one-dimensional")
    if mask is None:
        observed = ~np.isnan(array)
    else:
        observed = np.asarray(mask)
        if observed.dtype != np.bool_ or observed.shape != array.shape:
            raise ValueError("observation mask must contain one boolean per value")
    present = array[observed]
    if not present.size:
        raise ValueError("each time series must have at least one observed value")
    if not np.isfinite(present).all():
        raise ValueError("observed values must be finite")
    scaled = np.zeros(array.shape, dtype=np.float64)
    magnitude = float(np.abs(present).max())
    mean = std = 0.0
    if magnitude:
        unit = present / magnitude
        unit_mean, unit_std = float(unit.mean()), float(unit.std())
        scaled[observed] = (unit - unit_mean) / (unit_std + 1e-8 / magnitude)
        mean, std = unit_mean * magnitude, unit_std * magnitude
    return scaled.astype(np.float32), observed, mean, std


def _statistic(value: float) -> str:
    return f"{value:.4e}" if abs(value) >= 1e6 else f"{value:.4f}"


@contextmanager
def _default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)
