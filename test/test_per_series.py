# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
"""CPU fixtures for the Gemma 3 per-series checkpoint format (Transformers 4.57.x)."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, Gemma3Config, PreTrainedTokenizerFast

from opentslm import OpenTSLM
from opentslm.model.llm.per_series import END_OF_CHUNK_TOKEN, MEDIA_TOKEN, PerSeriesFlamingo, SeriesInput


@pytest.fixture
def model(tmp_path):
    torch.manual_seed(8)
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "yes": 3, "no": 4}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        additional_special_tokens=[END_OF_CHUNK_TOKEN, MEDIA_TOKEN],
    )
    language_config = Gemma3Config(
        text_config=dict(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=256,
            pad_token_id=1,
        ),
        vision_config=dict(
            hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, image_size=8, patch_size=4
        ),
        mm_tokens_per_image=1,
        image_token_index=999,
    )
    lang = AutoModelForCausalLM.from_config(language_config, attn_implementation="eager")
    config = {
        "encoder_config": {"output_dim": 8, "patch_size": 4, "max_patches": 16},
        "resampler_config": {
            "max_media": 4,
            "num_latents": 3,
            "depth": 1,
            "attention": {"dim": 8, "heads": 2, "dim_head": 4},
            "feedforward": {"mult": 2},
        },
        "cross_attention_every_n_layers": 1,
        "language_model_config": language_config.to_dict(),
    }
    model = PerSeriesFlamingo(config, tokenizer, lang).eval()
    for gate in lang.gated_cross_attn_layers:
        gate.attn_gate.data.fill_(0.3)
    (tmp_path / "config.json").write_text(json.dumps(config))
    lang.generation_config.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    model.files = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    return model


def test_preparation_preserves_missingness(model):
    a = SeriesInput([0, None, 2, 4, 777], name="A", observed=[True, False, True, True, False])
    b = SeriesInput([7], name="B")
    prompt, values, observed = model.prepare([a, b], "Which?")
    assert prompt.count(MEDIA_TOKEN) == 2
    assert "channel A, with mean 2.0000 and std 1.6330" in prompt
    assert "channel B, with mean 7.0000 and std 0.0000" in prompt
    assert values.shape == observed.shape == (2, 8)
    assert observed.tolist() == [[True, False, True, True, False, False, False, False], [True] + [False] * 7]
    assert (values[~observed] == 0).all()
    assert observed.reshape(2, 2, 4).any(-1).tolist() == [[True, False], [True, False]]
    assert values[0, 0] < 0 and values[1, 0] == 0
    _, _, partial = model.prepare([SeriesInput([1] * 5)], "Q")
    assert partial.reshape(2, 4).any(-1).tolist() == [True, True]
    for bad in [
        SeriesInput([None], name="empty"),
        SeriesInput([float("inf")], name="empty"),
        SeriesInput([1], name="empty", observed=[1]),
    ]:
        with pytest.raises(ValueError, match="empty"):
            model.prepare([bad], "Q")


def test_segments_and_patch_invariance(model):
    block = model.flamingo.lang_encoder.gated_cross_attn_layers[0]
    text = torch.randn(1, 5, 16)
    slots = torch.randn(1, 2, 3, 8)
    markers = torch.tensor([[0, 1, 0, 1, 0]])
    base = block.attn(text, slots, markers)
    slots[:, 1] += 2
    changed = block.attn(text, slots, markers)
    assert base[:, 0].count_nonzero() == 0
    torch.testing.assert_close(base[:, :3], changed[:, :3])
    assert not torch.allclose(base[:, 3:], changed[:, 3:])
    _, values, mask = model.prepare([SeriesInput([1, None, 2])], "Q")
    layer = model.flamingo.lang_encoder.language_model.layers[0]
    with model.conditioned_on(values, mask):
        latents = layer.vis_x
    values[~mask] = 102
    with model.conditioned_on(torch.nn.functional.pad(values, (0, 4)), torch.nn.functional.pad(mask, (0, 4))):
        torch.testing.assert_close(layer.vis_x, latents)


def test_generation_loss_and_cache(model):
    lang = model.flamingo.lang_encoder
    examples = [([SeriesInput([i, i + 1]) for i in range(n)], "Q") for n in (1, 3)]
    separate = [model.generate(series, q, max_new_tokens=2) for series, q in examples]
    assert model.generate_batch(examples, max_new_tokens=2) == separate
    series, question = examples[-1]
    prompt, values, mask = model.prepare(series, question)
    ids = model.tokenizer(prompt, return_tensors="pt").input_ids
    next_id = torch.tensor([[3]])
    with model.conditioned_on(values, mask):
        prefill = lang(ids, use_cache=True)
        cached = lang(next_id, past_key_values=prefill.past_key_values, cache_position=torch.tensor([ids.shape[1]]))
        full = lang(torch.cat((ids, next_id), dim=1), use_cache=False)
        torch.testing.assert_close(cached.logits[:, -1], full.logits[:, -1], atol=2e-5, rtol=2e-5)
    with pytest.raises(ValueError, match="markers"):
        model.generate(series, "<image> extra", max_new_tokens=1)
    assert all(layer.vis_x is None for layer in lang.language_model.layers)
    loss = model.loss(series, question, "yes", reduction="none")
    assert loss.shape == (2,)
    loss.sum().backward()
    grad = model.flamingo.vision_encoder.patch_embedding.weight.grad
    assert grad.isfinite().all() and grad.abs().sum() > 0


def test_checkpoint_forecast_and_plain_generation(model, tmp_path):
    history = torch.tensor([[1.0, 3.0, 2.0, 4.0], [4.0, 2.0, 3.0, 1.0]])
    masks = torch.ones_like(history, dtype=torch.bool)
    before = model.forecast_features(history, masks)
    assert before.shape == (2, 16)
    assert not torch.allclose(before[0], before[1])
    for i in range(len(history)):
        torch.testing.assert_close(before[i], model.forecast_features(history[i : i + 1], masks[i : i + 1])[0])
    model.save_pretrained(tmp_path)
    restored = OpenTSLM.load_pretrained(str(tmp_path), backend="per-series", device="cpu")
    torch.testing.assert_close(restored.forecast_features(history, masks), before)
    plain_ids = torch.tensor([[3, 4]])
    original = restored.flamingo.lang_encoder.generate(plain_ids, max_new_tokens=1, do_sample=False)
    restored.generate([SeriesInput([3])], "Q", max_new_tokens=1)
    again = restored.flamingo.lang_encoder.generate(plain_ids, max_new_tokens=1, do_sample=False)
    assert torch.equal(original, again)
    path = tmp_path / "model.safetensors"
    weights = load_file(path)
    del weights["flamingo.vision_encoder.patch_embedding.weight"]
    save_file(weights, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        OpenTSLM.load_pretrained(str(tmp_path), backend="per-series", device="cpu")


@pytest.mark.parametrize("layout", ["llm", "model_state", "ddp"])
def test_legacy_layouts_and_verified_restore(tmp_path, layout):
    from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo

    legacy = OpenTSLMFlamingo.__new__(OpenTSLMFlamingo)
    torch.nn.Module.__init__(legacy)
    legacy.model = torch.nn.ModuleDict(
        {k: torch.nn.Linear(2, 2) for k in ("vision_encoder", "perceiver", "lang_encoder")}
    )
    legacy.llm = legacy.model
    state = legacy.llm.state_dict() if layout == "llm" else legacy.state_dict()
    if layout == "ddp":
        state = {"module." + k: v for k, v in state.items()}
    path = tmp_path / "weights.pt"
    torch.save({"llm" if layout == "llm" else "model_state": state}, path)
    original = legacy.model.vision_encoder.weight.detach().clone()
    legacy.model.vision_encoder.weight.data.zero_()
    legacy.load_from_file(path, verify_weights=True)
    torch.testing.assert_close(original, legacy.model.vision_encoder.weight)
    torch.save({"llm": {"vision_encoder.weight": original}}, path)
    with pytest.raises(ValueError, match="lacks"):
        legacy.load_from_file(path, verify_weights=True)
