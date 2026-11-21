# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
"""Offline CPU integration for the ARFBench legacy OpenFlamingo runtime."""

from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from opentslm import OpenTSLM
from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo


def test_pinned_factory_restores_and_scores_multichannel_prefix(tmp_path):
    torch.manual_seed(13)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "yes": 3, "no": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    tokenizer.save_pretrained(tmp_path)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        pad_token_id=1,
        eos_token_id=2,
    )
    LlamaForCausalLM(config).save_pretrained(tmp_path)
    source = OpenTSLMFlamingo(device="cpu", llm_id=str(tmp_path)).eval()
    path = tmp_path / "forecast.pt"
    source.store_to_file(path)
    with (
        patch("opentslm.model.llm.OpenTSLM.hf_hub_download", return_value=str(path)) as download,
        patch.object(OpenTSLM, "_get_base_llm_id", return_value=str(tmp_path)),
    ):
        loaded = OpenTSLM.load_pretrained(
            "offline/llama-3.2-1b-tsqa-flamingo", device="cpu", revision="test-revision", verify_weights=True
        )
    assert download.call_args.kwargs["revision"] == "test-revision"
    samples = [
        {
            "pre_prompt": "",
            "post_prompt": "yes",
            "time_series": torch.arange(16, dtype=torch.float32).reshape(2, 8),
            "time_series_text": ["A", "B"],
            "answer": "no",
        }
    ]
    assert loaded.generate(samples, max_new_tokens=3) == source.generate(samples, max_new_tokens=3)
    ids, slots, _, _ = loaded.pad_and_apply_batch(samples, include_labels=True)
    assert slots.shape[1:3] == (1, 2)  # historical pooled-channel layout
    assert 4 not in ids  # the gold 'no' answer never enters the generation prefix
    loss = loaded.compute_loss(samples)
    torch.testing.assert_close(source.compute_loss(samples), loss)
    assert loss.isfinite()
