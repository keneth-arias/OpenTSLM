# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT

import json
from subprocess import CalledProcessError
from types import SimpleNamespace

import pytest
import torch

from opentslm import qwen_text
from opentslm.forecast_checkpoint import sha256


def stage_inputs(tmp_path, monkeypatch):
    weights = tmp_path / "weights"
    weights.mkdir()
    config = weights / "config.json"
    config.write_text("{}")
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(f"{sha256(config)}  weights/config.json\n")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    template = tokenizer / "chat_template.jinja"
    template.write_text("toy template")
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({
        "schema": "language-parity-v1",
        "max_new_tokens": 2,
        "cases": [["es", "Hola", "Buenos días"]],
    }))
    monkeypatch.setattr(qwen_text, "BASE_MANIFEST_SHA256", sha256(manifest))
    monkeypatch.setattr(qwen_text, "TOKENIZER_FILES", {template.name: sha256(template)})
    monkeypatch.setattr(qwen_text, "FIXTURE_SHA256", sha256(fixture))
    return weights, tokenizer, fixture


def test_verify_only_requires_no_model(tmp_path, monkeypatch, capsys):
    weights, tokenizer, fixture = stage_inputs(tmp_path, monkeypatch)
    qwen_text.main([
        "--base", str(weights), "--base-manifest-sha256", qwen_text.BASE_MANIFEST_SHA256,
        "--tokenizer", str(tokenizer), "--fixture", str(fixture), "--verify-only",
    ])
    result = json.loads(capsys.readouterr().out)
    assert result["base_manifest_sha256"] == qwen_text.BASE_MANIFEST_SHA256
    assert result["fixture_sha256"] == qwen_text.FIXTURE_SHA256


@pytest.mark.parametrize("corrupted", ["weights/config.json", "tokenizer/chat_template.jinja", "fixture.json"])
def test_verifies_weight_tokenizer_and_fixture_bytes_independently(tmp_path, monkeypatch, corrupted):
    weights, tokenizer, fixture = stage_inputs(tmp_path, monkeypatch)
    (tmp_path / corrupted).write_text("changed")
    with pytest.raises((ValueError, CalledProcessError)):
        qwen_text.verify_inputs(weights, qwen_text.BASE_MANIFEST_SHA256, tokenizer, fixture)


def test_rejects_files_that_can_override_pinned_tokenizer(tmp_path, monkeypatch):
    weights, tokenizer, fixture = stage_inputs(tmp_path, monkeypatch)
    (tokenizer / "special_tokens_map.json").write_text("{}")
    with pytest.raises(ValueError, match="Tokenizer"):
        qwen_text.verify_inputs(weights, qwen_text.BASE_MANIFEST_SHA256, tokenizer, fixture)


def test_rejects_unpinned_weights_even_with_self_consistent_manifest(tmp_path, monkeypatch):
    weights, tokenizer, fixture = stage_inputs(tmp_path, monkeypatch)
    (weights.parent / "SHA256SUMS").write_text("other manifest")
    with pytest.raises(ValueError, match="pinned"):
        qwen_text.verify_inputs(weights, sha256(weights.parent / "SHA256SUMS"), tokenizer, fixture)


class ToyTokenizer:
    eos_token_id = 5

    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{"role": "user", "content": "Hola"}]
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return "rendered"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"rendered": [0, 1], "Buenos días": [2, 3]}[text]


class ToyModel:
    def generate(self, *, input_ids, attention_mask, **kwargs):
        assert input_ids.tolist() == [[0, 1]]
        assert attention_mask.tolist() == [[1, 1]]
        assert kwargs == dict(max_new_tokens=2, do_sample=False, num_beams=1, pad_token_id=5)
        return torch.tensor([[0, 1, 4, 5]])

    def __call__(self, *, input_ids, attention_mask, use_cache):
        assert input_ids.tolist() == [[0, 1, 2, 3]]
        assert attention_mask.tolist() == [[1, 1, 1, 1]]
        assert use_cache is False
        return SimpleNamespace(logits=torch.tensor([[
            [0.0] * 6, [0, 0, 3, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0.0] * 6,
        ]]))


def test_scores_only_reference_tokens_using_previous_position():
    result = qwen_text.run_case(ToyModel(), ToyTokenizer(), "es", "Hola", "Buenos días", 2, "cpu")
    assert result["case"] == "es"
    assert result["prompt_token_ids"] == [0, 1]
    assert result["reference_ids"] == [2, 3]
    assert result["greedy_ids"] == [4, 5]
    assert result["reference_token_nll"] == pytest.approx([0.2222915, 1.0435918])


def test_packaged_command_writes_private_receipt_once(tmp_path, monkeypatch):
    weights, tokenizer, fixture = stage_inputs(tmp_path, monkeypatch)
    output = tmp_path / "receipt.json"
    calls = []

    def run_fixture(weights_path, tokenizer_path, fixture_doc, device):
        assert (weights_path, tokenizer_path, device) == (weights, tokenizer, "cpu")
        calls.append(device)
        cases = [
            qwen_text.run_case(ToyModel(), ToyTokenizer(), *case, fixture_doc["max_new_tokens"], device)
            for case in fixture_doc["cases"]
        ]
        return {"precision": "float32", "device": device, "cases": cases}

    monkeypatch.setattr(qwen_text, "run_fixture", run_fixture)
    argv = [
        "--base", str(weights), "--base-manifest-sha256", qwen_text.BASE_MANIFEST_SHA256,
        "--tokenizer", str(tokenizer), "--fixture", str(fixture), "--device", "cpu",
        "--output", str(output),
    ]
    qwen_text.main(argv)
    receipt = json.loads(output.read_text())
    assert receipt["fixture_sha256"] == qwen_text.FIXTURE_SHA256
    assert receipt["base_revision"] == qwen_text.BASE_REVISION
    assert receipt["cases"][0]["greedy_ids"] == [4, 5]
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        qwen_text.main(argv)
    assert calls == ["cpu"]
