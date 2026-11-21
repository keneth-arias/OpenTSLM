# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
"""CPU checkpoint tests for the legacy OpenFlamingo runtime."""

import pytest
import torch


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
