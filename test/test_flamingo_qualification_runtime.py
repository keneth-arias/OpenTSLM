# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT

import torch

from opentslm import flamingo_qualification as probe
from opentslm.flamingo_forecast import FlamingoForecaster
from opentslm.model.llm.per_series import SeriesInput

pytest_plugins = ("test.test_per_series",)


def test_gemma_probe_matches_qa_loss_and_survives_forecast(model):
    baseline = probe.measurements(model)
    for case, receipt in zip(probe.CASES, baseline, strict=True):
        _, question, reference, series = case
        if not series:
            continue
        inputs = [SeriesInput(values, name=label) for label, values in series]
        expected = model.loss(inputs, question, reference, reduction="none")
        torch.testing.assert_close(torch.tensor(receipt["reference_token_nll"]), expected)
        answer = model.generate(inputs, question, max_new_tokens=probe.MAX_NEW_TOKENS)
        assert model.tokenizer.decode(receipt["greedy_ids"], skip_special_tokens=True).strip() == answer

    wrapped = FlamingoForecaster(model).eval()
    adapters = wrapped.add_lora()
    with torch.no_grad():
        for name in adapters:
            wrapped.decoder.get_parameter(name).fill_(0.25)
        wrapped.head.weight.zero_()
        wrapped.head.bias.fill_(1.)
    result = probe.exercise_restore(wrapped)
    assert result["before"] == result["after"] == baseline
    raw = torch.tensor(probe.FORECAST_HISTORY, dtype=torch.float32)
    torch.testing.assert_close(torch.tensor(result["forecast"]), (raw.mean() + raw.std(correction=0)).expand(6))
