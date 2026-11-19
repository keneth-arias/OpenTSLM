# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
"""Check a single numerical forecast update on synthetic hourly history."""

import json

import torch

from opentslm.forecast import MODEL_REVISION, QwenForecaster, smoke_step

if __name__ == "__main__":
    torch.manual_seed(27)
    model = QwenForecaster.from_pretrained(local_files_only=True).cuda()
    hours = torch.arange(30, device="cuda").float().reshape(1, 30, 1) / 24
    history = hours[:, :24]
    report = smoke_step(model, history, torch.ones_like(history, dtype=torch.bool), hours[:, 24:])
    print(json.dumps(dict(report, revision=MODEL_REVISION, peak_bytes=torch.cuda.max_memory_allocated())))
