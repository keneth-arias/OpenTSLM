<!--
SPDX-FileCopyrightText: 2026 Keneth Arias
SPDX-License-Identifier: MIT
-->

# Numerical forecasts with Qwen3.5

From the repository root, create the isolated Python 3.13 environment and fetch the pinned Qwen snapshot once. Download only the configuration, weight index and weights.

```sh
uv sync --project environments/forecast --locked
uv run --project environments/forecast --locked hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --include 'config.json' --include '*.safetensors*'
PYTHONPATH=src uv run --project environments/forecast --locked python -m pytest -q environments/forecast/test_qwen_forecast.py
PYTHONPATH=src uv run --project environments/forecast --locked python scripts/smoke_qwen_forecast.py
```

The GPU smoke test uses the local checkpoint, a synthetic 24-hour history and a six-hour target. Four successive observations and their availability masks form each soft prompt. A learned final token summarizes the causal text decoder output for a linear regression head. The training step uses float32 and checks the gradients and updates in the decoder and head. The published multimodal checkpoint is loaded on CPU; only the text decoder, patch projection, readout and head are moved to the GPU. The image encoder and vocabulary projection take no part in the forecast.
