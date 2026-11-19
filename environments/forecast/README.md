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

For a univariate residual experiment, load `PersistenceForecaster.from_pretrained(local_files_only=True)` in place of `QwenForecaster`. Both accept the same history and mask. The residual head starts at zero, so each initial prediction equals the measured value at the forecast origin in input units. Use history and targets on the same scale. The first backward pass updates the head; the second can update the decoder. The one-step smoke script exercises the direct head. The offline persistence test checks both updates with a single optimizer.

## Evaluate a saved forecast

Train with `--validate-at final` to score only the last scheduled optimizer update. Periodic recovery checkpoints still follow `--save-seconds`. The default evaluates each saved checkpoint. An interrupted run can stop at an earlier step and resume with the same schedule and validation policy.

Use a published step directory to run inference. The base snapshot directory must have a sibling `SHA256SUMS` file listing its weights and configuration, as used during training. The loader checks that manifest and the saved checkpoint before restoring the model. It can load a full, sharded or LoRA training checkpoint on one GPU. Run from the repository root.

```sh
PYTHONPATH=src python -m opentslm.forecast_inference \
  --checkpoint /checkpoints/example/step-00000004 --weights /inputs/qwen/weights \
  --data /inputs/validation.npz --output validation-predictions.npz --batch 32
```

For two GPUs use `torchrun --standalone --nproc-per-node=2 --module opentslm.forecast_inference` with the same arguments. The input requires window IDs and a finite 24-hour history. The NPZ output carries the input order, forecast hours 1 through 6, predictions in original units and the checkpoint manifest SHA-256. Targets are only needed when independently scoring the export. Each output path can be published once.
