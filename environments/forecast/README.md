<!--
SPDX-FileCopyrightText: 2026 Keneth Arias
SPDX-License-Identifier: MIT
-->

# Numerical forecasts with Qwen3.5

The [AirAware improvement plan](../../tasks/plan.md) describes the proposed interval and six-hour episode outputs and their acceptance tests.

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

For the Lima pilot, add `--require-history-only` to the inference command. The point archive must contain exactly `window_id` and `history`. A two-channel archive also requires `channels`. Strict mode checks these fields before initializing CUDA or loading weights. Target-bearing archives remain available to the separate scoring job.

For matched two-channel regression, supply `--input-view first` or `--input-view both`
and `--scaler PATH` to the single-worker direct trainer and to checkpoint inference.
The training archive contains `history[N,24,2]`, `targets[N,6,2]`, ordered
`channels` and `window_id`. Inference only requires the histories, channel names and IDs.
The fitted JSON scaler records the input version, ordered channels, population means and
positive scales for training. The first-input view masks the second channel before scaling.
Both variants emit both targets in the original units and train on the final partial batch.
Restoration checks the scaler and input view recorded in the checkpoint manifest.

## Six-hour candidate heads

Use `scripts/train_qwen_forecast.py --mode single --head quantile` or `--head episode` with the usual training arguments. Both candidates use rank-4 LoRA, alpha 8 and the fixed final update for evaluation. The quantile task uses the univariate H24/F6 training archive and emits ordered q10, q50 and q90 for each forecast hour. The point estimate is the median.

An episode fit reads `window_id`, `history`, `threshold_ug_m3[N]`, a scalar `thresholds_sha256`, and one `event[N]` label (0, 1, or −1 for ineligible). Pass the frozen table digest with `--thresholds-sha256` for fit, resume and inference. Each station threshold is supplied in raw units and uses the same transform as its window. Use only eligible labels from the data owner. The history-only inference archive omits `event`; the episode output contains one raw `episode_logit` per ID. Calibration is a separate AirAware step.

For inference repeat the selected `--head` with the usual checkpoint, base weights, input and output paths. A versioned checkpoint records task definition, levels or threshold-table digest, and adapter configuration. A legacy point checkpoint keeps its original identity.

For candidate fits and inference, pass `--input-manifest PATH --input-manifest-sha256 SHA256`. The JSON manifest pins the SHA-256 of the NPZ under `files`, declares `schema=forecast-input-v1`, `unit=ug/m3`, the head and the phase (`fit` or `infer`). Inner validation uses separate `--validation-manifest` and digest with phase `inner_validation`. The episode bundle also contains a frozen `thresholds.json` with unit and station-to-threshold mapping. Include its digest in `files`, `thresholds_sha256` and the CLI. The NPZ supplies `station_key` per window. Fit and inference verify each raw threshold with the table. The model bounds the standardized threshold at ±100 standard deviations to keep near-constant histories numerically stable. Candidate exports record the input and manifest digests, model revision, units and checkpoint digest.

Use `--mode single --head direct --matched-point` when fitting a new control for the quantile comparison. It verifies the direct fit manifest, uses rank-4 LoRA and final-step evaluation, and includes the final partial batch. The same seed and batch size therefore visit the same IDs in the same order as the quantile trainer. The checkpoint records the full-epoch recipe and exact input manifest. Existing direct runs retain their original batching and identity. Unpinned runs can resume a checkpoint from the preceding trainer revision when every other identity field matches. Matched controls and runs with an explicitly pinned trainer require their recorded trainer identity. Final-step evaluation occurs when training reaches the scheduled optimizer step; intermediate recovery checkpoints preserve progress.


### Compare two direct objectives on pinned training cohorts

Use `--direct-objective normalized-mse` for the control and `--direct-objective physical-mae` for the candidate. Run each arm in a separate directory with a stable `--logical-run` ID. Supply `--mode single --head direct --precision fp32 --batch 16`, the matched seed, `--data` and `--validation` from their respective role directories, both manifest paths and SHA-256 values, and the pinned base. Set both the schedule and stop step to one epoch including its last partial batch.

The trainer evaluates the frozen fit sample and inner-validation set at initialization, every 500 updates and at the ceiling. The first eligible checkpoint is at update 500. A decrease in global validation MAE of at least 0.02 µg/m³ updates the selected checkpoint. Four consecutive checks without that decrease stop the fit. Read `direct-result.json` for the actual and selected steps, outcome and both checkpoint digests. `--resume` recovers the most recent complete compatible generation, repairs the selection pointer from its persisted history, and verifies any existing terminal result.

For cooperative shutdown the supervisor sets `EPISODE_STOP_FILE` to an empty, attempt-specific file before starting the trainer. Writing a byte or removing the armed file requests a stop at the next safe point. Before any update, the trainer emits `stopped_before_update`. After saving an interrupted update, it emits `interrupted_at_step` for the supervisor.
