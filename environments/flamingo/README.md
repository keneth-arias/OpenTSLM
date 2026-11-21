<!-- SPDX-FileCopyrightText: 2026 Keneth Arias -->
<!-- SPDX-License-Identifier: MIT -->

# Per-series Flamingo

Use this runtime for the Gemma per-series backend and six-horizon direct forecaster. The checkpoint loader accepts a local snapshot or a pinned Hub revision. For example, the public factory accepts `OpenTSLM.load_pretrained("opentime/tslm-nov26", backend="per-series", revision="5e030b001fb8d4be6d10ebb3a8b9bc3e6ef80983")`. The existing suffix-based loaders continue to serve legacy checkpoints.

From the project root, run the lightweight CPU fixtures:

```sh
uv sync --project environments/flamingo --locked
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH=src uv run --project environments/flamingo --locked python -m pytest -q \
  test/test_per_series.py test/test_flamingo_fit.py test/test_flamingo_qualification_runtime.py \
  environments/forecast/test_flamingo_forecast.py environments/forecast/test_flamingo_qualification.py
```

Stage the eight original model and tokenizer files under `weights/` and the [pinned checksum list](../../src/opentslm/tslm_nov26.sha256) as its sibling `SHA256SUMS`. From the parent directory, `sha256sum --check --strict SHA256SUMS` verifies the snapshot. The original weights are required both for forecast-adapter restore and unadapted text/series generation.

For this workspace, submit GPU qualification, fine-tuning and evaluation through ml-infra on the verified Nebius Managed Kubernetes cluster. The trainer accepts `--backbone flamingo`, the existing matched direct-fit arguments and a pinned role binding. Its two supported objectives are `normalized-mse` and `physical-mae`. A selected forecast restore checks the terminal receipt and the selected generation against the immutable base before producing six physical-unit predictions. [Forecast trainer](../../scripts/train_qwen_forecast.py) · [Reference parity check](../../scripts/check_per_series_reference.py) · [Text and forecast qualification commands](../../src/opentslm/flamingo_qualification.py)
