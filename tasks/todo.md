<!--
SPDX-FileCopyrightText: 2026 Keneth Arias
SPDX-License-Identifier: MIT
-->

# Forecast and episode implementation checklist

- [ ] **Agree on a synthetic interchange fixture with AirAware.** Include shuffled IDs, a threshold equality at +6 h, an event at +1 h, and a gap that AirAware excludes from scoring. Export from history and frozen context alone. Confirm that changing future targets leaves every inference output unchanged.
- [ ] **Deliver the quantile head through training and inference.** Extend `src/opentslm/forecast.py`, `scripts/train_qwen_forecast.py` and `src/opentslm/forecast_inference.py`. Test known pinball losses, finite gradients, monotone quantiles, shapes, scaling back to input units and the final partial inference batch.
- [ ] **Deliver one episode score per window.** Check a tiny labeled batch, the threshold's transformation from history, unknown label exclusion, mixed positive/negative examples, and ID alignment. For positive-scale affine changes of nonconstant univariate histories and their raw thresholds, check normalized context invariance. Check the flat-history fallback and reject a mismatched threshold manifest. Require finite logits and updates to the expected adapter and head weights. Reload the exported scores in AirAware and compare its calibrated score to a fixed example.
- [ ] **Qualify checkpoint compatibility.** Restore and resume an unversioned #28-style point checkpoint and compare its output to the legacy export. Test new-schema resume in a fresh process. For each new task, compare uninterrupted training with stop/resume and a fresh-process export. Reject a changed task, transform, threshold manifest or quantile order. Preserve existing PM10 masking and single-worker restrictions of joint mode.
- [ ] **Hand the trainer revision to ml-infra.** Run the locked forecast suite with `PYTHONPATH=src uv run --project environments/forecast --locked python -m pytest -q environments/forecast`. Supply the code SHA, fixture digests, output shapes and numerical tolerances before building the runtime. Compare the fixture IDs and hashes across all three repositories. The offline CPU fixtures precede the single-GPU release check.

**For every PR**, complete the [three-model review and follow-up](plan.md#review-before-merging-each-pr) before merging.
