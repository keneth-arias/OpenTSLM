<!-- SPDX-FileCopyrightText: 2026 Keneth Arias -->
<!-- SPDX-License-Identifier: MIT -->

# ARFBench QA

From the OpenTSLM repository root, use `python -m evaluation.opentslm.arfbench` for each command below. The benchmark environment uses OpenTSLM, PyTorch and the versions in `requirements.txt`. TimeNet is required only to materialize the release. For legacy runs, install the project's Open Flamingo dependencies, then install `open-flamingo==2.0.1` with `--no-deps` in the benchmark environment. The older 0.0.2 package has a different encoder contract. The legacy Llama backbone requires authorized Hub access or a local snapshot.

## Freeze and replay

```bash
python -m evaluation.opentslm.arfbench materialize --output manifest.json
python -m evaluation.opentslm.arfbench score --manifest manifest.json runs/legacy
python -m evaluation.opentslm.arfbench compare --manifest manifest.json runs/legacy runs/per-series
```

Materialization pins TimeNet's `datadog/arfbench` version 1.0.0 and uses the shipped incident split. Each row carries its task ID, split, incident IDs, target, released and evaluated option orders, original question, selected trace references and exclusion reason. The trace table stores ordered raw values with JSON nulls for gaps, source names, record IDs and rollups. The SHA-256 covers the canonical JSON content except the digest field. The public release produced digest `72fd43b5b9712cba2414b17d4981e6a7f44a78cc42e5f6b6e22859066452bf59` and retained 471/471 training, 68/68 validation and 133/135 test questions. The supplied split leaves another 76 release IDs unassigned.

Within each record, select the longest fitting rollup by maximum channel length, up to 4,096 values. Share 32 channel slots evenly across the question's records. Prefer channel names mentioned in the prompt or options, then larger population standard deviations, and present selected channels in source order. Exclude a task if any required record has no fitting observed trace or a positive identification answer cannot be mapped in full to retained channel names. Test IDs 516 and 543 meet the latter exclusion. Option shuffling for evaluation uses the raw question as the seed.

## Run on Nebius

Use the workspace ml-infra launcher and verify the Nebius Managed Kubernetes cluster identity before submitting. The following are workload entry points to execute **inside the managed job**. Each output directory must be new. Run each evaluation on the same manifest.

```bash
python -m evaluation.opentslm.arfbench evaluate --manifest manifest.json --backend legacy --output runs/legacy
python -m evaluation.opentslm.arfbench evaluate --manifest manifest.json --backend per-series --output runs/per-series
python -m evaluation.opentslm.arfbench evaluate --manifest manifest.json --backend per-series --split swap --output runs/per-series-swap
python -m evaluation.opentslm.arfbench evaluate --manifest manifest.json --backend per-series --split missing --output runs/per-series-missing
python -m evaluation.opentslm.arfbench train --manifest manifest.json --backend per-series --epochs 3 --output runs/tune
python -m evaluation.opentslm.arfbench evaluate --manifest manifest.json --backend per-series --checkpoint runs/tune/checkpoint --output runs/tuned
```

Use the same commands with `--backend legacy` for its probes and tuning. The pinned releases are OpenTSLM's Llama 3.2 1B TSQA Flamingo and opentime/tslm-nov26. The legacy loader verifies parameter coverage, including the encoder, Perceiver, cross-attention and embeddings. Both adapters use raw observed-value population statistics. Legacy fills normalized gaps and right padding with zeros. The per-series model also receives an observation mask. Each model constructs its own media prompt and sees the same question and ordered options. Decoding is greedy with a 64-token limit. Run metadata records checkpoint and backbone provenance, normalization, prompt format, decoding and package versions. Local snapshot identities use content hashes.

Every retained ID must have an output or an error. The scorer normalizes case, spaces, surrounding periods and the last `answer:` prefix, then matches the entire option. Accuracy uses every retained question as its denominator. `scored` counts completed generations, including unparseable outputs. Failures and unparseable outputs both earn zero credit. Within each category, balanced accuracy averages recalls across observed answer classes. Overall balanced accuracy averages the category balanced accuracies. The swap test moves the spike and its gap between two named channels and repeats both directions in reverse presentation order. The missing-data fixture compares a measured zero, an interior gap and a shorter signal requiring right padding. Diagnostic scores are separate from held-out accuracy.

Fine-tuning uses seeded example and option shuffles, token-weighted gradient accumulation (default 8 questions), AdamW, and the reference encoder/adapter/embedding learning rates. Validation loss selects the checkpoint. Use `compare` on pretrained and tuned predictions to measure each backend's change. The two released checkpoints also differ in backbone and pretraining. Independent estimates of routing or masking effects require separately trained matched-backbone ablations.

## Reference

The split, data selection, option ordering, scoring and training recipe are adapted from AWS Samples' `examples/opentslm-finetune-arfbench`, commit `7967032`, under MIT-0. Copyright Amazon.com, Inc. or its affiliates. The license is included in `LICENSE-MIT-0`. ARFBench data and model weights carry their respective upstream licenses.
