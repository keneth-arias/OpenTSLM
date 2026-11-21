# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
# Portions adapted from AWS Samples under MIT-0.

"""ARFBench materialization, adapted from AWS's opentslm-finetune-arfbench (MIT-0)."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

SPLIT = Path(__file__).with_name("arfbench-split.json")
DATASET = "datadog/arfbench"
VERSION = "1.0.0"


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def question(row, rng=None):
    options = list(row["options"])
    if rng is not None:
        options = list(row["released_options"])
        rng.shuffle(options)
    lines = "\n".join(f"{i}. {option}" for i, option in enumerate(options, 1))
    return f"answer_options:\n{lines} {row['prompt']}" if options else row["prompt"]


def named(name, text):
    return re.search(rf"(?<![\w:]){re.escape(name)}(?!\w)", text) is not None


def covered(answer, names):
    # Some options concatenate whole source names, which can themselves contain commas.
    remainder = answer
    names = sorted(set(names), key=len, reverse=True)
    while remainder:
        name = next(
            (n for n in names if n and remainder.startswith(n) and re.match(r"(?:$|,| and )", remainder[len(n) :])),
            None,
        )
        if not name:
            return False
        remainder = re.sub(r"^(?:,\s*| and )", "", remainder[len(name) :], count=1)
    return True


def candidates(record, max_values):
    lengths = {}
    for source in record.sources:
        for signal in source.signals:
            lengths[signal.name] = max(lengths.get(signal.name, 0), signal.n_values)
    fitting = [name for name, length in lengths.items() if length <= max_values]
    if not fitting:
        return []
    resolution = max(fitting, key=lengths.get)
    result = []
    for source in record.sources:
        for signal in source.signals:
            if signal.name != resolution:
                continue
            values = signal.to_arrow().to_numpy(zero_copy_only=False).astype(np.float64)
            if np.isinf(values).any():
                raise ValueError(f"infinite observation in {record.id}/{source.name}")
            if not np.isnan(values).all():
                result.append((source.name, float(np.nanstd(values)), resolution, values))
    return result


def materialize(tasks, split_ids, read_value, max_values=4096, max_series=32):
    if not 0 < max_values <= 4096 or not 0 < max_series <= 32:
        raise ValueError("budgets must fit the shared limits of 4096 values and 32 channels")
    if set(split_ids) != {"train", "validation", "test"}:
        raise ValueError("expected train, validation and test assignments")
    tasks = list(tasks)
    by_id = {int(task.id): task for task in tasks}
    assignment = {i: part for part, ids in split_ids.items() for i in ids}
    if len(by_id) != len(tasks) or len(assignment) != sum(map(len, split_ids.values())):
        raise ValueError("duplicate task ID in dataset or split")
    if missing := assignment.keys() - by_id.keys():
        raise ValueError(f"assigned IDs missing from dataset: {sorted(missing)}")
    traces, rows, cache, incident_partition = {}, [], {}, {}
    for task_id, part in assignment.items():
        task = by_id[task_id]
        incidents = []
        for record in task.inputs:
            ids = [str(read_value(a.value)) for a in record.annotations if a.key == "incident_id"]
            if not ids:
                raise ValueError(f"missing incident ID for task {task_id}")
            for incident in ids:
                if incident_partition.setdefault(incident, part) != part:
                    raise ValueError(f"incident {incident} overlaps splits")
            incidents.extend(ids)
        notes = {a.key: read_value(a.value) for a in task.annotations}
        labels = getattr(task, "label_set", None)
        options = (
            [label.name for label in labels.labels]
            if labels is not None
            else list(map(str, notes.get("answer_options", [])))
        )
        prompt = str(read_value(task.prompt))
        ordered = options.copy()
        random.Random(prompt).shuffle(ordered)
        answer = str(read_value(task.targets[0]))
        row = dict(
            id=task_id,
            split=part,
            incidents=sorted(set(incidents)),
            prompt=prompt,
            category=str(notes.get("task_category", "")),
            answer=answer,
            options=ordered,
            released_options=options,
            series=[],
            exclusion=None,
        )
        rows.append(row)
        if options and answer not in options:
            raise ValueError(f"target absent from options in task {task_id}")
        signals = [s for record in task.inputs for s in record.signals]
        if not signals or any(s.spec.modality == "image" for s in signals):
            row["exclusion"] = "missing series or image modality"
            continue
        limit = max_series // len(task.inputs)
        if not limit:
            row["exclusion"] = "record count exceeds channel budget"
            continue
        mentioned = "\n".join([prompt, *options])
        for record in task.inputs:
            rid = int(record.id)
            if rid not in cache:
                cache[rid] = candidates(record, max_values)
            group = cache[rid]
            if not group:
                row["exclusion"] = f"record {rid} has no observed series at a fitting resolution"
                break
            ranked = sorted(range(len(group)), key=lambda i: (not named(group[i][0], mentioned), -group[i][1]))
            for index in sorted(ranked[:limit]):
                name, _, resolution, values = group[index]
                key = json.dumps([rid, name, resolution], separators=(",", ":"))
                if key not in traces:
                    traces[key] = dict(
                        name=name,
                        resolution=resolution,
                        record_id=rid,
                        values=[None if np.isnan(v) else float(v) for v in values],
                    )
                row["series"].append(key)
        if (
            not row["exclusion"]
            and row["category"] == "Anomaly Identification"
            and answer != "No anomaly among the listed channels"
            and not covered(answer, [traces[k]["name"] for k in row["series"]])
        ):
            row["exclusion"] = "correct channel absent or answer cannot be mapped to selected channels"
    manifest = dict(
        schema=1,
        dataset=DATASET,
        version=VERSION,
        max_values=max_values,
        max_series=max_series,
        rows=rows,
        traces=traces,
        unassigned_ids=sorted(by_id.keys() - assignment.keys()),
    )
    manifest["sha256"] = digest(manifest)
    return manifest


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    expected = manifest.pop("sha256")
    if manifest["schema"] != 1 or digest(manifest) != expected:
        raise ValueError("unsupported or modified manifest")
    manifest["sha256"] = expected
    return manifest


def freeze(path):
    from timenet.client import TimeNet
    from timenet.types import read_value

    ids = json.loads(SPLIT.read_text())
    with TimeNet().open(DATASET, VERSION) as reader:
        manifest = materialize(reader.read().tasks, ids, read_value)
    with Path(path).open("x") as out:
        json.dump(manifest, out, separators=(",", ":"), allow_nan=False)
    print(manifest["sha256"], Counter(r["split"] for r in manifest["rows"] if not r["exclusion"]))
