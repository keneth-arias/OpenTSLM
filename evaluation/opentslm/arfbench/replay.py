"""Replay raw model outputs against a frozen split, including failed questions."""

from collections import Counter, defaultdict
from pathlib import Path
import json

from .protocol import digest, question

DECODE = dict(do_sample=False, num_beams=1, max_new_tokens=64)


def probe_manifest(kind):
    names = ["cpu", "memory"]
    rows, traces = [], {}

    def add(key, signals, prompt, options, answer, **extra):
        refs = []
        for i, (name, values) in enumerate(signals):
            ref = f"{key}/{i}"
            traces[ref] = dict(name=name, values=values, observed=[v is not None for v in values])
            refs.append(ref)
        rows.append(
            dict(
                id=key,
                split=kind,
                exclusion=None,
                category=kind,
                series=refs,
                prompt=prompt,
                released_options=options,
                options=options,
                answer=answer,
                **extra,
            )
        )

    if kind == "swap":
        spike, steady = [1, None, 1, 20, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]
        for order, labels in enumerate((names, names[::-1])):
            for target in names:
                signals = [(name, spike if target == name else steady) for name in labels]
                add(f"{order}/{target}", signals, "Which channel contains the spike?", names, target, pair=order)
    elif kind == "missing":
        for key, vals, answer in [("zero", [1, 0, 2], "Yes"), ("gap", [1, None, 2], "No"), ("pad", [1, 2], "No")]:
            add(key, [("sensor", vals)], "Does the series contain a measured zero?", ["Yes", "No"], answer)
    else:
        raise ValueError(f"unknown diagnostic {kind}")
    manifest = dict(rows=rows, traces=traces)
    manifest["sha256"] = digest(manifest)
    return manifest


def normalize(output):
    return " ".join(output.lower().rsplit("answer:", 1)[-1].strip(" .\n\t\r").split())


def read_run(path):
    path = Path(path)
    settings = json.loads((path / "run.json").read_text())
    with (path / "predictions.jsonl").open() as f:
        predictions = [json.loads(line) for line in f]
    return settings, predictions


def score(manifest, settings, predictions):
    split = settings["split"]
    if manifest["sha256"] != settings["manifest_sha256"]:
        raise ValueError("predictions belong to a different manifest")
    assigned = [r for r in manifest["rows"] if r["split"] == split]
    rows = {r["id"]: r for r in assigned if not r["exclusion"]}
    by_id = {p["id"]: p for p in predictions}
    if len(predictions) != len(by_id) or rows.keys() != by_id.keys():
        raise ValueError("predictions must contain every retained ID exactly once")
    right = {}
    categories = defaultdict(lambda: defaultdict(list))
    counts = Counter()
    for task_id, row in rows.items():
        pred = by_id[task_id]
        if pred["question"] != question(row):
            raise ValueError(f"question changed for {task_id}")
        if pred.get("error"):
            counts["failures"] += 1
            hit = False
        else:
            output = normalize(pred["output"])
            chosen = next((o for o in row["options"] or [row["answer"]] if normalize(o) == output), None)
            counts["scored"] += 1
            counts["unparseable"] += chosen is None
            hit = chosen == row["answer"]
        categories[row["category"]][row["answer"]].append(hit)
        right[task_id] = hit

    per_category = {}
    for category, labels in categories.items():
        n = sum(map(len, labels.values()))
        per_category[category] = dict(
            questions=n,
            correct=sum(map(sum, labels.values())),
            accuracy=sum(map(sum, labels.values())) / n,
            balanced_accuracy=sum(sum(v) / len(v) for v in labels.values()) / len(labels),
        )
    n = len(rows)
    report = dict(
        assigned=len(assigned),
        retained=n,
        attempted=len(predictions),
        scored=counts["scored"],
        failures=counts["failures"],
        unparseable=counts["unparseable"],
        correct=sum(right.values()),
        accuracy=sum(right.values()) / n if n else None,
        per_category=per_category,
    )
    report["balanced_accuracy"] = (
        sum(c["balanced_accuracy"] for c in per_category.values()) / len(per_category) if per_category else None
    )
    if split == "swap":
        pairs = defaultdict(list)
        for key, row in rows.items():
            pairs[row["pair"]].append(right[key])
        report["both_directions_correct"] = {str(order): all(hits) for order, hits in pairs.items()}
        report["fraction_pairs_both_correct"] = sum(map(all, pairs.values())) / len(pairs)
    return report, right


def compare(manifest, first, second):
    left, right = first[0], second[0]
    if (left["split"], left["decoding"]) != (right["split"], right["decoding"]):
        raise ValueError("paired runs require identical split and decoding")
    a, a_hits = score(manifest, *first)
    b, b_hits = score(manifest, *second)
    return dict(
        first=a,
        second=b,
        paired=len(a_hits),
        first_only=sum(a_hits[i] and not b_hits[i] for i in a_hits),
        second_only=sum(b_hits[i] and not a_hits[i] for i in a_hits),
        both_correct=sum(a_hits[i] and b_hits[i] for i in a_hits),
        both_wrong=sum(not a_hits[i] and not b_hits[i] for i in a_hits),
    )
