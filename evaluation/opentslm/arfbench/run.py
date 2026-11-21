# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
# Portions adapted from AWS Samples under MIT-0.

"""Execute a frozen QA protocol; training recipe adapted from AWS (MIT-0)."""

import json
import math
import random

import torch

from .backends import inputs
from .protocol import question
from .replay import DECODE, score


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


@torch.inference_mode()
def evaluate(net, manifest, settings, directory):
    net.eval()
    write_json(directory / "run.json", settings)
    results = []
    with (directory / "predictions.jsonl").open("x") as stream:
        for row in manifest["rows"]:
            if row["split"] != settings["split"] or row["exclusion"]:
                continue
            prompt = question(row)
            prediction = dict(id=row["id"], question=prompt)
            try:
                with torch.autocast("cuda", torch.bfloat16, enabled=settings["device"].startswith("cuda")):
                    prediction["output"] = net.generate(
                        inputs(manifest, row), prompt, max_new_tokens=DECODE["max_new_tokens"]
                    )
            except Exception as exc:
                prediction["error"] = f"{type(exc).__name__}: {exc}"
            stream.write(json.dumps(prediction) + "\n")
            stream.flush()
            results.append(prediction)
    metrics, _ = score(manifest, settings, results)
    write_json(directory / "metrics.json", metrics)
    return metrics


def fit(net, manifest, settings, directory):
    from transformers import get_wsd_schedule

    train = [r for r in manifest["rows"] if r["split"] == "train" and not r["exclusion"]]
    val = [r for r in manifest["rows"] if r["split"] == "validation" and not r["exclusion"]]
    epochs, accumulation = settings["epochs"], settings["accumulation"]
    if not train or not val or min(epochs, accumulation) < 1:
        raise ValueError("positive epochs and accumulation, training and validation rows required")
    rng = random.Random(settings["seed"])
    torch.manual_seed(settings["seed"])
    net.requires_grad_(False)
    encoder, perceiver, attention, embeddings = net.trained_modules()
    groups = []
    for name, modules, lr in [
        ("encoder", [encoder], 4e-4),
        ("adapter", [perceiver, attention], 2e-4),
        ("base", [embeddings], 4e-4),
    ]:
        for module in modules:
            module.float().requires_grad_(True)
        groups.append(dict(name=name, params=[p for mod in modules for p in mod.parameters()], lr=lr))
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = epochs * math.ceil(len(train) / accumulation)
    warmup = min(steps - 1, max(1, int(steps * 0.03)))
    decay = min(steps - warmup, max(1, int(steps * 0.2)))
    scheduler = get_wsd_schedule(
        optimizer, num_training_steps=steps, num_warmup_steps=warmup, num_decay_steps=decay, decay_type="linear"
    )
    settings.update(
        optimizer="AdamW",
        weight_decay=0.01,
        learning_rates={g["name"]: g["initial_lr"] for g in groups},
        warmup_steps=warmup,
        decay_steps=decay,
        total_steps=steps,
        training_questions=len(train),
        validation_questions=len(val),
    )
    write_json(directory / "training.json", settings)
    best_loss, best_epoch, best_weights = math.inf, None, None
    log = []
    for epoch in range(1, epochs + 1):
        order = train.copy()
        rng.shuffle(order)
        net.train()
        total = 0.0
        for start in range(0, len(order), accumulation):
            count = 0
            optimizer.zero_grad(set_to_none=True)
            for row in order[start : start + accumulation]:
                with torch.autocast("cuda", torch.bfloat16, enabled=settings["device"].startswith("cuda")):
                    losses = net.loss(inputs(manifest, row), question(row, rng), row["answer"], reduction="none")
                if not torch.isfinite(losses).all():
                    raise ValueError(f"nonfinite training loss at {row['id']}")
                losses.sum().backward()
                total += losses.detach().mean().item()
                count += losses.numel()
            for group in groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.div_(count)
            optimizer.step()
            scheduler.step()
        net.eval()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=settings["device"].startswith("cuda")):
            val_loss = sum(net.loss(inputs(manifest, r), question(r), r["answer"]).item() for r in val) / len(val)
        if not math.isfinite(val_loss):
            raise ValueError(f"nonfinite validation loss at epoch {epoch}")
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_weights = {k: p.detach().to("cpu", copy=True) for k, p in net.named_parameters() if p.requires_grad}
        log.append(dict(epoch=epoch, train_loss=total / len(order), validation_loss=val_loss))
        write_json(directory / "epochs.json", log)
    net.load_state_dict(best_weights, strict=False)
    net.save_pretrained(directory / "checkpoint")
    write_json(directory / "selection.json", dict(best_epoch=best_epoch, validation_loss=best_loss))
