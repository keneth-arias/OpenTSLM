# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
"""Evaluate the matched direct forecast recipes and checkpoint their selection history."""

import math
from decimal import Decimal

import numpy as np
import torch

from .forecast import task_loss
from .forecast_inference import forecast


OBJECTIVES = ("normalized-mse", "physical-mae")
EVAL_INTERVAL = 500
PATIENCE = 4
MIN_IMPROVEMENT = Decimal("0.02")


def selection(history, step, mae, ceiling):
    """Step zero is diagnostic. Improvements use the best accepted physical MAE."""
    if not math.isfinite(mae) or mae < 0:
        raise ValueError("Invalid validation MAE")
    if history and (step <= history[-1]["step"] or history[-1]["outcome"] != "running"):
        raise ValueError("Evaluation steps must advance")
    previous = history[-1] if history else {}
    best_step, best = previous.get("best_step"), previous.get("best_mae")
    patience = previous.get("patience", 0)
    if step >= EVAL_INTERVAL:
        if best is None or Decimal(str(best)) - Decimal(str(mae)) >= MIN_IMPROVEMENT:
            best_step, best, patience = step, mae, 0
        else:
            patience += 1
    terminal = step == ceiling or patience >= PATIENCE
    return {
        "step": step,
        "best_step": best_step,
        "best_mae": best,
        "patience": patience,
        "next_eval_step": None if terminal else min(ceiling, (step // EVAL_INTERVAL + 1) * EVAL_INTERVAL),
        "outcome": "early-stopped"
        if patience >= PATIENCE and step < ceiling
        else "completed"
        if step == ceiling
        else "running",
    }


@torch.no_grad()
def evaluate_direct(model, rows, args, device):
    indices = (
        np.flatnonzero(rows["graph_sample"]) if "graph_sample" in rows else np.arange(len(rows["window_id"]))
    )
    was_training = model.training
    model.eval()
    absolute = np.zeros(6)
    signed = np.zeros(6)
    persistence = np.zeros(6)
    objective_sum = high_error = high_persistence = 0.0
    high_count = 0
    high_hours = set()
    try:
        for start in range(0, len(indices), args.batch):
            batch = indices[start : start + args.batch]
            history = torch.as_tensor(rows["history"][batch], dtype=torch.float32, device=device).unsqueeze(
                -1
            )
            target = torch.as_tensor(rows["targets"][batch], dtype=torch.float32, device=device).unsqueeze(-1)
            normalized, center, scale = forecast(model, history, args)
            objective = task_loss(
                "direct", normalized, target, center, scale, direct_objective=args.direct_objective
            )
            objective_sum += objective.item() * len(batch) * 6
            predicted = (normalized * scale + center).squeeze(-1).cpu().numpy().astype(np.float64)
            actual = rows["targets"][batch].astype(np.float64)
            error = predicted - actual
            baseline = np.abs(rows["history"][batch, -1:] - actual)
            absolute += np.abs(error).sum(axis=0)
            signed += error.sum(axis=0)
            persistence += baseline.sum(axis=0)
            high = actual >= rows["threshold"][batch, None]
            high_error += np.abs(error[high]).sum()
            high_persistence += baseline[high].sum()
            high_count += int(high.sum())
            for row, horizon in zip(*np.nonzero(high), strict=True):
                i = batch[row]
                high_hours.add((str(rows["station_key"][i]), int(rows["origin_hour"][i]) + int(horizon) + 1))
    finally:
        model.train(was_training)
    n = len(indices)
    metrics = {
        "windows": n,
        "pairs": 6 * n,
        "objective": objective_sum / (6 * n),
        "mae": absolute.sum() / (6 * n),
        "mae_by_horizon": (absolute / n).tolist(),
        "signed_error_by_horizon": (signed / n).tolist(),
        "persistence_mae": persistence.sum() / (6 * n),
        "high_pairs": high_count,
        "high_station_hours": len(high_hours),
        "high_blocks_168h": len({hour // 168 for _, hour in high_hours}),
        "high_mae": high_error / high_count if high_count else None,
        "high_persistence_mae": high_persistence / high_count if high_count else None,
    }
    if not all(np.isfinite(v).all() for v in metrics.values() if v is not None):
        raise ValueError("Nonfinite direct evaluation")
    return metrics
