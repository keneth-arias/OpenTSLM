# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
"""Train a numerical Qwen head with reproducible single-node checkpoints."""

import argparse
import hashlib
import math
import fcntl
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from opentslm.forecast import FORECASTERS, MODEL_REVISION, task_loss
from opentslm.forecast_checkpoint import finish_direct, point_to, publish, read_manifest, reconcile_direct_resume, recover, rotate_direct, selected_generation, sha256
from opentslm import forecast_direct
from opentslm.forecast_inputs import direct_cohorts, verify_weight_snapshot
from opentslm.forecast_inference import JointInputs, add_lora, configure_device, forecast, forecast_batches, read_task_data, state_options, task_schema, verify_candidate_input
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel


# Script digest of the compatible unversioned direct/persistence trainer at d4a790e.
LEGACY_TRAINER_SHA256 = "4a17f9f4a9efc393767cfa0d42d89bd29165bae9794bcefa0e4bf1006e6dd9db"


PARENT_TRAINER_SHA256 = "e4a32f79f1533d2583c71f3b74619bd36202b532197f62bcf74f95ef0339e920"

PREVIOUS_TRAINER_SHA256 = "88beb175c7c0e4dec8480d9cd1d9fd10b5b2619cad6d377ef1337f73ecc0642e"


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--stop-step", required=True, type=int)
    p.add_argument("--schedule-steps", required=True, type=int)
    p.add_argument("--batch", type=int, default=4, help="Global batch size")
    p.add_argument("--mode", choices=("single", "ddp", "fsdp"), default="fsdp")
    p.add_argument("--head", choices=tuple(FORECASTERS), default="direct")
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--seed", type=int, default=27)
    p.add_argument("--save-seconds", type=int, default=900)
    p.add_argument("--min-free-gib", type=int, default=12)
    p.add_argument("--validation", type=Path)
    p.add_argument(
        "--validate-at",
        choices=("checkpoint", "final"),
        default="checkpoint",
        help="Validate at each save or only at the final scheduled update",
    )
    p.add_argument("--matched-point",action="store_true",help="Fit the one-GPU direct control on the same full epoch as the candidate heads")
    p.add_argument("--direct-objective", choices=forecast_direct.OBJECTIVES)
    p.add_argument("--logical-run", help="Stable identifier of this matched fit across attempts")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--lora", action="store_true")
    p.add_argument("--input-manifest", type=Path)
    p.add_argument("--input-manifest-sha256")
    p.add_argument("--validation-manifest", type=Path)
    p.add_argument("--validation-manifest-sha256")
    p.add_argument("--thresholds-sha256", help="Frozen threshold manifest SHA-256 for episode fits")
    p.add_argument("--scaler", type=Path)
    p.add_argument("--input-view", choices=("first", "both"))
    args = p.parse_args()
    if bool(args.logical_run) != bool(args.direct_objective):
        p.error("Supply --logical-run and --direct-objective together")
    if args.direct_objective:
        if (args.mode != "single" or args.head != "direct" or args.input_view or args.matched_point or
                args.validate_at != "checkpoint" or
                args.precision != "fp32" or args.batch != 16 or args.seed not in (27, 28) or
                args.stop_step != args.schedule_steps or args.schedule_steps < forecast_direct.EVAL_INTERVAL):
            p.error("Direct comparison requires a complete single-GPU FP32 batch-16 epoch and a matched seed")
        if not all((args.input_manifest, args.input_manifest_sha256, args.validation,
                    args.validation_manifest, args.validation_manifest_sha256)):
            p.error("The direct comparison requires both pinned cohort manifests")
        args.lora = True
    if args.matched_point and args.head!="direct":
        p.error("--matched-point requires the direct head")
    if args.matched_point or args.head in ("quantile", "episode"):
        if args.mode != "single" or args.input_view:
            p.error("Candidate heads require single-worker univariate training")
        args.lora = True
        args.validate_at = "final"
    if args.thresholds_sha256 and args.head != "episode":
        p.error("Only the episode head uses station thresholds")
    if args.head == "episode" and not args.thresholds_sha256:
        p.error("Supply --thresholds-sha256 for episode fits")
    args.task_schema = task_schema(args.head, args.thresholds_sha256)
    if bool(args.input_view) != bool(args.scaler):
        p.error("Supply both input view and fitted scaler")
    if args.scaler and (args.mode != "single" or args.head != "direct"):
        p.error("Matched two-channel fits require a single worker and direct head")
    args.joint = JointInputs.load(args.scaler, args.input_view) if args.scaler else None
    if not 0 < args.stop_step <= args.schedule_steps or args.save_seconds < 0:
        p.error("Invalid stopping step, schedule or save interval")
    if args.lora and args.mode == "fsdp":
        p.error("Use the single or ddp mode for LoRA")
    return args


def checkpoint_state(model, optimizer, lora):
    params, optim = get_state_dict(model, optimizer, options=state_options(lora))
    return {"model": params, "optimizer": optim}


def save(root, model, optimizer, scheduler, step, identity, rank, best_mae, metric, evaluations=None):
    started = time.monotonic()
    staging = [tempfile.mkdtemp(prefix=".saving-", dir=root) if rank == 0 else None]
    dist.broadcast_object_list(staging)
    staging = Path(staging[0])
    dcp.save(checkpoint_state(model, optimizer, identity["lora"]), checkpoint_id=staging / "dcp")
    torch.save(
        {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()},
        staging / f"rng-{rank}.pt",
    )
    dist.barrier()
    if rank == 0:
        metadata = {"scheduler": scheduler.state_dict()}
        if evaluations is not None:
            metadata["evaluations"] = evaluations
        elif "task_schema" in identity:
            metadata.update(best_metric=best_mae, metric="brier_uncalibrated" if identity["head"] == "episode" else "mae")
        else:
            metadata["best_mae"] = best_mae
        (staging / "training.pt.json").write_text(json.dumps(metadata, allow_nan=False))
        final, manifest = publish(root, staging, step, identity)
        if evaluations is not None:
            if evaluations[-1]["best_step"] == step:
                point_to(root, "best", final)
        elif metric is not None and metric <= best_mae:
            point_to(root, "best", final)
        if evaluations is not None:
            rotate_direct(root, identity, final)
        else:
            protected = {final.name}
            if (root / "best").exists():
                protected.add((root / "best").read_text().strip())
            # Rotate only our own published generations, after the new one is visible.
            candidates = [
                path
                for path in sorted(root.glob("step-????????"), reverse=True)
                if (info := read_manifest(path)) and info["identity"] == identity
            ]
            protected.update(p.name for p in candidates[:2])
            for path in candidates:
                if path.name not in protected:
                    shutil.rmtree(path)
        print(
            json.dumps(
                {
                    "checkpoint": final.name,
                    "checkpoint_bytes": sum(f["size"] for f in manifest["files"].values()),
                    "checkpoint_seconds": time.monotonic() - started,
                    "manifest_sha256": sha256(final / "manifest.json"),
                }
            ),
            flush=True,
        )
    dist.barrier()


def resume_generation(root, identity, pinned_source):
    generation = recover(root, identity)
    if generation is not None or pinned_source or "matched_point" in identity or "direct_objective" in identity:
        return generation, identity["trainer"]
    compatible = [PARENT_TRAINER_SHA256, PREVIOUS_TRAINER_SHA256]
    if "task_schema" not in identity:
        compatible.append(LEGACY_TRAINER_SHA256)
    for revision in compatible:
        generation = recover(root, {**identity, "trainer": revision})
        if generation is not None:
            return generation, revision
    return None, identity["trainer"]


def restore(root, model, optimizer, scheduler, args, identity, rank):
    selected = [None, identity["trainer"]]
    if rank == 0 and args.resume:
        selected[:] = resume_generation(root,identity,bool(os.environ.get("FORECAST_SOURCE_REVISION")))
    dist.broadcast_object_list(selected)
    identity["trainer"] = selected[1]
    if selected[0] is None:
        if args.resume:
            raise ValueError("No complete compatible checkpoint")
        return 0, None, []
    step, path = selected[0]
    if args.direct_objective:
        reconcile_direct_resume(root, identity, step)
    state = checkpoint_state(model, optimizer, identity["lora"])
    dcp.load(state, checkpoint_id=path / "dcp")
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
        options=state_options(identity["lora"]),
    )
    metadata = json.loads((path / "training.pt.json").read_text())
    scheduler.load_state_dict(metadata["scheduler"])
    rng = torch.load(path / f"rng-{rank}.pt", map_location="cpu", weights_only=True)
    torch.set_rng_state(rng["cpu"])
    torch.cuda.set_rng_state(rng["cuda"])
    if rank == 0:
        print(json.dumps({"restored_step": step, "checkpoint": path.name}), flush=True)
    if args.direct_objective:
        evaluations = metadata["evaluations"]
        if evaluations[-1]["best_step"] is not None:
            selected_generation(root, identity, evaluations[-1]["best_step"])
        return step, evaluations[-1]["best_mae"], evaluations
    return step, metadata["best_metric"] if "best_metric" in metadata else metadata["best_mae"], []


@torch.no_grad()
def validation(model, data, args, rank, world, device):
    # Every rank must execute the same number of FSDP gathers. Pad only the
    # final batch and discard those padding rows from metric reductions.
    ids, history, target, thresholds = data
    if args.head == "episode":
        model.eval()
        squared = torch.zeros((), dtype=torch.float64, device=device)
        for index, logit in forecast_batches(model, history, args, rank, world, device, thresholds):
            squared += (logit.sigmoid() - target[index].to(device)).square().sum(dtype=torch.float64)
        dist.all_reduce(squared)
        model.train()
        return {"validation_windows": len(ids), "brier_uncalibrated": squared.item() / len(ids)}
    absolute = torch.zeros((6, target.shape[-1]), dtype=torch.float64, device=device)
    persistence = torch.zeros_like(absolute)
    quantile_stats = torch.zeros((3, 6), dtype=torch.float64, device=device) if args.head == "quantile" else None
    model.eval()
    for index, predicted in forecast_batches(model, history, args, rank, world, device):
        y = target[index].to(device)
        if args.head == "quantile":
            low, median, high = predicted.unbind(-1)
            quantile_stats[0] += ((low <= y[..., 0]) & (y[..., 0] <= high)).sum(0)
            quantile_stats[1] += (high - low).sum(0, dtype=torch.float64)
            quantile_stats[2] += ((low > median) | (median > high)).sum(0)
            predicted = median.unsqueeze(-1)
        absolute += (predicted - y).abs().sum(dim=0, dtype=torch.float64)
        if args.joint is None:
            last = history[index, -1:].to(device)
            persistence += (last - y).abs().sum(dim=0, dtype=torch.float64)
    dist.all_reduce(absolute)
    dist.all_reduce(persistence)
    intervals = {}
    if quantile_stats is not None:
        dist.all_reduce(quantile_stats)
        intervals = {"coverage_by_horizon": (quantile_stats[0] / len(ids)).tolist(),
                     "width_by_horizon": (quantile_stats[1] / len(ids)).tolist(),
                     "crossed_by_horizon": quantile_stats[2].int().tolist()}
    model.train()
    if args.joint:
        return {"validation_windows": len(ids), "mae": absolute.mean().item() / len(ids),
                "channel_mae_by_horizon": (absolute / len(ids)).tolist()}
    return {
        "validation_windows": len(ids),
        "mae": absolute.mean().item() / len(ids),
        **intervals,
        "mae_by_horizon": (absolute[:, 0] / len(ids)).tolist(),
        "persistence_mae": persistence.mean().item() / len(ids),
    }


def epoch_batches(rows, batch, *, include_partial):
    return math.ceil(rows / batch) if include_partial else rows // batch


def epoch_order(ids, args, epoch):
    include_partial = bool(args.joint or args.task_schema or args.matched_point or args.direct_objective)
    batches = epoch_batches(len(ids), args.batch, include_partial=include_partial)
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(args.seed + epoch))
    return order[:batches * args.batch]


def stop_marked(path):
    # A vanished armed channel should stop the worker at its next safe point.
    try:
        return Path(path).stat().st_size > 0
    except FileNotFoundError:
        return True


def train(args, rank, world, device):
    if args.task_schema or args.matched_point:
        verify_candidate_input(args.data, args.input_manifest, args.input_manifest_sha256, "fit", args.head, args.thresholds_sha256)
        if args.validation:
            verify_candidate_input(args.validation, args.validation_manifest, args.validation_manifest_sha256, "inner_validation", args.head, args.thresholds_sha256)
    cohorts = evaluation = None
    if args.direct_objective:
        if (args.data != args.input_manifest.parent / "data.npz" or
                args.validation != args.validation_manifest.parent / "data.npz"):
            raise ValueError("Cohort data must come from its pinned role directory")
        cohorts = direct_cohorts(args.input_manifest, args.input_manifest_sha256, args.validation_manifest,
                                 args.validation_manifest_sha256, sha256(args.weights.parent / "SHA256SUMS"))
        fit = cohorts[0]
        ids = fit["window_id"]
        history = torch.from_numpy(fit["history"]).float().unsqueeze(-1)
        targets = torch.from_numpy(fit["targets"]).float().unsqueeze(-1)
        thresholds = None
    else:
        ids, history, targets, thresholds = read_task_data(args.data, args.joint, args.head, args.thresholds_sha256)
        evaluation = read_task_data(args.validation, args.joint, args.head, args.thresholds_sha256) if args.validation else None
        if evaluation is not None and (not len(evaluation[0]) or np.intersect1d(ids, evaluation[0]).size):
            raise ValueError("Validation must be nonempty and disjoint from training")
    window_order = epoch_order(ids, args, 0)
    batches = math.ceil(len(window_order) / args.batch)
    if batches < 1 or (cohorts and args.schedule_steps != batches):
        raise ValueError("Schedule does not match the available training batches")
    identity = {
        "model": MODEL_REVISION,
        "batch": args.batch,
        "head": args.head,
        "data": sha256(args.data),
        "validation": sha256(args.validation) if args.validation else None,
        "schedule": args.schedule_steps,
        "precision": args.precision,
        "world": world,
        "mode": args.mode,
        "lora": args.lora,
        "seed": args.seed,
        "torch": str(torch.__version__),
        "trainer": os.environ.get("FORECAST_SOURCE_REVISION", sha256(Path(__file__))),
        "weights": sha256(args.weights.parent / "SHA256SUMS"),
    }
    if args.direct_objective:
        verify_weight_snapshot(args.weights, identity["weights"], strict=True)
        identity.update(direct_objective=args.direct_objective, logical_run=args.logical_run,
                        input_manifest_sha256=args.input_manifest_sha256, validation_manifest_sha256=args.validation_manifest_sha256,
                        evaluation_interval=forecast_direct.EVAL_INTERVAL, patience_checks=forecast_direct.PATIENCE,
                        min_improvement_ug_m3=str(forecast_direct.MIN_IMPROVEMENT), optimizer={"name": "AdamW", "lr": 1e-4, "weight_decay": 0,
                            "gradient_clip": 1.0, "scheduler": "cosine", "lora_rank": 4, "lora_alpha": 8})
        recipe = {k: v for k, v in identity.items() if k != "logical_run"}
        identity["recipe_sha256"] = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    if args.task_schema:
        identity.update(task_schema=args.task_schema, input_manifest_sha256=args.input_manifest_sha256,
                        validation_manifest_sha256=args.validation_manifest_sha256)
    if args.matched_point:
        identity.update(matched_point="complete_epoch_v1", input_manifest_sha256=args.input_manifest_sha256,
                        validation_manifest_sha256=args.validation_manifest_sha256)
    if args.validate_at != "checkpoint":
        identity["validate_at"] = args.validate_at
    if rank == 0 and not args.resume and list(args.run_dir.glob("step-????????")):
        raise FileExistsError("Choose a new run directory or resume")
    torch.manual_seed(args.seed)
    forecaster = FORECASTERS[args.head]
    dimensions = {"channels": 2, "targets": 2} if args.joint else {}
    stop_file = os.environ.get("EPISODE_STOP_FILE")

    def stop_requested():
        if not stop_file:
            return False
        flag = torch.tensor(bool(rank == 0 and stop_marked(stop_file)), device=device)
        dist.broadcast(flag, src=0)
        return flag.item()

    if stop_requested():
        if rank == 0:
            print(json.dumps({"stopped_before_update": True}), flush=True)
        return
    load_started = time.monotonic()
    model = forecaster.from_pretrained(checkpoint_path=str(args.weights), local_files_only=True, **dimensions)
    loaded_at = time.monotonic()
    # The embedding table is unused with numerical soft prompts.
    model.decoder.get_input_embeddings().requires_grad_(False)
    adapters = add_lora(model) if args.lora else []
    if args.joint:
        identity["joint"] = args.joint.identity()
    if args.joint or args.direct_objective:
        initial = hashlib.sha256()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                initial.update(name.encode())
                initial.update(parameter.detach().contiguous().numpy().tobytes())
        identity["initial_trainable_sha256"] = initial.hexdigest()
    if adapters and rank == 0:
        print(json.dumps({"adapter_modules": adapters}), flush=True)
    model = model.to(device)
    if rank == 0:
        print(json.dumps({"model_load_seconds": loaded_at - load_started,
                          "model_placement_seconds": time.monotonic() - loaded_at}), flush=True)
    if args.mode == "fsdp":
        mesh = init_device_mesh("cuda", (world,))
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16 if args.precision == "bf16" else None, reduce_dtype=torch.float32
        )
        for layer in model.decoder.layers:
            fully_shard(layer, mesh=mesh, mp_policy=policy)
        fully_shard(model, mesh=mesh, mp_policy=policy)
    core = model
    if args.mode == "ddp":
        model = DistributedDataParallel(model, device_ids=[device.index])
    probe = core.decoder.get_parameter(adapters[0]) if adapters else core.decoder.layers[0].mlp.down_proj.weight
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0, foreach=False
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.schedule_steps)
    step, best, evaluations = restore(args.run_dir, model, optimizer, scheduler, args, identity, rank)
    def evaluate_pair(update):
        sample, inner = (forecast_direct.evaluate_direct(model, rows, args, device) for rows in cohorts)
        selected = forecast_direct.selection(evaluations, update, inner["mae"], args.schedule_steps)
        evaluations.append({**selected, "fit_sample": sample, "inner_validation": inner})
        print(json.dumps({"step": update, "direct_evaluation": evaluations[-1]}, allow_nan=False), flush=True)

    if cohorts and evaluations and evaluations[-1]["outcome"] != "running":
        finish_direct(args.run_dir, identity, step, evaluations)
        return
    if not step < args.stop_step:
        raise ValueError("Stopping step must advance the restored run")
    current_epoch = 0
    saved_at = time.monotonic()
    if stop_requested():
        if rank == 0:
            print(json.dumps({"stopped_before_update": True}), flush=True)
        return
    if cohorts and not evaluations:
        evaluate_pair(0)
    first_cursor = step
    for cursor in range(step, args.stop_step):
        epoch, position = divmod(cursor, batches)
        if current_epoch != epoch:
            window_order = epoch_order(ids,args,epoch)
            current_epoch = epoch
        indices = window_order[position * args.batch:(position + 1) * args.batch]
        subset = indices[rank::world]
        x, y = history[subset].to(device), targets[subset].to(device)
        torch.cuda.synchronize()
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        context = thresholds[subset].to(device) if thresholds is not None else None
        prediction, center, scale = forecast(model, x, args, context)
        loss = task_loss(args.head, prediction, y, center, scale, direct_objective=args.direct_objective or "normalized-mse")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True, foreach=False)
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        compute = time.monotonic() - started
        loss_value = loss.detach()
        dist.all_reduce(loss_value, op=dist.ReduceOp.AVG)
        elapsed = torch.tensor(compute, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        shard = [
            v.to_local().numel() if isinstance(v, DTensor) else v.numel()
            for v in (probe, probe.grad, optimizer.state[probe]["exp_avg"])
        ]
        peak = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()], device=device)
        per_rank = [torch.empty_like(peak) for _ in range(world)]
        dist.all_gather(per_rank, peak)
        step = cursor + 1
        action = 0
        if rank == 0:
            if stop_file and step < args.stop_step and stop_marked(stop_file):
                action = 2  # Publish this step and leave on all ranks.
            elif step == args.stop_step or (cohorts and step == evaluations[-1]["next_eval_step"]) or time.monotonic() - saved_at >= args.save_seconds:
                action = 1
            if cursor == first_cursor:
                print(json.dumps({"first_update_since_model_load_seconds": time.monotonic() - load_started}), flush=True)
        save_now = torch.tensor(action, device=device)
        dist.broadcast(save_now, src=0)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": loss_value.item(),
                        "learning_rate_next": scheduler.get_last_lr()[0],
                        **({"batch_window_ids_sha256": hashlib.sha256(
                            ("\n".join(ids[indices.numpy()]) + "\n").encode()).hexdigest()}
                           if args.direct_objective else {"window_ids": ids[indices.numpy()].tolist()}),
                        "windows_per_second": len(indices) / elapsed.item(),
                        "peak_bytes_per_rank": [v.tolist() for v in per_rank],
                        "probe_global_elements": probe.numel(),
                        "probe_local_elements": shard,
                    }
                ),
                flush=True,
            )
        if save_now.item():
            disk_ok = torch.tensor(
                int(shutil.disk_usage(args.run_dir).free >= args.min_free_gib * 2**30), device=device
            )
            dist.all_reduce(disk_ok, op=dist.ReduceOp.MIN)
            if not disk_ok.item():
                raise OSError("Insufficient free space for a new checkpoint generation")
            evaluate_now = save_now.item() != 2 and evaluation is not None and (args.validate_at == "checkpoint" or step == args.schedule_steps)
            metrics = validation(model, evaluation, args, rank, world, device) if evaluate_now else {}
            if rank == 0 and metrics:
                print(json.dumps({"step": step, **metrics}), flush=True)
            metric = metrics.get("brier_uncalibrated" if args.head == "episode" else "mae")
            if metric is not None:
                best = metric if best is None else min(best, metric)
            if cohorts and step == evaluations[-1]["next_eval_step"]:
                evaluate_pair(step)
            save(args.run_dir, model, optimizer, scheduler, step, identity, rank, best, metric,
                 evaluations if cohorts else None)
            if cohorts and evaluations[-1]["outcome"] != "running":
                finish_direct(args.run_dir, identity, step, evaluations)
                return
            if save_now.item() == 2 or (step < args.stop_step and stop_requested()):
                if rank == 0:
                    print(json.dumps({"interrupted_at_step": step}), flush=True)
                return
            saved_at = time.monotonic()


def main():
    args = arguments()
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != (1 if args.mode == "single" else 2) or args.batch < world or args.batch % world:
        raise ValueError("Launch one process per GPU with a divisible global batch size")
    device = torch.device("cuda", rank)
    configure_device(device)
    dist.init_process_group("nccl", device_id=device)
    try:
        value = torch.tensor(rank + 1, device=device)
        dist.all_reduce(value)
        print(
            json.dumps(
                {"rank": rank, "gpu": str(torch.cuda.get_device_properties(device)), "reduced_rank_sum": value.item()}
            ),
            flush=True,
        )
        args.run_dir.mkdir(parents=True, exist_ok=True)
        with (args.run_dir / ".writer.lock").open("a") as lock:
            if rank == 0:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            dist.barrier()
            train(args, rank, world, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
