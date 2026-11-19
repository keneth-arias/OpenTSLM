# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
"""Train a numerical Qwen head with reproducible single-node checkpoints."""

import argparse
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
from opentslm.forecast import MODEL_REVISION, PersistenceForecaster, QwenForecaster
from opentslm.forecast_checkpoint import point_to, publish, read_manifest, recover, sha256
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--stop-step", required=True, type=int)
    p.add_argument("--schedule-steps", required=True, type=int)
    p.add_argument("--batch", type=int, default=4, help="Global batch size")
    p.add_argument("--mode", choices=("single", "ddp", "fsdp"), default="fsdp")
    p.add_argument("--head", choices=("direct", "persistence"), default="direct")
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--seed", type=int, default=27)
    p.add_argument("--save-seconds", type=int, default=900)
    p.add_argument("--min-free-gib", type=int, default=12)
    p.add_argument("--validation", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--lora", action="store_true")
    args = p.parse_args()
    if not 0 < args.stop_step <= args.schedule_steps or args.save_seconds < 0:
        p.error("Invalid stopping step, schedule or save interval")
    if args.lora and args.mode == "fsdp":
        p.error("Use the single or ddp mode for LoRA")
    return args


def read_data(path):
    with np.load(path, allow_pickle=False) as data:
        ids = data["window_id"]
        history = torch.from_numpy(data["history"]).float().unsqueeze(-1)
        target = torch.from_numpy(data["targets"]).float().unsqueeze(-1)
    if (
        history.shape != (len(ids), 24, 1)
        or target.shape != (len(ids), 6, 1)
        or len(np.unique(ids)) != len(ids)
        or not history.isfinite().all()
        or not target.isfinite().all()
    ):
        raise ValueError("Expected unique, finite 24/6 forecast windows")
    return ids, history, target


def add_lora(model):
    from peft import LoraConfig, inject_adapter_in_model

    targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_a",
        "in_proj_b",
        "out_proj",
    ]
    model.decoder = inject_adapter_in_model(LoraConfig(r=4, lora_alpha=8, target_modules=targets), model.decoder)
    names = [name for name, _ in model.decoder.named_parameters() if ".lora_B." in name]
    expected = {
        index: 4 if kind == "full_attention" else 5 for index, kind in enumerate(model.decoder.config.layer_types)
    }
    for index, count in expected.items():
        if sum(name.startswith(f"layers.{index}.") for name in names) != count:
            raise RuntimeError(f"Unexpected LoRA coverage in decoder layer {index}")
    return names


def state_options(lora):
    return StateDictOptions(ignore_frozen_params=lora, strict=not lora)


def checkpoint_state(model, optimizer, lora):
    params, optim = get_state_dict(model, optimizer, options=state_options(lora))
    return {"model": params, "optimizer": optim}


def save(root, model, optimizer, scheduler, step, identity, rank, best_mae, metric):
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
        (staging / "training.pt.json").write_text(
            json.dumps({"scheduler": scheduler.state_dict(), "best_mae": best_mae})
        )
        final, manifest = publish(root, staging, step, identity)
        if metric is not None and metric <= best_mae:
            point_to(root, "best", final)
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


def restore(root, model, optimizer, scheduler, args, identity, rank):
    selected = [recover(root, identity) if rank == 0 and args.resume else None]
    dist.broadcast_object_list(selected)
    if selected[0] is None:
        if args.resume:
            raise ValueError("No complete compatible checkpoint")
        return 0, None
    step, path = selected[0]
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
    return step, metadata["best_mae"]


def forecast(model, history, args):
    center = history.mean(dim=1, keepdim=True)
    scale = history.std(dim=1, correction=0, keepdim=True)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalized = (history - center) / scale
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
        predicted = model(normalized, torch.ones_like(history, dtype=torch.bool))
    return predicted.float(), center, scale


@torch.no_grad()
def validation(model, data, args, rank, world, device):
    # Every rank must execute the same number of FSDP gathers. Pad only the
    # final batch and discard those padding rows from metric reductions.
    ids, history, target = data
    absolute = torch.zeros(6, dtype=torch.float64, device=device)
    persistence = torch.zeros_like(absolute)
    model.eval()
    for start in range(0, len(ids), args.batch):
        index = torch.arange(start + rank, start + args.batch, world)
        valid = index < len(ids)
        index = index.clamp_max(len(ids) - 1)
        x = history[index].to(device)
        y = target[index].to(device)
        p, center, scale = forecast(model, x, args)
        absolute += (p * scale + center - y).abs()[valid].sum(dim=(0, 2))
        persistence += (x[:, -1:] - y).abs()[valid].sum(dim=(0, 2))
    dist.all_reduce(absolute)
    dist.all_reduce(persistence)
    model.train()
    return {
        "validation_windows": len(ids),
        "mae": absolute.mean().item() / len(ids),
        "mae_by_horizon": (absolute / len(ids)).tolist(),
        "persistence_mae": persistence.mean().item() / len(ids),
    }


def train(args, rank, world, device):
    ids, history, targets = read_data(args.data)
    batches = len(ids) // args.batch
    if batches < 1:
        raise ValueError("One complete training batch is required")
    evaluation = read_data(args.validation) if args.validation else None
    if evaluation is not None and (not len(evaluation[0]) or np.intersect1d(ids, evaluation[0]).size):
        raise ValueError("Validation must be nonempty and disjoint from training")
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
    if rank == 0 and not args.resume and list(args.run_dir.glob("step-????????")):
        raise FileExistsError("Choose a new run directory or resume")
    torch.manual_seed(args.seed)
    forecaster = PersistenceForecaster if args.head == "persistence" else QwenForecaster
    model = forecaster.from_pretrained(checkpoint_path=str(args.weights), local_files_only=True)
    # The embedding table is unused with numerical soft prompts.
    model.decoder.get_input_embeddings().requires_grad_(False)
    adapters = add_lora(model) if args.lora else []
    if adapters and rank == 0:
        print(json.dumps({"adapter_modules": adapters}), flush=True)
    model = model.to(device)
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
    step, best = restore(args.run_dir, model, optimizer, scheduler, args, identity, rank)
    if not step < args.stop_step:
        raise ValueError("Stopping step must advance the restored run")
    window_order = None
    saved_at = time.monotonic()
    for cursor in range(step, args.stop_step):
        epoch, position = divmod(cursor, batches)
        if position == 0 or window_order is None:
            window_order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(args.seed + epoch))
        indices = window_order[position * args.batch : (position + 1) * args.batch]
        subset = indices[rank::world]
        x, y = history[subset].to(device), targets[subset].to(device)
        torch.cuda.synchronize()
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        prediction, center, scale = forecast(model, x, args)
        loss = torch.nn.functional.mse_loss(prediction, (y - center) / scale)
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
        save_now = torch.tensor(
            int(step == args.stop_step or time.monotonic() - saved_at >= args.save_seconds) if rank == 0 else 0,
            device=device,
        )
        dist.broadcast(save_now, src=0)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": loss_value.item(),
                        "learning_rate_next": scheduler.get_last_lr()[0],
                        "window_ids": ids[indices.numpy()].tolist(),
                        "windows_per_second": args.batch / elapsed.item(),
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
            metrics = validation(model, evaluation, args, rank, world, device) if evaluation else {}
            if rank == 0 and metrics:
                print(json.dumps({"step": step, **metrics}), flush=True)
            metric = metrics.get("mae")
            if metric is not None:
                best = metric if best is None else min(best, metric)
            save(args.run_dir, model, optimizer, scheduler, step, identity, rank, best, metric)
            saved_at = time.monotonic()


def main():
    args = arguments()
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != (1 if args.mode == "single" else 2) or args.batch < world or args.batch % world:
        raise ValueError("Launch one process per GPU with a divisible global batch size")
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
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
