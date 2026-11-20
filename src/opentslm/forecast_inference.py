# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
"""Restore numerical forecast checkpoints and export predictions in original units."""

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict

from opentslm.forecast import MODEL_REVISION, PersistenceForecaster, QwenForecaster
from opentslm.forecast_checkpoint import files, read_manifest, sha256


@dataclass(frozen=True)
class JointInputs:
    channels: tuple
    view: str
    mean: tuple
    scale: tuple
    version: str
    digest: str

    @classmethod
    def load(cls, path, view):
        raw = path.read_bytes()
        scaler = json.loads(raw)
        channels = tuple(scaler["channels"])
        mean, scale = (np.asarray(scaler[k], dtype=np.float64) for k in ("mean", "scale"))
        if (
            view not in {"first", "both"}
            or len(channels) != 2
            or len(set(channels)) != 2
            or mean.shape != (2,)
            or scale.shape != (2,)
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or not (scale > 0).all()
        ):
            raise ValueError("Invalid two-channel scaler or input view")
        return cls(
            channels,
            view,
            tuple(mean.tolist()),
            tuple(scale.tolist()),
            scaler["version"],
            hashlib.sha256(raw).hexdigest(),
        )

    def identity(self):
        return {
            "view": self.view,
            "channels": list(self.channels),
            "targets": list(self.channels),
            "horizon": 6,
            "version": self.version,
            "scaler_sha256": self.digest,
        }


def read_history(data, joint=None, *, require_history_only=False):
    if require_history_only:
        fields = {"window_id", "history", "channels"} if joint else {"window_id", "history"}
        if set(data) != fields:
            raise ValueError(f"Expected a history-only archive with fields {sorted(fields)}")
    ids = data["window_id"]
    raw = data["history"]
    if joint:
        if tuple(data["channels"].tolist()) != joint.channels or raw.shape != (len(ids), 24, 2):
            raise ValueError("Mismatched input channels or dimensions")
        if joint.view == "first":
            raw = raw.copy()
            raw[:, :, 1] = 0
    else:
        raw = raw[..., None]
    history = torch.from_numpy(raw).float()
    if (
        history.shape != (len(ids), 24, 2 if joint else 1)
        or ids.ndim != 1
        or ids.dtype.kind not in "Uifu"
        or not len(ids)
        or len(np.unique(ids)) != len(ids)
        or not history.isfinite().all()
    ):
        raise ValueError("Expected unique, finite 24/6 forecast windows")
    return ids, history


def read_data(path, joint=None):
    with np.load(path, allow_pickle=False) as data:
        ids, history = read_history(data, joint)
        target = torch.from_numpy(data["targets"]).float()
        if joint is None:
            target = target.unsqueeze(-1)
    if target.shape != (len(ids), 6, 2 if joint else 1) or not target.isfinite().all():
        raise ValueError("Expected six finite targets per window")
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


def forecast(model, history, args):
    joint = getattr(args, "joint", None)
    observed = torch.ones_like(history, dtype=torch.bool)
    if joint is None:
        center = history.mean(dim=1, keepdim=True)
        scale = history.std(dim=1, correction=0, keepdim=True)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        normalized = (history - center) / scale
    else:
        center = history.new_tensor(joint.mean)
        scale = history.new_tensor(joint.scale)
        if joint.view == "first":
            observed[:, :, 1] = False
        normalized = (torch.where(observed, history, center) - center) / scale
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
        predicted = model(normalized, observed)
    return predicted.float(), center, scale


@torch.no_grad()
def forecast_batches(model, history, args, rank, world, device):
    """All ranks run equally sized batches, including the padded final collective."""
    for start in range(0, len(history), args.batch):
        index = torch.arange(start + rank, start + args.batch, world)
        valid = index < len(history)
        x = history[index.clamp_max(len(history) - 1)].to(device)
        forecast_value, center, scale = forecast(model, x, args)
        yield index[valid], (forecast_value * scale + center)[valid]


def load_checkpoint_model(model, checkpoint, lora):
    options = state_options(lora)
    state = {"model": get_model_state_dict(model, options=options)}
    # Each inference worker loads a replica. DCP reads full and sharded training states.
    dcp.load(state, checkpoint_id=checkpoint / "dcp", no_dist=True)
    set_model_state_dict(model, state["model"], options=options)


def write_predictions(path, ids, shards, checkpoint_sha256=None, joint=None):
    indices = np.concatenate([idx for idx, _ in shards])
    values = np.concatenate([pred for _, pred in shards])
    order = indices.argsort()
    if (
        not np.array_equal(indices[order], np.arange(len(ids)))
        or values.shape != (len(ids), 6, 2 if joint else 1)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Expected one finite six-hour prediction per input window")
    with NamedTemporaryFile(dir=path.parent) as tmp:
        provenance = {"checkpoint_sha256": checkpoint_sha256} if checkpoint_sha256 else {}
        if joint:
            provenance.update(channels=np.asarray(joint.channels), view=joint.view, scaler_sha256=joint.digest)
        predicted = values[order] if joint else values[order, :, 0]
        np.savez(tmp, window_id=ids, forecast_hour=np.arange(1, 7), predicted=predicted, **provenance)
        tmp.flush()
        os.link(tmp.name, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Published step directory")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32, help="Global inference batch size")
    parser.add_argument("--input-view", choices=("first", "both"))
    parser.add_argument("--scaler", type=Path)
    parser.add_argument(
        "--require-history-only", action="store_true", help="Reject fields outside the history input schema"
    )
    args = parser.parse_args()
    if bool(args.input_view) != bool(args.scaler):
        parser.error("Supply both input view and fitted scaler")
    args.joint = JointInputs.load(args.scaler, args.input_view) if args.scaler else None
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if args.batch < world or args.batch % world:
        parser.error("Global batch must be a positive multiple of the inference world size")
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.data, allow_pickle=False) as data:
        ids, history = read_history(data, args.joint, require_history_only=args.require_history_only)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    try:
        manifest = read_manifest(args.checkpoint)
        if manifest is None:
            raise ValueError("Expected a published checkpoint generation")
        identity = manifest["identity"]
        if identity.get("joint") != (args.joint.identity() if args.joint else None):
            raise ValueError("Checkpoint input view, channels or fitted scaler mismatch")
        if identity["model"] != MODEL_REVISION or identity["head"] not in {"direct", "persistence"}:
            raise ValueError("Incompatible forecast checkpoint")
        if identity["precision"] not in {"fp32", "bf16"}:
            raise ValueError("Unsupported forecast precision")
        sums = args.weights.parent / "SHA256SUMS"
        if rank == 0:
            if manifest["files"] != files(args.checkpoint) or sha256(sums) != identity["weights"]:
                raise ValueError("Training checkpoint or base snapshot checksum mismatch")
            listed = {line.split(maxsplit=1)[1].removeprefix("*") for line in sums.read_text().splitlines()}
            actual = {str(p.relative_to(args.weights.parent)) for p in args.weights.rglob("*") if p.is_file()}
            if not actual or actual != listed:
                raise ValueError("Weight directory does not match the snapshot manifest")
            subprocess.run(["sha256sum", "--strict", "--check", sums.name], cwd=sums.parent, check=True)
        if world > 1:
            dist.barrier()
        forecaster = PersistenceForecaster if identity["head"] == "persistence" else QwenForecaster
        dimensions = {"channels": 2, "targets": 2} if args.joint else {}
        model = forecaster.from_pretrained(checkpoint_path=str(args.weights), local_files_only=True, **dimensions)
        if identity["lora"]:
            add_lora(model)
        load_checkpoint_model(model, args.checkpoint, identity["lora"])
        model.to(device).eval()
        args.precision = identity["precision"]
        indices, values = [], []
        for idx, prediction in forecast_batches(model, history, args, rank, world, device):
            indices.append(idx.numpy())
            values.append(prediction.cpu().numpy())
        shard = (np.concatenate(indices), np.concatenate(values))
        shards = [shard]
        if world > 1:
            shards = [None] * world if rank == 0 else None
            dist.gather_object(shard, shards, dst=0)
        if rank == 0:
            digest = sha256(args.checkpoint / "manifest.json")
            write_predictions(args.output, ids, shards, digest, args.joint)
            print(json.dumps({"windows": len(ids), "checkpoint_sha256": digest, "output_sha256": sha256(args.output)}))
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
