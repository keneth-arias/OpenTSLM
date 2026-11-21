# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
"""Restore numerical forecast checkpoints and export predictions in original units."""

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from opentslm.forecast_inputs import verify_candidate_input, verify_weight_snapshot
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict

from opentslm.forecast import FORECASTERS, MODEL_REVISION, QUANTILES
from opentslm.forecast_checkpoint import files, read_manifest, sha256


LORA_RANK = 4
LORA_ALPHA = 8
MAX_THRESHOLD_STD = 100.0


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


def read_history(data, joint=None, *, require_history_only=False, episode=False):
    if require_history_only:
        fields = {"window_id", "history", "channels"} if joint else {"window_id", "history"}
        if episode:
            fields.update(("station_key", "threshold_ug_m3", "thresholds_sha256"))
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
        or (ids.dtype.kind == "f" and not np.isfinite(ids).all())
        or (ids.dtype.kind == "U" and np.any(ids == ""))
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


def task_schema(head, threshold_hash=None):
    if head not in ("quantile", "episode"):
        return None
    schema = {"version": 2, "transform": "history_mean_population_std_flat_one_v1", "unit": "ug/m3", "head": head, "horizon": 6, "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA}
    if head == "episode":
        if not isinstance(threshold_hash, str) or not re.fullmatch("[0-9a-f]{64}", threshold_hash):
            raise ValueError("Expected a frozen threshold manifest SHA-256")
        schema.update(target="any_ge_h1_h6_v1", max_threshold_std=MAX_THRESHOLD_STD, thresholds_sha256=threshold_hash)
    else:
        schema.update(levels=list(QUANTILES), target="hourly_concentration_h1_h6_v1")
    return schema


def episode_thresholds(data, ids, expected_hash):
    if str(data["thresholds_sha256"].item()) != expected_hash:
        raise ValueError("Mismatched threshold manifest")
    values = torch.from_numpy(data["threshold_ug_m3"]).float()
    if values.shape != (len(ids),) or not values.isfinite().all():
        raise ValueError("Expected one finite PM2.5 threshold per window")
    return values


def read_task_data(path, joint, head, threshold_hash):
    if head != "episode":
        return (*read_data(path, joint), None)
    with np.load(path, allow_pickle=False) as data:
        fields = {"window_id", "station_key", "history", "threshold_ug_m3", "thresholds_sha256", "event"}
        if set(data.files) != fields:
            raise ValueError(f"Expected episode training fields {sorted(fields)}")
        ids, history = read_history(data)
        thresholds = episode_thresholds(data, ids, threshold_hash)
        event = data["event"]
        if event.shape != (len(ids),) or event.dtype.kind not in "biuf" or not np.isin(event, (-1, 0, 1)).all():
            raise ValueError("Expected one 0/1 event label or -1 for ineligible windows")
        valid = event != -1
        if not valid.any():
            raise ValueError("No eligible episode labels")
        return ids[valid], history[valid], torch.from_numpy(event[valid]).float(), thresholds[valid]


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
    model.decoder = inject_adapter_in_model(LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=targets), model.decoder)
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


def forecast(model, history, args, thresholds=None):
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
        if getattr(args, "head", None) == "episode":
            context = ((thresholds - center.flatten()) / scale.flatten()).clamp(-MAX_THRESHOLD_STD, MAX_THRESHOLD_STD)
            predicted = model(normalized, observed, context)
        else:
            predicted = model(normalized, observed)
    return predicted.float(), center, scale


@torch.no_grad()
def forecast_batches(model, history, args, rank, world, device, thresholds=None):
    """All ranks run equally sized batches, including the padded final collective."""
    for start in range(0, len(history), args.batch):
        index = torch.arange(start + rank, start + args.batch, world)
        valid = index < len(history)
        rows = index.clamp_max(len(history) - 1)
        x = history[rows].to(device)
        context = thresholds[rows].to(device) if thresholds is not None else None
        value, center, scale = forecast(model, x, args, context)
        if getattr(args, "head", None) != "episode":
            value = value * scale + center
        yield index[valid], value[valid]


def load_checkpoint_model(model, checkpoint, lora):
    options = state_options(lora)
    state = {"model": get_model_state_dict(model, options=options)}
    # Each inference worker loads a replica. DCP reads full and sharded training states.
    dcp.load(state, checkpoint_id=checkpoint / "dcp", no_dist=True)
    set_model_state_dict(model, state["model"], options=options)


def checkpoint_head(identity, schema, head, joint):
    if identity.get("joint") != (joint.identity() if joint else None):
        raise ValueError("Checkpoint input view, channels or fitted scaler mismatch")
    if identity["model"] != MODEL_REVISION or identity["head"] not in FORECASTERS:
        raise ValueError("Incompatible forecast checkpoint")
    if identity.get("task_schema") != schema or (head and identity["head"] != head):
        raise ValueError("Checkpoint task schema or threshold manifest mismatch")
    if identity["head"] in ("quantile", "episode") and schema is None:
        raise ValueError("Versioned checkpoints require the matching task schema")
    if identity["precision"] not in {"fp32", "bf16"}:
        raise ValueError("Unsupported forecast precision")
    if type(identity.get("lora")) is not bool or (schema and not identity["lora"]):
        raise ValueError("Invalid LoRA mode in checkpoint")
    return identity["head"]


def write_predictions(path, ids, shards, checkpoint_sha256=None, joint=None, *, head="direct",
                      input_sha256=None, thresholds_sha256=None, input_manifest_sha256=None):
    indices = np.concatenate([idx for idx, _ in shards])
    values = np.concatenate([pred for _, pred in shards])
    order = indices.argsort()
    values = values[order]
    shape = (len(ids), 6, 2 if joint else 1)
    if head == "episode":
        shape = (len(ids),)
    elif head == "quantile":
        shape = (len(ids), 6, 3)
    if not np.array_equal(indices[order], np.arange(len(ids))) or values.shape != shape or not np.isfinite(values).all():
        raise ValueError("Expected one finite task output per input window")
    payload = {"window_id": ids}
    if checkpoint_sha256:
        payload["checkpoint_sha256"] = checkpoint_sha256
    if head in ("episode", "quantile"):
        if any(not isinstance(v, str) or not re.fullmatch("[a-f0-9]{64}",v) for v in (checkpoint_sha256,input_sha256,input_manifest_sha256)):
            raise ValueError("Missing candidate export provenance")
        if head=="episode" and (not isinstance(thresholds_sha256,str) or not re.fullmatch("[a-f0-9]{64}",thresholds_sha256)):
            raise ValueError("Missing episode threshold digest")
        payload.update(task=head, input_sha256=input_sha256, schema_version=2, model_revision=MODEL_REVISION,
                       input_manifest_sha256=input_manifest_sha256, unit="ug/m3" if head == "quantile" else "logit")
    if head == "episode":
        payload.update(episode_logit=values, thresholds_sha256=thresholds_sha256)
    else:
        payload["forecast_hour"] = np.arange(1, 7)
        if head == "quantile":
            crossed = (np.diff(values, axis=-1) < 0).any(axis=-1)
            if crossed.any():
                row, horizon = np.argwhere(crossed)[0]
                raise ValueError(f"Crossed forecast quantiles at {int(crossed.sum())} window-hours; first {ids[row]} +{horizon + 1}h")
            payload.update(quantiles=values, levels=np.asarray(QUANTILES), predicted=values[:, :, 1])
        else:
            payload["predicted"] = values if joint else values[:, :, 0]
    if joint:
        payload.update(channels=np.asarray(joint.channels), view=joint.view, scaler_sha256=joint.digest)
    with NamedTemporaryFile(dir=path.parent) as tmp:
        np.savez(tmp, **payload)
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
    parser.add_argument("--head", choices=tuple(FORECASTERS), help="Required for versioned candidate outputs")
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--input-manifest-sha256")
    parser.add_argument("--thresholds-sha256", help="Frozen threshold manifest for episode inference")
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
    if args.head == "episode" and not args.thresholds_sha256:
        parser.error("Supply --thresholds-sha256 for episode inference")
    schema = task_schema(args.head, args.thresholds_sha256)
    if (schema and args.joint) or (args.thresholds_sha256 and args.head != "episode"):
        parser.error("Candidate heads use univariate inputs; only episodes use thresholds")
    if schema:
        verify_candidate_input(args.data, args.input_manifest, args.input_manifest_sha256, "infer", args.head, args.thresholds_sha256)
    with np.load(args.data, allow_pickle=False) as data:
        ids, history = read_history(data, args.joint, require_history_only=args.require_history_only or bool(schema),
                                    episode=args.head == "episode")
        thresholds = episode_thresholds(data, ids, args.thresholds_sha256) if args.head == "episode" else None
    manifest = read_manifest(args.checkpoint)
    if manifest is None:
        raise ValueError("Expected a published checkpoint generation")
    identity = manifest["identity"]
    args.head = checkpoint_head(identity, schema, args.head, args.joint)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    try:
        if rank == 0:
            if manifest["files"] != files(args.checkpoint):
                raise ValueError("Training checkpoint checksum mismatch")
            verify_weight_snapshot(args.weights, identity["weights"])
        if world > 1:
            dist.barrier()
        forecaster = FORECASTERS[args.head]
        dimensions = {"channels": 2, "targets": 2} if args.joint else {}
        model = forecaster.from_pretrained(checkpoint_path=str(args.weights), local_files_only=True, **dimensions)
        if identity["lora"]:
            add_lora(model)
        load_checkpoint_model(model, args.checkpoint, identity["lora"])
        model.to(device).eval()
        args.precision = identity["precision"]
        indices, values = [], []
        for idx, prediction in forecast_batches(model, history, args, rank, world, device, thresholds):
            indices.append(idx.numpy())
            values.append(prediction.cpu().numpy())
        shard = (np.concatenate(indices), np.concatenate(values))
        shards = [shard]
        if world > 1:
            shards = [None] * world if rank == 0 else None
            dist.gather_object(shard, shards, dst=0)
        if rank == 0:
            digest = sha256(args.checkpoint / "manifest.json")
            write_predictions(args.output, ids, shards, digest, args.joint, head=args.head,
                              input_sha256=sha256(args.data) if schema else None,
                              thresholds_sha256=args.thresholds_sha256, input_manifest_sha256=args.input_manifest_sha256)
            print(json.dumps({"windows": len(ids), "checkpoint_sha256": digest, "output_sha256": sha256(args.output)}))
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
