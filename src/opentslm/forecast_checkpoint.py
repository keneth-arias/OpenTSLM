# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
"""Publish and verify complete generations of a shared training checkpoint."""

import hashlib
import json
import os
import uuid
from pathlib import Path


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def files(directory):
    return {
        str(path.relative_to(directory)): {"size": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != directory / "manifest.json"
    }


def sync_path(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish(root, staging, step, identity):
    manifest = {"step": step, "identity": identity, "files": files(staging)}
    (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    for directory, _, names in os.walk(staging, topdown=False):
        for name in names:
            sync_path(Path(directory) / name)
        sync_path(directory)
    final = root / f"step-{step:08d}"
    if final.exists():
        existing = read_manifest(final)
        if existing and (existing["identity"] != identity or existing["files"] == files(final)):
            raise FileExistsError(final)
        final.rename(root / f".damaged-{final.name}-{uuid.uuid4().hex}")
    staging.rename(final)
    point_to(root, "latest", final)
    return final, manifest


def point_to(root, name, final):
    pointer = root / f".{name}.tmp"
    pointer.write_text(final.name + "\n")
    sync_path(pointer)
    pointer.replace(root / name)
    sync_path(root)


def read_manifest(path):
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        if path.name == f"step-{manifest['step']:08d}" and manifest["files"] and "identity" in manifest:
            return manifest
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def recover(root: Path, identity):
    for path in sorted(root.glob("step-????????"), reverse=True):
        manifest = read_manifest(path)
        if manifest and manifest["identity"] == identity and manifest["files"] == files(path):
            return manifest["step"], path
    return None


def selected_generation(root, identity, best_step):
    """Reconcile the best pointer from the latest durable evaluation history."""
    selected = root / f"step-{best_step:08d}"
    info = read_manifest(selected)
    if not info or info["identity"] != identity or files(selected) != info["files"]:
        raise ValueError("Selected direct checkpoint is missing, damaged or belongs to another run")
    point_to(root, "best", selected)
    return selected


def finish_direct(root, identity, latest_step, history):
    """Publish one idempotent receipt after checking both restorable generations."""
    last = history[-1]
    if last["step"] != latest_step or last["outcome"] not in ("early-stopped", "completed"):
        raise ValueError("Training has no terminal evaluation")
    selected = selected_generation(root, identity, last["best_step"])
    latest = root / f"step-{latest_step:08d}"
    if latest != selected:
        info = read_manifest(latest)
        if not info or info["identity"] != identity or files(latest) != info["files"]:
            raise ValueError("Terminal checkpoint is missing or damaged")
    receipt = {**last, "identity": identity,
               "selected_manifest_sha256": sha256(selected / "manifest.json"),
               "latest_manifest_sha256": sha256(latest / "manifest.json")}
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(dir=root, mode="w") as tmp:
        json.dump(receipt, tmp, sort_keys=True, allow_nan=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        path = root / "direct-result.json"
        try:
            os.link(tmp.name, path)
            sync_path(root)
        except FileExistsError:
            if json.loads(path.read_bytes()) != receipt:
                raise ValueError("Existing terminal result belongs to a different fit")
    return receipt
