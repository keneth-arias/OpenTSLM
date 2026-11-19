# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
"""Publish and verify complete generations of a shared training checkpoint."""

import hashlib
import json
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


def publish(root, staging, step, identity):
    manifest = {"step": step, "identity": identity, "files": files(staging)}
    (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
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
    pointer.replace(root / name)


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
