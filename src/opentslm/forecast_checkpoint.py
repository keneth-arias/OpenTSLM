# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
"""Publish and verify complete generations of a shared training checkpoint."""

import hashlib
import json
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
    staging.rename(final)
    point_to(root, "latest", final)
    return final, manifest


def point_to(root, name, final):
    pointer = root / f".{name}.tmp"
    pointer.write_text(final.name + "\n")
    pointer.replace(root / name)


def recover(root: Path, identity):
    for path in sorted(root.glob("step-????????"), reverse=True):
        try:
            manifest = json.loads((path / "manifest.json").read_text())
            if (
                manifest["identity"] == identity
                and path.name == f"step-{manifest['step']:08d}"
                and manifest["files"]
                and manifest["files"] == files(path)
            ):
                return manifest["step"], path
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return None
