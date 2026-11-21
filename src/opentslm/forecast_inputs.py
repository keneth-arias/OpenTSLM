# SPDX-FileCopyrightText: 2026 Keneth Arias
# SPDX-License-Identifier: MIT
"""CPU-only verification shared by the trainer and episode preflight."""

import hashlib
import json
import re
import subprocess

import numpy as np

from .forecast_checkpoint import sha256


def verify_candidate_input(data_path, manifest_path, expected_digest, phase, head, threshold_hash):
    """Verify the phase artifact and independently join frozen station thresholds."""
    if not manifest_path or not expected_digest or sha256(manifest_path) != expected_digest:
        raise ValueError("Expected the pinned candidate input manifest")
    spec = json.loads(manifest_path.read_text())
    if (spec.get("schema"), spec.get("unit"), spec.get("phase"), spec.get("head")) != (
        "forecast-input-v1",
        "ug/m3",
        phase,
        head,
    ):
        raise ValueError("Candidate input phase, head, units or schema mismatch")
    if spec.get("files", {}).get(data_path.name) != sha256(data_path):
        raise ValueError("Input archive checksum mismatch")
    if head == "episode":
        source = manifest_path.parent / "thresholds.json"
        if (
            source.is_symlink()
            or spec.get("thresholds_sha256") != threshold_hash
            or spec["files"].get(source.name) != threshold_hash
            or sha256(source) != threshold_hash
        ):
            raise ValueError("Frozen station threshold manifest mismatch")
        table = json.loads(source.read_text())
        if table.get("unit") != "ug/m3" or not isinstance(table.get("stations"), dict):
            raise ValueError("Invalid station threshold units or table")
        with np.load(data_path, allow_pickle=False) as rows:
            stations = rows["station_key"]
            if stations.dtype.kind != "U" or stations.shape != rows["window_id"].shape:
                raise ValueError("Invalid station keys for threshold lookup")
            expected = np.array([table["stations"].get(station, np.nan) for station in stations], dtype=float)
            if (
                not np.isfinite(expected).all()
                or not np.array_equal(expected, rows["threshold_ug_m3"])
                or rows["thresholds_sha256"].item() != threshold_hash
            ):
                raise ValueError("Station threshold differs from the frozen table")


def direct_cohorts(fit_manifest, fit_digest, validation_manifest, validation_digest, weights_digest):
    """Load exactly the pinned training roles and their fit-derived thresholds."""
    cohorts = []
    manifests = []
    for role, path, digest in (
        ("fit", fit_manifest, fit_digest),
        ("inner_validation", validation_manifest, validation_digest),
    ):
        if path.is_symlink() or sha256(path) != digest:
            raise ValueError("Direct cohort manifest checksum mismatch")
        spec = json.loads(path.read_bytes())
        if (
            spec.get("schema"),
            spec.get("unit"),
            spec.get("head"),
            spec.get("phase"),
            spec.get("weights_manifest_sha256"),
        ) != ("forecast-direct-cohort-v1", "ug/m3", "direct", role, weights_digest):
            raise ValueError("Direct cohort role, base or schema mismatch")
        for key in ("protocol_sha256", "source_manifest_sha256", "ids_sha256", "weights_manifest_sha256"):
            if not re.fullmatch(r"[a-f0-9]{64}", str(spec.get(key, ""))):
                raise ValueError("Invalid cohort fingerprint")
        required = {"data.npz", "thresholds.json"}
        if set(spec["files"]) != required or {f.name for f in path.parent.iterdir()} != required | {
            "manifest.json"
        }:
            raise ValueError("Unexpected direct cohort files")
        for name in required:
            file = path.parent / name
            if file.is_symlink() or sha256(file) != spec["files"][name]:
                raise ValueError("Corrupt direct cohort file")
        if manifests and (
            any(spec[k] != manifests[0][k] for k in ("protocol_sha256", "source_manifest_sha256"))
            or spec["files"]["thresholds.json"] != manifests[0]["files"]["thresholds.json"]
        ):
            raise ValueError("Direct cohorts have different protocols, sources or thresholds")
        manifests.append(spec)
        thresholds = json.loads((path.parent / "thresholds.json").read_bytes())
        if (
            thresholds.get("fit_ids_sha256") != manifests[0]["ids_sha256"]
            or thresholds.get("unit") != "ug/m3"
            or thresholds.get("method") != "linear_p95_unique_fit_history_station_hours"
        ):
            raise ValueError("Expected thresholds frozen on fit histories")
        expected = {"window_id", "station_key", "origin_hour", "history", "targets"}
        if role == "fit":
            expected.add("graph_sample")
        with np.load(path.parent / "data.npz", allow_pickle=False) as archive:
            if set(archive.files) != expected:
                raise ValueError("Unexpected fields for the direct cohort role")
            rows = {name: archive[name] for name in expected}
        ids = rows["window_id"]
        n = len(ids)
        if (
            ids.ndim != 1
            or ids.dtype.kind != "U"
            or n == 0
            or n != spec["windows"]
            or np.any(ids[1:] <= ids[:-1])
        ):
            raise ValueError("Expected nonempty sorted unique direct cohort IDs")
        fingerprint = hashlib.sha256()
        for value in ids:
            if not value or "\n" in value or "\r" in value:
                raise ValueError("Invalid cohort ID")
            fingerprint.update(value.encode("utf-8") + b"\n")
        if fingerprint.hexdigest() != spec["ids_sha256"]:
            raise ValueError("Cohort membership differs from its manifest")
        for name, width in (("history", 24), ("targets", 6)):
            values = rows[name]
            if values.shape != (n, width) or values.dtype.kind not in "ifu" or not np.isfinite(values).all():
                raise ValueError("Expected finite 24-hour histories and six-hour targets")
        stations, hours = rows["station_key"], rows["origin_hour"]
        if (
            stations.shape != (n,)
            or stations.dtype.kind != "U"
            or hours.shape != (n,)
            or hours.dtype.kind != "i"
        ):
            raise ValueError("Invalid direct cohort station or UTC hour")
        rows["threshold"] = np.asarray(
            [thresholds["stations"].get(s, np.nan) for s in stations], dtype=np.float64
        )
        if not np.isfinite(rows["threshold"]).all():
            raise ValueError("Missing fit-derived station threshold")
        cohorts.append(rows)
    fit, validation = cohorts
    mask = fit["graph_sample"]
    if (
        mask.shape != fit["window_id"].shape
        or mask.dtype != np.bool_
        or mask.sum() != len(validation["window_id"])
    ):
        raise ValueError("Invalid frozen fit graph sample")
    if (
        int(fit["origin_hour"].max()) + 6 >= int(validation["origin_hour"].min()) - 23
        or np.intersect1d(fit["window_id"], validation["window_id"], assume_unique=True).size
    ):
        raise ValueError("Fit and inner validation overlap")
    return fit, validation


def verify_weight_snapshot(weights, expected, *, strict=False):
    sums = weights.parent / "SHA256SUMS"
    if (strict and (sums.is_symlink() or weights.is_symlink())) or sha256(sums) != expected:
        raise ValueError("Base snapshot checksum mismatch")
    listed = [line.split(maxsplit=1)[1].removeprefix("*") for line in sums.read_text().splitlines()]
    paths = list(weights.rglob("*"))
    actual = {str(p.relative_to(weights.parent)) for p in paths if p.is_file()}
    if (
        not actual
        or actual != set(listed)
        or (strict and (len(listed) != len(actual) or any(p.is_symlink() for p in paths)))
    ):
        raise ValueError("Weight directory does not match the snapshot manifest")
    subprocess.run(["sha256sum", "--strict", "--check", "--status", sums.name], cwd=sums.parent, check=True)
