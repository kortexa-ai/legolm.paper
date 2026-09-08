"""Publish and fetch exact adapter checkpoints independently of full retraining."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
from urllib.parse import urlparse

from .artifacts import checked_path, load, verify
from .bootstrap import download_verified, validate_file
from .provenance import digest, file_hash, write_json
from .outcomes import terminal_record


def evidence_checkpoints(evidence):
    evidence = Path(evidence)
    result = {}
    for trajectory in load(evidence / "index.json")["trajectories"]:
        record = load(checked_path(evidence, trajectory["path"]))
        recipe = record["recipe"]
        modalities = ["imu", "audio", "vision"] if recipe["cohort"] == "U" and recipe["kind"] in ("none", "static") else [recipe["modality"]]
        for point in record["checkpoints"]:
            entry = {"sha256": point["sha256"], "bytes": point["checkpoint_bytes"],
                     "step": point["step"], "training_id": record["training_id"],
                     "recipe": recipe, "evaluation_modalities": modalities}
            if point["sha256"] in result and result[point["sha256"]] != entry:
                raise ValueError("One checkpoint hash has conflicting scientific identities")
            result[point["sha256"]] = entry
    return result


def validate_index(index, evidence=None):
    if index["schema"] != "paper1-checkpoint-index-v2":
        raise ValueError("Unknown checkpoint index schema")
    if index["index_id"] != digest({k: v for k, v in index.items() if k != "index_id"}):
        raise ValueError("Checkpoint index integrity mismatch")
    actual = {}
    for entry in [*index["checkpoints"], *index["feature_caches"]]:
        sha = entry["sha256"]
        url = urlparse(entry["url"])
        if (not re.fullmatch("[0-9a-f]{64}", sha) or sha in actual or entry["bytes"] <= 0
            or url.scheme != "https" or not url.netloc or url.username or url.password
            or url.query or url.fragment or not url.path.endswith("/" + sha + ".pt")):
            raise ValueError("Invalid or repeated checkpoint download identity")
        actual[sha] = {k: v for k, v in entry.items() if k != "url"}
    caches = {r["modality"]: r for r in index["feature_caches"]}
    if len(index["feature_caches"]) != 3 or set(caches) != {"imu", "audio", "vision"}:
        raise ValueError("The index requires all three frozen feature caches")
    checkpoints = {r["sha256"]: {k: v for k, v in r.items() if k != "url"} for r in index["checkpoints"]}
    if evidence is not None:
        evidence = Path(evidence)
        manifest = load(evidence / "manifest.json")
        if (manifest["status"] != "complete" or manifest["matrix_id"] != index["matrix_id"]
            or file_hash(evidence / "manifest.json") != index["evidence_manifest_sha256"]
            or checkpoints != evidence_checkpoints(evidence)):
            raise ValueError("Checkpoint index does not cover the exact complete evidence")
        for modality, entry in caches.items():
            metadata = checked_path(evidence, "data/" + modality + ".json")
            validate_file(metadata, manifest["files"]["data/" + modality + ".json"])
            if load(metadata)["cache_sha256"] != entry["sha256"]:
                raise ValueError("Frozen feature cache differs from the released encoder inputs")
    return {"checkpoints": len(checkpoints), "feature_caches": 3, "bytes": sum(r["bytes"] for r in actual.values())}


def materialize(source, objects, expected):
    """Hard-link completed immutable tensors; never overwrite or duplicate a TB."""
    source, objects = Path(source), Path(objects)
    if objects.is_symlink():
        raise ValueError("Checkpoint object directory must not be a symbolic link")
    objects.mkdir(parents=True, exist_ok=True)
    target = objects / (expected["sha256"] + ".pt")
    if target.is_symlink():
        raise ValueError("Checkpoint object must be a regular immutable file")
    validate_file(source, expected)
    if target.exists():
        validate_file(target, expected)
        return target
    if source.stat().st_dev != objects.stat().st_dev:
        raise ValueError("Checkpoint archive must share the checkpoint filesystem for hard links")
    os.link(source, target)
    return target


def publish(evidence, matrix, features, objects, base_url, output):
    evidence, matrix = Path(evidence), Path(matrix)
    verify(evidence)
    manifest = load(evidence / "manifest.json")
    completed = load(matrix / "matrix-complete.json")
    if not completed["passed"] or completed["matrix_id"] != manifest["matrix_id"]:
        raise ValueError("Checkpoint archive matrix differs from the verified release")
    entries = evidence_checkpoints(evidence)
    locations = {}
    for path in sorted(matrix.glob("job-*.json")):
        job = load(path)
        outcome = terminal_record(job)
        directory = Path(outcome["directory"])
        for point in outcome["checkpoints"]:
            locations[point["sha256"]] = checked_path(directory, point["path"])
    if set(locations) != set(entries):
        raise ValueError("Checkpoint files do not match every released milestone")
    index = {"schema": "paper1-checkpoint-index-v2", "matrix_id": manifest["matrix_id"],
             "evidence_manifest_sha256": file_hash(evidence / "manifest.json"),
             "checkpoints": [{**entries[sha], "url": base_url.rstrip("/") + "/" + sha + ".pt"} for sha in sorted(entries)],
             "feature_caches": []}
    for modality in ("imu", "audio", "vision"):
        sha = load(evidence / "data" / (modality + ".json"))["cache_sha256"]
        source = Path(features) / (modality + ".pt")
        locations[sha] = source
        index["feature_caches"].append({"modality": modality, "sha256": sha, "bytes": source.stat().st_size,
                                        "url": base_url.rstrip("/") + "/" + sha + ".pt"})
    index["index_id"] = digest(index)
    coverage = validate_index(index, evidence)
    output = Path(output)
    if output.exists() and load(output) != index:
        raise ValueError("Refusing to replace a different published checkpoint index")
    for entry in [*index["checkpoints"], *index["feature_caches"]]:
        materialize(locations[entry["sha256"]], objects, entry)
    write_json(output, index)
    return {"passed": True, "index_id": index["index_id"], **coverage}


def fetch_features(index_path, evidence, output):
    """Use published frozen features when encoder arithmetic differs by device."""
    index, evidence, output = load(index_path), Path(evidence), Path(output)
    validate_index(index, evidence)
    manifest = load(evidence / "manifest.json")
    for entry in index["feature_caches"]:
        modality = entry["modality"]
        download_verified(entry["url"], output / (modality + ".pt"), entry)
        source = evidence / "data" / (modality + ".json")
        target = output / (modality + ".json")
        if target.exists():
            validate_file(target, manifest["files"]["data/" + modality + ".json"])
        else:
            shutil.copyfile(source, target)
    return {"feature_caches": 3, "directory": str(output), "index_id": index["index_id"]}


def fetch(index_path, output, *, host, cohort, modality, kind, feature_mode, seed, step, diversity):
    index = load(index_path)
    validate_index(index)
    fields = {"host": host, "cohort": cohort, "kind": kind, "feature_mode": feature_mode, "seed": seed, "diversity": diversity}
    matches = [r for r in index["checkpoints"] if r["step"] == step and modality in r["evaluation_modalities"]
               and all(r["recipe"][k] == v for k, v in fields.items())]
    if len(matches) != 1:
        raise ValueError("Select exactly one registered checkpoint; the requested recipe is absent or ambiguous")
    entry = matches[0]
    target = Path(output) / (entry["sha256"] + ".pt")
    download_verified(entry["url"], target, entry)
    return {"checkpoint": str(target), "sha256": entry["sha256"], "bytes": entry["bytes"],
            "training_id": entry["training_id"], "step": step}
