"""Verify regenerated token arrays despite platform-specific torch containers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .bootstrap import bundled_assets
from .provenance import file_hash


def reference_path():
    return bundled_assets() / "text-cache-identities.json"


def content_identity(cache):
    if set(cache) != {"identity", "train", "val"}:
        raise ValueError("Unexpected text-cache fields")
    tensors = {}
    for split in ("train", "val"):
        value = cache[split]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.int64 or value.ndim != 2:
            raise ValueError("Text caches require two-dimensional int64 token arrays")
        raw = value.detach().cpu().contiguous().numpy().astype("<i8", copy=False).tobytes()
        tensors[split] = {"dtype": str(value.dtype), "shape": list(value.shape),
                          "sha256": hashlib.sha256(raw).hexdigest()}
    return {"identity": cache["identity"], "tensor_hashes": tensors}


def load_verified_text(directory, host, expected_sha256):
    directory = Path(directory)
    manifest = json.loads((directory / f"text-{host}.json").read_text())
    if Path(manifest["file"]).name != manifest["file"]:
        raise ValueError("Text-cache manifest requires a local filename")
    path = directory / manifest["file"]
    observed = file_hash(path)
    if observed != manifest["sha256"]:
        raise ValueError("Text-cache file failed its local integrity check")
    cache = torch.load(path, map_location="cpu", weights_only=True)
    identity = {k: v for k, v in manifest.items() if k not in ("file", "sha256")}
    if cache["identity"] != identity or identity["host"] != host:
        raise ValueError("Text-cache manifest identity does not match its contents")
    mode = "identical_file"
    if observed != expected_sha256:
        reference = json.loads(reference_path().read_text())["hosts"].get(host)
        if reference is None or reference["file_sha256"] != expected_sha256:
            raise ValueError("No registered token-array identity for this checkpoint's text cache")
        if content_identity(cache) != {k: reference[k] for k in ("identity", "tensor_hashes")}:
            raise ValueError("Text token arrays or input identity changed since training")
        mode = "identical_token_arrays"
    return cache, {"mode": mode, "training_file_sha256": expected_sha256,
                   "evaluation_file_sha256": observed}
