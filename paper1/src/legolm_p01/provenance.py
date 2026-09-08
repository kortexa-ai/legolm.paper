"""Content identities and atomic artifacts; paths do not define a run."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def source_identity() -> dict:
    import importlib.metadata as metadata
    root = Path(__file__).resolve().parent
    files = {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*.py"))}
    package = root.parents[1]
    lock = package / "uv.lock"
    result = subprocess.run(["git", "-C", str(package), "rev-parse", "HEAD"], capture_output=True, text=True)
    return {"files": files, "source_hash": digest(files), "git_commit": result.stdout.strip(),
            "lock_sha256": file_hash(lock) if lock.exists() else None,
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {n: metadata.version(n) for n in ("torch", "torchvision", "transformers", "numpy")}}
