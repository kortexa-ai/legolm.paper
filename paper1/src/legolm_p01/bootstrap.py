"""Materialize exact Paper 1 inputs without importing another paper's runner."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
import urllib.request

from .provenance import file_hash, write_json


def bundled_assets():
    here = Path(__file__).resolve()
    candidates = (here.parents[2] / "assets", here.parent / "assets")
    return next(p for p in candidates if (p / "inputs.json").is_file())


def validate_file(path, expected):
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
        raise ValueError(f"Input differs from this release: {path}. Materialize the pinned Git LFS asset or use a new cache directory.")


def download_verified(url, target, expected):
    if target.exists():
        validate_file(target, expected)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".paper1-download")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024*1024)
            validate_file(temporary, expected)
            temporary.replace(target)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def run(asset_root, data_cache):
    assets = bundled_assets()
    manifest = json.loads((assets / "inputs.json").read_text())
    for relative, expected in manifest["repository_assets"].items():
        validate_file(asset_root / relative, expected)
    for name, expected in manifest["tokenizer"].items():
        source = assets / name
        validate_file(source, expected)
        target = data_cache / "tokenizer" / name
        if target.exists():
            validate_file(target, expected)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    for name, expected in manifest["climbmix"]["shards"].items():
        download_verified(expected["url"], data_cache / "data" / name, expected)
        print(f"Verified text input {name}", flush=True)
    # These isolated, attributed snapshots retain the original raw-data loaders.
    # Preparation subsequently fingerprints every extracted source file.
    from .legacy import audio_snapshot as audio, imu_snapshot as imu
    audio.CACHE_DIR = data_cache / "audio-encoder"
    audio.load_esc50()
    imu.CACHE_DIR = data_cache / "sensor-fusion"
    imu.download_uci_har()
    write_json(data_cache / "paper1-inputs.json", manifest)
    return {"verified_repository_assets": len(manifest["repository_assets"]), "verified_text_shards": len(manifest["climbmix"]["shards"]),
            "tokenizer": "exact bundled checkpoint tokenizer", "data_cache": str(data_cache)}
