"""Fetch or verify the exact inputs used by the public experiment commands."""

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from .analysis import PACKAGE


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing input: {path}")
    if digest(path) != expected:
        raise ValueError(f"input hash mismatch: {path}")


def download(url: str, path: Path, expected: str | None = None) -> None:
    if path.is_file():
        if expected:
            require_hash(path, expected)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, 1024 * 1024)
    if expected:
        require_hash(partial, expected)
    partial.replace(path)


def coco(directory: Path, fetch: bool) -> dict:
    manifest = json.loads((PACKAGE / "provenance/coco-inputs.json").read_text())
    if fetch:
        archives = PACKAGE / "downloads"
        specifications = [
            ("https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip",
             "annotations_trainval2017.zip", lambda n: n == "annotations/instances_val2017.json"),
            ("https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip", "val2017.zip",
             lambda n: n.startswith("val2017/") and n.endswith(".jpg")),
        ]
        for url, name, selected in specifications:
            archive = archives / name
            download(url, archive)
            with zipfile.ZipFile(archive) as source:
                for member in source.infolist():
                    if not selected(member.filename):
                        continue
                    relative = Path(member.filename)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("unsafe input archive path")
                    target = directory / relative
                    if target.exists():
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    partial = target.with_name(target.name + ".part")
                    with source.open(member) as inp, partial.open("wb") as out:
                        shutil.copyfileobj(inp, out)
                    partial.replace(target)
    require_hash(directory / "annotations/instances_val2017.json", manifest["annotation_sha256"])
    ids = set()
    for name, images in manifest["image_sets"].items():
        current = {r["image_id"] for r in images}
        if len(current) != len(images) or ids & current:
            raise ValueError(f"duplicate COCO image across splits: {name}")
        ids.update(current)
        for image in images:
            require_hash(directory / "val2017" / image["file"], image["sha256"])
    return {"verified_images": len(ids), "splits": {k: len(v) for k, v in manifest["image_sets"].items()}}


def models(host: str, fetch: bool) -> dict:
    from huggingface_hub import snapshot_download
    records = json.loads((PACKAGE / "provenance/models.json").read_text())
    selected = records if host == "all" else {host: records[host]}
    results = {}
    for key, record in selected.items():
        directory = Path(snapshot_download(record["model_id"], revision=record["revision"],
                                           local_files_only=not fetch))
        require_hash(directory / "config.json", record["config_sha256"])
        results[key] = {"revision": record["revision"], "cache": str(directory)}
    return results


def checkpoints(fetch: bool) -> dict:
    records = json.loads((PACKAGE / "provenance/checkpoints.json").read_text())["owned"]
    for record in records:
        path = PACKAGE / "harness/checkpoints/experiments" / Path(record["path"]).name
        if fetch:
            download(record["url"], path, record["sha256"])
        require_hash(path, record["sha256"])
    return {"verified_checkpoints": len(records), "bytes": sum(r["bytes"] for r in records)}


def detectors(fetch: bool) -> dict:
    records = json.loads((PACKAGE / "provenance/checkpoints.json").read_text())["detectors"]
    for record in records:
        path = PACKAGE / "harness/checkpoints/experiments" / Path(record["path"]).name
        if fetch and not path.exists():
            from ultralytics.utils.downloads import attempt_download_asset
            path.parent.mkdir(parents=True, exist_ok=True)
            attempt_download_asset(str(path))
        require_hash(path, record["sha256"])
    return {"verified_detectors": len(records)}


def renders(directory: Path, kind: str) -> dict:
    record = json.loads((PACKAGE / "provenance/rendered-inputs.json").read_text())[kind]
    if kind == "mixed_training":
        manifest = json.loads((directory / "mesh/manifest.json").read_text())
        if manifest["sha256"] != record["expected_manifest_sha256"]:
            raise ValueError("mixed-training rendered pixels differ from the study")
        candidates = {record["expected_manifest_sha256"]: manifest["records"]}
    else:
        variants = record.get("manifest_variants") or {record["manifest"]["sha256"]: record}
        candidates = {key: value["manifest"]["records"] for key, value in variants.items()}
    actual = {}
    failures = {}
    for key, rows in candidates.items():
        recomputed_manifest = hashlib.sha256()
        mismatches = []
        for row in rows:
            recomputed_manifest.update(row["sha256"].encode())
            recomputed_manifest.update(row["label"].encode())
            subdir = "mesh" if row["level"] == "mesh" else f"l{row['level']}"
            path = directory / subdir / Path(row["file"]).name
            if path not in actual:
                actual[path] = digest(path) if path.is_file() else None
            if actual[path] != row["sha256"]:
                mismatches.append(str(path))
        if recomputed_manifest.hexdigest() != key:
            raise ValueError("recorded render manifest is internally inconsistent")
        if not mismatches:
            return {"kind": kind, "verified_images": len(rows), "matched_pixel_manifest": key}
        failures[key] = len(mismatches)
    raise ValueError(f"rendered images differ or are missing; counts by recorded pixel version: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", choices=("coco", "models", "checkpoints", "detectors", "renders"))
    parser.add_argument("--download", action="store_true", help="fetch missing inputs (otherwise verify only)")
    parser.add_argument("--directory", type=Path, default=Path("~/data/coco").expanduser())
    parser.add_argument("--host", choices=("450m", "3b", "27b", "all"), default="450m")
    parser.add_argument("--kind", choices=("capsule", "mesh", "mixed_training"), default="mesh")
    args = parser.parse_args()
    if args.input == "coco":
        result = coco(args.directory.expanduser(), args.download)
    elif args.input == "models":
        result = models(args.host, args.download)
    elif args.input == "checkpoints":
        result = checkpoints(args.download)
    elif args.input == "detectors":
        result = detectors(args.download)
    else:
        if args.download:
            parser.error("rendered inputs are regenerated with the archived harness; see README")
        result = renders(args.directory.expanduser(), args.kind)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
