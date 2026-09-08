"""Run one archived experiment with pinned model revisions."""

import argparse
import importlib
import json
import sys

from .analysis import PACKAGE, sha256

SCRIPTS = ("vm_rung1", "vm_rung1b", "vm_rung2", "vm_detectors", "vm_rung2y",
           "vm_rung2b", "vm_rung2c", "vm_rung2t", "vm_rung3", "vm_door2x")


def verify_harness() -> None:
    manifest = json.loads((PACKAGE / "provenance/harness-manifest.json").read_text())
    for item in manifest["files"]:
        path = PACKAGE / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"archived harness changed: {item['path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", choices=SCRIPTS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    verify_harness()
    sys.path.insert(0, str(PACKAGE / "harness/tracks/vision-mode"))
    # Import once: every experiment shares this exact module instance. The
    # archived source stays unchanged; only recovered input identities are set.
    r1 = importlib.import_module("vm_rung1")
    models = json.loads((PACKAGE / "provenance/models.json").read_text())
    for host, record in models.items():
        if r1.HOSTS[host]["model_id"] != record["model_id"]:
            raise ValueError("host identifier differs from the input manifest")
        r1.HOSTS[host]["revision"] = record["revision"]
    module = importlib.import_module(args.script)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    sys.argv = [args.script + ".py", *arguments]
    module.main()


if __name__ == "__main__":
    main()
