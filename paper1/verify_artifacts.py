#!/usr/bin/env python3
"""Verify released evidence with Python's standard library, without model assets."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from legolm_p01.artifacts import verify

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
args = parser.parse_args()
print(json.dumps(verify(args.directory), indent=2))
