"""List or run the recorded experiment groups in dependency order."""

import argparse
import json
import shlex
import subprocess
import sys

from .analysis import PACKAGE


def commands(run: dict, device: str = "cuda") -> list[list[str]]:
    prefix = [sys.executable, "-m", "legolm_lookout.experiment", run["script"],
              *run["arguments"], "--device", device]
    seeds = run["seeds"]
    if seeds == [42, 1042, 2042] and run["script"] != "vm_door2x":
        return [prefix + ["--all-seeds"]]
    return [prefix + ["--seed", str(seed)] for seed in seeds] if seeds else [prefix]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", metavar="ID", help="execute this group, or all; default only lists commands")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    runs = json.loads((PACKAGE / "experiments.json").read_text())["runs"]
    if args.run and args.run != "all":
        runs = [r for r in runs if r["id"] == args.run]
        if not runs:
            parser.error("unknown experiment ID")
    for run in runs:
        print(f"# {run['id']}", flush=True)
        for command in commands(run, args.device):
            print(shlex.join(command), flush=True)
            if args.run:
                subprocess.run(command, cwd=PACKAGE, check=True)


if __name__ == "__main__":
    main()
