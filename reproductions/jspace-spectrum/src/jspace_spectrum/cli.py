"""Command-line entry point for the standalone reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .data import validate_data
from .experiment import regenerate_artifacts, run_reproduction
from .metrics import compare_models
from .model import PROFILES, atomic_json_dump


HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parents[2]
REPOSITORY = HERE.parents[4]


def command_check(_args: argparse.Namespace) -> None:
    report = validate_data()
    report["profiles"] = {
        name: {
            "model_id": profile.model_id,
            "revision": profile.revision,
            "source_layer": profile.source_layer,
            "target_layer": profile.target_layer,
            "trace_layers": list(profile.trace_layers),
            "paper_target": profile.paper_target,
        }
        for name, profile in PROFILES.items()
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def command_reproduce(args: argparse.Namespace) -> None:
    profile = args.model
    if profile is None:
        profile = "qwen35-2b-smoke" if args.suite == "smoke" else "qwen36-35b"
    summary = run_reproduction(
        suite=args.suite,
        profile_name=profile,
        output_dir=Path(args.output_dir),
        device_name=args.device,
        repository=REPOSITORY,
        command=sys.argv,
    )
    print(
        json.dumps(
            {
                "event": "reproduction_complete",
                "output_directory": args.output_dir,
                "model": profile,
                "decisions": summary["metrics"]["decisions"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def command_artifacts(args: argparse.Namespace) -> None:
    result = regenerate_artifacts(
        summary_path=Path(args.summary),
        measurements_path=(Path(args.measurements) if args.measurements else None),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def command_compare(args: argparse.Namespace) -> None:
    reference = json.loads(Path(args.reference).read_text())
    extension = json.loads(Path(args.extension).read_text())
    if reference["metrics"]["axis_names"] != extension["metrics"]["axis_names"]:
        raise ValueError("model summaries have different axis orders")
    payload = {
        "format": "jspace-spectrum-model-comparison-v1",
        "reference": {
            "model": reference["model"],
            "summary": args.reference,
        },
        "extension": {
            "model": extension["model"],
            "summary": args.extension,
        },
        "comparison": compare_models(
            reference["metrics"],
            extension["metrics"],
        ),
    }
    output = Path(args.output)
    atomic_json_dump(payload, output)
    print(json.dumps({"output": str(output)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jspace-spectrum-paper",
        description="Fit and replay a twelve-axis residual-stream spectrum.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="validate the frozen inventory")
    check.set_defaults(function=command_check)

    reproduce = commands.add_parser(
        "reproduce",
        help="run or resume a smoke or paper experiment",
    )
    reproduce.add_argument("--suite", choices=("smoke", "full"), required=True)
    reproduce.add_argument("--model", choices=tuple(PROFILES))
    reproduce.add_argument("--output-dir", required=True)
    reproduce.add_argument("--device", default="auto")
    reproduce.set_defaults(function=command_reproduce)

    artifacts = commands.add_parser(
        "artifacts",
        help="regenerate figures and the HTML replay from JSON",
    )
    artifacts.add_argument("summary")
    artifacts.add_argument("--measurements")
    artifacts.add_argument("--output-dir")
    artifacts.set_defaults(function=command_artifacts)

    compare = commands.add_parser(
        "compare",
        help="compare atlas centroids from two completed model summaries",
    )
    compare.add_argument("reference")
    compare.add_argument("extension")
    compare.add_argument("--output", required=True)
    compare.set_defaults(function=command_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
