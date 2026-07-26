"""Command-line entry point for the causal-expression reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import validate_style_data
from .experiment import reproduce
from .figures import render_figures


DEFAULT_SEEDS = (20260724, 20260725, 20260726)


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from error
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be non-empty and unique")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check",
        help="validate the frozen synthetic corpus",
    )
    check.set_defaults(handler=command_check)

    run = subparsers.add_parser(
        "reproduce",
        help="train and audit fresh prefix writers",
    )
    run.add_argument("--suite", choices=("smoke", "full"), required=True)
    run.add_argument(
        "--seeds",
        type=parse_seeds,
        default=DEFAULT_SEEDS,
    )
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    run.set_defaults(handler=command_reproduce)

    figures = subparsers.add_parser(
        "figures",
        help="regenerate figures from a summary artifact",
    )
    figures.add_argument("summary", type=Path)
    figures.add_argument("--output-dir", type=Path, required=True)
    figures.set_defaults(handler=command_figures)
    return parser


def command_check(_args: argparse.Namespace) -> None:
    print(json.dumps(validate_style_data(), indent=2, sort_keys=True))


def command_reproduce(args: argparse.Namespace) -> None:
    summary = reproduce(
        suite=args.suite,
        seeds=args.seeds,
        output_dir=args.output_dir.resolve(),
        requested_device=args.device,
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir.resolve()),
                "scientific_target": summary["scientific_target"],
                "decisions": summary["consolidated"]["decisions"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def command_figures(args: argparse.Namespace) -> None:
    paths = render_figures(
        args.summary.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(paths, indent=2, sort_keys=True))


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
