"""Command line for the expression ladder.

    expression-ladder check
    expression-ladder reproduce --suite smoke --device cpu
    expression-ladder reproduce --suite full --device cuda --output-dir results/<run>

The reproduce path runs stages A through G in order and writes one summary
artifact. Stages B and C feed C and D respectively, so they are not
independently selectable; a partial run would not produce a comparable number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .data import build_style_pairs, validate_style_data
from .runtime import (
    FULL_CONFIG,
    FULL_PROFILE,
    SMOKE_CONFIG,
    SMOKE_PROFILE,
    atomic_json_dump,
    choose_device,
    cuda_gate,
    emit,
    environment_report,
    load_model,
    memory_snapshot,
    release_model,
)
from .stages import (
    StageConfig,
    stage_a_prompt_upper_bound,
    stage_b_fit,
    stage_c_dose_screen,
    stage_d_generation_audit,
    stage_f_contextual,
    stage_g_train_residual,
)

AXES = ("warmth", "patience", "goodwill")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce the expression ladder")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Validate the frozen corpus without loading a model")

    run = sub.add_parser("reproduce", help="Run stages A through G")
    run.add_argument("--suite", choices=["smoke", "full"], default="smoke")
    run.add_argument("--device", default="auto")
    run.add_argument("--output-dir", type=Path, default=Path("results/ladder"))
    run.add_argument("--seed", type=int, default=20260727)
    run.add_argument(
        "--layers",
        default="19,27,35",
        help="Comma-separated residual layers to fit and screen",
    )
    run.add_argument("--components", default="residual,attention")
    run.add_argument("--fractions", default="0.005,0.01,0.02,0.04,0.08")
    run.add_argument("--windows", default="1,8,0", help="0 means every response position")
    run.add_argument("--stage-f-total-fraction", type=float, default=0.16)
    run.add_argument("--stage-g-steps", type=int, default=72)
    run.add_argument("--stage-g-lr", type=float, default=1e-3)
    run.add_argument("--stage-g-fraction", type=float, default=0.16)
    run.add_argument("--stage-g-cross-axis-weight", type=float, default=1.0)
    return parser


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(v) for v in value.split(",") if v.strip())


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(v) for v in value.split(",") if v.strip())


def _names(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


def command_check(_args: argparse.Namespace) -> None:
    report = validate_style_data()
    emit("corpus_validated", **report)


def command_reproduce(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    profile = SMOKE_PROFILE if args.suite == "smoke" else FULL_PROFILE
    run_config = SMOKE_CONFIG if args.suite == "smoke" else FULL_CONFIG
    device = choose_device(args.device, profile)

    layers = _ints(args.layers)
    if args.suite == "smoke":
        # The smoke model is shallower than the paper target; keep the deepest
        # layer in range rather than silently indexing past the end.
        layers = tuple(l for l in layers if l < 24) or (11,)

    config = StageConfig(
        axes=AXES,
        layers=layers,
        components=_names(args.components),
        fractions=_floats(args.fractions),
        windows=_ints(args.windows),
        max_seq_len=run_config.max_score_seq_len,
        max_new_tokens=run_config.max_new_tokens,
        max_context_tokens=run_config.max_context_tokens,
    )

    splits = build_style_pairs(AXES, ("fit", "dev", "test"))
    fit_pairs = [p for p in splits if p.split == "fit"]
    heldout_pairs = [p for p in splits if p.split == "test"]
    if run_config.test_case_limit is not None:
        heldout_pairs = heldout_pairs[: run_config.test_case_limit * len(AXES)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "suite": args.suite,
        "seed": args.seed,
        "profile": profile.name,
        "model_id": profile.model_id,
        "revision": profile.revision,
        "config": {
            "layers": list(config.layers),
            "components": list(config.components),
            "fractions": list(config.fractions),
            "windows": list(config.windows),
        },
        "corpus": validate_style_data(),
    }

    model, tokenizer = load_model(profile, device=device)
    try:
        summary["environment"] = environment_report(profile, device)
        summary["memory_after_load"] = memory_snapshot(device)
        cuda_gate(device, stage="after_load", minimum_free_gib=4.0)

        summary["stage_a"] = stage_a_prompt_upper_bound(model, tokenizer, heldout_pairs, config)

        stage_b = stage_b_fit(model, tokenizer, fit_pairs, config, device)
        directions = stage_b["fitted"]["directions"]
        summary["stage_b"] = {"geometry": stage_b["geometry"]}

        stage_c = stage_c_dose_screen(model, tokenizer, directions, heldout_pairs, config)
        summary["stage_c"] = {"rows": stage_c["rows"], "best": stage_c["best"]}

        summary["stage_d"] = stage_d_generation_audit(
            model, tokenizer, directions, stage_c["best"], heldout_pairs, config
        )

        summary["stage_f"] = stage_f_contextual(
            model,
            tokenizer,
            heldout_pairs,
            config,
            device,
            total_fraction=args.stage_f_total_fraction,
        )

        init = {
            axis: directions[axis]["residual"][config.layers[-1]] for axis in config.axes
        }
        stage_g = stage_g_train_residual(
            model,
            tokenizer,
            fit_pairs,
            init,
            config,
            layer=config.layers[-1],
            fraction=args.stage_g_fraction,
            steps=args.stage_g_steps if args.suite == "full" else 4,
            lr=args.stage_g_lr,
            cross_axis_weight=args.stage_g_cross_axis_weight,
        )
        trained = {
            axis: {"residual": {config.layers[-1]: stage_g["writer"].direction_for(axis).detach().cpu()}}
            for axis in config.axes
        }
        best_g = {
            axis: {
                "layer": config.layers[-1],
                "component": "residual",
                "fraction": args.stage_g_fraction,
                "window": 0,
            }
            for axis in config.axes
        }
        summary["stage_g"] = {
            "geometry": stage_g["geometry"],
            "loss_history": stage_g["loss_history"],
            "audit": stage_d_generation_audit(
                model, tokenizer, trained, best_g, heldout_pairs, config, stage="G"
            ),
        }
        summary["memory_peak"] = memory_snapshot(device)
    finally:
        release_model(device, model, tokenizer)

    path = args.output_dir / "summary.json"
    atomic_json_dump(summary, path)
    emit("run_complete", path=str(path))


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "check":
        command_check(args)
    else:
        command_reproduce(args)


if __name__ == "__main__":
    main()
