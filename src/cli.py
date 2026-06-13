from __future__ import annotations

import argparse

from .common import DEFAULT_MINI_CHECKPOINT
from .experiments import (
    dump_result,
    run_bridge_experiment,
    run_composition,
    run_diversity_experiment,
    run_prefix_experiment,
    run_static_lora,
    run_task_eval,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper-specific experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(cmd, *, default_steps=300, include_lr=False):
        cmd.add_argument("--checkpoint", default=str(DEFAULT_MINI_CHECKPOINT))
        cmd.add_argument("--steps", type=int, default=default_steps)
        cmd.add_argument("--rank", type=int, default=4)
        cmd.add_argument("--target", default="all", choices=["attn", "ffn", "all"])
        cmd.add_argument("--eval-tokens", type=int, default=None)
        cmd.add_argument("--sensor-limit", type=int, default=None)
        cmd.add_argument("--seed", type=int, default=42)
        cmd.add_argument("--log-csv", default=None)
        if include_lr:
            cmd.add_argument("--lr", type=float, default=1e-3)

    static = sub.add_parser("static-lora", help="Run the static LoRA baseline")
    static.add_argument("--modality", required=True, choices=["audio", "vision", "imu"])
    add_common(static, include_lr=True)

    descriptions = {
        "bridge": "Run the conditional bridge experiment",
        "shuffled": "Run the shuffled-features control",
        "random": "Run the random-features control",
        "constant": "Run the constant-feature (capacity-matched) control",
        "diversity": "Run the diversity-regularized bridge experiment",
    }
    for name in ("bridge", "shuffled", "random", "constant", "diversity"):
        cmd = sub.add_parser(name, help=descriptions[name])
        if name == "diversity":
            cmd.add_argument("--modality", default="imu", choices=["audio", "vision", "imu"])
        else:
            cmd.add_argument("--modality", required=True, choices=["audio", "vision", "imu"])
        add_common(cmd, include_lr=True)
        cmd.add_argument("--diversity-weight", type=float, default=0.1 if name == "diversity" else 0.0)
        if name == "diversity":
            cmd.add_argument("--probe-max-items-per-activity", type=int, default=32)
            cmd.add_argument("--probe-seed", type=int, default=None)

    composition = sub.add_parser("composition", help="Run additive bridge composition")
    composition.add_argument("--bricks", default="vision,audio,imu")
    composition.add_argument("--steps-per-brick", type=int, default=150)
    composition.add_argument("--checkpoint", default=str(DEFAULT_MINI_CHECKPOINT))
    composition.add_argument("--rank", type=int, default=4)
    composition.add_argument("--target", default="all", choices=["attn", "ffn", "all"])
    composition.add_argument("--lr", type=float, default=1e-3)
    composition.add_argument("--eval-tokens", type=int, default=None)
    composition.add_argument("--sensor-limit", type=int, default=None)
    composition.add_argument("--seed", type=int, default=42)
    composition.add_argument("--log-csv", default=None)
    composition.add_argument("--eval-mode", default="conditioned", choices=["conditioned", "fixed"])

    prefix = sub.add_parser("prefix", help="Run the prefix-tuning baseline")
    prefix.add_argument("--modality", required=True, choices=["audio", "vision", "imu"])
    add_common(prefix)
    prefix.add_argument("--n-prefix", type=int, default=8)
    prefix.add_argument("--lr", type=float, default=1e-3)

    task = sub.add_parser("task-eval", help="Run task-aligned evaluation")
    task.add_argument("--modality", required=True, choices=["audio", "imu"])
    task.add_argument("--checkpoint", default=str(DEFAULT_MINI_CHECKPOINT))
    task.add_argument("--steps", type=int, default=600)
    task.add_argument("--rank", type=int, default=4)
    task.add_argument("--target", default="all", choices=["attn", "ffn", "all"])
    task.add_argument("--lr", type=float, default=1e-3)
    task.add_argument("--max-eval-items", type=int, default=200)
    task.add_argument("--seed", type=int, default=42)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "static-lora":
        result = run_static_lora(
            modality=args.modality,
            checkpoint=args.checkpoint,
            train_steps=args.steps,
            rank=args.rank,
            target=args.target,
            lr=args.lr,
            log_csv=args.log_csv,
            eval_tokens=args.eval_tokens,
            seed=args.seed,
        )
    elif args.command == "diversity":
        result = run_diversity_experiment(
            checkpoint=args.checkpoint,
            train_steps=args.steps,
            rank=args.rank,
            target=args.target,
            lr=args.lr,
            diversity_weight=args.diversity_weight,
            log_csv=args.log_csv,
            eval_tokens=args.eval_tokens,
            sensor_limit=args.sensor_limit,
            seed=args.seed,
            probe_max_items_per_activity=args.probe_max_items_per_activity,
            probe_seed=args.probe_seed,
        )
    elif args.command in {"bridge", "shuffled", "random", "constant"}:
        feature_mode = "true" if args.command == "bridge" else args.command
        result = run_bridge_experiment(
            modality=args.modality,
            feature_mode=feature_mode,
            checkpoint=args.checkpoint,
            train_steps=args.steps,
            rank=args.rank,
            target=args.target,
            lr=args.lr,
            diversity_weight=args.diversity_weight,
            log_csv=args.log_csv,
            eval_tokens=args.eval_tokens,
            sensor_limit=args.sensor_limit,
            seed=args.seed,
        )
    elif args.command == "composition":
        result = run_composition(
            bricks=[brick.strip() for brick in args.bricks.split(",") if brick.strip()],
            checkpoint=args.checkpoint,
            steps_per_brick=args.steps_per_brick,
            rank=args.rank,
            target=args.target,
            lr=args.lr,
            eval_tokens=args.eval_tokens,
            sensor_limit=args.sensor_limit,
            seed=args.seed,
            log_csv=args.log_csv,
            eval_mode=args.eval_mode,
        )
    elif args.command == "prefix":
        result = run_prefix_experiment(
            modality=args.modality,
            checkpoint=args.checkpoint,
            train_steps=args.steps,
            n_prefix=args.n_prefix,
            lr=args.lr,
            log_csv=args.log_csv,
            eval_tokens=args.eval_tokens,
            sensor_limit=args.sensor_limit,
            seed=args.seed,
        )
    elif args.command == "task-eval":
        result = run_task_eval(
            modality=args.modality,
            checkpoint=args.checkpoint,
            train_steps=args.steps,
            rank=args.rank,
            target=args.target,
            lr=args.lr,
            max_eval_items=args.max_eval_items,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unhandled command: {args.command}")

    print(dump_result(result))


if __name__ == "__main__":
    main()
