from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

if __package__ in (None, ""):
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.bootstrap_setup import ensure_tokenizer
    from src.common import DEFAULT_MINI_CHECKPOINT, REPO_ROOT
    from src.experiments import (
        results_to_jsonable,
        run_bridge_experiment,
        run_composition,
        run_diversity_experiment,
        run_prefix_experiment,
        run_static_lora,
        run_task_eval,
    )
else:
    from .bootstrap_setup import ensure_tokenizer
    from .common import DEFAULT_MINI_CHECKPOINT, REPO_ROOT
    from .experiments import (
        results_to_jsonable,
        run_bridge_experiment,
        run_composition,
        run_diversity_experiment,
        run_prefix_experiment,
        run_static_lora,
        run_task_eval,
    )


ARTIFACT_ROOT = REPO_ROOT / "results"


@dataclass
class Job:
    name: str
    group: str
    run: Callable[[], dict]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stddev(values: list[float]) -> float:
    if not values:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper experiment suite and save artifacts")
    parser.add_argument(
        "--suite",
        default="all",
        choices=["smoke", "benchmark", "controls", "diversity", "composition", "prefix", "task", "repro", "all"],
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_MINI_CHECKPOINT))
    parser.add_argument("--output-dir", default=None, help="Directory to store JSON summaries and CSV logs")
    parser.add_argument("--tag", default=None, help="Optional name prefix for the output directory")
    parser.add_argument("--resume", action="store_true", help="Skip jobs whose JSON result already exists")
    parser.add_argument("--force", action="store_true", help="Re-run jobs even if result files exist")
    parser.add_argument("--quick", action="store_true", help="Use tiny step counts for a smoke pass")
    parser.add_argument("--benchmark-steps", type=int, default=300)
    parser.add_argument("--composition-steps-per-brick", type=int, default=150)
    parser.add_argument("--task-steps", type=int, default=600)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--target", default="all", choices=["attn", "ffn", "all"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-tokens", type=int, default=None)
    parser.add_argument("--sensor-limit", type=int, default=None)
    parser.add_argument("--max-eval-items", type=int, default=200)
    parser.add_argument("--composition-eval-mode", default="conditioned", choices=["conditioned", "fixed"])
    parser.add_argument("--probe-max-items-per-activity", type=int, default=32)
    parser.add_argument("--probe-seed", type=int, default=None)
    parser.add_argument("--n-prefix", type=int, default=8)
    parser.add_argument("--prefix-lr", type=float, default=None, help="Defaults to --lr so the prefix baseline stays matched")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repro-seeds", default="42,1042,2042")
    parser.add_argument("--task-modalities", default="imu", help="Comma-separated list from audio,imu")
    parser.add_argument("--include-audio-task", action="store_true", help="Shortcut to include audio in task eval")
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> None:
    if args.quick:
        args.benchmark_steps = 10
        args.composition_steps_per_brick = 10
        args.task_steps = 10
        args.eval_tokens = 4096 if args.eval_tokens is None else min(args.eval_tokens, 4096)
        args.sensor_limit = 8 if args.sensor_limit is None else min(args.sensor_limit, 8)
        args.max_eval_items = min(args.max_eval_items, 8)
        args.probe_max_items_per_activity = min(args.probe_max_items_per_activity, 2)

    if args.prefix_lr is None:
        args.prefix_lr = args.lr

    task_modalities = [item.strip() for item in args.task_modalities.split(",") if item.strip()]
    if args.include_audio_task and "audio" not in task_modalities:
        task_modalities.append("audio")
    args.task_modalities = task_modalities
    args.repro_seeds = [int(item.strip()) for item in args.repro_seeds.split(",") if item.strip()]


def build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = args.tag or args.suite
    return ARTIFACT_ROOT / f"{tag}-{stamp}"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def link_latest(output_dir: Path) -> None:
    latest = ARTIFACT_ROOT / "latest"
    latest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(output_dir.resolve(), target_is_directory=True)
    except OSError:
        pass


def log_path(output_dir: Path, job_name: str) -> str:
    return str(output_dir / "logs" / f"{job_name}.csv")


def repro_job(
    name: str,
    modality: str,
    *,
    checkpoint: str,
    train_steps: int,
    rank: int,
    target: str,
    lr: float,
    eval_tokens: int | None,
    sensor_limit: int | None,
    seeds: list[int],
    output_dir: Path,
) -> Callable[[], dict]:
    def run() -> dict:
        runs = []
        for seed in seeds:
            result = run_bridge_experiment(
                modality=modality,
                feature_mode="true",
                checkpoint=checkpoint,
                train_steps=train_steps,
                rank=rank,
                target=target,
                lr=lr,
                eval_tokens=eval_tokens,
                sensor_limit=sensor_limit,
                seed=seed,
                log_csv=log_path(output_dir, f"{name}-seed{seed}"),
            )
            runs.append(results_to_jsonable(result))
        improvements = [run["improvement"] for run in runs]
        return {
            "experiment": name,
            "modality": modality,
            "seeds": seeds,
            "runs": runs,
            "mean_improvement": mean(improvements),
            "std_improvement": stddev(improvements),
        }

    return run


def build_jobs(args: argparse.Namespace, output_dir: Path) -> list[Job]:
    jobs: list[Job] = []

    def add(name: str, group: str, run: Callable[[], dict]) -> None:
        jobs.append(Job(name=name, group=group, run=run))

    suite = args.suite
    smoke = suite == "smoke"

    if suite in {"benchmark", "controls", "all"} or smoke:
        benchmark_modalities = ["audio", "vision", "imu"] if not smoke else ["imu"]
        if suite in {"benchmark", "all"} or smoke:
            for modality in benchmark_modalities:
                add(
                    f"static-lora-{modality}",
                    "benchmark",
                    lambda modality=modality: run_static_lora(
                        modality=modality,
                        checkpoint=args.checkpoint,
                        train_steps=args.benchmark_steps,
                        rank=args.rank,
                        target=args.target,
                        lr=args.lr,
                        log_csv=log_path(output_dir, f"static-lora-{modality}"),
                        eval_tokens=args.eval_tokens,
                        seed=args.seed,
                    ),
                )
                add(
                    f"bridge-{modality}",
                    "benchmark",
                    lambda modality=modality: run_bridge_experiment(
                        modality=modality,
                        feature_mode="true",
                        checkpoint=args.checkpoint,
                        train_steps=args.benchmark_steps,
                        rank=args.rank,
                        target=args.target,
                        lr=args.lr,
                        eval_tokens=args.eval_tokens,
                        sensor_limit=args.sensor_limit,
                        seed=args.seed,
                        log_csv=log_path(output_dir, f"bridge-{modality}"),
                    ),
                )

        if suite in {"controls", "all"} or smoke:
            control_modalities = benchmark_modalities
            for feature_mode in ("shuffled", "random", "constant"):
                for modality in control_modalities:
                    add(
                        f"{feature_mode}-{modality}",
                        "controls",
                        lambda feature_mode=feature_mode, modality=modality: run_bridge_experiment(
                            modality=modality,
                            feature_mode=feature_mode,
                            checkpoint=args.checkpoint,
                            train_steps=args.benchmark_steps,
                            rank=args.rank,
                            target=args.target,
                            lr=args.lr,
                            eval_tokens=args.eval_tokens,
                            sensor_limit=args.sensor_limit,
                            seed=args.seed,
                            log_csv=log_path(output_dir, f"{feature_mode}-{modality}"),
                        ),
                    )

    if suite in {"diversity", "all"} or smoke:
        lambdas = [0.0, 0.01, 0.05, 0.1, 0.2] if not smoke else [0.0, 0.1]
        for lam in lambdas:
            label = f"diversity-imu-l{lam:.2f}"
            add(
                label,
                "diversity",
                lambda lam=lam, label=label: run_diversity_experiment(
                    checkpoint=args.checkpoint,
                    train_steps=args.benchmark_steps,
                    rank=args.rank,
                    target=args.target,
                    lr=args.lr,
                    diversity_weight=lam,
                    log_csv=log_path(output_dir, label),
                    eval_tokens=args.eval_tokens,
                    sensor_limit=args.sensor_limit,
                    seed=args.seed,
                    probe_max_items_per_activity=args.probe_max_items_per_activity,
                    probe_seed=args.probe_seed,
                ),
            )

    if suite in {"composition", "all"} or smoke:
        brick_sets = [["vision"], ["audio"], ["imu"], ["vision", "audio"], ["vision", "imu"], ["audio", "imu"], ["vision", "audio", "imu"]]
        if smoke:
            brick_sets = [["vision"], ["vision", "audio", "imu"]]
        for bricks in brick_sets:
            label = f"compose-{''.join(brick[0].upper() for brick in bricks)}"
            add(
                label,
                "composition",
                lambda bricks=bricks, label=label: run_composition(
                    bricks=bricks,
                    checkpoint=args.checkpoint,
                    steps_per_brick=args.composition_steps_per_brick,
                    rank=args.rank,
                    target=args.target,
                    lr=args.lr,
                    eval_tokens=args.eval_tokens,
                    sensor_limit=args.sensor_limit,
                    seed=args.seed,
                    log_csv=log_path(output_dir, label),
                    eval_mode=args.composition_eval_mode,
                ),
            )

    if suite in {"prefix", "all"} or smoke:
        prefix_modalities = ["audio", "vision", "imu"] if not smoke else ["imu"]
        for modality in prefix_modalities:
            label = f"prefix-{modality}-{args.n_prefix}tok"
            add(
                label,
                "prefix",
                lambda modality=modality, label=label: run_prefix_experiment(
                    modality=modality,
                    checkpoint=args.checkpoint,
                    train_steps=args.benchmark_steps,
                    n_prefix=args.n_prefix,
                    lr=args.prefix_lr,
                    log_csv=log_path(output_dir, label),
                    eval_tokens=args.eval_tokens,
                    sensor_limit=args.sensor_limit,
                    seed=args.seed,
                ),
            )

    if suite in {"task", "all"} or smoke:
        task_modalities = args.task_modalities if not smoke else ["imu"]
        for modality in task_modalities:
            add(
                f"task-{modality}",
                "task",
                lambda modality=modality: run_task_eval(
                    modality=modality,
                        checkpoint=args.checkpoint,
                        train_steps=args.task_steps,
                        rank=args.rank,
                        target=args.target,
                        lr=args.lr,
                        max_eval_items=args.max_eval_items,
                        seed=args.seed,
                    ),
                )

    if suite in {"repro", "all"} or smoke:
        repro_modalities = ["audio", "vision", "imu"] if not smoke else ["imu"]
        for modality in repro_modalities:
            label = f"repro-{modality}"
            add(
                label,
                "repro",
                repro_job(
                    label,
                    modality,
                    checkpoint=args.checkpoint,
                    train_steps=args.benchmark_steps,
                    rank=args.rank,
                    target=args.target,
                    lr=args.lr,
                    eval_tokens=args.eval_tokens,
                    sensor_limit=args.sensor_limit,
                    seeds=args.repro_seeds,
                    output_dir=output_dir,
                ),
            )

    return jobs


def summarize(results: dict[str, dict]) -> dict:
    groups: dict[str, list[str]] = {}
    for name, payload in results.items():
        groups.setdefault(payload["group"], []).append(name)
    return {"count": len(results), "groups": groups}


def main() -> None:
    args = parse_args()
    normalize_args(args)
    # Self-heal a stale token_bytes.pt (e.g. after a git pull without setup.sh) so
    # BPB denominators always match the shipped tokenizer.
    if args.checkpoint.startswith("hf:"):
        if __package__ in (None, ""):
            from src.runtime_lfm import configure_tokenizer
        else:
            from .runtime_lfm import configure_tokenizer
        configure_tokenizer(args.checkpoint[len("hf:"):])
    else:
        ensure_tokenizer()
    output_dir = build_output_dir(args)
    results_dir = output_dir / "results"
    logs_dir = output_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args, output_dir)
    summary_path = output_dir / "summary.json"
    metadata = {
        "suite": args.suite,
        "quick": args.quick,
        "checkpoint": args.checkpoint,
        "benchmark_steps": args.benchmark_steps,
        "composition_steps_per_brick": args.composition_steps_per_brick,
        "task_steps": args.task_steps,
        "rank": args.rank,
        "target": args.target,
        "lr": args.lr,
        "eval_tokens": args.eval_tokens,
        "sensor_limit": args.sensor_limit,
        "max_eval_items": args.max_eval_items,
        "composition_eval_mode": args.composition_eval_mode,
        "probe_max_items_per_activity": args.probe_max_items_per_activity,
        "probe_seed": args.probe_seed,
        "n_prefix": args.n_prefix,
        "prefix_lr": args.prefix_lr,
        "seed": args.seed,
        "repro_seeds": args.repro_seeds,
        "task_modalities": args.task_modalities,
        "created_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
    }

    aggregate: dict[str, dict] = {}
    for index, job in enumerate(jobs, start=1):
        result_path = results_dir / f"{job.name}.json"
        if args.resume and not args.force and result_path.exists():
            payload = load_json(result_path)
            aggregate[job.name] = payload
            print(f"[{index}/{len(jobs)}] skip {job.name} (cached)")
            continue

        print(f"[{index}/{len(jobs)}] run {job.name}")
        result = results_to_jsonable(job.run())
        payload = {"name": job.name, "group": job.group, "result": result}
        write_json(result_path, payload)
        aggregate[job.name] = payload
        write_json(summary_path, {"metadata": metadata, "summary": summarize(aggregate), "results": aggregate})

    final_payload = {"metadata": metadata, "summary": summarize(aggregate), "results": aggregate}
    write_json(summary_path, final_payload)
    link_latest(output_dir)
    print(json.dumps(final_payload["summary"], indent=2, sort_keys=True))
    print(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
