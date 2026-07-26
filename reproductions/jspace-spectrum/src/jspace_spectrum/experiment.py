"""End-to-end experiment orchestration and artifact persistence."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import torch

from .data import (
    AXES,
    CALIBRATION_MESSAGES,
    FIT_PROMPTS,
    SYSTEM_PROMPTS,
    EvalCase,
    build_cases,
    validate_data,
)
from .figures import render_figures
from .metrics import BOOTSTRAP_SAMPLES, summarize_measurements
from .model import (
    PROFILES,
    ModelProfile,
    atomic_json_dump,
    atomic_torch_save,
    choose_device,
    environment_report,
    fit_calibration,
    fit_targeted_lens,
    load_model,
    measure_case,
    memory_snapshot,
    resolve_axis_token_ids,
    runtime_gate,
    serializable_lens,
)
from .viewer import render_viewer, summarize_groups


FORMAT = "jspace-spectrum-experiment-v1"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_line(handle: Any, payload: Mapping[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _axis_poles() -> list[dict[str, str]]:
    return [
        {
            "axis": axis.name,
            "positive": axis.positive_label,
            "negative": axis.negative_label,
        }
        for axis in AXES
    ]


def _smoke_cases() -> tuple[EvalCase, ...]:
    selected: list[EvalCase] = []
    seen_landmark: set[tuple[str, str]] = set()
    for case in build_cases():
        if case.kind == "landmark":
            key = (str(case.axis), str(case.pole))
            if key not in seen_landmark:
                selected.append(case)
                seen_landmark.add(key)
        elif case.group in {"meh", "neutral"}:
            selected.append(case)
        elif case.variant == 1:
            selected.append(case)
    return tuple(selected)


def suite_config(suite: str) -> dict[str, Any]:
    if suite == "full":
        return {
            "fit_prompts": FIT_PROMPTS,
            "calibration_messages": CALIBRATION_MESSAGES,
            "system_prompts": SYSTEM_PROMPTS,
            "cases": build_cases(),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "fit_max_seq_len": 96,
            "measure_max_seq_len": 128,
        }
    if suite == "smoke":
        return {
            "fit_prompts": FIT_PROMPTS[:1],
            "calibration_messages": CALIBRATION_MESSAGES[:4],
            "system_prompts": dict(list(SYSTEM_PROMPTS.items())[:2]),
            "cases": _smoke_cases(),
            "bootstrap_samples": 100,
            "fit_max_seq_len": 96,
            "measure_max_seq_len": 128,
        }
    raise ValueError(f"unknown suite {suite!r}")


def _run_signature(
    *,
    suite: str,
    profile: ModelProfile,
    data_report: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "suite": suite,
        "profile": profile.name,
        "model_id": profile.model_id,
        "revision": profile.revision,
        "source_layer": profile.source_layer,
        "target_layer": profile.target_layer,
        "trace_layers": list(profile.trace_layers),
        "data_sha256": data_report["sha256"],
        "fit_prompt_count": len(config["fit_prompts"]),
        "calibration_message_count": len(config["calibration_messages"]),
        "systems": list(config["system_prompts"]),
        "case_ids": [case.case_id for case in config["cases"]],
    }


def _check_signature(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    source: Path,
) -> None:
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(
                f"{source} has {key}={actual.get(key)!r}; expected {value!r}"
            )


def _load_events(
    path: Path,
    signature: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], bool]:
    if not path.exists():
        return {}, False
    rows: dict[str, dict[str, Any]] = {}
    saw_start = False
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "start":
            _check_signature(event["signature"], signature, source=path)
            saw_start = True
        elif event.get("event") == "case":
            row = event["row"]
            rows[str(row["id"])] = row
        elif event.get("event") == "complete":
            continue
        else:
            raise ValueError(f"{path}:{line_number} has an unknown event")
    if rows and not saw_start:
        raise ValueError(f"{path} contains cases without a start event")
    return rows, saw_start


def _ordered_rows(
    rows: Mapping[str, dict[str, Any]],
    *,
    cases: Sequence[EvalCase],
    systems: Mapping[str, str],
) -> list[dict[str, Any]]:
    expected = [
        f"{case.case_id}--{system_id}" for case in cases for system_id in systems
    ]
    missing = [row_id for row_id in expected if row_id not in rows]
    extras = sorted(set(rows) - set(expected))
    if missing or extras:
        raise ValueError(
            f"measurement inventory mismatch: {len(missing)} missing, "
            f"{len(extras)} extra"
        )
    return [rows[row_id] for row_id in expected]


def _load_lens(
    path: Path,
    *,
    signature: Mapping[str, Any],
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _check_signature(payload["signature"], signature, source=path)
    return payload["lens"]


def _recording_payload(
    *,
    rows: Sequence[Mapping[str, Any]],
    profile: ModelProfile,
    suite: str,
    signature: Mapping[str, Any],
    data_report: Mapping[str, Any],
    environment: Mapping[str, Any],
    runtime_seconds: float,
) -> dict[str, Any]:
    return {
        "format": "jspace-spectrum-recording-v1",
        "status": "complete",
        "created_at": utc_timestamp(),
        "model": {
            "name": profile.name,
            "id": profile.model_id,
            "revision": profile.revision,
            "device": environment["device"],
            "dtype": "bfloat16" if profile.paper_target else "float32",
        },
        "lens": {
            "format": "targeted-affect-lens-v1",
            "source_layer": profile.source_layer,
            "target_layer": profile.target_layer,
            "trace_layers": list(profile.trace_layers),
        },
        "measurement": {
            "unit": "neutral-calibration standard deviations",
            "span": "exact user-content tokens",
            "case_coordinate": "mean over user-content tokens",
            "frontier_coordinate": "final user-content token",
            "generation": False,
            "steering": False,
        },
        "suite": suite,
        "signature": dict(signature),
        "data": dict(data_report),
        "axis_names": [axis.name for axis in AXES],
        "axis_poles": _axis_poles(),
        "cases": list(rows),
        "groups": summarize_groups(rows),
        "runtime_seconds": round(runtime_seconds, 3),
    }


def _summary_payload(
    *,
    metrics: Mapping[str, Any],
    profile: ModelProfile,
    suite: str,
    signature: Mapping[str, Any],
    data_report: Mapping[str, Any],
    environment: Mapping[str, Any],
    runtime_seconds: float,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "status": "complete",
        "scientific_target": suite == "full" and profile.paper_target,
        "suite": suite,
        "model": asdict(profile),
        "signature": dict(signature),
        "data": dict(data_report),
        "environment": dict(environment),
        "runtime_seconds": round(runtime_seconds, 3),
        "metrics": dict(metrics),
        "artifacts": {
            "lens": "lens.pt",
            "events": "events.jsonl",
            "measurements": "measurements.json",
            "viewer": "jspace-spectrum.html",
            "figures": [
                "figures/landmark-separation.png",
                "figures/meh-radar.png",
                "figures/atlas-heatmap.png",
                "figures/meh-depth.png",
            ],
        },
        "output_directory": str(output_dir),
        "completed_at": utc_timestamp(),
    }


def run_reproduction(
    *,
    suite: str,
    profile_name: str,
    output_dir: Path,
    device_name: str = "auto",
    repository: Path,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run or resume one model and write all evidence-bearing artifacts."""
    if profile_name not in PROFILES:
        raise ValueError(f"unknown model profile {profile_name!r}")
    profile = PROFILES[profile_name]
    if suite == "full" and not profile.paper_target:
        raise ValueError("the full suite requires a paper model")
    if suite == "smoke" and profile.paper_target:
        raise ValueError("the smoke suite requires qwen35-2b-smoke")

    data_report = validate_data()
    config = suite_config(suite)
    signature = _run_signature(
        suite=suite,
        profile=profile,
        data_report=data_report,
        config=config,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    lens_path = output_dir / "lens.pt"
    fit_state_path = output_dir / "lens-fit-state.pt"
    measurements_path = output_dir / "measurements.json"
    summary_path = output_dir / "summary.json"

    device = choose_device(device_name, profile)
    environment = environment_report(
        profile,
        device,
        repository=repository,
        command=command or sys.argv,
    )
    started = time.perf_counter()
    model = tokenizer = lens_model = None
    try:
        model, tokenizer, lens_model = load_model(profile, device=device)
        if lens_path.exists():
            lens = _load_lens(lens_path, signature=signature)
            print(
                json.dumps(
                    {"event": "lens_resumed", "path": str(lens_path)},
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            axis_token_ids = resolve_axis_token_ids(tokenizer)
            fitted = fit_targeted_lens(
                lens_model,
                profile=profile,
                prompts=config["fit_prompts"],
                axis_token_ids=axis_token_ids,
                max_seq_len=config["fit_max_seq_len"],
                fit_state=fit_state_path,
            )
            calibration = fit_calibration(
                lens_model,
                tokenizer,
                system_prompts=config["system_prompts"],
                messages=config["calibration_messages"],
                read_matrix=fitted["read_directions"],
                trace_layers=profile.trace_layers,
                max_seq_len=config["measure_max_seq_len"],
                profile=profile,
            )
            lens = serializable_lens(fitted, calibration)
            atomic_torch_save(
                {
                    "format": "jspace-spectrum-lens-artifact-v1",
                    "signature": signature,
                    "lens": lens,
                },
                lens_path,
            )
            print(
                json.dumps(
                    {"event": "lens_saved", "path": str(lens_path)},
                    sort_keys=True,
                ),
                flush=True,
            )

        rows, saw_start = _load_events(events_path, signature)
        event_mode = "a" if saw_start else "w"
        with events_path.open(event_mode) as event_handle:
            if not saw_start:
                _json_line(
                    event_handle,
                    {
                        "event": "start",
                        "at": utc_timestamp(),
                        "signature": signature,
                        "environment": environment,
                    },
                )
            total = len(config["cases"]) * len(config["system_prompts"])
            index = 0
            for case in config["cases"]:
                for system_id, system_prompt in config["system_prompts"].items():
                    index += 1
                    row_id = f"{case.case_id}--{system_id}"
                    if row_id in rows:
                        continue
                    case_started = time.perf_counter()
                    row = measure_case(
                        lens_model,
                        tokenizer,
                        case=case,
                        system_id=system_id,
                        system_prompt=system_prompt,
                        read_matrix=lens["read_directions"],
                        calibration=lens["calibration"][system_id],
                        profile=profile,
                        max_seq_len=config["measure_max_seq_len"],
                    )
                    rows[row_id] = row
                    _json_line(
                        event_handle,
                        {"event": "case", "row": row},
                    )
                    if index % 25 == 0 or index == total:
                        memory = runtime_gate(
                            device,
                            stage=f"{profile.name} measurement {index}/{total}",
                            minimum_cuda_headroom_gib=(
                                20.0 if profile.paper_target else 0.5
                            ),
                        )
                    else:
                        memory = None
                    print(
                        json.dumps(
                            {
                                "event": "measurement",
                                "index": index,
                                "total": total,
                                "id": row_id,
                                "seconds": round(
                                    time.perf_counter() - case_started,
                                    3,
                                ),
                                "memory": memory,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

            ordered = _ordered_rows(
                rows,
                cases=config["cases"],
                systems=config["system_prompts"],
            )
            elapsed = time.perf_counter() - started
            recording = _recording_payload(
                rows=ordered,
                profile=profile,
                suite=suite,
                signature=signature,
                data_report=data_report,
                environment=environment,
                runtime_seconds=elapsed,
            )
            atomic_json_dump(recording, measurements_path)
            metrics = summarize_measurements(
                ordered,
                axis_names=[axis.name for axis in AXES],
                trace_layers=profile.trace_layers,
                bootstrap_samples=config["bootstrap_samples"],
            )
            summary = _summary_payload(
                metrics=metrics,
                profile=profile,
                suite=suite,
                signature=signature,
                data_report=data_report,
                environment=environment,
                runtime_seconds=elapsed,
                output_dir=output_dir,
            )
            atomic_json_dump(summary, summary_path)
            render_figures(summary, output_dir / "figures")
            render_viewer(recording, output_dir / "jspace-spectrum.html")
            _json_line(
                event_handle,
                {
                    "event": "complete",
                    "at": utc_timestamp(),
                    "summary": str(summary_path),
                    "measurements": str(measurements_path),
                    "decisions": metrics["decisions"],
                    "memory": memory_snapshot(device),
                },
            )
            return summary
    finally:
        del lens_model, tokenizer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def regenerate_artifacts(
    *,
    summary_path: Path,
    measurements_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    target = output_dir or summary_path.parent
    render_figures(summary, target / "figures")
    source = measurements_path or summary_path.parent / "measurements.json"
    recording = json.loads(source.read_text())
    render_viewer(recording, target / "jspace-spectrum.html")
    return {
        "summary": str(summary_path),
        "measurements": str(source),
        "output_directory": str(target),
    }
