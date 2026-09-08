"""V-line rung 2t: the drowsy captain — the lookout gates the big model's compute.

Registration: tracks/vision-mode/PLAN.md, "Rung 2t registration — the drowsy
captain" (2026-08-23). 60 episodes × 16 steps from rung 2's eval pool: each
step re-rendered with camera jitter (94–100% crop + brightness ×U(0.96,
1.04)); cuts to a uniform pool frame with p = 0.2. The system is rung 2y's
switch with a LATCH: at a wake the detector's verdict picks the packet, the
big model decodes, and the action stands until the next wake; between wakes
only the lookout runs. Gates: ALWAYS (baseline), NEVER (floor), VERDICT
(verdict vs the last wake), BOX (geometry vs the last wake), PIXEL (the
understanding-free control), CUT-ORACLE (true cuts — the ceiling; its
disagreement with ALWAYS is the big model's own jitter-instability).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import vm_rung1 as r1  # noqa: E402
import vm_rung2 as r2  # noqa: E402
import vm_rung2y as r2y  # noqa: E402
from vm_rung1 import (  # noqa: E402
    HOSTS, HOST_ORDER, MODE_ACTION, SEEDS, TRAIN_DOSE, FrameBank, atomic_json_dump, bank_features,
    cuda_snapshot, display_path, emit, generate_with_packet, git_head, held_out_position, layer_plan,
    load_door, load_host, oracle_action, parameter_samples, parse_action, payloads_of,
    release_device_memory, samples_unchanged, seed_everything, select_frames, sha256_of,
)

REGISTRATION = "tracks/vision-mode/PLAN.md, Rung 2t registration — the drowsy captain (2026-08-23)"
CANONICAL_STEM = "vision-mode-r2t"
PERSON_MODES = r2.PERSON_MODES
DETECTOR_NAME = "yolo26s"  # the operating detector (rung 2y-26s decision rule)
GATES = ("always", "never", "verdict", "box", "pixel", "cut_oracle")

#: the registered draw
EPISODES = 60
STEPS = 16
CUT_P = 0.2
EPISODE_SEED = 2026
REGISTERED_EPISODE_SHA256 = "2aa0665e7dc2442ee86231031d530c9398ba458f3ccf47725e165b418dfe25d6"
JITTER_CROP_MIN = 0.94
BRIGHTNESS = (0.96, 1.04)
SMOKE_EPISODES = 6
SMOKE_STEPS = 8

#: the gates, by number
BOX_FRACTION_DELTA = 0.10
BOX_CENTER_DELTA = 0.15
MAX_BOXES = 5
PIXEL_THRESHOLD = 0.04

#: readings, by number (on the VERDICT gate)
AGREE_BAND = 0.95
CEILING_MARGIN = 0.02
STALE_BAND = 0.85
WAKE_BAND = 0.40
JITTERY_BAND = 0.70


# --------------------------------------------------------------------------
# episodes
# --------------------------------------------------------------------------

def build_structure(pool_size: int, rng: random.Random, *, episodes: int, steps: int) -> list[list[dict[str, Any]]]:
    """Cut structure first, one rng: step 0 always cuts; later steps cut with p."""

    out = []
    for _ in range(episodes):
        episode = []
        base = None
        for t in range(steps):
            cut = t == 0 or rng.random() < CUT_P
            if cut:
                base = rng.randrange(pool_size)
            episode.append({"base": base, "cut": cut})
        out.append(episode)
    return out


def render_episodes(pool: Sequence[Mapping[str, Any]], structure, rng: random.Random,
                    out_dir: Path) -> list[list[dict[str, Any]]]:
    """Jitter every step (the same rng continues after the structure draws);
    files are ep{e}-s{t}.png in `out_dir`. Returns per-step records."""

    from PIL import Image, ImageEnhance

    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for e, episode in enumerate(structure):
        rows = []
        for t, step in enumerate(episode):
            frame = pool[step["base"]]
            image = Image.open(frame["file"]).convert("RGB")
            width, height = image.size
            scale = rng.uniform(JITTER_CROP_MIN, 1.0)
            crop_w, crop_h = max(1, int(width * scale)), max(1, int(height * scale))
            left = rng.randrange(width - crop_w + 1)
            top = rng.randrange(height - crop_h + 1)
            jittered = image.crop((left, top, left + crop_w, top + crop_h)).resize((width, height))
            jittered = ImageEnhance.Brightness(jittered).enhance(rng.uniform(*BRIGHTNESS))
            path = out_dir / f"ep{e:02d}-s{t:02d}.png"
            jittered.save(path, format="PNG")
            rows.append({"file": str(path), "image_id": e * 1000 + t, "episode": e, "step": t,
                         "cut": step["cut"], "base_image_id": int(frame["image_id"]),
                         "label": frame["label"]})
        rendered.append(rows)
    return rendered


def reuse_episodes(pool: Sequence[Mapping[str, Any]], structure,
                   out_dir: Path) -> list[list[dict[str, Any]]]:
    """Rebuild the registered episode records without touching the pinned PNGs."""

    rendered = []
    for e, episode in enumerate(structure):
        rows = []
        for t, step in enumerate(episode):
            frame = pool[step["base"]]
            path = out_dir / f"ep{e:02d}-s{t:02d}.png"
            if not path.is_file():
                raise SystemExit(f"episode cache missing: {display_path(path)}")
            rows.append({"file": str(path), "image_id": e * 1000 + t, "episode": e, "step": t,
                         "cut": step["cut"], "base_image_id": int(frame["image_id"]),
                         "label": frame["label"]})
        rendered.append(rows)
    return rendered


def manifest_sha(rendered) -> str:
    digest = hashlib.sha256()
    for rows in rendered:
        for row in rows:
            digest.update(sha256_of(Path(row["file"])).encode())
            digest.update(f'{row["base_image_id"]}:{row["cut"]}'.encode())
    return digest.hexdigest()


def pixel_diffs(rendered) -> dict[int, float]:
    """image_id -> mean |Δ| to the previous step (full scale = 1); step 0 -> 1.0."""

    import numpy as np
    from PIL import Image

    out: dict[int, float] = {}
    for rows in rendered:
        previous = None
        for row in rows:
            current = np.asarray(Image.open(row["file"]).convert("RGB"), dtype=np.float32) / 255.0
            if previous is None or previous.shape != current.shape:
                out[row["image_id"]] = 1.0
            else:
                out[row["image_id"]] = float(np.abs(current - previous).mean())
            previous = current
    return out


# --------------------------------------------------------------------------
# the gates
# --------------------------------------------------------------------------

def box_summary(boxes: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    verdict = r2y.verdict_of(boxes)
    largest = max(boxes, key=lambda b: b["fraction"]) if boxes else None
    return {"verdict": verdict["verdict"], "count": min(len(boxes), MAX_BOXES),
            "fraction": largest["fraction"] if largest else 0.0,
            "cx": largest["cx"] if largest else 0.0, "cy": largest["cy"] if largest else 0.0}


def box_changed(now: Mapping[str, Any], at_wake: Mapping[str, Any]) -> bool:
    return (now["verdict"] != at_wake["verdict"] or now["count"] != at_wake["count"]
            or abs(now["fraction"] - at_wake["fraction"]) > BOX_FRACTION_DELTA
            or abs(now["cx"] - at_wake["cx"]) > BOX_CENTER_DELTA
            or abs(now["cy"] - at_wake["cy"]) > BOX_CENTER_DELTA)


def wake_decision(gate: str, *, step_row, summary, diffs, at_wake_summary) -> bool:
    if step_row["step"] == 0:
        return True
    if gate == "always":
        return True
    if gate == "never":
        return False
    if gate == "verdict":
        return summary["verdict"] != at_wake_summary["verdict"]
    if gate == "box":
        return box_changed(summary, at_wake_summary)
    if gate == "pixel":
        return diffs[step_row["image_id"]] > PIXEL_THRESHOLD
    if gate == "cut_oracle":
        return bool(step_row["cut"])
    raise ValueError(gate)


def run_gate(*, gate, person_mode, rendered, summaries, diffs, decode) -> list[dict[str, Any]]:
    """Simulate one gate over every episode; `decode(row, packet_mode)` is the
    (cached) big-model call. Returns one record per step."""

    rows: list[dict[str, Any]] = []
    for e, episode in enumerate(rendered):
        at_wake_summary = None
        latched: dict[str, Any] | None = None
        last_wake_step = None
        pending_cut_steps: list[int] = []
        for row in episode:
            summary = summaries[row["image_id"]]
            woke = wake_decision(gate, step_row=row, summary=summary, diffs=diffs,
                                 at_wake_summary=at_wake_summary)
            if row["cut"]:
                pending_cut_steps.append(row["step"])
            false_wake = woke and row["step"] > 0 and not pending_cut_steps
            lags = None
            if woke:
                packet_mode = r2.policy_mode(person_mode, summary["verdict"])
                latched = decode(row, packet_mode, e)
                at_wake_summary = summary
                last_wake_step = row["step"]
                lags = [row["step"] - cut for cut in pending_cut_steps]
                pending_cut_steps = []
            rows.append({"episode": e, "step": row["step"], "image_id": row["image_id"],
                         "base_image_id": row["base_image_id"], "label": row["label"],
                         "cut": row["cut"], "verdict": summary["verdict"], "woke": woke,
                         "false_wake": false_wake, "lags": lags, **latched})
    return rows


def gate_metrics(rows, baseline, person_mode: str) -> dict[str, Any]:
    action = MODE_ACTION[person_mode]
    total = len(rows)
    by_key = {(r["episode"], r["step"]): r for r in baseline}
    agree = sum(r["action"] == by_key[(r["episode"], r["step"])]["action"] for r in rows)
    wakes = sum(r["woke"] for r in rows)
    correct = sum(r["action"] == oracle_action(person_mode, r["label"]) for r in rows)
    lags = [lag for r in rows if r["lags"] for lag in r["lags"]]
    return {
        "action": action,
        "agreement": round(agree / total, 6),
        "wake_fraction": round(wakes / total, 6),
        "compute_saved": round(1 - wakes / total, 6),
        "oracle_correctness": round(correct / total, 6),
        "mean_cut_lag": round(statistics.fmean(lags), 4) if lags else None,
        "false_wakes": sum(r["false_wake"] for r in rows),
        "wakes": wakes, "steps": total,
    }


def reading_of(per_mode: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    verdict = {m: per_mode[m]["verdict"] for m in PERSON_MODES}
    ceiling = {m: per_mode[m]["cut_oracle"]["agreement"] for m in PERSON_MODES}
    agreement = {m: verdict[m]["agreement"] for m in PERSON_MODES}
    wake = {m: verdict[m]["wake_fraction"] for m in PERSON_MODES}
    threshold = {m: round(min(AGREE_BAND, ceiling[m] - CEILING_MARGIN), 6) for m in PERSON_MODES}
    if all(agreement[m] >= threshold[m] and wake[m] <= WAKE_BAND for m in PERSON_MODES):
        reading = "(i) FREE LUNCH"
    elif any(agreement[m] < STALE_BAND for m in PERSON_MODES):
        reading = "(ii) STALE"
    elif any(wake[m] > JITTERY_BAND for m in PERSON_MODES):
        reading = "(iii) JITTERY"
    else:
        reading = "(iv) MIXED"
    return {"reading": reading, "agreement": agreement, "wake_fraction": wake,
            "ceiling": ceiling, "threshold": threshold,
            "compute_saved": {m: verdict[m]["compute_saved"] for m in PERSON_MODES},
            "bands": {"agree": AGREE_BAND, "ceiling_margin": CEILING_MARGIN, "stale": STALE_BAND,
                      "wake": WAKE_BAND, "jittery": JITTERY_BAND}}


# --------------------------------------------------------------------------
# one (host, seed)
# --------------------------------------------------------------------------

def run_seed(args, *, host_key, seed, model, tower, processor, device, dtype, depth, hidden,
             rendered, summaries, diffs, detector_record, manifest, bank, base_samples,
             plan) -> dict[str, Any]:
    started = time.perf_counter()
    tokenizer = processor.tokenizer
    seed_everything(seed, device)
    source_layer, inject_layers, dose = plan["source_layer"], plan["inject_layers"], TRAIN_DOSE
    door, door_record = load_door(host_key, "mode", seed, hidden=hidden, inject_layers=inject_layers, device=device)
    if door_record["source_layer"] != source_layer:
        raise SystemExit("the door's source layer is not this plan's")
    emit("vm2t_seed_start", host=host_key, seed=seed, door=door_record["sha256"], **cuda_snapshot(device))
    features = bank_features(model, tokenizer, device, source_layer)
    if not samples_unchanged(model, base_samples):
        raise RuntimeError("the host moved during the mode-assignment taps")

    eos = int(tokenizer.eos_token_id)
    decode_cache: dict[tuple, dict[str, Any]] = {}
    decode_seconds = [0.0, 0]

    def decode(row, packet_mode, episode_index):
        position = held_out_position(episode_index)
        key = (row["image_id"], packet_mode, position)
        if key not in decode_cache:
            t0 = time.perf_counter()
            _p, payloads = payloads_of(door, features[(packet_mode, position)], device)
            new_ids = generate_with_packet(model, tower, bank.inputs(row, mode_text=None), payloads,
                                           inject_layers, dose, eos=eos)
            decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
            action, how = parse_action(decoded)
            decode_cache[key] = {"action": action, "parse": how, "packet_mode": packet_mode}
            decode_seconds[0] += time.perf_counter() - t0
            decode_seconds[1] += 1
        return decode_cache[key]

    cells: dict[str, dict[str, list[dict[str, Any]]]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for person_mode in PERSON_MODES:
        cells[person_mode] = {}
        baseline = run_gate(gate="always", person_mode=person_mode, rendered=rendered,
                            summaries=summaries, diffs=diffs, decode=decode)
        cells[person_mode]["always"] = baseline
        for gate in GATES:
            if gate != "always":
                cells[person_mode][gate] = run_gate(gate=gate, person_mode=person_mode, rendered=rendered,
                                                    summaries=summaries, diffs=diffs, decode=decode)
        metrics[person_mode] = {gate: gate_metrics(cells[person_mode][gate], baseline, person_mode)
                                for gate in GATES}
        emit("vm2t_mode", host=host_key, seed=seed, person_mode=person_mode,
             **{gate: {k: metrics[person_mode][gate][k] for k in ("agreement", "wake_fraction", "oracle_correctness")}
                for gate in GATES})
    if not samples_unchanged(model, base_samples):
        raise RuntimeError("the host moved during the gates")
    reading = reading_of(metrics)
    mean_decode = round(decode_seconds[0] / decode_seconds[1], 4) if decode_seconds[1] else None
    emit("vm2t_seed_reading", host=host_key, seed=seed, reading=reading["reading"],
         agreement=reading["agreement"], compute_saved=reading["compute_saved"],
         mean_decode_seconds=mean_decode)
    return {
        "track": r1.TRACK, "experiment": f"{CANONICAL_STEM}-{host_key}-s{seed}", "registration": REGISTRATION,
        "reuses": [r1.REGISTRATION, r2y.REGISTRATION],
        "claim_scope": (f"rung 2t's six gates x two person-modes at {HOSTS[host_key]['label']} on the s{seed} "
                        f"door; {len(rendered)} episodes x {len(rendered[0])} steps, one door seed"),
        "git_head": git_head(), "host_machine": platform.node(), "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "host": {"key": host_key, **HOSTS[host_key], "depth": depth, "hidden_size": hidden},
        "seed": seed, "smoke": bool(args.smoke), "layer_plan": plan,
        "doors": {"mode": door_record, "dose": dose, "frozen": True},
        "detector": {k: v for k, v in detector_record.items() if k != "verdicts"},
        "episodes": {"episodes": len(rendered), "steps": len(rendered[0]), "cut_p": CUT_P,
                     "seed": EPISODE_SEED, "manifest_sha256": manifest,
                     "jitter": {"crop_min": JITTER_CROP_MIN, "brightness": list(BRIGHTNESS)},
                     "cuts": sum(r["cut"] for rows in rendered for r in rows)},
        "gates": {"box": {"fraction_delta": BOX_FRACTION_DELTA, "center_delta": BOX_CENTER_DELTA},
                  "pixel_threshold": PIXEL_THRESHOLD},
        "decode": {"greedy": True, "max_new_tokens": r1.MAX_NEW, "unique_decodes": len(decode_cache),
                   "mean_decode_seconds": mean_decode},
        "arms": cells, "metrics": metrics, "reading": reading,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }


def pooled_payload(host_key: str, per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pooled_metrics: dict[str, dict[str, Any]] = {}
    for m in PERSON_MODES:
        pooled_metrics[m] = {}
        for gate in GATES:
            rows = [p["metrics"][m][gate] for p in per_seed]
            pooled_metrics[m][gate] = {
                "action": rows[0]["action"],
                "agreement": round(statistics.fmean(r["agreement"] for r in rows), 6),
                "wake_fraction": round(statistics.fmean(r["wake_fraction"] for r in rows), 6),
                "compute_saved": round(statistics.fmean(r["compute_saved"] for r in rows), 6),
                "oracle_correctness": round(statistics.fmean(r["oracle_correctness"] for r in rows), 6),
                "false_wakes": sum(r["false_wakes"] for r in rows),
                "mean_cut_lag": (round(statistics.fmean(r["mean_cut_lag"] for r in rows
                                                        if r["mean_cut_lag"] is not None), 4)
                                 if any(r["mean_cut_lag"] is not None for r in rows) else None),
            }
    seed_readings = {str(p["seed"]): p["reading"]["reading"] for p in per_seed}
    votes: dict[str, int] = {}
    for value in seed_readings.values():
        votes[value] = votes.get(value, 0) + 1
    majority = max(votes.items(), key=lambda item: (item[1], item[0]))
    return {
        "track": r1.TRACK, "experiment": f"{CANONICAL_STEM}-{host_key}-pooled", "registration": REGISTRATION,
        "git_head": git_head(), "host_machine": platform.node(), "host": per_seed[0]["host"],
        "seeds": [p["seed"] for p in per_seed], "smoke": any(p["smoke"] for p in per_seed),
        "detector": per_seed[0]["detector"], "episodes": per_seed[0]["episodes"],
        "per_seed_readings": seed_readings,
        "majority_reading": {"reading": majority[0], "votes": majority[1], "of": len(per_seed),
                             "decisive": majority[1] * 2 > len(per_seed)},
        "pooled_metrics": pooled_metrics, "pooled_reading": reading_of(pooled_metrics),
        "per_seed": {str(p["seed"]): p["reading"] for p in per_seed},
        "wall_seconds": round(sum(p["wall_seconds"] for p in per_seed), 3),
    }


def fmt(value: Any) -> str:
    return "—" if value is None else (f"{value:.2f}" if isinstance(value, float) else str(value))


def table(metrics: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = ["| person-mode | gate | agreement | wake fraction | saved | oracle correct | cut lag | false wakes |",
             "|---|---|---|---|---|---|---|---|"]
    for m in PERSON_MODES:
        for gate in GATES:
            x = metrics[m][gate]
            lines.append(f"| {m} | {gate} | {fmt(x['agreement'])} | {fmt(x['wake_fraction'])} | "
                         f"{fmt(x['compute_saved'])} | {fmt(x['oracle_correctness'])} | "
                         f"{fmt(x['mean_cut_lag'])} | {x['false_wakes']} |")
    return lines


def write_markdown(payload: Mapping[str, Any], path: Path, *, pooled: bool) -> None:
    host = payload["host"]["label"]
    title = (f"# V-line rung 2t — {host}, pooled over door seeds {payload['seeds']}" if pooled
             else f"# V-line rung 2t — {host}, door seed {payload['seed']}")
    lines = [title, ""]
    if payload["smoke"]:
        lines += ["**SMOKE ARTIFACT**", ""]
    e = payload["episodes"]
    lines += [f"Registration: {payload['registration']}.", "",
              f"{e['episodes']} episodes × {e['steps']} steps, {e['cuts']} cuts; lookout {DETECTOR_NAME}.", ""]
    if pooled:
        maj = payload["majority_reading"]
        lines += [f"Per-seed readings: {json.dumps(payload['per_seed_readings'])} — majority **{maj['reading']}** "
                  f"({maj['votes']} of {maj['of']}). Pooled-number reading: {payload['pooled_reading']['reading']}.", ""]
        lines += table(payload["pooled_metrics"])
    else:
        lines += [f"Reading at this seed (the VERDICT gate): **{payload['reading']['reading']}**.", ""]
        lines += table(payload["metrics"])
    lines += ["", "agreement = share of steps whose latched action equals ALWAYS's; saved = 1 − wake fraction; "
              "cut lag = mean steps from a true cut to the next wake; the CUT-ORACLE row is the ceiling "
              "(its disagreement with ALWAYS is the big model's own jitter-instability)."]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def run_host(args, host_key, seeds, rendered, summaries, diffs, detector_record, manifest, base):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    processor, model, tower, depth, hidden, dtype = load_host(host_key, device, local_files_only=args.local_files_only)
    plan = layer_plan(depth)
    emit("vm2t_host", host=host_key, depth=depth, plan=plan, **cuda_snapshot(device))
    bank = FrameBank(processor, device, dtype)
    base_samples = parameter_samples(model)
    per_seed = []
    for seed in seeds:
        payload = run_seed(args, host_key=host_key, seed=seed, model=model, tower=tower, processor=processor,
                           device=device, dtype=dtype, depth=depth, hidden=hidden, rendered=rendered,
                           summaries=summaries, diffs=diffs, detector_record=detector_record,
                           manifest=manifest, bank=bank, base_samples=base_samples, plan=plan)
        prefix = base.with_name(f"{base.name}-{host_key}-s{seed}")
        atomic_json_dump(payload, prefix.with_suffix(".json"))
        write_markdown(payload, prefix.with_suffix(".md"), pooled=False)
        emit("vm2t_seed_written", host=host_key, seed=seed, json=display_path(prefix.with_suffix(".json")))
        per_seed.append(payload)
    pooled = pooled_payload(host_key, per_seed)
    prefix = base.with_name(f"{base.name}-{host_key}")
    atomic_json_dump(pooled, prefix.with_suffix(".json"))
    write_markdown(pooled, prefix.with_suffix(".md"), pooled=True)
    emit("vm2t_host_written", host=host_key, json=display_path(prefix.with_suffix(".json")),
         majority=pooled["majority_reading"])
    del model, tower, bank
    release_device_memory(device)
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=(*HOSTS, "all"), default="450m")
    parser.add_argument("--seed", type=int, default=None, choices=SEEDS)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--frames-dir", default="/tmp/vm-r2t-frames")
    parser.add_argument("--reuse-frames", action="store_true")
    parser.add_argument("--coco-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.all_seeds and args.seed is not None:
        raise SystemExit("--all-seeds runs every registered seed; drop --seed")
    seeds = list(SEEDS) if args.all_seeds else [args.seed if args.seed is not None else SEEDS[0]]
    if args.output_prefix is None:
        if args.smoke:
            raise SystemExit("--smoke requires a non-canonical --output-prefix")
        base = r1.ROOT / "findings" / CANONICAL_STEM
    else:
        base = Path(args.output_prefix).expanduser()
        if args.smoke and base.name.startswith(CANONICAL_STEM) and base.parent.resolve() == (r1.ROOT / "findings").resolve():
            raise SystemExit("a smoke may not write the canonical family")
    coco_dir = Path(args.coco_dir).expanduser() if args.coco_dir else r1.default_coco_dir()
    frames = select_frames(coco_dir, per_class_train=r1.TRAIN_PER_CLASS, per_class_eval=r1.EVAL_PER_CLASS,
                           rung2_per_class=r2.EVAL_PER_CLASS)
    pool = frames["rung2"]

    episodes = SMOKE_EPISODES if args.smoke else EPISODES
    steps = SMOKE_STEPS if args.smoke else STEPS
    rng = random.Random(EPISODE_SEED)
    structure = build_structure(len(pool), rng, episodes=episodes, steps=steps)
    frames_dir = Path(args.frames_dir).expanduser()
    if args.smoke:
        frames_dir = frames_dir.with_name(frames_dir.name + "-smoke")
    rendered = (reuse_episodes(pool, structure, frames_dir) if args.reuse_frames else
                render_episodes(pool, structure, rng, frames_dir))
    manifest = manifest_sha(rendered)
    if args.reuse_frames and manifest != REGISTERED_EPISODE_SHA256:
        print(f"reused episode manifest {manifest} != registered "
              f"{REGISTERED_EPISODE_SHA256}; refusing to load models")
        raise SystemExit(2)
    diffs = pixel_diffs(rendered)
    emit("vm2t_episodes", episodes=episodes, steps=steps, manifest_sha256=manifest,
         cuts=sum(r["cut"] for rows in rendered for r in rows),
         pixel_diff_median=round(statistics.median(diffs.values()), 5))

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    detector, detector_record = r2y.load_detector(r2y.DETECTOR_PATH.with_name(f"{DETECTOR_NAME}.pt"), device)
    started = time.perf_counter()
    all_steps = [row for rows in rendered for row in rows]
    boxes = r2y.person_boxes_for(detector, all_steps, device)
    summaries = {image_id: box_summary(rows) for image_id, rows in boxes.items()}
    lookout_wall = time.perf_counter() - started
    detector_record["lookout"] = {"frames": len(all_steps), "wall_seconds": round(lookout_wall, 3),
                                  "ms_per_frame": round(1000 * lookout_wall / len(all_steps), 3)}
    verdict_by_label = {"present": 0, "absent": 0, "present_frames": 0, "absent_frames": 0}
    for row in all_steps:
        verdict_by_label[row["label"] + "_frames"] += 1
        if summaries[row["image_id"]]["verdict"] == "present":
            verdict_by_label[row["label"]] += 1
    detector_record["episode_verdicts"] = verdict_by_label
    emit("vm2t_detector", **{k: v for k, v in detector_record.items() if k != "verdicts"})
    del detector
    release_device_memory(device)

    hosts = list(HOST_ORDER) if args.host == "all" else [args.host]
    for host_key in hosts:
        run_host(args, host_key, seeds, rendered, summaries, diffs, detector_record, manifest, base)


if __name__ == "__main__":
    main()
