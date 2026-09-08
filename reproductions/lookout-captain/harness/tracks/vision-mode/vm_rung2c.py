"""V-line rung 2c: the blind captain — boxes on the bus where the host cannot see.

Registration: tracks/vision-mode/PLAN.md, "Rung 2c registration — the blind
captain" (2026-08-23). The host's image input is a registered uniform black
frame for EVERY training and eval row; the detector (yolo26n) sees the real
frames and supplies verdicts and box vectors. Everything else is rung 2b
amendment 2's harness: the frozen mode door, the zero-initialized projection
in the door's normalized source space, lr 3e-4, oracle-action targets; arms
BOXES / SWITCH / LYING / HABIT. New metric DELIVERY = the person-mode's
action share on present frames minus on absent frames — the bus's
discrimination with no eyes to check it. CAPTURE (lying absent-share >=
0.50) is recorded, not read: the blind host obeying the lie is the expected
confirmation that sight, not the bus, was rung 2's safety.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import vm_rung2b as r2b  # noqa: E402
import vm_rung1 as r1  # noqa: E402
import vm_rung2 as r2  # noqa: E402
import vm_rung2y as r2y  # noqa: E402
from vm_rung1 import (  # noqa: E402
    HOSTS, HOST_ORDER, MODE_ACTION, SEEDS, TRAIN_DOSE, FrameBank, atomic_json_dump, bank_features,
    cuda_snapshot, display_path, emit, git_head, layer_plan, load_door, load_host, mix,
    parameter_samples, release_device_memory, samples_unchanged, seed_everything, select_frames,
    sha256_of, share,
)

REGISTRATION = "tracks/vision-mode/PLAN.md, Rung 2c registration — the blind captain (2026-08-23)"
CANONICAL_STEM = "vision-mode-r2c"
PERSON_MODES = r2.PERSON_MODES
ARMS = r2b.ARMS  # boxes, switch, lying, habit
BLANK_SIZE = (640, 480)

#: readings, by number
DELIVERY_CARRIES = 0.50
DELIVERY_WEAK = 0.25
CAPTURE_FLOOR = 0.50


def blank_frame_path(directory: Path) -> tuple[Path, str]:
    """The registered blank: a uniform black PNG, written deterministically."""

    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "vm-r2c-blank.png"
    Image.new("RGB", BLANK_SIZE, (0, 0, 0)).save(path, format="PNG")
    return path, sha256_of(path)


def blinded(frames: Sequence[Mapping[str, Any]], blank: Path) -> list[dict[str, Any]]:
    return [{**frame, "file": str(blank)} for frame in frames]


def projection_path(host_key: str, seed: int) -> Path:
    return r1.ROOT / "checkpoints" / "experiments" / f"vm-r2c-box-projection-{host_key}-s{seed}.pt"


def arm_metrics(cells: Mapping[str, Sequence[Mapping[str, Any]]], person_mode: str) -> dict[str, Any]:
    action = MODE_ACTION[person_mode]
    present = {arm: share(cells[arm], "present", action) for arm in ARMS}
    absent = {arm: share(cells[arm], "absent", action) for arm in ARMS}
    return {
        "action": action, "present_share": present, "absent_share": absent,
        "present_mix": {arm: mix(cells[arm], "present") for arm in ARMS},
        "absent_mix": {arm: mix(cells[arm], "absent") for arm in ARMS},
        "delivery": {arm: round(present[arm] - absent[arm], 6) for arm in ARMS},
        "gain": round(present["boxes"] - present["switch"], 6),
        "capture": absent["lying"],
        "parse_fallback_share": {arm: round(sum(r["parse"] == "fallback" for r in cells[arm]) / len(cells[arm]), 6)
                                 for arm in ARMS},
    }


def reading_of(per_mode: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    switch_delivery = {m: per_mode[m]["delivery"]["switch"] for m in PERSON_MODES}
    boxes_delivery = {m: per_mode[m]["delivery"]["boxes"] for m in PERSON_MODES}
    capture = {m: per_mode[m]["capture"] for m in PERSON_MODES}
    gain = {m: per_mode[m]["gain"] for m in PERSON_MODES}
    if all(switch_delivery[m] >= DELIVERY_CARRIES for m in PERSON_MODES):
        reading = "(i) BUS CARRIES THE FACT"
    elif any(switch_delivery[m] < DELIVERY_WEAK for m in PERSON_MODES):
        reading = "(ii) BUS TOO WEAK"
    else:
        reading = "(iii) MIXED"
    return {"reading": reading, "switch_delivery": switch_delivery, "boxes_delivery": boxes_delivery,
            "gain": gain, "capture": capture,
            "captured_modes": [m for m in PERSON_MODES if capture[m] is not None and capture[m] >= CAPTURE_FLOOR],
            "bands": {"delivery_carries": DELIVERY_CARRIES, "delivery_weak": DELIVERY_WEAK,
                      "capture_floor": CAPTURE_FLOOR}}


def run_seed(args, *, host_key, seed, model, tower, processor, device, dtype, depth, hidden,
             blind_train, blind_eval, boxes, verdicts, detector_record, blank_record, bank,
             base_samples, plan) -> dict[str, Any]:
    started = time.perf_counter()
    tokenizer = processor.tokenizer
    seed_everything(seed, device)
    rng = random.Random(seed)
    source_layer, inject_layers, dose = plan["source_layer"], plan["inject_layers"], TRAIN_DOSE
    door, door_record = load_door(host_key, "mode", seed, hidden=hidden, inject_layers=inject_layers, device=device)
    constant_door, constant_record = load_door(host_key, "constant", seed, hidden=hidden,
                                               inject_layers=inject_layers, device=device)
    if door_record["source_layer"] != source_layer:
        raise SystemExit("the door's source layer is not this plan's")
    emit("vm2c_seed_start", host=host_key, seed=seed, doors={"mode": door_record["sha256"],
                                                             "constant": constant_record["sha256"]},
         **cuda_snapshot(device))
    features = bank_features(model, tokenizer, device, source_layer)
    if not samples_unchanged(model, base_samples):
        raise RuntimeError("the host moved during the mode-assignment taps")
    constant_feature = r1.train_tap_stack(features).mean(0)

    steps = r1.SMOKE_STEPS if args.smoke else r1.STEPS
    train_started = time.perf_counter()
    projection, history = r2b.train_projection(
        model=model, tower=tower, tokenizer=tokenizer, device=device, bank=bank, frames=blind_train,
        features=features, boxes=boxes, door=door, door_sha256=door_record["sha256"], hidden=hidden,
        inject_layers=inject_layers, dose=dose, steps=steps, rng=rng,
    )
    train_wall = time.perf_counter() - train_started
    if not samples_unchanged(model, base_samples):
        raise RuntimeError("the host moved during projection training")
    gate_frames = [f for f in blind_train if f["label"] == "present"][:4] + \
                  [f for f in blind_train if f["label"] == "absent"][:4]
    gate = r2b.held_out_gate(model=model, tower=tower, tokenizer=tokenizer, device=device, bank=bank,
                             frames=gate_frames, features=features, boxes=boxes, door=door,
                             projection=projection, inject_layers=inject_layers, dose=dose)
    emit("vm2c_trained", host=host_key, seed=seed, final=history[-1], gate=gate, wall=round(train_wall, 1))
    if args.smoke:
        projection_record: dict[str, Any] = {"path": None, "sha256": None, "note": "SMOKE: not saved"}
    else:
        projection_record = r2b.save_projection(projection, history, projection_path(host_key, seed),
                                                host_key=host_key, seed=seed, door_sha256=door_record["sha256"],
                                                hidden=hidden, inject_layers=inject_layers,
                                                source_layer=source_layer)
    projection_record.update({"parameters": r2b.BOX_DIM * hidden + hidden, "steps": steps,
                              "learning_rate": r2b.PROJECTION_LR,
                              "space": "the door's normalized source (unit RMS)",
                              "door_alone": history[0], "history": history, "gate": gate,
                              "weight_norm": round(float(projection.weight.norm()), 6)})

    lies = r2b.lie_boxes(blind_eval, boxes)
    decode_cache: dict[tuple, dict[str, Any]] = {}
    cells: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for person_mode in PERSON_MODES:
        cells[person_mode] = {}
        for arm in ARMS:
            cells[person_mode][arm] = r2b.run_arm(
                model=model, tower=tower, tokenizer=tokenizer, device=device, bank=bank, frames=blind_eval,
                arm=arm, person_mode=person_mode, features=features, boxes=boxes, lies=lies, verdicts=verdicts,
                door=door, projection=projection, constant_door=constant_door, constant_feature=constant_feature,
                inject_layers=inject_layers, dose=dose, decode_cache=decode_cache,
            )
            emit("vm2c_cell", host=host_key, seed=seed, person_mode=person_mode, arm=arm,
                 present_mix=mix(cells[person_mode][arm], "present"),
                 absent_mix=mix(cells[person_mode][arm], "absent"))
    if not samples_unchanged(model, base_samples):
        raise RuntimeError("the host moved during the cells")
    metrics = {m: arm_metrics(cells[m], m) for m in PERSON_MODES}
    reading = reading_of(metrics)
    emit("vm2c_seed_reading", host=host_key, seed=seed, reading=reading["reading"],
         switch_delivery=reading["switch_delivery"], boxes_delivery=reading["boxes_delivery"],
         capture=reading["capture"])
    return {
        "track": r1.TRACK, "experiment": f"{CANONICAL_STEM}-{host_key}-s{seed}", "registration": REGISTRATION,
        "reuses": [r1.REGISTRATION, r2b.REGISTRATION],
        "claim_scope": (f"rung 2c's four arms x two person-modes at {HOSTS[host_key]['label']} on the s{seed} "
                        f"door, the host blind; {len(blind_eval)} eval frames, one door seed"),
        "git_head": git_head(), "host_machine": platform.node(), "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "host": {"key": host_key, **HOSTS[host_key], "depth": depth, "hidden_size": hidden},
        "seed": seed, "smoke": bool(args.smoke), "layer_plan": plan,
        "doors": {"mode": door_record, "constant": constant_record, "dose": dose, "frozen": True},
        "projection": projection_record, "detector": detector_record, "blank": blank_record,
        "box_vector": "as rung 2b (vm_rung2b.box_vector)", "lie": {"rule": "as rung 2b"},
        "frames": {"train": len(blind_train), "eval": len(blind_eval),
                   "note": "every host input is the registered blank; labels and boxes are the real frames'"},
        "decode": {"greedy": True, "max_new_tokens": r1.MAX_NEW, "decodes": len(decode_cache)},
        "arms": cells, "metrics": metrics, "reading": reading,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }


def pooled_payload(host_key: str, per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pooled_metrics = {}
    for m in PERSON_MODES:
        rows = {arm: [row for p in per_seed for row in p["arms"][m][arm]] for arm in ARMS}
        pooled_metrics[m] = arm_metrics(rows, m)
    seed_readings = {str(p["seed"]): p["reading"]["reading"] for p in per_seed}
    votes: dict[str, int] = {}
    for value in seed_readings.values():
        votes[value] = votes.get(value, 0) + 1
    majority = max(votes.items(), key=lambda item: (item[1], item[0]))
    return {
        "track": r1.TRACK, "experiment": f"{CANONICAL_STEM}-{host_key}-pooled", "registration": REGISTRATION,
        "git_head": git_head(), "host_machine": platform.node(), "host": per_seed[0]["host"],
        "seeds": [p["seed"] for p in per_seed], "smoke": any(p["smoke"] for p in per_seed),
        "detector": per_seed[0]["detector"], "blank": per_seed[0]["blank"],
        "per_seed_readings": seed_readings,
        "majority_reading": {"reading": majority[0], "votes": majority[1], "of": len(per_seed),
                             "decisive": majority[1] * 2 > len(per_seed)},
        "pooled_metrics": pooled_metrics, "pooled_reading": reading_of(pooled_metrics),
        "per_seed": {str(p["seed"]): {k: p["reading"][k] for k in ("switch_delivery", "boxes_delivery", "gain", "capture")}
                     for p in per_seed},
        "wall_seconds": round(sum(p["wall_seconds"] for p in per_seed), 3),
    }


def fmt(value: Any) -> str:
    return "—" if value is None else (f"{value:.2f}" if isinstance(value, float) else str(value))


def table(metrics: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = ["| person-mode | action | boxes P/A | switch P/A | lying P/A | habit P/A | switch delivery | boxes delivery | capture |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in PERSON_MODES:
        x = metrics[m]; p = x["present_share"]; a = x["absent_share"]
        cells = " | ".join(f"{fmt(p[arm])}/{fmt(a[arm])}" for arm in ARMS)
        lines.append(f"| {m} | {x['action']} | {cells} | {fmt(x['delivery']['switch'])} | "
                     f"{fmt(x['delivery']['boxes'])} | {fmt(x['capture'])} |")
    return lines


def write_markdown(payload: Mapping[str, Any], path: Path, *, pooled: bool) -> None:
    host = payload["host"]["label"]
    title = (f"# V-line rung 2c — {host} blind, pooled over door seeds {payload['seeds']}" if pooled
             else f"# V-line rung 2c — {host} blind, door seed {payload['seed']}")
    lines = [title, ""]
    if payload["smoke"]:
        lines += ["**SMOKE ARTIFACT**", ""]
    lines += [f"Registration: {payload['registration']}.", "",
              f"The host sees only the registered blank ({payload['blank']['sha256'][:12]}…); "
              f"the detector sees the real frames.", ""]
    if pooled:
        maj = payload["majority_reading"]
        lines += [f"Per-seed readings: {json.dumps(payload['per_seed_readings'])} — majority **{maj['reading']}** "
                  f"({maj['votes']} of {maj['of']}). Pooled-number reading: {payload['pooled_reading']['reading']}.", ""]
        lines += table(payload["pooled_metrics"])
    else:
        lines += [f"Reading at this seed: **{payload['reading']['reading']}**.", ""]
        lines += table(payload["metrics"])
    lines += ["", "P/A = the person-mode's action share on present / absent frames. delivery = P − A (the bus's "
              "discrimination with no eyes); capture = the action's share on empty frames when a person frame's "
              "boxes ride the bus (the blind host obeying the lie — expected)."]
    path.write_text("\n".join(lines) + "\n")


def run_host(args, host_key, seeds, blind_train, blind_eval, boxes, verdicts, detector_record,
             blank_record, base):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    processor, model, tower, depth, hidden, dtype = load_host(host_key, device, local_files_only=args.local_files_only)
    plan = layer_plan(depth)
    emit("vm2c_host", host=host_key, depth=depth, plan=plan, **cuda_snapshot(device))
    bank = FrameBank(processor, device, dtype)
    base_samples = parameter_samples(model)
    per_seed = []
    for seed in seeds:
        payload = run_seed(args, host_key=host_key, seed=seed, model=model, tower=tower, processor=processor,
                           device=device, dtype=dtype, depth=depth, hidden=hidden, blind_train=blind_train,
                           blind_eval=blind_eval, boxes=boxes, verdicts=verdicts,
                           detector_record=detector_record, blank_record=blank_record, bank=bank,
                           base_samples=base_samples, plan=plan)
        prefix = base.with_name(f"{base.name}-{host_key}-s{seed}")
        atomic_json_dump(payload, prefix.with_suffix(".json"))
        write_markdown(payload, prefix.with_suffix(".md"), pooled=False)
        emit("vm2c_seed_written", host=host_key, seed=seed, json=display_path(prefix.with_suffix(".json")))
        per_seed.append(payload)
    pooled = pooled_payload(host_key, per_seed)
    prefix = base.with_name(f"{base.name}-{host_key}")
    atomic_json_dump(pooled, prefix.with_suffix(".json"))
    write_markdown(pooled, prefix.with_suffix(".md"), pooled=True)
    emit("vm2c_host_written", host=host_key, json=display_path(prefix.with_suffix(".json")),
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
                           rung2_per_class=r2.SMOKE_EVAL_PER_CLASS if args.smoke else r2.EVAL_PER_CLASS)
    train_frames = frames["train"]
    if args.smoke:
        train_frames = train_frames[:4] + train_frames[-4:]
    blank, blank_sha = blank_frame_path(Path(args.output_prefix).expanduser().parent if args.output_prefix
                                        else r1.ROOT / "checkpoints" / "experiments")
    blank_record = {"path": display_path(blank), "sha256": blank_sha, "size": list(BLANK_SIZE),
                    "color": [0, 0, 0]}
    emit("vm2c_frames", train=len(train_frames), eval=len(frames["rung2"]), blank=blank_record)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    detector, detector_record = r2y.load_detector(r2y.DETECTOR_PATH, device)
    real_frames = train_frames + frames["rung2"]
    raw = r2y.person_boxes_for(detector, real_frames, device)
    boxes = {image_id: r2b.box_vector(rows) for image_id, rows in raw.items()}
    verdicts = {}
    for frame in frames["rung2"]:
        row = r2y.verdict_of(raw[int(frame["image_id"])])
        row["correct"] = row["verdict"] == frame["label"]
        verdicts[int(frame["image_id"])] = row
    detector_record["eval"] = r2.confusion(frames["rung2"], verdicts)
    detector_record["verdicts"] = {str(k): v for k, v in verdicts.items()}
    emit("vm2c_detector", **{k: v for k, v in detector_record.items() if k not in ("verdicts",)})
    del detector
    release_device_memory(device)

    blind_train = blinded(train_frames, blank)
    blind_eval = blinded(frames["rung2"], blank)
    hosts = list(HOST_ORDER) if args.host == "all" else [args.host]
    for host_key in hosts:
        run_host(args, host_key, seeds, blind_train, blind_eval, boxes, verdicts, detector_record,
                 blank_record, base)


if __name__ == "__main__":
    main()
