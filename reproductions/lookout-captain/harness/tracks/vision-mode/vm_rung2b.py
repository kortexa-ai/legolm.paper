"""V-line rung 2b: boxes on the bus.

Registration: tracks/vision-mode/PLAN.md, "Rung 2b registration — boxes on
the bus (2026-08-23)", as amended the same day after attempt 1 diverged.
The detector's boxes enter the door: the emitter is rung 1's mode tap, in
the door's NORMALIZED source space (unit RMS per dimension), PLUS a learned
projection of an 8-number box vector, through rung 1's mode door (loaded,
FROZEN); only the projection trains (lr 3e-4), on the oracle action of
(mode, label) — the policy is learned, not written. Step 1 of the training
history is the door alone (the projection is zero during it). Arms per person-mode: BOXES (the frame's boxes), SWITCH (rung
2y's verdict picks rung 1's packet), LYING BOXES (a present frame's boxes
on every empty frame, zeros on every person frame), HABIT (the constant
packet). The mode-only control is the door itself (b = 0).
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import vm_rung1 as r1  # noqa: E402
import vm_rung2 as r2  # noqa: E402
import vm_rung2y as r2y  # noqa: E402
from vm_rung1 import (  # noqa: E402
    HOSTS, HOST_ORDER, MODES, MODE_ACTION, SEEDS, TRAIN_DOSE, FrameBank, atomic_json_dump, bank_features,
    cuda_snapshot, display_path, emit, encode_template, generate_with_packet, git_head,
    held_out_position, layer_plan, load_door, load_host, mix, oracle_action, parameter_samples,
    parse_action, payloads_of, release_device_memory, samples_unchanged, seed_everything,
    select_frames, sha256_of, share, target_text, teacher_forced_loss,
)

REGISTRATION = "tracks/vision-mode/PLAN.md, Rung 2b registration — boxes on the bus (2026-08-23)"
CANONICAL_STEM = "vision-mode-r2b"
PERSON_MODES = r2.PERSON_MODES
ARMS = ("boxes", "switch", "lying", "habit")
BOX_DIM = 8
MAX_BOXES = 5
PROJECTION_FORMAT = "legolm-vision-mode-r2b-box-projection-v1"
LIE_DRAW_SEED = 2026

#: readings, by number
GAIN_HELP = 0.10
GAIN_ONE_BIT = 0.10
LIE_CLEAN = 0.20
LIE_OWNS = 0.50


# --------------------------------------------------------------------------
# the box vector and the emitter
# --------------------------------------------------------------------------

def box_vector(boxes: Sequence[Mapping[str, float]]) -> torch.Tensor:
    """[present by the rule, min(n, 5)/5, largest fraction, best confidence,
    cx, cy, w, h of the largest person box]; zeros with no box."""

    if not boxes:
        return torch.zeros(BOX_DIM)
    largest = max(boxes, key=lambda b: b["fraction"])
    verdict = r2y.verdict_of(boxes)
    return torch.tensor([
        1.0 if verdict["verdict"] == "present" else 0.0, min(len(boxes), MAX_BOXES) / MAX_BOXES,
        largest["fraction"], verdict["p_present"], largest["cx"], largest["cy"], largest["w"], largest["h"],
    ], dtype=torch.float32)


def new_projection(hidden: int, device: torch.device) -> nn.Linear:
    projection = nn.Linear(BOX_DIM, hidden).to(device=device, dtype=torch.float32)
    nn.init.zeros_(projection.weight)
    nn.init.zeros_(projection.bias)
    return projection


PROJECTION_LR = 3e-4  # rung 1's 3e-3 drifted the zero-init projection past the tap's scale (attempt 1)


def emitter_payloads(door, projection: nn.Linear, tap: torch.Tensor, box: torch.Tensor,
                     device: torch.device):
    """door.encode with the projection added AFTER the door's source
    normalization: gelu(emitter(normalize(tap) + projection(b))). With the
    projection at zero this is exactly rung 1's door."""

    source = door.normalize_source(tap.unsqueeze(0).to(device))
    packet = torch.nn.functional.gelu(door.emitter(source + projection(box.to(device)).unsqueeze(0)))
    return packet, door.deliver(packet)


def projection_path(host_key: str, seed: int) -> Path:
    return r1.ROOT / "checkpoints" / "experiments" / f"vm-r2b-box-projection-{host_key}-s{seed}.pt"


def train_projection(*, model, tower, tokenizer, device, bank, frames, features, boxes, door,
                     door_sha256, hidden, inject_layers, dose, steps, rng) -> tuple[nn.Linear, list[dict[str, Any]]]:
    """Rung 1's loop by number, with the door frozen and the projection the
    only parameter; the batch draws a mode, a frame and a training phrasing."""

    eos = int(tokenizer.eos_token_id)
    projection = new_projection(hidden, device)
    optimizer = torch.optim.AdamW(projection.parameters(), lr=PROJECTION_LR, weight_decay=0.0)
    history: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        accuracy = 0.0
        for _ in range(r1.BATCH):
            mode = rng.choice(MODES)
            frame = rng.choice(frames)
            tap = features[(mode, rng.randrange(r1.TRAIN_POSITIONS))]
            target_ids = encode_template(tokenizer, target_text(oracle_action(mode, frame["label"]))) + [eos]
            _packet, payloads = emitter_payloads(door, projection, tap, boxes[int(frame["image_id"])], device)
            loss, logits, targets = teacher_forced_loss(
                model, tower, bank.inputs(frame, mode_text=None), target_ids, payloads, inject_layers, dose,
            )
            (loss / r1.BATCH).backward()
            with torch.no_grad():
                accuracy += float((logits.argmax(-1) == targets).float().mean()) / r1.BATCH
            loss_total += float(loss.detach()) / r1.BATCH
        grad_norm = torch.nn.utils.clip_grad_norm_(projection.parameters(), r1.CLIP_NORM)
        optimizer.step()
        history.append({"step": step, "loss": round(loss_total, 4), "token_accuracy": round(accuracy, 4),
                        "grad_norm": round(float(grad_norm), 4)})
    projection.eval()
    for parameter in projection.parameters():
        parameter.requires_grad_(False)
    return projection, history


@torch.no_grad()
def held_out_gate(*, model, tower, tokenizer, device, bank, frames, features, boxes, door,
                  projection, inject_layers, dose) -> dict[str, Any]:
    eos = int(tokenizer.eos_token_id)
    per_mode: dict[str, float] = {}
    for mode in MODES:
        hits = 0.0
        count = 0
        for position in r1.HELD_OUT:
            for frame in frames:
                _packet, payloads = emitter_payloads(door, projection, features[(mode, position)],
                                                     boxes[int(frame["image_id"])], device)
                target_ids = encode_template(tokenizer, target_text(oracle_action(mode, frame["label"]))) + [eos]
                _loss, logits, targets = teacher_forced_loss(
                    model, tower, bank.inputs(frame, mode_text=None), target_ids, payloads, inject_layers, dose,
                )
                hits += float((logits.argmax(-1) == targets).float().mean())
                count += 1
        per_mode[mode] = round(hits / count, 4) if count else None
    pooled = round(statistics.fmean(per_mode.values()), 4)
    return {"per_mode": per_mode, "pooled": pooled, "passed": pooled >= r1.GATE_MIN_HELD_OUT_ACCURACY,
            "threshold": r1.GATE_MIN_HELD_OUT_ACCURACY, "frames": len(frames), "positions": list(r1.HELD_OUT)}


def save_projection(projection: nn.Linear, history, path: Path, *, host_key, seed, door_sha256,
                    hidden, inject_layers, source_layer) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": PROJECTION_FORMAT, "host": HOSTS[host_key]["model_id"], "seed": seed,
                "door_sha256": door_sha256, "hidden_size": hidden, "inject_layers": list(inject_layers),
                "source_layer": source_layer, "box_dim": BOX_DIM, "state_dict": projection.state_dict(),
                "history": history}, path)
    return {"path": display_path(path), "sha256": sha256_of(path)}


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------

def lie_boxes(eval_frames: Sequence[Mapping[str, Any]], boxes: Mapping[int, torch.Tensor]) -> dict[int, dict[str, Any]]:
    """image_id -> {box, source}: a present frame's vector (drawn by the fixed
    rng) on every absent frame, zeros on every present frame."""

    rng = random.Random(LIE_DRAW_SEED)
    present = [int(f["image_id"]) for f in eval_frames if f["label"] == "present"]
    out: dict[int, dict[str, Any]] = {}
    for frame in eval_frames:
        image_id = int(frame["image_id"])
        if frame["label"] == "present":
            out[image_id] = {"box": torch.zeros(BOX_DIM), "source": None}
        else:
            source = rng.choice(present)
            out[image_id] = {"box": boxes[source], "source": source}
    return out


@torch.no_grad()
def run_arm(*, model, tower, tokenizer, device, bank, frames, arm, person_mode, features, boxes, lies,
            verdicts, door, projection, constant_door, constant_feature, inject_layers, dose,
            decode_cache) -> list[dict[str, Any]]:
    eos = int(tokenizer.eos_token_id)
    rows: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        image_id = int(frame["image_id"])
        position = held_out_position(index)
        if arm == "habit":
            key: tuple = (image_id, "habit")
            builder = lambda: payloads_of(constant_door, constant_feature, device)[1]  # noqa: E731
        elif arm == "switch":
            chosen = r2.policy_mode(person_mode, verdicts[image_id]["verdict"])
            key = (image_id, "switch", chosen)
            builder = lambda chosen=chosen: payloads_of(door, features[(chosen, position)], device)[1]  # noqa: E731
        else:
            box = boxes[image_id] if arm == "boxes" else lies[image_id]["box"]
            key = (image_id, arm, person_mode)
            builder = lambda box=box: emitter_payloads(  # noqa: E731
                door, projection, features[(person_mode, position)], box, device)[1]
        if key not in decode_cache:
            new_ids = generate_with_packet(model, tower, bank.inputs(frame, mode_text=None), builder(),
                                           inject_layers, dose, eos=eos)
            decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
            action, how = parse_action(decoded)
            decode_cache[key] = {"decoded": decoded, "action": action, "parse": how}
        rows.append({"image_id": image_id, "label": frame["label"], "verdict": verdicts[image_id]["verdict"],
                     "position": position, "lie_source": lies[image_id]["source"] if arm == "lying" else None,
                     **decode_cache[key]})
    return rows


def arm_metrics(cells: Mapping[str, Sequence[Mapping[str, Any]]], person_mode: str) -> dict[str, Any]:
    action = MODE_ACTION[person_mode]
    present = {arm: share(cells[arm], "present", action) for arm in ARMS}
    absent = {arm: share(cells[arm], "absent", action) for arm in ARMS}
    return {
        "action": action, "present_share": present, "absent_share": absent,
        "present_mix": {arm: mix(cells[arm], "present") for arm in ARMS},
        "absent_mix": {arm: mix(cells[arm], "absent") for arm in ARMS},
        "gain": round(present["boxes"] - present["switch"], 6),
        "lie_cell": absent["lying"],
        "silence_cost": round(present["boxes"] - present["lying"], 6),
        "boxes_grounding": absent["boxes"],
        "switch_miss_cost": round(present["switch"] - present["habit"], 6),
        "parse_fallback_share": {arm: round(sum(r["parse"] == "fallback" for r in cells[arm]) / len(cells[arm]), 6) for arm in ARMS},
    }


def reading_of(per_mode: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    gain = {m: per_mode[m]["gain"] for m in PERSON_MODES}
    lie = {m: per_mode[m]["lie_cell"] for m in PERSON_MODES}
    clean = all(lie[m] <= LIE_CLEAN for m in PERSON_MODES)
    if all(gain[m] >= GAIN_HELP for m in PERSON_MODES) and clean:
        reading = "(i) BOXES HELP"
    elif any(lie[m] > LIE_OWNS for m in PERSON_MODES):
        reading = "(ii) CHANNEL OWNS"
    elif all(abs(gain[m]) <= GAIN_ONE_BIT for m in PERSON_MODES) and clean:
        reading = "(iii) ONE BIT"
    else:
        reading = "(iv) MIXED"
    return {"reading": reading, "gain": gain, "lie_cell": lie,
            "silence_cost": {m: per_mode[m]["silence_cost"] for m in PERSON_MODES},
            "boxes_grounding": {m: per_mode[m]["boxes_grounding"] for m in PERSON_MODES},
            "lie_clean": clean,
            "bands": {"gain_help": GAIN_HELP, "gain_one_bit": GAIN_ONE_BIT, "lie_clean": LIE_CLEAN, "lie_owns": LIE_OWNS}}


# --------------------------------------------------------------------------
# one (host, seed)
# --------------------------------------------------------------------------

def run_seed(args, *, host_key, seed, model, tower, processor, device, dtype, depth, hidden, frames,
             boxes, verdicts, detector_record, bank, base_samples, plan) -> dict[str, Any]:
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
    emit("vm2b_seed_start", host=host_key, seed=seed, doors={"mode": door_record["sha256"],
                                                             "constant": constant_record["sha256"]},
         **cuda_snapshot(device))
    features = bank_features(model, tokenizer, device, source_layer)
    if not samples_unchanged(model, base_samples):
        raise RuntimeError("the host moved during the mode-assignment taps")
    constant_feature = r1.train_tap_stack(features).mean(0)

    train_frames = frames["train"]
    steps = r1.SMOKE_STEPS if args.smoke else r1.STEPS
    train_started = time.perf_counter()
    projection, history = train_projection(
        model=model, tower=tower, tokenizer=tokenizer, device=device, bank=bank, frames=train_frames,
        features=features, boxes=boxes, door=door, door_sha256=door_record["sha256"], hidden=hidden,
        inject_layers=inject_layers, dose=dose, steps=steps, rng=rng,
    )
    train_wall = time.perf_counter() - train_started
    if not samples_unchanged(model, base_samples):
        raise RuntimeError("the host moved during projection training")
    gate_frames = [f for f in train_frames if f["label"] == "present"][:4] + \
                  [f for f in train_frames if f["label"] == "absent"][:4]
    gate = held_out_gate(model=model, tower=tower, tokenizer=tokenizer, device=device, bank=bank,
                         frames=gate_frames, features=features, boxes=boxes, door=door, projection=projection,
                         inject_layers=inject_layers, dose=dose)
    emit("vm2b_trained", host=host_key, seed=seed, final=history[-1], gate=gate, wall=round(train_wall, 1))
    if args.smoke:
        projection_record: dict[str, Any] = {"path": None, "sha256": None, "note": "SMOKE: not saved"}
    else:
        projection_record = save_projection(projection, history, projection_path(host_key, seed), host_key=host_key,
                                            seed=seed, door_sha256=door_record["sha256"], hidden=hidden,
                                            inject_layers=inject_layers, source_layer=source_layer)
    projection_record.update({"parameters": BOX_DIM * hidden + hidden, "steps": steps, "batch": r1.BATCH,
                              "learning_rate": PROJECTION_LR, "space": "the door's normalized source (unit RMS)",
                              "door_alone": history[0], "history": history, "gate": gate,
                              "weight_norm": round(float(projection.weight.norm()), 6)})

    eval_frames = frames["rung2"]
    lies = lie_boxes(eval_frames, boxes)
    decode_cache: dict[tuple, dict[str, Any]] = {}
    cells: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for person_mode in PERSON_MODES:
        cells[person_mode] = {}
        for arm in ARMS:
            cells[person_mode][arm] = run_arm(
                model=model, tower=tower, tokenizer=tokenizer, device=device, bank=bank, frames=eval_frames,
                arm=arm, person_mode=person_mode, features=features, boxes=boxes, lies=lies, verdicts=verdicts,
                door=door, projection=projection, constant_door=constant_door, constant_feature=constant_feature,
                inject_layers=inject_layers, dose=dose, decode_cache=decode_cache,
            )
            emit("vm2b_cell", host=host_key, seed=seed, person_mode=person_mode, arm=arm,
                 present_mix=mix(cells[person_mode][arm], "present"), absent_mix=mix(cells[person_mode][arm], "absent"))
    if not samples_unchanged(model, base_samples):
        raise RuntimeError("the host moved during the cells")
    metrics = {m: arm_metrics(cells[m], m) for m in PERSON_MODES}
    reading = reading_of(metrics)
    emit("vm2b_seed_reading", host=host_key, seed=seed, reading=reading["reading"], gain=reading["gain"],
         lie_cell=reading["lie_cell"], silence_cost=reading["silence_cost"])
    return {
        "track": r1.TRACK, "experiment": f"{CANONICAL_STEM}-{host_key}-s{seed}", "registration": REGISTRATION,
        "reuses": [r1.REGISTRATION, r2.REGISTRATION, r2y.REGISTRATION],
        "claim_scope": (f"rung 2b's four arms x two person-modes at {HOSTS[host_key]['label']} on the s{seed} "
                        f"door; {len(eval_frames)} eval frames, one door seed"),
        "git_head": git_head(), "host_machine": platform.node(), "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "host": {"key": host_key, **HOSTS[host_key], "depth": depth, "hidden_size": hidden},
        "seed": seed, "smoke": bool(args.smoke), "layer_plan": plan,
        "doors": {"mode": door_record, "constant": constant_record, "dose": dose, "frozen": True},
        "projection": projection_record, "detector": detector_record,
        "box_vector": "[present, min(n,5)/5, largest fraction, best confidence, cx, cy, w, h of the largest person box]",
        "lie": {"rule": "absent frames: a present eval frame's vector by random.Random(%d); present frames: zeros" % LIE_DRAW_SEED},
        "frames": {"train": len(train_frames), "eval": len(eval_frames), "rule": frames["rule"], "pool": frames["pool"]},
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
        "detector": per_seed[0]["detector"],
        "per_seed_readings": seed_readings,
        "majority_reading": {"reading": majority[0], "votes": majority[1], "of": len(per_seed),
                             "decisive": majority[1] * 2 > len(per_seed)},
        "pooled_metrics": pooled_metrics, "pooled_reading": reading_of(pooled_metrics),
        "per_seed": {str(p["seed"]): {**{k: p["reading"][k] for k in ("gain", "lie_cell", "silence_cost")},
                                      "gate": p["projection"]["gate"]["pooled"]} for p in per_seed},
        "wall_seconds": round(sum(p["wall_seconds"] for p in per_seed), 3),
    }


def fmt(value: Any) -> str:
    return "—" if value is None else (f"{value:.2f}" if isinstance(value, float) else str(value))


def table(metrics: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = ["| person-mode | action | boxes | switch | lying | habit | gain | lie cell | silence cost | boxes on empty |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m in PERSON_MODES:
        x = metrics[m]; p = x["present_share"]
        lines.append(f"| {m} | {x['action']} | {fmt(p['boxes'])} | {fmt(p['switch'])} | {fmt(p['lying'])} | {fmt(p['habit'])} | "
                     f"{fmt(x['gain'])} | {fmt(x['lie_cell'])} | {fmt(x['silence_cost'])} | {fmt(x['boxes_grounding'])} |")
    return lines


def write_markdown(payload: Mapping[str, Any], path: Path, *, pooled: bool) -> None:
    host = payload["host"]["label"]
    title = (f"# V-line rung 2b — {host}, pooled over door seeds {payload['seeds']}" if pooled
             else f"# V-line rung 2b — {host}, door seed {payload['seed']}")
    lines = [title, ""]
    if payload["smoke"]:
        lines += ["**SMOKE ARTIFACT**", ""]
    lines += [f"Registration: {payload['registration']}.", ""]
    d = payload["detector"]
    lines += [f"Detector: {d['training']['backbone']}; eval accuracy {fmt(d['eval']['accuracy'])} "
              f"({d['eval']['false_alarms']} false alarms, {d['eval']['misses']} misses of {d['eval']['frames']}).", ""]
    if pooled:
        maj = payload["majority_reading"]
        lines += [f"Per-seed readings: {json.dumps(payload['per_seed_readings'])} — majority **{maj['reading']}** "
                  f"({maj['votes']} of {maj['of']}). Pooled-number reading: {payload['pooled_reading']['reading']}.",
                  f"Per-seed gates: {json.dumps({s: v['gate'] for s, v in payload['per_seed'].items()})}.", ""]
        lines += table(payload["pooled_metrics"])
    else:
        g = payload["projection"]["gate"]
        lines += [f"Reading at this seed: **{payload['reading']['reading']}**. Projection gate {fmt(g['pooled'])} "
                  f"({'passed' if g['passed'] else 'FAILED'}); weight norm {fmt(payload['projection']['weight_norm'])}.", ""]
        lines += table(payload["metrics"])
    lines += ["", "Present-frame share of the person-mode's action per arm. gain = boxes − switch; lie cell = the "
              "action's share on ABSENT frames when a person frame's boxes ride the bus; silence cost = boxes − "
              "lying on present frames (zeros on the bus); boxes on empty = the BOXES arm's action share on absent frames."]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def run_host(args, host_key, seeds, frames, boxes, verdicts, detector_record, base):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    processor, model, tower, depth, hidden, dtype = load_host(host_key, device, local_files_only=args.local_files_only)
    plan = layer_plan(depth)
    emit("vm2b_host", host=host_key, depth=depth, plan=plan, **cuda_snapshot(device))
    bank = FrameBank(processor, device, dtype)
    base_samples = parameter_samples(model)
    per_seed = []
    for seed in seeds:
        payload = run_seed(args, host_key=host_key, seed=seed, model=model, tower=tower, processor=processor,
                           device=device, dtype=dtype, depth=depth, hidden=hidden, frames=frames, boxes=boxes,
                           verdicts=verdicts, detector_record=detector_record, bank=bank,
                           base_samples=base_samples, plan=plan)
        prefix = base.with_name(f"{base.name}-{host_key}-s{seed}")
        atomic_json_dump(payload, prefix.with_suffix(".json"))
        write_markdown(payload, prefix.with_suffix(".md"), pooled=False)
        emit("vm2b_seed_written", host=host_key, seed=seed, json=display_path(prefix.with_suffix(".json")))
        per_seed.append(payload)
    pooled = pooled_payload(host_key, per_seed)
    prefix = base.with_name(f"{base.name}-{host_key}")
    atomic_json_dump(pooled, prefix.with_suffix(".json"))
    write_markdown(pooled, prefix.with_suffix(".md"), pooled=True)
    emit("vm2b_host_written", host=host_key, json=display_path(prefix.with_suffix(".json")),
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
    if args.smoke:
        frames["train"] = frames["train"][:4] + frames["train"][-4:]
    emit("vm2b_frames", train=len(frames["train"]), eval=len(frames["rung2"]))

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    detector, detector_record = r2y.load_detector(r2y.DETECTOR_PATH, device)
    all_frames = frames["train"] + frames["rung2"]
    raw = r2y.person_boxes_for(detector, all_frames, device)
    boxes = {image_id: box_vector(rows) for image_id, rows in raw.items()}
    verdicts = {}
    for frame in frames["rung2"]:
        row = r2y.verdict_of(raw[int(frame["image_id"])])
        row["correct"] = row["verdict"] == frame["label"]
        verdicts[int(frame["image_id"])] = row
    detector_record["eval"] = r2.confusion(frames["rung2"], verdicts)
    detector_record["verdicts"] = {str(k): v for k, v in verdicts.items()}
    detector_record["box_vectors"] = {str(k): [round(x, 4) for x in v.tolist()] for k, v in boxes.items()}
    emit("vm2b_detector", **{k: v for k, v in detector_record.items() if k not in ("verdicts", "box_vectors")})
    del detector
    release_device_memory(device)

    hosts = list(HOST_ORDER) if args.host == "all" else [args.host]
    for host_key in hosts:
        run_host(args, host_key, seeds, frames, boxes, verdicts, detector_record, base)


if __name__ == "__main__":
    main()
