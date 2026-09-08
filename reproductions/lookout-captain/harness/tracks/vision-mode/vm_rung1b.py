"""V-line rung 1b: the hostile prompt on the VL host.

Registration: tracks/vision-mode/PLAN.md, "Rung 1b registration — the
hostile prompt on the VL host (2026-08-22)". GF-H's six cells on rung 1's
doors, frames and decode contract: the client's prompt carries the
OPPONENT mode's phrasing while the operator's packet carries the mode.
Two dose rows (`--dose-scale 1` and `1.5`), two emitters (`--emitter
oracle`: the label picks the packet; `--emitter classifier`: rung 2's head;
`--emitter detector`: rung 2y's YOLO26n verdicts (rung 1b-y registration)
picks it). Nothing trains here.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import vm_rung1 as r1  # noqa: E402
import vm_rung2 as r2  # noqa: E402
from vm_rung1 import (  # noqa: E402
    HOSTS, HOST_ORDER, MODES, MODE_ACTION, PHRASES, SEEDS, TRAIN_DOSE, FrameBank,
    atomic_json_dump, bank_features, cuda_snapshot, display_path, emit,
    generate_with_packet, git_head, held_out_position, layer_plan, load_door,
    load_host, mix, parameter_samples, parse_action, payloads_of,
    release_device_memory, samples_unchanged, seed_everything, select_frames,
    share, train_tap_stack,
)

REGISTRATION = (
    "tracks/vision-mode/PLAN.md, Rung 1b registration — the hostile prompt on "
    "the VL host (2026-08-22)"
)
CANONICAL_STEM = "vision-mode-r1b"
OPPONENT = {"caution": "explore", "explore": "caution", "social": "explore"}
ARMS = ("packet", "hostile-text", "packet+hostile", "packet+congruent", "habit",
        "shuffled+hostile")
OWNERSHIP_GUARD = 0.05
FALLBACK_CEILING = 0.05
GROUNDING_CEILING = 0.20
HOLDS = 0.70
TEXT_WINS = 0.30
GROUNDED_MODES = ("social", "caution")


def scale_label(scale: float) -> str:
    return f"x{scale:g}".replace(".", "p")


def cell_key(arm: str, mode: str) -> tuple[str | None, str | None]:
    """(packet source mode or None for the constant, text source mode or None)."""

    other = OPPONENT[mode]
    return {
        "packet": (mode, None),
        "hostile-text": (None, other),
        "packet+hostile": (mode, other),
        "packet+congruent": (mode, mode),
        "habit": (None, None),
        "shuffled+hostile": (other, other),
    }[arm]


@torch.no_grad()
def run_cell(*, model, tower, tokenizer, device, bank, frames, key, emitter, verdicts,
             features, mode_bridge, constant_feature, constant_bridge, inject_layers,
             dose) -> list[dict[str, Any]]:
    eos = int(tokenizer.eos_token_id)
    packet_mode, text_mode = key
    rows: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        position = held_out_position(index)
        chosen = packet_mode
        if packet_mode is not None and emitter in ("classifier", "detector"):
            chosen = r2.policy_mode(packet_mode, verdicts[frame["image_id"]]["verdict"])
        if chosen is None:
            _p, payloads = payloads_of(constant_bridge, constant_feature, device)
        else:
            _p, payloads = payloads_of(mode_bridge, features[(chosen, position)], device)
        mode_text = None if text_mode is None else PHRASES[text_mode][position]
        new_ids = generate_with_packet(
            model, tower, bank.inputs(frame, mode_text=mode_text), payloads,
            inject_layers, dose, eos=eos,
        )
        decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
        action, how = parse_action(decoded)
        rows.append({"image_id": frame["image_id"], "label": frame["label"],
                     "position": position, "packet_mode": chosen, "text_mode": text_mode,
                     "verdict": verdicts[frame["image_id"]]["verdict"] if verdicts else None,
                     "decoded": decoded, "action": action, "parse": how})
    return rows


def ownership_of(shares: Mapping[str, float]) -> dict[str, Any]:
    numerator = shares["packet+hostile"] - shares["hostile-text"]
    denominator = shares["packet"] - shares["habit"]
    guarded = abs(denominator) < OWNERSHIP_GUARD
    value = None if guarded else round(numerator / denominator, 6)
    return {"value": value, "numerator": round(numerator, 6), "denominator": round(denominator, 6),
            "guard": OWNERSHIP_GUARD, "guarded": guarded,
            "reason": "no lean to own" if guarded else None,
            "definition": "(share(packet+hostile) - share(hostile-text)) / (share(packet) - share(habit))"}


def band_of(value: float | None) -> str | None:
    if value is None:
        return None
    if value >= HOLDS:
        return "packet-holds"
    if value < TEXT_WINS:
        return "text-wins"
    return "split"


def mode_metrics(cells: Mapping[str, Sequence[Mapping[str, Any]]], mode: str) -> dict[str, Any]:
    action = MODE_ACTION[mode]
    present = {arm: share(cells[arm], "present", action) for arm in ARMS}
    absent = {arm: share(cells[arm], "absent", action) for arm in ARMS}
    ownership = ownership_of(present)
    return {
        "action": action, "present_share": present, "absent_share": absent,
        "present_mix": {arm: mix(cells[arm], "present") for arm in ARMS},
        "absent_mix": {arm: mix(cells[arm], "absent") for arm in ARMS},
        "ownership": ownership, "band": band_of(ownership["value"]),
        "lean": round(present["packet"] - present["habit"], 6),
        "additivity": round(present["packet+congruent"] - present["packet"], 6),
        "grounding": absent["packet"] if mode in GROUNDED_MODES else None,
        "hostile_text_on_absent": absent["hostile-text"],
        "parse_fallback_share": {arm: round(sum(r["parse"] == "fallback" for r in cells[arm]) / len(cells[arm]), 6) for arm in ARMS},
    }


def admissibility(per_mode: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    fallback = max(v for m in per_mode for v in per_mode[m]["parse_fallback_share"].values())
    grounding = {m: per_mode[m]["grounding"] for m in GROUNDED_MODES}
    grounded = all(g is not None and g <= GROUNDING_CEILING for g in grounding.values())
    return {"max_parse_fallback_share": round(fallback, 6), "fallback_ceiling": FALLBACK_CEILING,
            "grounding": grounding, "grounding_ceiling": GROUNDING_CEILING,
            "admissible": fallback <= FALLBACK_CEILING and grounded}


def reading_of(per_mode: Mapping[str, Mapping[str, Any]], seeds_bands: Mapping[str, Mapping[str, str | None]] | None = None) -> dict[str, Any]:
    """Pooled-number bands per mode; with per-seed bands, the registered
    2-of-3 rule on the modes that have a lean."""

    bands = {m: per_mode[m]["band"] for m in MODES}
    ownership = {m: per_mode[m]["ownership"]["value"] for m in MODES}
    read_modes = [m for m in MODES if ownership[m] is not None]
    majority: dict[str, str | None] = {}
    if seeds_bands:
        for m in MODES:
            votes: dict[str, int] = {}
            for seed in seeds_bands:
                b = seeds_bands[seed].get(m)
                if b:
                    votes[b] = votes.get(b, 0) + 1
            top = max(votes.items(), key=lambda kv: (kv[1], kv[0]))[0] if votes else None
            majority[m] = top if top and (votes[top] >= 2 or len(seeds_bands) == 1) else None
    else:
        majority = dict(bands)
    read = [majority[m] for m in read_modes if majority.get(m)]
    if not read:
        reading = "no mode with a lean to own"
    elif all(b == "packet-holds" for b in read):
        reading = "(i) PACKET HOLDS"
    elif all(b == "split" for b in read):
        reading = "(ii) SPLIT"
    elif all(b == "text-wins" for b in read):
        reading = "(iii) TEXT WINS"
    else:
        reading = "(iv) ASYMMETRY: " + ", ".join(f"{m}={majority[m]}" for m in read_modes if majority.get(m))
    return {"reading": reading, "ownership": ownership, "pooled_band": bands,
            "majority_band": majority, "modes_with_lean": read_modes,
            "bands": {"holds": HOLDS, "text_wins": TEXT_WINS, "guard": OWNERSHIP_GUARD}}


def run_seed(args, *, host_key, seed, model, tower, processor, device, dtype, depth, hidden,
             frames, verdicts, bank, base_samples, plan, dose_scale, emitter,
             classifier_record, artifact_suffix) -> dict[str, Any]:
    started = time.perf_counter()
    tokenizer = processor.tokenizer
    seed_everything(seed, device)
    source_layer = plan["source_layer"]
    inject_layers = plan["inject_layers"]
    dose = TRAIN_DOSE * dose_scale
    mode_bridge, mode_record = load_door(host_key, "mode", seed, hidden=hidden,
                                         inject_layers=inject_layers, device=device)
    constant_bridge, constant_record = load_door(host_key, "constant", seed, hidden=hidden,
                                                 inject_layers=inject_layers, device=device)
    emit("vm1b_seed_start", host=host_key, seed=seed, dose=dose, dose_scale=dose_scale,
         emitter=emitter, **cuda_snapshot(device))
    features = bank_features(model, tokenizer, device, source_layer)
    if not samples_unchanged(model, base_samples):
        raise RuntimeError("the host moved during the mode-assignment taps")
    constant_feature = train_tap_stack(features).mean(0)

    eval_frames = frames["eval"]
    by_key: dict[tuple[str | None, str | None], list[dict[str, Any]]] = {}
    cells: dict[str, dict[str, list[dict[str, Any]]]] = {m: {} for m in MODES}
    for mode in MODES:
        for arm in ARMS:
            key = cell_key(arm, mode)
            if key not in by_key:
                by_key[key] = run_cell(
                    model=model, tower=tower, tokenizer=tokenizer, device=device, bank=bank,
                    frames=eval_frames, key=key, emitter=emitter, verdicts=verdicts,
                    features=features, mode_bridge=mode_bridge,
                    constant_feature=constant_feature, constant_bridge=constant_bridge,
                    inject_layers=inject_layers, dose=dose,
                )
                emit("vm1b_cell", host=host_key, seed=seed, packet=key[0], text=key[1],
                     present_mix=mix(by_key[key], "present"), absent_mix=mix(by_key[key], "absent"))
            cells[mode][arm] = by_key[key]
    if not samples_unchanged(model, base_samples):
        raise RuntimeError("the host moved during the cells")

    metrics = {m: mode_metrics(cells[m], m) for m in MODES}
    admissible = admissibility(metrics)
    reading = reading_of(metrics)
    emit("vm1b_seed_reading", host=host_key, seed=seed, dose_scale=dose_scale, emitter=emitter,
         reading=reading["reading"], ownership=reading["ownership"], admissible=admissible["admissible"],
         fallback=admissible["max_parse_fallback_share"], grounding=admissible["grounding"])
    experiment_suffix = (artifact_suffix if args.order_seed is not None or
                         args.constant_variant != "same" else "")
    payload = {
        "track": r1.TRACK,
        "experiment": (f"{CANONICAL_STEM}-{host_key}-{scale_label(dose_scale)}-"
                       f"{emitter}{experiment_suffix}-s{seed}"),
        "registration": REGISTRATION, "reuses": [r1.REGISTRATION, r2.REGISTRATION],
        "claim_scope": (f"rung 1b's six arms x three modes at {HOSTS[host_key]['label']} on the "
                        f"s{seed} doors, dose x{dose_scale:g}, emitter {emitter}; one door seed"),
        "git_head": git_head(), "host_machine": platform.node(), "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "host": {"key": host_key, **HOSTS[host_key], "depth": depth, "hidden_size": hidden},
        "seed": seed, "smoke": bool(args.smoke), "layer_plan": plan,
        "doors": {"mode": mode_record, "constant": constant_record, "train_dose": TRAIN_DOSE,
                  "dose": dose, "dose_scale": dose_scale, "trained_here": False},
        "emitter": emitter, "dose_scale": dose_scale, "door_stem": r1.DOOR_STEM,
        "classifier": classifier_record,
        "opponents": dict(OPPONENT), "frames": {"eval": len(eval_frames), "rule": frames["rule"]},
        "decode": {"greedy": True, "max_new_tokens": r1.MAX_NEW, "decoded_cells": len(by_key)},
        "arms": cells, "metrics": metrics, "admissibility": admissible, "reading": reading,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    if args.order_seed is not None or args.constant_variant != "same":
        payload.update({"order_seed": args.order_seed,
                        "constant_variant": args.constant_variant})
    return payload


def pooled_payload(host_key, per_seed, *, dose_scale, emitter, artifact_suffix):
    seeds = [p["seed"] for p in per_seed]
    pooled_metrics = {}
    for m in MODES:
        rows = {arm: [row for p in per_seed for row in p["arms"][m][arm]] for arm in ARMS}
        pooled_metrics[m] = mode_metrics(rows, m)
    seeds_bands = {str(p["seed"]): {m: p["metrics"][m]["band"] for m in MODES} for p in per_seed}
    reading = reading_of(pooled_metrics, seeds_bands)
    admissible = admissibility(pooled_metrics)
    experiment_suffix = artifact_suffix if "constant_variant" in per_seed[0] else ""
    payload = {
        "track": r1.TRACK,
        "experiment": (f"{CANONICAL_STEM}-{host_key}-{scale_label(dose_scale)}-"
                       f"{emitter}{experiment_suffix}-pooled"),
        "registration": REGISTRATION, "git_head": git_head(), "host_machine": platform.node(),
        "host": per_seed[0]["host"], "seeds": seeds, "smoke": any(p["smoke"] for p in per_seed),
        "dose_scale": dose_scale, "emitter": emitter, "classifier": per_seed[0]["classifier"],
        "per_seed_ownership": {str(p["seed"]): p["reading"]["ownership"] for p in per_seed},
        "per_seed_bands": seeds_bands,
        "per_seed_admissible": {str(p["seed"]): p["admissibility"]["admissible"] for p in per_seed},
        "pooled_metrics": pooled_metrics, "admissibility": admissible, "reading": reading,
        "wall_seconds": round(sum(p["wall_seconds"] for p in per_seed), 3),
    }
    if "constant_variant" in per_seed[0]:
        payload.update({"order_seed": per_seed[0]["order_seed"],
                        "constant_variant": per_seed[0]["constant_variant"]})
    return payload


def fmt(v):
    return "—" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def table(metrics):
    lines = ["| mode | action | packet | hostile text | packet+hostile | packet+congruent | habit | shuffled+hostile | ownership | band | grounding | hostile text on absent |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in MODES:
        x = metrics[m]; p = x["present_share"]
        lines.append(f"| {m} | {x['action']} | {fmt(p['packet'])} | {fmt(p['hostile-text'])} | {fmt(p['packet+hostile'])} | "
                     f"{fmt(p['packet+congruent'])} | {fmt(p['habit'])} | {fmt(p['shuffled+hostile'])} | "
                     f"{fmt(x['ownership']['value'])} | {x['band'] or '—'} | {fmt(x['grounding'])} | {fmt(x['hostile_text_on_absent'])} |")
    return lines


def write_markdown(payload, path, *, pooled):
    host = payload["host"]["label"]
    head = f"dose ×{payload['dose_scale']:g}, emitter {payload['emitter']}"
    title = (f"# V-line rung 1b — {host}, {head}, pooled over door seeds {payload['seeds']}" if pooled
             else f"# V-line rung 1b — {host}, {head}, door seed {payload['seed']}")
    lines = [title, ""]
    if payload["smoke"]:
        lines += ["**SMOKE ARTIFACT**", ""]
    lines += [f"Registration: {payload['registration']}.", "",
              f"Reading: **{payload['reading']['reading']}** — admissible: {payload['admissibility']['admissible']} "
              f"(max parse fallback {fmt(payload['admissibility']['max_parse_fallback_share'])}, grounding "
              f"{json.dumps(payload['admissibility']['grounding'])}).", ""]
    lines += table(payload["pooled_metrics"] if pooled else payload["metrics"])
    if pooled:
        lines += ["", "Per-seed ownership: " + json.dumps(payload["per_seed_ownership"]),
                  "Per-seed bands: " + json.dumps(payload["per_seed_bands"])]
    lines += ["", "Present-frame share of the mode's action per arm; ownership = (packet+hostile − hostile) / (packet − habit)."]
    path.write_text("\n".join(lines) + "\n")


def run_host(args, host_key, seeds, frames, base, *, dose_scale, emitter, verdicts,
             classifier_record, artifact_suffix):
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    processor, model, tower, depth, hidden, dtype = load_host(host_key, device, local_files_only=args.local_files_only)
    plan = layer_plan(depth)
    emit("vm1b_host", host=host_key, depth=depth, plan=plan, dose_scale=dose_scale, emitter=emitter,
         **cuda_snapshot(device))
    bank = FrameBank(processor, device, dtype)
    base_samples = parameter_samples(model)
    per_seed = []
    stem = f"{base.name}-{host_key}-{scale_label(dose_scale)}-{emitter}{artifact_suffix}"
    for seed in seeds:
        payload = run_seed(args, host_key=host_key, seed=seed, model=model, tower=tower,
                           processor=processor, device=device, dtype=dtype, depth=depth,
                           hidden=hidden, frames=frames, verdicts=verdicts, bank=bank,
                           base_samples=base_samples, plan=plan, dose_scale=dose_scale,
                           emitter=emitter, classifier_record=classifier_record,
                           artifact_suffix=artifact_suffix)
        prefix = base.with_name(f"{stem}-s{seed}")
        atomic_json_dump(payload, prefix.with_suffix(".json"))
        write_markdown(payload, prefix.with_suffix(".md"), pooled=False)
        emit("vm1b_seed_written", host=host_key, seed=seed, json=display_path(prefix.with_suffix(".json")))
        per_seed.append(payload)
    pooled = pooled_payload(host_key, per_seed, dose_scale=dose_scale, emitter=emitter,
                            artifact_suffix=artifact_suffix)
    prefix = base.with_name(stem)
    atomic_json_dump(pooled, prefix.with_suffix(".json"))
    write_markdown(pooled, prefix.with_suffix(".md"), pooled=True)
    emit("vm1b_host_written", host=host_key, json=display_path(prefix.with_suffix(".json")),
         reading=pooled["reading"]["reading"], admissible=pooled["admissibility"]["admissible"])
    del model, tower, bank
    release_device_memory(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=(*HOSTS, "all"), default="450m")
    parser.add_argument("--seed", type=int, default=None, choices=SEEDS)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--dose-scale", type=float, default=1.0)
    parser.add_argument("--emitter", choices=("oracle", "classifier", "detector"), default="oracle")
    parser.add_argument("--door-variant", choices=("registered", "x2", "mx", "mx2"), default="registered",
                        help="x2 loads vm-r1x2; mx loads vm-r1mx; each writes its matching "
                             "artifact family")
    parser.add_argument("--order-seed", type=int, choices=r1.SEEDS, default=None)
    parser.add_argument("--constant-variant", choices=("same", "registered"), default="same")
    parser.add_argument("--classifier-path", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--coco-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.all_seeds and args.seed is not None:
        raise SystemExit("--all-seeds runs every registered seed; drop --seed")
    if args.dose_scale <= 0:
        raise SystemExit("--dose-scale must be positive")
    if args.order_seed is not None and args.door_variant != "mx2":
        raise SystemExit("--order-seed requires --door-variant mx2")
    if args.constant_variant == "registered" and args.door_variant != "mx2":
        raise SystemExit("--constant-variant registered requires --door-variant mx2")
    if args.constant_variant == "registered" and args.order_seed is not None:
        raise SystemExit("--constant-variant registered cannot be combined with --order-seed")
    artifact_suffix = ""
    if args.door_variant != "registered":
        r1.DOOR_STEM = {"x2": "vm-r1x2", "mx": "vm-r1mx", "mx2": "vm-r1mx2"}[args.door_variant]
        artifact_suffix = {"x2": "-x2door", "mx": "-mxdoor",
                           "mx2": "-mx2door"}[args.door_variant]
    if args.order_seed is not None:
        r1.DOOR_STEM = f"vm-r1mx2o{args.order_seed}"
        artifact_suffix = f"-mx2o{args.order_seed}door"
    elif args.constant_variant == "registered":
        r1.CONSTANT_DOOR_STEM = "vm-r1"
        artifact_suffix = "-mx2ndoor"
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
    per_class_eval = r1.SMOKE_FRAMES if args.smoke else r1.EVAL_PER_CLASS
    per_class_train = r1.SMOKE_FRAMES if args.smoke else r1.TRAIN_PER_CLASS
    frames = select_frames(coco_dir, per_class_train=per_class_train, per_class_eval=per_class_eval)
    if args.smoke:
        #: the smoke frames are not rung 1's 30+30; the doors are still rung 1's
        emit("vm1b_smoke_note", note="smoke frames differ from the registered eval frames")

    verdicts = None
    classifier_record = None
    if args.emitter == "classifier":
        device = torch.device(args.device if args.device != "auto"
                              else ("cuda" if torch.cuda.is_available() else "cpu"))
        path = Path(args.classifier_path).expanduser() if args.classifier_path else r2.CLASSIFIER_PATH
        classifier, classifier_record = r2.load_classifier(path, device)
        verdicts = r2.verdicts_for(classifier, frames["eval"], device)
        classifier_record["eval"] = r2.confusion(frames["eval"], verdicts)
        classifier_record["verdicts"] = {str(k): v for k, v in verdicts.items()}
        emit("vm1b_classifier", **{k: v for k, v in classifier_record.items() if k != "verdicts"})
        del classifier
        release_device_memory(device)
    elif args.emitter == "detector":
        import vm_rung2y as r2y

        device = torch.device(args.device if args.device != "auto"
                              else ("cuda" if torch.cuda.is_available() else "cpu"))
        detector, classifier_record = r2y.load_detector(r2y.DETECTOR_PATH, device)
        verdicts = r2y.verdicts_for(detector, frames["eval"], device)
        classifier_record["eval"] = r2.confusion(frames["eval"], verdicts)
        classifier_record["verdicts"] = {str(k): v for k, v in verdicts.items()}
        emit("vm1b_detector", **{k: v for k, v in classifier_record.items() if k != "verdicts"})
        del detector
        release_device_memory(device)
    else:
        verdicts = {int(f["image_id"]): {"verdict": f["label"]} for f in frames["eval"]}

    hosts = list(HOST_ORDER) if args.host == "all" else [args.host]
    for host_key in hosts:
        run_host(args, host_key, seeds, frames, base, dose_scale=args.dose_scale,
                 emitter=args.emitter, verdicts=verdicts, classifier_record=classifier_record,
                 artifact_suffix=artifact_suffix)


if __name__ == "__main__":
    main()
