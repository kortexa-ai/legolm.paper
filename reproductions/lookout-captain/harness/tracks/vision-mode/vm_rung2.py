"""V-line rung 2: the fast classifier emits the mode.

Registration: tracks/vision-mode/PLAN.md, "Rung 2 registration — the fast
classifier emits the mode (2026-08-22)". A MobileNetV3-small head (the
small model, ~1 ms a frame) says PRESENT / ABSENT; a policy turns the
verdict into one of rung 1's packets (the operator's person-mode on
"present", explore on "absent"); the VL host looks for itself. Rung 2 is a
SWITCH over rung 1's doors — nothing trains but the classifier head.

Arms per person-mode: ORACLE (the label drives the policy), CLASSIFIER
(the verdict drives it), FLIPPED (the verdict inverted on every frame —
the wrong-classifier cell at full size), HABIT (the constant packet).
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
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import vm_rung1 as r1  # noqa: E402
from vm_rung1 import (  # noqa: E402
    HOSTS, HOST_ORDER, MODES, MODE_ACTION, SEEDS, TRAIN_DOSE, FrameBank, atomic_json_dump, bank_features,
    cuda_snapshot, display_path, emit, generate_with_packet, git_head,
    held_out_position, layer_plan, load_door, load_host, mix, parse_action,
    payloads_of, release_device_memory, samples_unchanged, parameter_samples,
    seed_everything, select_frames, sha256_of, share, train_tap_stack,
)

REGISTRATION = (
    "tracks/vision-mode/PLAN.md, Rung 2 registration — the fast classifier "
    "emits the mode (2026-08-22)"
)
CANONICAL_STEM = "vision-mode-r2"
RUNG_LABEL = "rung 2"  # rung 2y rebinds this, CANONICAL_STEM, REGISTRATION and REUSES
REUSES: Any = r1.REGISTRATION
PERSON_MODES = ("social", "caution")
ABSENT_MODE = "explore"
ARMS = ("oracle", "classifier", "flipped", "habit")
EVAL_PER_CLASS = 100
SMOKE_EVAL_PER_CLASS = 3

#: the classifier, by number
CLASSIFIER_SEED = 42
CLASSIFIER_EPOCHS = 3
CLASSIFIER_BATCH = 64
CLASSIFIER_LR = 1e-3
CLASSIFIER_IMAGE = 224
CLASSIFIER_PATH = r1.ROOT / "checkpoints" / "experiments" / "vm-r2-person-head.pt"
CLASSIFIER_FORMAT = "legolm-vision-mode-r2-person-head-v1"
LABEL_INDEX = {"absent": 0, "present": 1}
INDEX_LABEL = {index: label for label, index in LABEL_INDEX.items()}

#: readings, by number
FIDELITY_BAND = 0.10
FALSE_ALARM_CLEAN = 0.20
FALSE_ALARM_CAPTURED = 0.50
INERT_BAND = 0.10
ORACLE_LEAN_FLOOR = 0.25


# --------------------------------------------------------------------------
# the classifier
# --------------------------------------------------------------------------

class PersonHead(nn.Module):
    """MobileNetV3-small's frozen ImageNet features -> a linear 2-way head."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.pool = backbone.avgpool
        self.head = nn.Linear(576, 2)
        self.transforms = weights.transforms()
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        self.features.eval()

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.flatten(self.pool(self.features(images)), 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(images))


def frame_tensor(classifier: PersonHead, path: str) -> torch.Tensor:
    from PIL import Image

    return classifier.transforms(Image.open(path).convert("RGB"))


@torch.no_grad()
def embed_frames(classifier: PersonHead, frames: Sequence[Mapping[str, Any]],
                 device: torch.device, batch: int = 64) -> torch.Tensor:
    out: list[torch.Tensor] = []
    for start in range(0, len(frames), batch):
        chunk = torch.stack([frame_tensor(classifier, f["file"]) for f in frames[start:start + batch]])
        out.append(classifier.embed(chunk.to(device)).float().cpu())
    return torch.cat(out) if out else torch.zeros(0, 576)


def train_classifier(frames: Sequence[Mapping[str, Any]], device: torch.device,
                     *, epochs: int) -> tuple[PersonHead, dict[str, Any]]:
    """The head on the frozen features; balanced by per-epoch subsampling of
    the larger class; one fixed seed — the classifier is one object shared by
    every host and door seed."""

    torch.manual_seed(CLASSIFIER_SEED)
    rng = random.Random(CLASSIFIER_SEED)
    classifier = PersonHead().to(device)
    started = time.perf_counter()
    features = embed_frames(classifier, frames, device)
    labels = torch.tensor([LABEL_INDEX[f["label"]] for f in frames])
    by_class = {index: [i for i, label in enumerate(labels.tolist()) if label == index]
                for index in (0, 1)}
    smallest = min(len(rows) for rows in by_class.values())
    optimizer = torch.optim.Adam(classifier.head.parameters(), lr=CLASSIFIER_LR)
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        chosen = [i for index in (0, 1) for i in rng.sample(by_class[index], smallest)]
        rng.shuffle(chosen)
        total = 0.0
        hits = 0
        for start in range(0, len(chosen), CLASSIFIER_BATCH):
            rows = chosen[start:start + CLASSIFIER_BATCH]
            x = features[rows].to(device)
            y = labels[rows].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier.head(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(rows)
            hits += int((logits.argmax(-1) == y).sum())
        history.append({"epoch": epoch, "loss": round(total / len(chosen), 4),
                        "train_accuracy": round(hits / len(chosen), 4),
                        "examples": len(chosen)})
    classifier.eval()
    return classifier, {
        "backbone": "torchvision mobilenet_v3_small IMAGENET1K_V1, frozen",
        "head": "Linear(576, 2)", "parameters_trained": 576 * 2 + 2,
        "train_frames": {"present": len(by_class[1]), "absent": len(by_class[0])},
        "balanced_per_epoch": smallest * 2, "epochs": epochs,
        "batch": CLASSIFIER_BATCH, "learning_rate": CLASSIFIER_LR,
        "seed": CLASSIFIER_SEED, "image_size": CLASSIFIER_IMAGE,
        "history": history, "wall_seconds": round(time.perf_counter() - started, 3),
    }


def save_classifier(classifier: PersonHead, training: Mapping[str, Any], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": CLASSIFIER_FORMAT, "head_state_dict": classifier.head.state_dict(),
                "training": dict(training)}, path)
    return {"path": display_path(path), "sha256": sha256_of(path)}


def load_classifier(path: Path, device: torch.device) -> tuple[PersonHead, dict[str, Any]]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format") != CLASSIFIER_FORMAT:
        raise SystemExit(f"{display_path(path)} is not a rung 2 classifier")
    classifier = PersonHead().to(device)
    classifier.head.load_state_dict(saved["head_state_dict"])
    classifier.eval()
    return classifier, {"path": display_path(path), "sha256": sha256_of(path),
                        "training": saved["training"]}


@torch.no_grad()
def verdicts_for(classifier: PersonHead, frames: Sequence[Mapping[str, Any]],
                 device: torch.device) -> dict[int, dict[str, Any]]:
    """image_id -> {verdict, p_present, correct} on `frames`."""

    features = embed_frames(classifier, frames, device)
    probabilities = torch.softmax(classifier.head(features.to(device)).float(), -1).cpu()
    out: dict[int, dict[str, Any]] = {}
    for frame, row in zip(frames, probabilities):
        verdict = INDEX_LABEL[int(row.argmax())]
        out[int(frame["image_id"])] = {
            "verdict": verdict, "p_present": round(float(row[1]), 6),
            "correct": verdict == frame["label"],
        }
    return out


def confusion(frames: Sequence[Mapping[str, Any]], verdicts: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    table = {"present": {"present": 0, "absent": 0}, "absent": {"present": 0, "absent": 0}}
    for frame in frames:
        table[frame["label"]][verdicts[frame["image_id"]]["verdict"]] += 1
    total = len(frames)
    correct = table["present"]["present"] + table["absent"]["absent"]
    return {"label_x_verdict": table, "accuracy": round(correct / total, 6) if total else None,
            "false_alarms": table["absent"]["present"], "misses": table["present"]["absent"],
            "frames": total}


# --------------------------------------------------------------------------
# the policy and the arms
# --------------------------------------------------------------------------

def policy_mode(person_mode: str, verdict: str) -> str:
    return person_mode if verdict == "present" else ABSENT_MODE


def flipped(verdict: str) -> str:
    return "absent" if verdict == "present" else "present"


def packet_mode_for(arm: str, person_mode: str, frame: Mapping[str, Any],
                    verdicts: Mapping[int, Mapping[str, Any]]) -> str | None:
    """Which mode's packet this frame gets under `arm`; None = constant."""

    if arm == "habit":
        return None
    if arm == "oracle":
        return policy_mode(person_mode, frame["label"])
    verdict = verdicts[frame["image_id"]]["verdict"]
    if arm == "classifier":
        return policy_mode(person_mode, verdict)
    if arm == "flipped":
        return policy_mode(person_mode, flipped(verdict))
    raise ValueError(arm)


@torch.no_grad()
def run_arm(*, model, tower, tokenizer, device, bank, frames, arm, person_mode,
            verdicts, features, mode_bridge, constant_feature, constant_bridge,
            inject_layers, dose, decode_cache) -> list[dict[str, Any]]:
    """One arm's rows; decodes are cached by (frame, packet mode) so a frame
    that gets the same packet under two arms is decoded once."""

    eos = int(tokenizer.eos_token_id)
    rows: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        position = held_out_position(index)
        packet_mode = packet_mode_for(arm, person_mode, frame, verdicts)
        key = (frame["image_id"], packet_mode)
        if key not in decode_cache:
            if packet_mode is None:
                _p, payloads = payloads_of(constant_bridge, constant_feature, device)
            else:
                _p, payloads = payloads_of(mode_bridge, features[(packet_mode, position)], device)
            new_ids = generate_with_packet(
                model, tower, bank.inputs(frame, mode_text=None), payloads,
                inject_layers, dose, eos=eos,
            )
            decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
            action, how = parse_action(decoded)
            decode_cache[key] = {"decoded": decoded, "action": action, "parse": how}
        cached = decode_cache[key]
        rows.append({"image_id": frame["image_id"], "label": frame["label"],
                     "verdict": verdicts[frame["image_id"]]["verdict"],
                     "packet_mode": packet_mode, "position": position, **cached})
    return rows


def arm_metrics(cells: Mapping[str, Sequence[Mapping[str, Any]]], person_mode: str) -> dict[str, Any]:
    action = MODE_ACTION[person_mode]
    present = {arm: share(cells[arm], "present", action) for arm in ARMS}
    absent = {arm: share(cells[arm], "absent", action) for arm in ARMS}
    wrong = [row for row in cells["classifier"] if row["verdict"] != row["label"]]
    wrong_present = [row for row in wrong if row["label"] == "present"]
    wrong_absent = [row for row in wrong if row["label"] == "absent"]
    return {
        "action": action,
        "present_share": present, "absent_share": absent,
        "present_mix": {arm: mix(cells[arm], "present") for arm in ARMS},
        "absent_mix": {arm: mix(cells[arm], "absent") for arm in ARMS},
        "oracle_lean": round(present["oracle"] - present["habit"], 6),
        "fidelity": round(present["classifier"] - present["oracle"], 6),
        "false_alarm_cell": absent["flipped"],
        "miss_cell": present["flipped"],
        "miss_cost": round(present["oracle"] - present["flipped"], 6),
        "real_errors": {
            "misses": {"n": len(wrong_present),
                       "action_share": round(sum(r["action"] == action for r in wrong_present) / len(wrong_present), 6) if wrong_present else None},
            "false_alarms": {"n": len(wrong_absent),
                             "action_share": round(sum(r["action"] == action for r in wrong_absent) / len(wrong_absent), 6) if wrong_absent else None},
        },
        "parse_fallback_share": {arm: round(sum(r["parse"] == "fallback" for r in cells[arm]) / len(cells[arm]), 6) for arm in ARMS},
    }


def reading_of(per_mode: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    fidelity = {m: per_mode[m]["fidelity"] for m in PERSON_MODES}
    false_alarm = {m: per_mode[m]["false_alarm_cell"] for m in PERSON_MODES}
    oracle_lean = {m: per_mode[m]["oracle_lean"] for m in PERSON_MODES}
    classifier_vs_habit = {m: round(per_mode[m]["present_share"]["classifier"] - per_mode[m]["present_share"]["habit"], 6)
                           for m in PERSON_MODES}
    faithful = all(abs(fidelity[m]) <= FIDELITY_BAND for m in PERSON_MODES)
    clean = all(false_alarm[m] <= FALSE_ALARM_CLEAN for m in PERSON_MODES)
    captured = [m for m in PERSON_MODES if false_alarm[m] > FALSE_ALARM_CAPTURED]
    inert = all(abs(classifier_vs_habit[m]) <= INERT_BAND and oracle_lean[m] >= ORACLE_LEAN_FLOOR
                for m in PERSON_MODES)
    if faithful and clean:
        reading = "(i) ROBUST"
    elif captured:
        reading = "(ii) CAPTURED"
    elif inert:
        reading = "(iii) INERT"
    else:
        reading = "(iv) MIXED"
    return {"reading": reading, "fidelity": fidelity, "false_alarm": false_alarm,
            "miss_cost": {m: per_mode[m]["miss_cost"] for m in PERSON_MODES},
            "oracle_lean": oracle_lean, "classifier_vs_habit": classifier_vs_habit,
            "faithful": faithful, "grounding_clean": clean, "captured_modes": captured,
            "bands": {"fidelity": FIDELITY_BAND, "false_alarm_clean": FALSE_ALARM_CLEAN,
                      "false_alarm_captured": FALSE_ALARM_CAPTURED, "inert": INERT_BAND,
                      "oracle_lean_floor": ORACLE_LEAN_FLOOR}}


# --------------------------------------------------------------------------
# one (host, seed)
# --------------------------------------------------------------------------

def run_seed(args, *, host_key, seed, model, tower, processor, device, dtype, depth,
             hidden, frames, verdicts, classifier_record, bank, base_samples, plan) -> dict[str, Any]:
    started = time.perf_counter()
    tokenizer = processor.tokenizer
    seed_everything(seed, device)
    source_layer = plan["source_layer"]
    inject_layers = plan["inject_layers"]
    dose = TRAIN_DOSE

    mode_bridge, mode_record = load_door(host_key, "mode", seed, hidden=hidden,
                                         inject_layers=inject_layers, device=device)
    constant_bridge, constant_record = load_door(host_key, "constant", seed, hidden=hidden,
                                                 inject_layers=inject_layers, device=device)
    if mode_record["source_layer"] != source_layer:
        raise SystemExit("the door's source layer is not this plan's")
    emit("vm2_seed_start", host=host_key, seed=seed, doors={"mode": mode_record["sha256"],
                                                            "constant": constant_record["sha256"]},
         **cuda_snapshot(device))

    features = bank_features(model, tokenizer, device, source_layer)
    if not samples_unchanged(model, base_samples):
        raise RuntimeError("the host moved during the mode-assignment taps")
    constant_feature = train_tap_stack(features).mean(0)

    eval_frames = frames["rung2"]
    decode_cache: dict[tuple[int, str | None], dict[str, Any]] = {}
    cells: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for person_mode in PERSON_MODES:
        cells[person_mode] = {}
        for arm in ARMS:
            cells[person_mode][arm] = run_arm(
                model=model, tower=tower, tokenizer=tokenizer, device=device, bank=bank,
                frames=eval_frames, arm=arm, person_mode=person_mode, verdicts=verdicts,
                features=features, mode_bridge=mode_bridge, constant_feature=constant_feature,
                constant_bridge=constant_bridge, inject_layers=inject_layers, dose=dose,
                decode_cache=decode_cache,
            )
            emit("vm2_cell", host=host_key, seed=seed, person_mode=person_mode, arm=arm,
                 present_mix=mix(cells[person_mode][arm], "present"),
                 absent_mix=mix(cells[person_mode][arm], "absent"))
    if not samples_unchanged(model, base_samples):
        raise RuntimeError("the host moved during the cells")

    metrics = {m: arm_metrics(cells[m], m) for m in PERSON_MODES}
    reading = reading_of(metrics)
    emit("vm2_seed_reading", host=host_key, seed=seed, reading=reading["reading"],
         fidelity=reading["fidelity"], false_alarm=reading["false_alarm"],
         miss_cost=reading["miss_cost"])
    return {
        "track": r1.TRACK, "experiment": f"{CANONICAL_STEM}-{host_key}-s{seed}",
        "registration": REGISTRATION, "reuses": REUSES,
        "claim_scope": (f"rung 2's four arms x two person-modes at {HOSTS[host_key]['label']} "
                        f"on the s{seed} doors; {len(eval_frames)} eval frames, one door seed"),
        "git_head": git_head(), "host_machine": platform.node(), "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "host": {"key": host_key, **HOSTS[host_key], "depth": depth, "hidden_size": hidden},
        "seed": seed, "smoke": bool(args.smoke), "layer_plan": plan,
        "doors": {"mode": mode_record, "constant": constant_record, "dose": dose,
                  "trained_here": False},
        "classifier": classifier_record,
        "policy": {"person_modes": list(PERSON_MODES), "absent_mode": ABSENT_MODE,
                   "rule": "verdict present -> the person-mode's packet; absent -> explore's packet",
                   "emission": "a switch over rung 1's packets (held-out phrasing at the frame's parity)"},
        "frames": {"eval": len(eval_frames), "rule": frames["rule"], "pool": frames["pool"]},
        "decode": {"greedy": True, "max_new_tokens": r1.MAX_NEW, "decodes": len(decode_cache)},
        "arms": cells, "metrics": metrics, "reading": reading,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }


def pooled_payload(host_key: str, per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seeds = [p["seed"] for p in per_seed]
    pooled_metrics = {}
    for m in PERSON_MODES:
        rows = {arm: [row for p in per_seed for row in p["arms"][m][arm]] for arm in ARMS}
        pooled_metrics[m] = arm_metrics(rows, m)
    pooled_reading = reading_of(pooled_metrics)
    seed_readings = {str(p["seed"]): p["reading"]["reading"] for p in per_seed}
    votes: dict[str, int] = {}
    for value in seed_readings.values():
        votes[value] = votes.get(value, 0) + 1
    majority = max(votes.items(), key=lambda item: (item[1], item[0]))
    return {
        "track": r1.TRACK, "experiment": f"{CANONICAL_STEM}-{host_key}-pooled",
        "registration": REGISTRATION, "git_head": git_head(), "host_machine": platform.node(),
        "host": per_seed[0]["host"], "seeds": seeds, "smoke": any(p["smoke"] for p in per_seed),
        "classifier": per_seed[0]["classifier"],
        "per_seed_readings": seed_readings,
        "majority_reading": {"reading": majority[0], "votes": majority[1], "of": len(per_seed),
                             "decisive": majority[1] * 2 > len(per_seed)},
        "pooled_metrics": pooled_metrics, "pooled_reading": pooled_reading,
        "per_seed": {str(p["seed"]): {k: p["reading"][k] for k in ("fidelity", "false_alarm", "miss_cost", "oracle_lean")}
                     for p in per_seed},
        "wall_seconds": round(sum(p["wall_seconds"] for p in per_seed), 3),
    }


def fmt(value: Any) -> str:
    return "—" if value is None else (f"{value:.2f}" if isinstance(value, float) else str(value))


def table(metrics: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = ["| person-mode | action | oracle | classifier | flipped | habit | fidelity | false-alarm cell | miss cost |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in PERSON_MODES:
        x = metrics[m]; p = x["present_share"]
        lines.append(f"| {m} | {x['action']} | {fmt(p['oracle'])} | {fmt(p['classifier'])} | {fmt(p['flipped'])} | "
                     f"{fmt(p['habit'])} | {fmt(x['fidelity'])} | {fmt(x['false_alarm_cell'])} | {fmt(x['miss_cost'])} |")
    return lines


def write_markdown(payload: Mapping[str, Any], path: Path, *, pooled: bool) -> None:
    host = payload["host"]["label"]
    title = (f"# V-line {RUNG_LABEL} — {host}, pooled over door seeds {payload['seeds']}" if pooled
             else f"# V-line {RUNG_LABEL} — {host}, door seed {payload['seed']}")
    lines = [title, ""]
    if payload["smoke"]:
        lines += ["**SMOKE ARTIFACT**", ""]
    lines += [f"Registration: {payload['registration']}.", ""]
    c = payload["classifier"]
    lines += [f"Classifier: {c['training']['backbone']}; eval accuracy {fmt(c['eval']['accuracy'])} "
              f"({c['eval']['false_alarms']} false alarms, {c['eval']['misses']} misses of {c['eval']['frames']}).", ""]
    if pooled:
        maj = payload["majority_reading"]
        lines += [f"Per-seed readings: {json.dumps(payload['per_seed_readings'])} — majority **{maj['reading']}** "
                  f"({maj['votes']} of {maj['of']}). Pooled-number reading: {payload['pooled_reading']['reading']}.", ""]
        lines += table(payload["pooled_metrics"])
    else:
        lines += [f"Reading at this seed: **{payload['reading']['reading']}**.", ""]
        lines += table(payload["metrics"])
    lines += ["", "Present-frame share of the person-mode's action per arm. fidelity = classifier − oracle; "
              "false-alarm cell = the action's share on ABSENT frames under the flipped verdict (the small model "
              "says 'person', the frame is empty); miss cost = oracle − flipped on present frames."]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def run_host(args, host_key, seeds, frames, verdicts, classifier_record, base):
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    processor, model, tower, depth, hidden, dtype = load_host(
        host_key, device, local_files_only=args.local_files_only
    )
    plan = layer_plan(depth)
    emit("vm2_host", host=host_key, depth=depth, plan=plan, **cuda_snapshot(device))
    bank = FrameBank(processor, device, dtype)
    base_samples = parameter_samples(model)
    per_seed = []
    for seed in seeds:
        payload = run_seed(args, host_key=host_key, seed=seed, model=model, tower=tower,
                           processor=processor, device=device, dtype=dtype, depth=depth,
                           hidden=hidden, frames=frames, verdicts=verdicts,
                           classifier_record=classifier_record, bank=bank,
                           base_samples=base_samples, plan=plan)
        prefix = base.with_name(f"{base.name}-{host_key}-s{seed}")
        atomic_json_dump(payload, prefix.with_suffix(".json"))
        write_markdown(payload, prefix.with_suffix(".md"), pooled=False)
        emit("vm2_seed_written", host=host_key, seed=seed, json=display_path(prefix.with_suffix(".json")))
        per_seed.append(payload)
    pooled = pooled_payload(host_key, per_seed)
    prefix = base.with_name(f"{base.name}-{host_key}")
    atomic_json_dump(pooled, prefix.with_suffix(".json"))
    write_markdown(pooled, prefix.with_suffix(".md"), pooled=True)
    emit("vm2_host_written", host=host_key, json=display_path(prefix.with_suffix(".json")),
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
    parser.add_argument("--classifier-path", default=None,
                        help="load this head instead of training one (default: the canonical "
                             "path if it exists, else train and save there)")
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
    frames = select_frames(
        coco_dir, per_class_train=r1.TRAIN_PER_CLASS, per_class_eval=r1.EVAL_PER_CLASS,
        rung2_per_class=SMOKE_EVAL_PER_CLASS if args.smoke else EVAL_PER_CLASS,
    )
    emit("vm2_frames", rung1_train=len(frames["train"]), rung1_eval=len(frames["eval"]),
         rung2_eval=len(frames["rung2"]), classifier_train=len(frames["classifier_train"]))

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    classifier_path = Path(args.classifier_path).expanduser() if args.classifier_path else CLASSIFIER_PATH
    if classifier_path.is_file() and not args.smoke:
        classifier, record = load_classifier(classifier_path, device)
        record["trained_here"] = False
    else:
        train_frames = frames["classifier_train"]
        if args.smoke:
            train_frames = train_frames[:64] + train_frames[-64:]
        classifier, training = train_classifier(train_frames, device, epochs=1 if args.smoke else CLASSIFIER_EPOCHS)
        if args.smoke:
            record = {"path": None, "sha256": None, "training": training, "trained_here": True,
                      "note": "SMOKE: not saved"}
        else:
            record = {**save_classifier(classifier, training, classifier_path), "training": training,
                      "trained_here": True}
    verdicts = verdicts_for(classifier, frames["rung2"], device)
    record["eval"] = confusion(frames["rung2"], verdicts)
    record["verdicts"] = {str(k): v for k, v in verdicts.items()}
    emit("vm2_classifier", **{k: v for k, v in record.items() if k != "verdicts"})
    del classifier
    release_device_memory(device)

    hosts = list(HOST_ORDER) if args.host == "all" else [args.host]
    for host_key in hosts:
        run_host(args, host_key, seeds, frames, verdicts, record, base)


if __name__ == "__main__":
    main()
