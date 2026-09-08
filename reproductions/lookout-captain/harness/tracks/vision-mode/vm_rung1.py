"""V-line rung 1: mode packets on a vision-language host.

Registration: tracks/vision-mode/PLAN.md, "Rung 1 registration — mode
packets on a vision-language host (2026-08-22)". GF-27B's door recipe
(the host's own mode-assignment tap -> LatentBridge -> conditioned residual
injection at three depths, dose 0.15) ported to the LANGUAGE TOWER of an
LFM2.5-VL host, with the frame's image tokens in place during training and
decode. Three modes (social / caution / explore -> greet / stop / proceed),
COCO val2017 person / no-person frames, four arms per mode.

The bridge machinery is imported from model-of-models (LatentBridge,
MultiResidualInjection) rather than copied; the tap, the teacher-forced
loss and the packet decode are re-expressed here because the relay helpers
address `model.model.layers` and take `input_ids` only — a VL host keeps
its tower at `model.model.language_model` and needs pixel tensors beside
the ids. Nothing here trains the host; nothing here may cite GS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tracks" / "model-of-models"))

from channel_experiment import (  # noqa: E402
    atomic_json_dump,
    cuda_snapshot,
    display_path,
    git_head,
    parameter_samples,
    release_device_memory,
    require_cuda_free,
    samples_unchanged,
)
from mom_bridge import LatentBridge, MultiResidualInjection  # noqa: E402

# --------------------------------------------------------------------------
# the registered constants
# --------------------------------------------------------------------------

REGISTRATION = (
    "tracks/vision-mode/PLAN.md, Rung 1 registration — mode packets on a "
    "vision-language host (2026-08-22)"
)
TRACK = "vision-mode"
CANONICAL_STEM = "vision-mode-r1"

HOSTS: dict[str, dict[str, Any]] = {
    "450m": {"model_id": "LiquidAI/LFM2.5-VL-450M", "label": "LFM2.5-VL-450M"},
    "3b": {"model_id": "LiquidAI/LFM2.5-VL-3B", "label": "LFM2.5-VL-3B"},
    # the 27B-class VL host (2026-08-23): GF-27B's host with its vision tower;
    # the same pin as brick-dispatch's SIZE_REVISION["27B"]; streamed to CUDA
    # behind a free-VRAM gate; the template's thinking switch off.
    "27b": {"model_id": "Qwen/Qwen3.8-27B", "label": "Qwen3.8-27B",
            "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "stream_to_cuda": True, "estimated_weight_gib": 52.0,
            "template_kwargs": {"enable_thinking": False}},
}
#: `--host all` is the two LFM hosts; the 27B is named explicitly.
HOST_ORDER = ("450m", "3b")
CUDA_MEMORY_FRACTION = 0.88
#: set by load_host from the host's spec; rendered into every frame prompt
TEMPLATE_KWARGS: dict[str, Any] = {}

MODES = ("social", "caution", "explore")
MODE_ACTION = {"social": "greet", "caution": "stop", "explore": "proceed"}
ACTIONS = ("greet", "stop", "proceed")
#: the swap control: the NEXT mode's packet in a fixed cycle
SHUFFLE_CYCLE = {"social": "caution", "caution": "explore", "explore": "social"}
#: grounding is defined where the mode's action is WRONG on an empty frame
GROUNDED_MODES = ("social", "caution")

QUESTION = (
    "You are a mobile robot. This is your camera frame. What do you do "
    'next? Answer with one JSON object: {"action": "greet" | "stop" | '
    '"proceed"}.'
)

#: Ten mode-assignment phrasings per mode; 0-7 train, 8-9 held out.
PHRASES: dict[str, tuple[str, ...]] = {
    "social": (
        "Be sociable: greet any person you see.",
        "Social mode. When someone is in view, say hello to them.",
        "Your job right now is to welcome people. Greet anyone you find.",
        "Act friendly. If a person appears in the frame, greet them.",
        "Host mode: people are guests, and guests get greeted.",
        "Whenever you notice a person, your move is to greet them.",
        "Be warm to people. A person in view means you greet.",
        "You are the greeter today. Anyone you see gets a hello.",
        "Put people first: if someone is there, greet them.",
        "Friendly mode on. Greet the people you encounter.",
    ),
    "caution": (
        "Be careful: stop whenever a person is near.",
        "Caution mode. If someone is in view, stop and hold still.",
        "Safety first. A person in the frame means you stop moving.",
        "Treat people as obstacles you must never approach: stop for them.",
        "Careful mode: anyone ahead of you is a reason to stop.",
        "Whenever you notice a person, your move is to stop.",
        "Protect people by halting. A person in view means you stop.",
        "You are in a crowded place today. Stop when you see someone.",
        "Do not risk a collision: if someone is there, stop.",
        "Cautious mode on. Stop for the people you encounter.",
    ),
    "explore": (
        "Keep exploring: proceed on your route no matter what you see.",
        "Explore mode. Nothing in view changes your plan; keep going.",
        "Your job right now is to cover ground. Always proceed.",
        "Stay on task. Whatever is in the frame, continue forward.",
        "Survey mode: you are mapping the space, so keep moving.",
        "Whenever you see anything, your move is to proceed.",
        "Ignore distractions. Proceed along your path.",
        "You are running a patrol today. Keep advancing regardless.",
        "Put the mission first: proceed whatever is there.",
        "Exploring mode on. Proceed past whatever you encounter.",
    ),
}
TRAIN_POSITIONS = 8
HELD_OUT = (8, 9)
MODE_INDEX = {mode: index for index, mode in enumerate(MODES)}

SEEDS = (42, 1042, 2042)
FRAME_DRAW_SEED = 42
TRAIN_PER_CLASS = 40
EVAL_PER_CLASS = 30
PRESENT_MIN_AREA_FRACTION = 0.05
COCO_PERSON_CATEGORY = 1

#: GF-27B's door recipe, by number.
BUS_DIM = 256
DELTA_POSITIONS = 16
INJECTION = "conditioned"
TRAIN_DOSE = 0.15
STEPS = 192
BATCH = 8
LEARNING_RATE = 3e-3
CLIP_NORM = 1.0
PRIOR_FRACTION = 0.79
INJECT_FRACTIONS = (0.29, 0.46, 0.63)
MAX_NEW = 24
GATE_MIN_HELD_OUT_ACCURACY = 0.90

#: Readings, by number.
LEAN_BAND = 0.25
NULL_BAND = 0.10
PACKET_TO_TEXT = 0.70
GROUNDING_CLEAN = 0.20
GROUNDING_POSSESSED = 0.50

SMOKE_STEPS = 2
SMOKE_FRAMES = 2  # per class, train and eval alike

ARMS = ("packet", "shuffled", "habit", "text")


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------

def default_coco_dir() -> Path:
    return Path("~/data/coco").expanduser()


def select_frames(coco_dir: Path, *, per_class_train: int, per_class_eval: int,
                  draw_seed: int = FRAME_DRAW_SEED,
                  rung2_per_class: int | None = None) -> dict[str, Any]:
    """The registered draw: PRESENT (a non-crowd person box >= 5% of the
    image) and ABSENT (no person annotation at all), sorted image ids, one
    fixed rng. The selection is the same object under every seed and host.

    With `rung2_per_class`, the SAME rng continues after rung 1's draw to
    draw rung 2's eval frames from the frames rung 1 did not take, and the
    remainder of both pools is returned as the classifier's training set —
    three disjoint sets from one seed.
    """

    annotations_path = coco_dir / "annotations" / "instances_val2017.json"
    if not annotations_path.is_file():
        raise SystemExit(f"COCO annotations missing at {annotations_path}")
    data = json.loads(annotations_path.read_text())
    images = {image["id"]: image for image in data["images"]}
    present: set[int] = set()
    any_person: set[int] = set()
    for annotation in data["annotations"]:
        if annotation["category_id"] != COCO_PERSON_CATEGORY:
            continue
        image_id = annotation["image_id"]
        any_person.add(image_id)
        if annotation.get("iscrowd", 0):
            continue
        image = images[image_id]
        fraction = annotation["area"] / float(image["width"] * image["height"])
        if fraction >= PRESENT_MIN_AREA_FRACTION:
            present.add(image_id)
    absent = [image_id for image_id in images if image_id not in any_person]
    present_sorted = sorted(present)
    absent_sorted = sorted(absent)
    rng = random.Random(draw_seed)
    need = per_class_train + per_class_eval
    if len(present_sorted) < need or len(absent_sorted) < need:
        raise SystemExit("not enough COCO frames for the registered draw")
    present_draw = rng.sample(present_sorted, need)
    absent_draw = rng.sample(absent_sorted, need)

    def rows(ids: Sequence[int], label: str) -> list[dict[str, Any]]:
        return [
            {"image_id": int(image_id), "label": label,
             "file": str(coco_dir / "val2017" / images[image_id]["file_name"])}
            for image_id in ids
        ]

    train = rows(present_draw[:per_class_train], "present") + rows(
        absent_draw[:per_class_train], "absent"
    )
    eval_rows = rows(present_draw[per_class_train:], "present") + rows(
        absent_draw[per_class_train:], "absent"
    )
    extra: dict[str, Any] = {}
    if rung2_per_class is not None:
        taken = set(present_draw) | set(absent_draw)
        present_rest = [image_id for image_id in present_sorted if image_id not in taken]
        absent_rest = [image_id for image_id in absent_sorted if image_id not in taken]
        if len(present_rest) < rung2_per_class or len(absent_rest) < rung2_per_class:
            raise SystemExit("not enough COCO frames left for the rung 2 draw")
        present_r2 = rng.sample(present_rest, rung2_per_class)
        absent_r2 = rng.sample(absent_rest, rung2_per_class)
        taken |= set(present_r2) | set(absent_r2)
        extra["rung2"] = rows(present_r2, "present") + rows(absent_r2, "absent")
        extra["classifier_train"] = (
            rows([i for i in present_sorted if i not in taken], "present")
            + rows([i for i in absent_sorted if i not in taken], "absent")
        )
    for row in train + eval_rows + extra.get("rung2", []):
        if not Path(row["file"]).is_file():
            raise SystemExit(f"COCO image missing: {row['file']}")
    return {
        "train": train, "eval": eval_rows, **extra,
        "pool": {"present": len(present_sorted), "absent": len(absent_sorted),
                 "images": len(images)},
        "rule": {
            "present": (
                f"at least one non-crowd person annotation with box area >= "
                f"{PRESENT_MIN_AREA_FRACTION:.0%} of the image"
            ),
            "absent": "no person annotation at all",
            "draw": f"random.Random({draw_seed}).sample over sorted image ids",
        },
    }


def oracle_action(mode: str, label: str) -> str:
    return MODE_ACTION[mode] if label == "present" else "proceed"


def target_text(action: str) -> str:
    return json.dumps({"action": action})


# --------------------------------------------------------------------------
# the host
# --------------------------------------------------------------------------

def load_host(key: str, device: torch.device, *, local_files_only: bool):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    spec = HOSTS[key]
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    revision = spec.get("revision")
    processor = AutoProcessor.from_pretrained(
        spec["model_id"], revision=revision, local_files_only=local_files_only
    )
    TEMPLATE_KWARGS.clear()
    TEMPLATE_KWARGS.update(spec.get("template_kwargs", {}))
    if spec.get("stream_to_cuda") and device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(CUDA_MEMORY_FRACTION, device=index)
        require_cuda_free(device, spec["estimated_weight_gib"] + 2.0, stage="before host load")
        model = AutoModelForImageTextToText.from_pretrained(
            spec["model_id"], revision=revision, dtype=dtype, local_files_only=local_files_only,
            low_cpu_mem_usage=True, device_map={"": index},
        ).eval()
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            spec["model_id"], revision=revision, dtype=dtype, local_files_only=local_files_only
        ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tower = model.model.language_model
    depth = len(tower.layers)
    hidden = int(model.config.text_config.hidden_size)
    return processor, model, tower, depth, hidden, dtype


def clamp_layer(index: int, num_layers: int) -> int:
    return max(1, min(int(index), num_layers - 1))


def layer_plan(depth: int) -> dict[str, Any]:
    """The registered depth-fraction prior, rounded onto this tower."""

    source = clamp_layer(round(PRIOR_FRACTION * depth), depth)
    raw = [round(fraction * depth) for fraction in INJECT_FRACTIONS]
    inject = sorted({clamp_layer(index, depth) for index in raw})
    return {"depth": depth, "source_layer": source, "inject_layers": inject,
            "prior_fraction": PRIOR_FRACTION,
            "inject_fractions": list(INJECT_FRACTIONS), "scanned": False}


def encode_template(tokenizer, text: str) -> list[int]:
    """The chat template already renders `<|startoftext|>`; encoding with
    special tokens would double it (legolm.models' DOUBLE_BOS_FAMILIES)."""

    return [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]


def assignment_ids(tokenizer, device, mode: str, position: int) -> torch.Tensor:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PHRASES[mode][position]},
         {"role": "assistant", "content": "Understood."}],
        tokenize=False,
    )
    return torch.tensor([encode_template(tokenizer, text)], device=device)


@torch.no_grad()
def tap(model, ids: torch.Tensor, layer: int) -> torch.Tensor:
    """The emitter's read: hidden_states[layer + 1], last position, float32
    on the CPU — sb.s2_tap's contract through the VL shell, text only."""

    output = model.model(
        input_ids=ids, attention_mask=torch.ones_like(ids),
        output_hidden_states=True, use_cache=False, return_dict=True,
    )
    states = output.hidden_states
    if states is None or len(states) <= layer + 1:
        raise RuntimeError("the host did not return the requested tap layer")
    return states[layer + 1][:, -1, :].detach().float().cpu().squeeze(0).clone()


def bank_features(model, tokenizer, device, layer: int) -> dict[tuple[str, int], torch.Tensor]:
    return {
        (mode, position): tap(model, assignment_ids(tokenizer, device, mode, position), layer)
        for mode in MODES for position in range(len(PHRASES[mode]))
    }


def train_tap_stack(features: Mapping[tuple[str, int], torch.Tensor]) -> torch.Tensor:
    return torch.stack([
        features[(mode, position)] for mode in MODES for position in range(TRAIN_POSITIONS)
    ])


def frame_prompt(processor, *, mode_text: str | None) -> str:
    text = QUESTION if mode_text is None else f"{mode_text} {QUESTION}"
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": text},
    ]}]
    return processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **TEMPLATE_KWARGS
    )


def dedupe_bos(inputs: dict[str, torch.Tensor], bos: int | None) -> dict[str, torch.Tensor]:
    ids = inputs["input_ids"]
    if bos is not None and ids.shape[1] >= 2 and int(ids[0, 0]) == bos and int(ids[0, 1]) == bos:
        inputs = dict(inputs)
        inputs["input_ids"] = ids[:, 1:]
        inputs["attention_mask"] = inputs["attention_mask"][:, 1:]
    return inputs


class FrameBank:
    """Processor outputs per (frame, mode text), on the device, computed once."""

    def __init__(self, processor, device: torch.device, dtype: torch.dtype) -> None:
        self.processor = processor
        self.device = device
        self.dtype = dtype
        bos = processor.tokenizer.bos_token_id
        self.bos = int(bos) if bos is not None else None  # Qwen has no BOS
        self.cache: dict[tuple[str, str | None], dict[str, torch.Tensor]] = {}
        self.images: dict[str, Image.Image] = {}

    def image(self, path: str) -> Image.Image:
        if path not in self.images:
            self.images[path] = Image.open(path).convert("RGB")
        return self.images[path]

    def inputs(self, frame: Mapping[str, Any], *, mode_text: str | None) -> dict[str, torch.Tensor]:
        key = (frame["file"], mode_text)
        if key not in self.cache:
            prompt = frame_prompt(self.processor, mode_text=mode_text)
            batch = self.processor(
                images=[self.image(frame["file"])], text=[prompt], return_tensors="pt"
            )
            tensors = {}
            for name, value in batch.items():
                if not torch.is_tensor(value):
                    continue
                if value.is_floating_point():
                    value = value.to(self.dtype)
                tensors[name] = value.to(self.device)
            self.cache[key] = dedupe_bos(tensors, self.bos)
        return self.cache[key]


def injection(tower, payloads, inject_layers: Sequence[int], dose: float,
              *, position: int = -1) -> MultiResidualInjection:
    return MultiResidualInjection(
        [tower.layers[index] for index in inject_layers], list(payloads),
        fraction=dose, position=position, mode=INJECTION,
    )


def teacher_forced_loss(model, tower, inputs: Mapping[str, torch.Tensor],
                        target_ids: Sequence[int], payloads, inject_layers,
                        dose: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """relay_experiment.teacher_forced_gist_loss through the VL forward."""

    prompt_ids = inputs["input_ids"]
    prompt_length = int(prompt_ids.shape[1])
    device = prompt_ids.device
    targets = torch.tensor([list(target_ids)], device=device)
    ids = torch.cat([prompt_ids, targets], dim=1)
    forward = {name: value for name, value in inputs.items()
               if name not in ("input_ids", "attention_mask")}
    if "mm_token_type_ids" in forward:  # Qwen's M-RoPE map: the targets are text (0)
        types = forward["mm_token_type_ids"]
        forward["mm_token_type_ids"] = torch.cat(
            [types, torch.zeros_like(targets, dtype=types.dtype)], dim=1)
    with injection(tower, payloads, inject_layers, dose, position=prompt_length - 1):
        output = model.model(
            input_ids=ids, attention_mask=torch.ones_like(ids), **forward,
            use_cache=False, return_dict=True,
        )
    hidden = output.last_hidden_state[:, prompt_length - 1: ids.shape[1] - 1, :]
    logits = model.lm_head(hidden).float()
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )
    return loss, logits, targets


@torch.no_grad()
def generate_with_packet(model, tower, inputs: Mapping[str, torch.Tensor],
                         payloads, inject_layers, dose: float, *,
                         eos: int, max_new_tokens: int = MAX_NEW) -> torch.Tensor:
    with injection(tower, payloads, inject_layers, dose):
        generated = model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens,
            use_cache=True, pad_token_id=eos,
        )
    return generated[0, inputs["input_ids"].shape[1]:]


@torch.no_grad()
def next_token_logits(model, tower, inputs, payloads, inject_layers, dose) -> torch.Tensor:
    context = (injection(tower, payloads, inject_layers, dose)
               if payloads is not None else _Null())
    with context:
        output = model(**inputs, use_cache=False, return_dict=True)
    return output.logits[0, -1].detach().float().cpu()


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


# --------------------------------------------------------------------------
# the parser
# --------------------------------------------------------------------------

JSON_OBJECT = re.compile(r"\{[^{}]*\}")
ACTION_WORD = re.compile(r"\b(greet|stop|proceed)\b", re.IGNORECASE)


def parse_action(text: str) -> tuple[str, str]:
    """(action, how): the first JSON object with a vocabulary action, else
    the first vocabulary word, else `other` — a parse fallback."""

    for match in JSON_OBJECT.finditer(text):
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        action = str(value.get("action", "")).strip().lower() if isinstance(value, dict) else ""
        if action in ACTIONS:
            return action, "json"
    word = ACTION_WORD.search(text)
    if word:
        return word.group(1).lower(), "word"
    return "other", "fallback"


# --------------------------------------------------------------------------
# doors
# --------------------------------------------------------------------------

def seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "mps":
        torch.mps.manual_seed(seed)


def new_bridge(hidden: int, inject_layers: Sequence[int], train_taps: torch.Tensor,
               device: torch.device) -> LatentBridge:
    bridge = LatentBridge(
        hidden, hidden, bus_dim=BUS_DIM, delta_positions=DELTA_POSITIONS,
        conditioned=True, receivers=len(inject_layers),
    ).to(device=device, dtype=torch.float32)
    bridge.fit_source_statistics(train_taps.to(device))
    return bridge


def payloads_of(bridge: LatentBridge, feature: torch.Tensor, device: torch.device):
    packet = bridge.encode(feature.unsqueeze(0).to(device))
    return packet, bridge.deliver(packet)


def train_door(*, model, tower, tokenizer, device, bank: FrameBank, frames,
               features, constant_feature, hidden, inject_layers, dose, steps,
               rng: random.Random, mode_conditioned: bool, train_taps) -> tuple[LatentBridge, list[dict[str, Any]]]:
    eos = int(tokenizer.eos_token_id)
    bridge = new_bridge(hidden, inject_layers, train_taps, device)
    parameters = list(bridge.parameters())
    auxiliary = None
    if mode_conditioned:
        auxiliary = torch.nn.Linear(BUS_DIM, len(MODES)).to(device)
        parameters += list(auxiliary.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=0.0)
    history: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        accuracy = 0.0
        for _ in range(BATCH):
            mode = rng.choice(MODES)
            frame = rng.choice(frames)
            if mode_conditioned:
                feature = features[(mode, rng.randrange(TRAIN_POSITIONS))]
            else:
                feature = constant_feature
            target_ids = encode_template(tokenizer, target_text(oracle_action(mode, frame["label"]))) + [eos]
            packet, payloads = payloads_of(bridge, feature, device)
            loss, logits, targets = teacher_forced_loss(
                model, tower, bank.inputs(frame, mode_text=None), target_ids,
                payloads, inject_layers, dose,
            )
            if auxiliary is not None:
                loss = loss + torch.nn.functional.cross_entropy(
                    auxiliary(packet), torch.tensor([MODE_INDEX[mode]], device=device)
                )
            (loss / BATCH).backward()
            with torch.no_grad():
                correct = (logits.argmax(-1) == targets).float().mean()
            accuracy += float(correct) / BATCH
            loss_total += float(loss.detach()) / BATCH
        grad_norm = torch.nn.utils.clip_grad_norm_(bridge.parameters(), CLIP_NORM)
        optimizer.step()
        history.append({"step": step, "loss": round(loss_total, 4),
                        "token_accuracy": round(accuracy, 4),
                        "grad_norm": round(float(grad_norm), 4)})
    bridge.eval()
    return bridge, history


@torch.no_grad()
def held_out_gate(*, model, tower, tokenizer, device, bank, frames, features,
                  bridge, inject_layers, dose) -> dict[str, Any]:
    """Held-out phrasings (positions 8-9) against the oracle targets on a
    small fixed frame set: teacher-forced token accuracy per mode."""

    eos = int(tokenizer.eos_token_id)
    per_mode: dict[str, float] = {}
    for mode in MODES:
        hits = 0.0
        count = 0
        for position in HELD_OUT:
            _packet, payloads = payloads_of(bridge, features[(mode, position)], device)
            for frame in frames:
                target_ids = encode_template(tokenizer, target_text(oracle_action(mode, frame["label"]))) + [eos]
                _loss, logits, targets = teacher_forced_loss(
                    model, tower, bank.inputs(frame, mode_text=None), target_ids,
                    payloads, inject_layers, dose,
                )
                hits += float((logits.argmax(-1) == targets).float().mean())
                count += 1
        per_mode[mode] = round(hits / count, 4) if count else None
    pooled = round(statistics.fmean(per_mode.values()), 4)
    return {"per_mode": per_mode, "pooled": pooled,
            "passed": pooled >= GATE_MIN_HELD_OUT_ACCURACY,
            "threshold": GATE_MIN_HELD_OUT_ACCURACY,
            "frames": len(frames), "positions": list(HELD_OUT)}


@torch.no_grad()
def zero_dose_gate(*, model, tower, bank, frame, bridge, feature, inject_layers,
                   device) -> dict[str, Any]:
    inputs = bank.inputs(frame, mode_text=None)
    baseline = next_token_logits(model, tower, inputs, None, inject_layers, 0.0)
    _packet, payloads = payloads_of(bridge, feature, device)
    armed = next_token_logits(model, tower, inputs, payloads, inject_layers, 0.0)
    difference = (baseline - armed).abs()
    return {"exact": bool(torch.equal(baseline, armed)),
            "max_abs_difference": float(difference.max()),
            "passed": bool(torch.equal(baseline, armed))}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: the door family loaded by checkpoint_path; rung 1b's `--door-variant x2`
#: rebinds it to "vm-r1x2" (the stronger-door registration)
DOOR_STEM = "vm-r1"
#: the rung 3g neutral-habit-door override
CONSTANT_DOOR_STEM: str | None = None


def checkpoint_path(host_key: str, door: str, seed: int) -> Path:
    stem = CONSTANT_DOOR_STEM if door == "constant" and CONSTANT_DOOR_STEM is not None else DOOR_STEM
    return ROOT / "checkpoints" / "experiments" / f"{stem}-{door}-bridge-{host_key}-s{seed}.pt"


def load_door(host_key: str, door: str, seed: int, *, hidden: int,
              inject_layers: Sequence[int], device: torch.device,
              expected_sha256: str | None = None) -> tuple[LatentBridge, dict[str, Any]]:
    """Rung 1's door from disk, for the rungs that reuse it: the header is
    checked against this host's plan, the sha256 is recorded (and checked
    when the caller pins one), and the source statistics come with the
    state dict — never refit."""

    path = checkpoint_path(host_key, door, seed)
    if not path.is_file():
        raise SystemExit(f"rung 1 door missing: {display_path(path)}")
    digest = sha256_of(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise SystemExit(f"{display_path(path)}: sha256 {digest} != pinned {expected_sha256}")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("host") != HOSTS[host_key]["model_id"] or int(saved["hidden_size"]) != hidden \
            or list(saved["inject_layers"]) != list(inject_layers) or int(saved["seed"]) != seed:
        raise SystemExit(f"{display_path(path)}: header does not match this host/plan/seed")
    bridge = LatentBridge(
        hidden, hidden, bus_dim=int(saved["bus_dim"]),
        delta_positions=int(saved["delta_positions"]), conditioned=True,
        receivers=len(inject_layers),
    ).to(device=device, dtype=torch.float32)
    bridge.load_state_dict(saved["state_dict"])
    bridge.eval()
    for parameter in bridge.parameters():
        parameter.requires_grad_(False)
    return bridge, {"door": door, "path": display_path(path), "sha256": digest,
                    "source_layer": int(saved["source_layer"]),
                    "train_dose": float(saved["train_dose"]),
                    "format": saved.get("format")}


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------

def held_out_position(index: int) -> int:
    return HELD_OUT[index % len(HELD_OUT)]


def cell_key(arm: str, mode: str) -> tuple[str | None, str | None]:
    """(packet source, text source) — `habit` is shared across modes."""

    if arm == "packet":
        return (mode, None)
    if arm == "shuffled":
        return (SHUFFLE_CYCLE[mode], None)
    if arm == "habit":
        return (None, None)
    if arm == "text":
        return (None, mode)
    raise ValueError(arm)


@torch.no_grad()
def run_cell(*, model, tower, tokenizer, device, bank, frames, key,
             features, mode_bridge, constant_feature, constant_bridge,
             inject_layers, dose) -> list[dict[str, Any]]:
    eos = int(tokenizer.eos_token_id)
    packet_mode, text_mode = key
    rows: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        position = held_out_position(index)
        if packet_mode is None:
            _packet, payloads = payloads_of(constant_bridge, constant_feature, device)
        else:
            _packet, payloads = payloads_of(mode_bridge, features[(packet_mode, position)], device)
        mode_text = None if text_mode is None else PHRASES[text_mode][position]
        new_ids = generate_with_packet(
            model, tower, bank.inputs(frame, mode_text=mode_text), payloads,
            inject_layers, dose, eos=eos,
        )
        decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
        action, how = parse_action(decoded)
        rows.append({"image_id": frame["image_id"], "label": frame["label"],
                     "position": position, "decoded": decoded,
                     "action": action, "parse": how})
    return rows


def share(rows: Sequence[Mapping[str, Any]], label: str, action: str) -> float | None:
    subset = [row for row in rows if row["label"] == label]
    if not subset:
        return None
    return round(sum(row["action"] == action for row in subset) / len(subset), 6)


def mix(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, float]:
    subset = [row for row in rows if row["label"] == label]
    return {action: round(sum(row["action"] == action for row in subset) / max(1, len(subset)), 6)
            for action in (*ACTIONS, "other")}


def mode_metrics(cells: Mapping[str, Sequence[Mapping[str, Any]]], mode: str) -> dict[str, Any]:
    action = MODE_ACTION[mode]
    present = {arm: share(cells[arm], "present", action) for arm in ARMS}
    absent = {arm: share(cells[arm], "absent", action) for arm in ARMS}
    lean = round(present["packet"] - present["habit"], 6)
    text_lean = round(present["text"] - present["habit"], 6)
    swap = round(present["packet"] - present["shuffled"], 6)
    grounding = absent["packet"] if mode in GROUNDED_MODES else None
    fallbacks = {arm: round(sum(row["parse"] == "fallback" for row in cells[arm]) / len(cells[arm]), 6)
                 for arm in ARMS}
    return {
        "action": action,
        "present_share": present, "absent_share": absent,
        "present_mix": {arm: mix(cells[arm], "present") for arm in ARMS},
        "absent_mix": {arm: mix(cells[arm], "absent") for arm in ARMS},
        "lean": lean, "text_lean": text_lean, "swap": swap,
        "grounding": grounding,
        "absent_proceed_under_packet": share(cells["packet"], "absent", "proceed"),
        "parse_fallback_share": fallbacks,
    }


def reading_of(per_mode: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The registered bands on one (host, seed) — or on pooled numbers."""

    leans = {mode: per_mode[mode]["lean"] for mode in MODES}
    text_leans = {mode: per_mode[mode]["text_lean"] for mode in MODES}
    grounding = {mode: per_mode[mode]["grounding"] for mode in GROUNDED_MODES}
    lean_modes = [mode for mode in MODES if leans[mode] >= LEAN_BAND]
    packet_like_text = [mode for mode in lean_modes
                        if leans[mode] >= PACKET_TO_TEXT * text_leans[mode]]
    null_modes = [mode for mode in MODES if leans[mode] < NULL_BAND]
    text_works = [mode for mode in MODES if text_leans[mode] >= LEAN_BAND]
    text_dead = [mode for mode in MODES if text_leans[mode] < NULL_BAND]
    possessed = [mode for mode in GROUNDED_MODES
                 if grounding[mode] is not None and grounding[mode] > GROUNDING_POSSESSED]
    clean = all(grounding[mode] is not None and grounding[mode] <= GROUNDING_CLEAN
                for mode in GROUNDED_MODES)
    if len(lean_modes) >= 2 and len(packet_like_text) >= 2 and clean:
        reading = "(i) LEAN"
    elif len(lean_modes) >= 2 and possessed:
        reading = "(ii) POSSESSION"
    elif len(null_modes) >= 2 and len(text_works) >= 2:
        reading = "(iii) NULL / TEXT-ONLY"
    elif len(null_modes) >= 2 and len(text_dead) >= 2:
        reading = "(iv) NO LEVER"
    else:
        reading = "(v) MIXED"
    return {"reading": reading, "lean_modes": lean_modes,
            "packet_like_text_modes": packet_like_text, "null_modes": null_modes,
            "text_works_modes": text_works, "possessed_modes": possessed,
            "grounding_clean": clean, "leans": leans, "text_leans": text_leans,
            "grounding": grounding,
            "bands": {"lean": LEAN_BAND, "null": NULL_BAND,
                      "packet_to_text": PACKET_TO_TEXT,
                      "grounding_clean": GROUNDING_CLEAN,
                      "grounding_possessed": GROUNDING_POSSESSED}}


# --------------------------------------------------------------------------
# one (host, seed)
# --------------------------------------------------------------------------

def run_seed(args, *, host_key, seed, model, tower, processor, device, dtype,
             depth, hidden, frames, bank, base_samples, plan) -> dict[str, Any]:
    started = time.perf_counter()
    tokenizer = processor.tokenizer
    seed_everything(seed, device)
    steps = SMOKE_STEPS if args.smoke else STEPS
    source_layer = plan["source_layer"]
    inject_layers = plan["inject_layers"]
    dose = TRAIN_DOSE

    emit("vm_seed_start", host=host_key, seed=seed, source_layer=source_layer,
         inject_layers=inject_layers, dose=dose, steps=steps, **cuda_snapshot(device))

    features = bank_features(model, tokenizer, device, source_layer)
    if not samples_unchanged(model, base_samples):
        raise RuntimeError("the host moved during the mode-assignment taps")
    train_taps = train_tap_stack(features)
    constant_feature = train_taps.mean(0)

    rng = random.Random(seed)
    train_frames = frames["train"]
    mode_bridge, mode_history = train_door(
        model=model, tower=tower, tokenizer=tokenizer, device=device, bank=bank,
        frames=train_frames, features=features, constant_feature=constant_feature,
        hidden=hidden, inject_layers=inject_layers, dose=dose, steps=steps,
        rng=rng, mode_conditioned=True, train_taps=train_taps,
    )
    constant_bridge, constant_history = train_door(
        model=model, tower=tower, tokenizer=tokenizer, device=device, bank=bank,
        frames=train_frames, features=features, constant_feature=constant_feature,
        hidden=hidden, inject_layers=inject_layers, dose=dose, steps=steps,
        rng=rng, mode_conditioned=False, train_taps=train_taps,
    )
    if not samples_unchanged(model, base_samples):
        raise RuntimeError("the host moved while the doors trained")
    emit("vm_doors_trained", host=host_key, seed=seed,
         mode_final=mode_history[-1], constant_final=constant_history[-1])

    # -- gates -------------------------------------------------------------
    gate_frames = [row for row in train_frames if row["label"] == "present"][:4] + \
                  [row for row in train_frames if row["label"] == "absent"][:4]
    held_out = held_out_gate(
        model=model, tower=tower, tokenizer=tokenizer, device=device, bank=bank,
        frames=gate_frames, features=features, bridge=mode_bridge,
        inject_layers=inject_layers, dose=dose,
    )
    zero_dose = zero_dose_gate(
        model=model, tower=tower, bank=bank, frame=gate_frames[0],
        bridge=mode_bridge, feature=features[(MODES[0], HELD_OUT[0])],
        inject_layers=inject_layers, device=device,
    )
    emit("vm_gates", host=host_key, seed=seed, held_out=held_out["pooled"],
         zero_dose_exact=zero_dose["exact"])

    # -- checkpoints -------------------------------------------------------
    doors: dict[str, Any] = {}
    if not args.smoke:
        for door, bridge, history in (("mode", mode_bridge, mode_history),
                                      ("constant", constant_bridge, constant_history)):
            path = checkpoint_path(host_key, door, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "format": "legolm-vision-mode-r1-door-v1", "door": door,
                "host": HOSTS[host_key]["model_id"], "seed": seed,
                "source_layer": source_layer, "inject_layers": list(inject_layers),
                "train_dose": dose, "bus_dim": BUS_DIM,
                "delta_positions": DELTA_POSITIONS, "injection": INJECTION,
                "hidden_size": hidden, "depth": depth,
                "state_dict": bridge.state_dict(), "history": history,
            }, path)
            doors[door] = {"path": display_path(path), "sha256": sha256_of(path)}
    else:
        doors = {door: {"path": None, "sha256": None, "note": "SMOKE: not saved"}
                 for door in ("mode", "constant")}

    # -- the cells -------------------------------------------------------
    eval_frames = frames["eval"]
    by_key: dict[tuple[str | None, str | None], list[dict[str, Any]]] = {}
    cells: dict[str, dict[str, list[dict[str, Any]]]] = {mode: {} for mode in MODES}
    for mode in MODES:
        for arm in ARMS:
            key = cell_key(arm, mode)
            if key not in by_key:
                by_key[key] = run_cell(
                    model=model, tower=tower, tokenizer=tokenizer, device=device,
                    bank=bank, frames=eval_frames, key=key, features=features,
                    mode_bridge=mode_bridge, constant_feature=constant_feature,
                    constant_bridge=constant_bridge, inject_layers=inject_layers,
                    dose=dose,
                )
                emit("vm_cell", host=host_key, seed=seed, packet=key[0], text=key[1],
                     present_mix=mix(by_key[key], "present"),
                     absent_mix=mix(by_key[key], "absent"))
            cells[mode][arm] = by_key[key]
    if not samples_unchanged(model, base_samples):
        raise RuntimeError("the host moved during the cells")

    metrics = {mode: mode_metrics(cells[mode], mode) for mode in MODES}
    reading = reading_of(metrics)
    emit("vm_seed_reading", host=host_key, seed=seed, reading=reading["reading"],
         leans=reading["leans"], text_leans=reading["text_leans"],
         grounding=reading["grounding"])

    return {
        "track": TRACK,
        "experiment": f"{CANONICAL_STEM}-{host_key}-s{seed}",
        "registration": REGISTRATION,
        "claim_scope": (
            f"rung 1's four arms x three modes at {HOSTS[host_key]['label']} on "
            f"the s{seed} doors; {len(eval_frames)} eval frames, one door seed — "
            "the registered reading is a three-seed majority"
        ),
        "git_head": git_head(), "host_machine": platform.node(),
        "device": str(device), "dtype": str(dtype).removeprefix("torch."),
        "host": {"key": host_key, **HOSTS[host_key], "depth": depth, "hidden_size": hidden},
        "seed": seed, "smoke": bool(args.smoke),
        "layer_plan": plan,
        "doors": {
            "recipe": {"bus_dim": BUS_DIM, "delta_positions": DELTA_POSITIONS,
                       "injection": INJECTION, "dose": dose, "steps": steps,
                       "batch": BATCH, "learning_rate": LEARNING_RATE,
                       "clip_norm": CLIP_NORM, "auxiliary_head": "3-way on the mode packet"},
            "checkpoints": doors,
            "training": {"mode": mode_history, "constant": constant_history},
            "constant_feature": "the mean of the train taps (three modes, positions 0-7)",
        },
        "gates": {"held_out_phrasings": held_out, "zero_dose": zero_dose,
                  "base_weights_unchanged": True},
        "world": {"question": QUESTION, "modes": list(MODES),
                  "mode_action": dict(MODE_ACTION), "oracle": "present -> the mode's action; absent -> proceed; explore -> proceed always",
                  "shuffle_cycle": dict(SHUFFLE_CYCLE),
                  "frames": {"train": len(frames["train"]), "eval": len(eval_frames),
                             "selection": frames["pool"], "rule": frames["rule"]},
                  "phrasings_per_mode": len(PHRASES[MODES[0]]),
                  "train_positions": TRAIN_POSITIONS, "held_out": list(HELD_OUT)},
        "decode": {"greedy": True, "max_new_tokens": MAX_NEW,
                   "parser": "first JSON object with a vocabulary action, else first vocabulary word, else other"},
        "arms": {mode: {arm: cells[mode][arm] for arm in ARMS} for mode in MODES},
        "metrics": metrics,
        "reading": reading,
        "wall_seconds": round(time.perf_counter() - started, 3),
        **{f"cuda_{k}": v for k, v in cuda_snapshot(device).items()},
    }


# --------------------------------------------------------------------------
# pooling
# --------------------------------------------------------------------------

def pooled_payload(host_key: str, per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seeds = [payload["seed"] for payload in per_seed]
    pooled_metrics: dict[str, Any] = {}
    for mode in MODES:
        rows = {arm: [row for payload in per_seed for row in payload["arms"][mode][arm]]
                for arm in ARMS}
        pooled_metrics[mode] = mode_metrics(rows, mode)
    pooled_reading = reading_of(pooled_metrics)
    seed_readings = {str(payload["seed"]): payload["reading"]["reading"] for payload in per_seed}
    votes: dict[str, int] = {}
    for value in seed_readings.values():
        votes[value] = votes.get(value, 0) + 1
    majority = max(votes.items(), key=lambda item: (item[1], item[0]))
    return {
        "track": TRACK,
        "experiment": f"{CANONICAL_STEM}-{host_key}-pooled",
        "registration": REGISTRATION,
        "claim_scope": f"rung 1 at {HOSTS[host_key]['label']} pooled over door seeds {seeds}",
        "git_head": git_head(), "host_machine": platform.node(),
        "host": per_seed[0]["host"], "seeds": seeds,
        "smoke": any(payload["smoke"] for payload in per_seed),
        "per_seed_readings": seed_readings,
        "majority_reading": {"reading": majority[0], "votes": majority[1],
                             "of": len(per_seed), "decisive": majority[1] * 2 > len(per_seed)},
        "pooled_metrics": pooled_metrics,
        "pooled_reading": pooled_reading,
        "per_seed_leans": {str(payload["seed"]): payload["reading"]["leans"] for payload in per_seed},
        "per_seed_text_leans": {str(payload["seed"]): payload["reading"]["text_leans"] for payload in per_seed},
        "per_seed_grounding": {str(payload["seed"]): payload["reading"]["grounding"] for payload in per_seed},
        "gates": {str(payload["seed"]): payload["gates"] for payload in per_seed},
        "wall_seconds": round(sum(payload["wall_seconds"] for payload in per_seed), 3),
    }


# --------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------

def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def metrics_table(metrics: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = ["| mode | action | packet | shuffled | habit | text | lean | text lean | swap | grounding |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for mode in MODES:
        m = metrics[mode]
        p = m["present_share"]
        lines.append(
            f"| {mode} | {m['action']} | {fmt(p['packet'])} | {fmt(p['shuffled'])} | "
            f"{fmt(p['habit'])} | {fmt(p['text'])} | {fmt(m['lean'])} | "
            f"{fmt(m['text_lean'])} | {fmt(m['swap'])} | {fmt(m['grounding'])} |"
        )
    return lines


def write_seed_markdown(payload: Mapping[str, Any], path: Path) -> None:
    host = payload["host"]["label"]
    lines = [f"# V-line rung 1 — {host}, door seed {payload['seed']}", ""]
    if payload["smoke"]:
        lines += ["**SMOKE ARTIFACT** — two-step doors on a handful of frames; not the registered run.", ""]
    lines += [f"Registration: {payload['registration']}.", "",
              f"Reading at this seed: **{payload['reading']['reading']}**.", "",
              "Present-frame share of the mode's action per arm (the lean is packet − habit; "
              "grounding is the mode's action share on ABSENT frames under the packet):", ""]
    lines += metrics_table(payload["metrics"])
    gates = payload["gates"]
    lines += ["", f"Gates: held-out phrasing token accuracy {fmt(gates['held_out_phrasings']['pooled'])} "
              f"({'pass' if gates['held_out_phrasings']['passed'] else 'FAIL'}); zero-dose "
              f"{'exact' if gates['zero_dose']['exact'] else 'NOT exact'}; base weights unchanged.",
              f"Layer plan: tower depth {payload['layer_plan']['depth']}, source {payload['layer_plan']['source_layer']}, "
              f"inject {payload['layer_plan']['inject_layers']} (prior, no scan). Dose {payload['doors']['recipe']['dose']}, "
              f"{payload['doors']['recipe']['steps']} steps.",
              f"Wall {payload['wall_seconds']} s on {payload['host_machine']} ({payload['device']}, {payload['dtype']})."]
    lines += ["", "Habit mix on present / absent frames: "
              + json.dumps(payload["metrics"][MODES[0]]["present_mix"]["habit"]) + " / "
              + json.dumps(payload["metrics"][MODES[0]]["absent_mix"]["habit"]) + "."]
    path.write_text("\n".join(lines) + "\n")


def write_pooled_markdown(payload: Mapping[str, Any], path: Path) -> None:
    host = payload["host"]["label"]
    lines = [f"# V-line rung 1 — {host}, pooled over door seeds {payload['seeds']}", ""]
    if payload["smoke"]:
        lines += ["**SMOKE ARTIFACT**", ""]
    maj = payload["majority_reading"]
    lines += [f"Registration: {payload['registration']}.", "",
              f"Per-seed readings: {json.dumps(payload['per_seed_readings'])} — majority "
              f"**{maj['reading']}** ({maj['votes']} of {maj['of']}"
              f"{'' if maj['decisive'] else ', NOT a majority'}).",
              f"Pooled-number reading: {payload['pooled_reading']['reading']}.", "",
              "Pooled present-frame shares:", ""]
    lines += metrics_table(payload["pooled_metrics"])
    lines += ["", "Per-seed leans: " + json.dumps(payload["per_seed_leans"]),
              "Per-seed text leans: " + json.dumps(payload["per_seed_text_leans"]),
              "Per-seed grounding: " + json.dumps(payload["per_seed_grounding"])]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, default=str), flush=True)


def run_host(args, host_key: str, seeds: Sequence[int], frames, base: Path) -> dict[str, Any]:
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    processor, model, tower, depth, hidden, dtype = load_host(
        host_key, device, local_files_only=args.local_files_only
    )
    plan = layer_plan(depth)
    emit("vm_host", host=host_key, model_id=HOSTS[host_key]["model_id"], depth=depth,
         hidden=hidden, device=str(device), dtype=str(dtype), plan=plan,
         **cuda_snapshot(device))
    bank = FrameBank(processor, device, dtype)
    base_samples = parameter_samples(model)
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        payload = run_seed(
            args, host_key=host_key, seed=seed, model=model, tower=tower,
            processor=processor, device=device, dtype=dtype, depth=depth,
            hidden=hidden, frames=frames, bank=bank, base_samples=base_samples,
            plan=plan,
        )
        prefix = base.with_name(f"{base.name}-{host_key}-s{seed}")
        atomic_json_dump(payload, prefix.with_suffix(".json"))
        write_seed_markdown(payload, prefix.with_suffix(".md"))
        emit("vm_seed_written", host=host_key, seed=seed, json=display_path(prefix.with_suffix(".json")))
        per_seed.append(payload)
    pooled = pooled_payload(host_key, per_seed)
    prefix = base.with_name(f"{base.name}-{host_key}")
    atomic_json_dump(pooled, prefix.with_suffix(".json"))
    write_pooled_markdown(pooled, prefix.with_suffix(".md"))
    emit("vm_host_written", host=host_key, json=display_path(prefix.with_suffix(".json")),
         majority=pooled["majority_reading"])
    del model, tower, bank
    release_device_memory(device)
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=(*HOSTS, "all"), default="450m")
    parser.add_argument("--seed", type=int, default=None, choices=SEEDS)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="two-step doors, two frames per class; scratch prefix required")
    parser.add_argument("--output-prefix", default=None,
                        help="the BASE; host and seed names derive from it")
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
        base = ROOT / "findings" / CANONICAL_STEM
    else:
        base = Path(args.output_prefix).expanduser()
        if base.name.startswith(CANONICAL_STEM) and base.parent.resolve() == (ROOT / "findings").resolve():
            if args.smoke:
                raise SystemExit("a smoke may not write the canonical family")
    coco_dir = Path(args.coco_dir).expanduser() if args.coco_dir else default_coco_dir()
    per_class_train = SMOKE_FRAMES if args.smoke else TRAIN_PER_CLASS
    per_class_eval = SMOKE_FRAMES if args.smoke else EVAL_PER_CLASS
    frames = select_frames(coco_dir, per_class_train=per_class_train, per_class_eval=per_class_eval)
    emit("vm_frames", train=len(frames["train"]), eval=len(frames["eval"]), pool=frames["pool"])

    hosts = list(HOST_ORDER) if args.host == "all" else [args.host]
    for host_key in hosts:
        run_host(args, host_key, seeds, frames, base)


if __name__ == "__main__":
    main()
