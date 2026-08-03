"""Stage J: on-policy neutral distillation of the shared center.

Stage I fixed the tensor-path discontinuity — zero state is the same prefix
path a decaying coordinate approaches — but its neutral-reference
cross-entropy loss taught the center *another plausible neutral continuation*
rather than the base decoder policy. Its 35B center had zero exact matches to
memory-off and drifted 12.5–33.4% of the explicit span.

Stage J trains the always-present center against the frozen no-prefix model's
own token distribution instead:

* token-level KL from the shared-center path to the no-prefix path, measured
  at the first decoding frontier and along sampled on-policy continuations;
* signed contrastive losses retained only for the axis deltas;
* prefix dropout plus explicit zero-state distillation batches, so the center
  is optimized on the same schedule the deltas are;
* doubled negative patience and goodwill pole coverage, without touching the
  warmth events that carried Stage H.

Unlike stages A–I this rung was designed but never run in the development
tree, so it is the one rung of the ladder that can carry honest frozen rules.
The thresholds below are copied from ``PREREGISTRATION-stage-j.md``; the run
must not edit either copy after the fact.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import gc
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
import zlib

import torch
from torch import nn
import torch.nn.functional as F

from .data import (
    BASE_SYSTEM_PROMPT,
    STYLE_AXIS_NAMES,
    StylePair,
    build_style_pairs,
    lexical_style_score,
    style_system_prompt,
    validate_style_data,
)
from .metrics import aggregate_responses, axis_response_metrics, word_jaccard
from .prefix import (
    NeutralAnchoredPrefixBank,
    SoftPrefixBank,
    insert_prefix_embeddings,
    pole_training_schedule,
)
from .runtime import (
    FULL_PROFILE,
    SMOKE_PROFILE,
    ModelProfile,
    assistant_header_start,
    atomic_json_dump,
    atomic_torch_save,
    bounded_context,
    choose_device,
    cuda_gate,
    emit,
    environment_report,
    generate_response,
    load_model,
    memory_snapshot,
    model_input_device,
    pair_margin,
    prefix_response_nll,
    release_model,
    render_prompt,
    response_attribution,
)


PRIMARY_AXES = ("warmth", "patience")
SECONDARY_AXES = ("goodwill",)
TRAJECTORY_STATES = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)

# Frozen decision rules. Every number is justified against a measured Stage
# H/I value in PREREGISTRATION-stage-j.md; the verdict below reads only this
# dictionary, so a run cannot pass under thresholds the document does not
# carry.
STAGE_J_PREREG: dict[str, Any] = {
    "registered": "2026-08-01",
    "primary_axes": list(PRIMARY_AXES),
    "secondary_axes": list(SECONDARY_AXES),
    "model_id": "Qwen/Qwen3.6-35B-A3B",
    "model_revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
    "corpus_sha256": (
        "85f6e3599474ad33c37092b583b21ccfc3175fd85b168c65b735c2dd75332346"
    ),
    "seeds": [20260801, 20260802, 20260803],
    "sampled_temperature": 0.8,
    "trajectory_states": list(TRAJECTORY_STATES),
    "trajectory_max_adjacent_inversions": 1,
    "center_drift_fraction_max": 0.10,
    "center_greedy_jaccard_min": 0.60,
    "relative_span_min": {"warmth": 0.40, "patience": 0.20},
    "signed_min_of_24": {"warmth": 21, "patience": 21},
    "specific_min_of_24": {"warmth": 21, "patience": 16},
    "center_specific_min_of_24": {"warmth": 21, "patience": 16},
}


@dataclass(frozen=True)
class StageJConfig:
    suite: str
    stage_h_steps: int
    pole_steps: int
    distill_steps: int
    dropout_rate: float
    distill_new_tokens: int
    learning_rate: float
    sampled_temperature: float
    snapshot_count: int
    test_case_limit: int | None
    max_new_tokens: int
    max_context_tokens: int
    max_score_seq_len: int
    max_train_seq_len: int


FULL_STAGE_J_CONFIG = StageJConfig(
    suite="full",
    stage_h_steps=36,
    pole_steps=48,
    distill_steps=18,
    dropout_rate=0.25,
    distill_new_tokens=24,
    learning_rate=0.001,
    sampled_temperature=0.8,
    snapshot_count=3,
    test_case_limit=None,
    max_new_tokens=96,
    max_context_tokens=512,
    max_score_seq_len=512,
    max_train_seq_len=256,
)

SMOKE_STAGE_J_CONFIG = StageJConfig(
    suite="smoke",
    stage_h_steps=6,
    pole_steps=6,
    distill_steps=3,
    dropout_rate=0.25,
    distill_new_tokens=8,
    learning_rate=0.001,
    sampled_temperature=0.8,
    snapshot_count=2,
    test_case_limit=1,
    max_new_tokens=24,
    max_context_tokens=256,
    max_score_seq_len=256,
    max_train_seq_len=256,
)


def _stop_tokens(tokenizer: Any) -> set[int]:
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, (tuple, list, set)):
        return {int(value) for value in eos}
    return {int(eos)} if eos is not None else set()


def condition_generator(seed: int, condition: str, pair_id: str) -> torch.Generator:
    """A CPU generator whose stream is fixed by (run seed, condition, prompt).

    Sampled decoding must be reproducible and must give every condition its
    own stream, otherwise two conditions with near-identical distributions
    would emit identical text for the wrong reason.
    """
    generator = torch.Generator()
    generator.manual_seed(zlib.crc32(f"{seed}:{condition}:{pair_id}".encode()))
    return generator


def release_decode_transients(device: torch.device) -> None:
    """Hand decode caches back to the allocator between generations.

    Every Stage J generation runs in inference mode: on the 35B target, a
    single greedy decode in plain grad mode after the training phase's
    backward passes retained about 190 MB per generated token and hit the
    allocator cap at 83.6 GiB (measured on smarty, 2026-08-01), while the
    identical decode under inference mode stayed flat at 64.7 GiB. Returning
    the freed KV/conv cache blocks eagerly keeps the free-memory gates
    reading reality rather than allocator history.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()


def phase_cleanup(device: torch.device) -> None:
    """Collect and release between run phases (training, audits, seeds)."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def token_kl(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """Mean token-level KL(no-prefix ‖ shared-center), in float32.

    Forward KL against the full teacher distribution is the distillation
    objective proper: matching the whole distribution, not one sampled
    reference continuation, is exactly what Stage I's cross-entropy loss
    failed to do.
    """
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher and student logits must share a shape")
    teacher_log = F.log_softmax(teacher_logits.float(), dim=-1)
    student_log = F.log_softmax(student_logits.float(), dim=-1)
    return torch.sum(teacher_log.exp() * (teacher_log - student_log), dim=-1).mean()


@torch.inference_mode()
def sample_continuation_ids(
    model: nn.Module,
    tokenizer: Any,
    *,
    user_prompt: str,
    prefix: torch.Tensor,
    new_tokens: int,
    temperature: float,
    generator: torch.Generator,
    max_seq_len: int,
) -> torch.Tensor:
    """Sample an on-policy continuation from the prefixed path.

    Recomputes the full forward at every step instead of using the KV cache:
    this runs inside the training loop, where gradient checkpointing may
    silently disable caching, and correctness beats speed at 24 tokens.
    """
    if temperature <= 0.0:
        raise ValueError("sampling temperature must be positive")
    prompt = bounded_context(
        render_prompt(
            tokenizer,
            system_prompt=BASE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        ),
        max_seq_len,
    )
    device = model_input_device(model)
    insertion = assistant_header_start(
        tokenizer,
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    fixed_prefix = prefix.detach()
    stop = _stop_tokens(tokenizer)
    ids = prompt.to(device)
    tokens: list[int] = []
    for _ in range(new_tokens):
        if ids.shape[1] + fixed_prefix.shape[0] + 1 > max_seq_len:
            break
        embeddings = model.get_input_embeddings()(ids)
        combined = insert_prefix_embeddings(
            embeddings,
            fixed_prefix,
            insertion=insertion,
        )
        logits = model(inputs_embeds=combined, use_cache=False).logits
        values = logits[0, -1].float().cpu()
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError("on-policy sampling logits are non-finite")
        probabilities = F.softmax(values / temperature, dim=-1)
        token = int(torch.multinomial(probabilities, 1, generator=generator))
        tokens.append(token)
        if token in stop:
            break
        ids = torch.cat(
            [ids, torch.tensor([[token]], dtype=torch.long, device=device)],
            dim=1,
        )
    return torch.tensor(tokens, dtype=torch.long)


def distillation_kl(
    model: nn.Module,
    tokenizer: Any,
    *,
    user_prompt: str,
    prefix: torch.Tensor,
    continuation: torch.Tensor,
    max_seq_len: int,
) -> tuple[torch.Tensor, int]:
    """KL between the center path and the no-prefix path on one trajectory.

    Positions covered: the first decoding frontier plus every position along
    the supplied continuation. The teacher pass has no prefix and no gradient;
    the student pass inserts the center prefix, so its logits sit
    ``prefix_tokens`` positions later in its own sequence.
    """
    if continuation.ndim != 1:
        raise ValueError("continuation must be a token vector")
    prompt = bounded_context(
        render_prompt(
            tokenizer,
            system_prompt=BASE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        ),
        max_seq_len,
    )
    full = prompt
    if continuation.numel():
        full = torch.cat([prompt, continuation.reshape(1, -1).long()], dim=1)
    if full.shape[1] + prefix.shape[0] > max_seq_len:
        raise ValueError("distillation trajectory exceeds the training token limit")
    device = model_input_device(model)
    inputs = full.to(device)
    frontier = prompt.shape[1] - 1
    with torch.no_grad():
        teacher = model(input_ids=inputs, use_cache=False).logits[:, frontier:, :]
        teacher = teacher.float()
    insertion = assistant_header_start(
        tokenizer,
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    with torch.no_grad():
        token_embeddings = model.get_input_embeddings()(inputs)
    combined = insert_prefix_embeddings(
        token_embeddings,
        prefix,
        insertion=insertion,
    )
    student = model(inputs_embeds=combined, use_cache=False).logits
    student = student[:, frontier + prefix.shape[0] :, :].float()
    if student.shape[1] != teacher.shape[1]:
        raise AssertionError("distillation teacher and student are misaligned")
    loss = token_kl(teacher, student)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("distillation KL is non-finite")
    return loss, int(teacher.shape[1])


def distill_training_schedule(
    pairs: Sequence[StylePair],
    *,
    pole_steps: int,
    distill_steps: int,
    dropout_rate: float,
    extra_negative_axes: Sequence[str] = ("patience", "goodwill"),
    seed: int,
) -> list[tuple[str, StylePair, int]]:
    """The Stage J event schedule.

    Pole events keep the Stage I signed inventory but double the negative
    poles of ``extra_negative_axes`` — the registration's demand for more
    negative patience and goodwill coverage without touching warmth. Prefix
    dropout then converts a seeded fraction of pole events into zero-state
    distillation events on the same pair, and the explicit distillation steps
    are appended on top, so the center trains against the no-prefix path
    throughout the schedule rather than in a separate phase.
    """
    if pole_steps < 0 or distill_steps < 0 or pole_steps + distill_steps <= 0:
        raise ValueError("invalid Stage J step counts")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("prefix dropout rate must be in [0, 1)")
    base = [(pair, sign) for pair in pairs for sign in (1, -1)]
    base.extend((pair, -1) for pair in pairs if pair.axis in set(extra_negative_axes))
    if not base:
        raise ValueError("training pairs are empty")
    randomizer = random.Random(seed)
    pole_events: list[tuple[StylePair, int]] = []
    while len(pole_events) < pole_steps:
        epoch = list(base)
        randomizer.shuffle(epoch)
        pole_events.extend(epoch)
    events: list[tuple[str, StylePair, int]] = []
    for pair, sign in pole_events[:pole_steps]:
        if randomizer.random() < dropout_rate:
            events.append(("distill", pair, 0))
        else:
            events.append(("pole", pair, sign))
    distill_pairs: list[StylePair] = []
    while len(distill_pairs) < distill_steps:
        epoch = list(pairs)
        randomizer.shuffle(epoch)
        distill_pairs.extend(epoch)
    events.extend(("distill", pair, 0) for pair in distill_pairs[:distill_steps])
    randomizer.shuffle(events)
    return events


def train_stage_h(
    model: nn.Module,
    tokenizer: Any,
    bank: SoftPrefixBank,
    *,
    pairs: Sequence[StylePair],
    steps: int,
    seed: int,
    max_seq_len: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """The frozen Stage H recipe, run only to give Stage J its initialization.

    Identical to the confirmatory H/I runner: contrastive paired NLL on signed
    prefixes at learning rate 0.001, diversity 0.01, anchor 0.001, clip 1.0.
    Stage J does not audit this bank; it exists so the neutral-anchored bank
    starts from the same lineage Stage I started from.
    """
    optimizer = torch.optim.AdamW(bank.parameters(), lr=0.001, weight_decay=0.0)
    schedule = pole_training_schedule(pairs, steps=steps, seed=seed)
    history: list[dict[str, Any]] = []
    checkpointing = hasattr(model, "gradient_checkpointing_enable")
    if checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.train()
    try:
        for step, (pair, sign) in enumerate(schedule, start=1):
            optimizer.zero_grad(set_to_none=True)
            desired_response = (
                pair.positive_response if sign > 0 else pair.negative_response
            )
            rejected_response = (
                pair.negative_response if sign > 0 else pair.positive_response
            )
            desired, desired_tokens = prefix_response_nll(
                model,
                tokenizer,
                pair=pair,
                response=desired_response,
                prefix=bank.prefix(pair.axis, sign),
                max_seq_len=max_seq_len,
            )
            desired_value = float(desired.detach().cpu())
            desired.backward()
            del desired
            rejected, rejected_tokens = prefix_response_nll(
                model,
                tokenizer,
                pair=pair,
                response=rejected_response,
                prefix=bank.prefix(pair.axis, sign),
                max_seq_len=max_seq_len,
            )
            rejected_value = float(rejected.detach().cpu())
            (-rejected).backward()
            del rejected
            diversity = bank.diversity_loss()
            anchor = bank.anchor_loss()
            (0.01 * diversity + 0.001 * anchor).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(f"non-finite Stage H gradient at {step}")
            optimizer.step()
            bank.project_norms_()
            snapshot = cuda_gate(
                device,
                stage=f"Stage H step {step}",
                minimum_free_gib=20.0 if device.type == "cuda" else 0.0,
            )
            record = {
                "step": step,
                "pair_id": pair.pair_id,
                "axis": pair.axis,
                "sign": sign,
                "desired_nll": desired_value,
                "rejected_nll": rejected_value,
                "training_margin": rejected_value - desired_value,
                "desired_tokens": desired_tokens,
                "rejected_tokens": rejected_tokens,
                "grad_norm": float(grad_norm.detach().cpu()),
                "memory": snapshot,
            }
            history.append(record)
            if step == 1 or step == steps or step % 6 == 0:
                emit("stage_j_h_init_step", seed=seed, **record)
    finally:
        model.eval()
        if checkpointing:
            model.gradient_checkpointing_disable()
        del optimizer
    return history


def _dev_selection(
    model: nn.Module,
    tokenizer: Any,
    bank: NeutralAnchoredPrefixBank,
    *,
    dev_pairs: Sequence[StylePair],
    config: StageJConfig,
) -> dict[str, float]:
    """The frozen snapshot-selection score: dev margin minus dev center KL.

    Both terms are in nats. The margin term rewards signed control on the
    primary axes; the KL term is the frontier-only distillation loss on the
    dev prompts, so a snapshot cannot buy margin by letting the center drift.
    """
    margins: list[float] = []
    for pair in dev_pairs:
        if pair.axis not in PRIMARY_AXES:
            continue
        positive = pair_margin(
            model,
            tokenizer,
            pair=pair,
            prefix=bank.prefix(pair.axis, 1),
            max_seq_len=config.max_train_seq_len,
        )["margin"]
        negative = pair_margin(
            model,
            tokenizer,
            pair=pair,
            prefix=bank.prefix(pair.axis, -1),
            max_seq_len=config.max_train_seq_len,
        )["margin"]
        margins.append(positive - negative)
    if not margins:
        raise ValueError("dev selection requires primary-axis dev pairs")
    seen: set[str] = set()
    kl_values: list[float] = []
    with torch.no_grad():
        for pair in dev_pairs:
            if pair.prompt in seen:
                continue
            seen.add(pair.prompt)
            loss, _ = distillation_kl(
                model,
                tokenizer,
                user_prompt=pair.prompt,
                prefix=bank.neutral_prefix(),
                continuation=torch.empty(0, dtype=torch.long),
                max_seq_len=config.max_train_seq_len,
            )
            kl_values.append(float(loss.cpu()))
    dev_margin = sum(margins) / len(margins)
    dev_kl = sum(kl_values) / len(kl_values)
    return {
        "dev_margin": dev_margin,
        "dev_kl": dev_kl,
        "score": dev_margin - dev_kl,
    }


def train_stage_j(
    model: nn.Module,
    tokenizer: Any,
    bank: NeutralAnchoredPrefixBank,
    *,
    pairs: Sequence[StylePair],
    dev_pairs: Sequence[StylePair],
    config: StageJConfig,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Distill the center on-policy while training deltas contrastively.

    Pole steps update only the deltas; distillation steps update only the
    center. The center's loss never sees a curated reference response — its
    only teacher is the frozen model's own no-prefix distribution on the
    frontier and on continuations the center path itself sampled.
    """
    optimizer = torch.optim.AdamW(
        bank.parameters(),
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    schedule = distill_training_schedule(
        pairs,
        pole_steps=config.pole_steps,
        distill_steps=config.distill_steps,
        dropout_rate=config.dropout_rate,
        seed=seed,
    )
    total = len(schedule)
    snapshot_steps = sorted(
        {
            max(1, round(total * index / config.snapshot_count))
            for index in range(1, config.snapshot_count + 1)
        }
    )
    history: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    checkpointing = hasattr(model, "gradient_checkpointing_enable")
    if checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.train()
    try:
        for step, (kind, pair, sign) in enumerate(schedule, start=1):
            optimizer.zero_grad(set_to_none=True)
            desired_value = None
            rejected_value = None
            kl_value = None
            desired_tokens = rejected_tokens = kl_positions = 0
            if kind == "pole":
                desired_response = (
                    pair.positive_response if sign > 0 else pair.negative_response
                )
                rejected_response = (
                    pair.negative_response if sign > 0 else pair.positive_response
                )
                desired, desired_tokens = prefix_response_nll(
                    model,
                    tokenizer,
                    pair=pair,
                    response=desired_response,
                    prefix=bank.prefix(pair.axis, sign),
                    max_seq_len=config.max_train_seq_len,
                )
                desired_value = float(desired.detach().cpu())
                desired.backward()
                del desired
                rejected, rejected_tokens = prefix_response_nll(
                    model,
                    tokenizer,
                    pair=pair,
                    response=rejected_response,
                    prefix=bank.prefix(pair.axis, sign),
                    max_seq_len=config.max_train_seq_len,
                )
                rejected_value = float(rejected.detach().cpu())
                (-rejected).backward()
                del rejected
            else:
                generator = torch.Generator()
                generator.manual_seed(
                    zlib.crc32(f"{seed}:distill:{step}:{pair.pair_id}".encode())
                )
                # Sample in eval mode so gradient checkpointing does not wrap
                # a forward that carries no gradient.
                model.eval()
                continuation = sample_continuation_ids(
                    model,
                    tokenizer,
                    user_prompt=pair.prompt,
                    prefix=bank.neutral_prefix(),
                    new_tokens=config.distill_new_tokens,
                    temperature=1.0,
                    generator=generator,
                    max_seq_len=config.max_train_seq_len,
                )
                model.train()
                kl, kl_positions = distillation_kl(
                    model,
                    tokenizer,
                    user_prompt=pair.prompt,
                    prefix=bank.neutral_prefix(),
                    continuation=continuation,
                    max_seq_len=config.max_train_seq_len,
                )
                kl_value = float(kl.detach().cpu())
                kl.backward()
                del kl, continuation
            diversity = bank.diversity_loss()
            anchor = bank.anchor_loss()
            (0.01 * diversity + 0.001 * anchor).backward()
            if kind == "pole":
                bank.neutral.grad = None
            else:
                bank.delta.grad = None
            parameters = [
                parameter
                for parameter in bank.parameters()
                if parameter.grad is not None
            ]
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(f"non-finite Stage J gradient at {step}")
            optimizer.step()
            bank.project_norms_()
            snapshot = cuda_gate(
                device,
                stage=f"Stage J step {step}",
                minimum_free_gib=20.0 if device.type == "cuda" else 0.0,
            )
            record = {
                "step": step,
                "kind": kind,
                "pair_id": pair.pair_id,
                "axis": pair.axis,
                "sign": sign,
                "desired_nll": desired_value,
                "rejected_nll": rejected_value,
                "training_margin": (
                    rejected_value - desired_value
                    if desired_value is not None and rejected_value is not None
                    else None
                ),
                "distillation_kl": kl_value,
                "desired_tokens": desired_tokens,
                "rejected_tokens": rejected_tokens,
                "kl_positions": kl_positions,
                "diversity_loss": float(diversity.detach().cpu()),
                "anchor_loss": float(anchor.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "geometry": bank.geometry(),
                "memory": snapshot,
            }
            history.append(record)
            if step == 1 or step == total or step % 9 == 0:
                emit("stage_j_step", seed=seed, **record)
            if step in snapshot_steps:
                model.eval()
                selection = _dev_selection(
                    model,
                    tokenizer,
                    bank,
                    dev_pairs=dev_pairs,
                    config=config,
                )
                model.train()
                snapshots.append(
                    {
                        "step": step,
                        **selection,
                        "neutral": bank.neutral.detach().float().cpu().clone(),
                        "delta": bank.delta.detach().float().cpu().clone(),
                    }
                )
                emit(
                    "stage_j_snapshot",
                    seed=seed,
                    step=step,
                    dev_margin=selection["dev_margin"],
                    dev_kl=selection["dev_kl"],
                    score=selection["score"],
                )
    finally:
        model.eval()
        if checkpointing:
            model.gradient_checkpointing_disable()
        del optimizer
    best = max(snapshots, key=lambda entry: entry["score"])
    with torch.no_grad():
        bank.neutral.copy_(best["neutral"].to(bank.neutral.device))
        bank.delta.copy_(best["delta"].to(bank.delta.device))
    selection = {
        "selected_step": best["step"],
        "snapshots": [
            {key: entry[key] for key in ("step", "dev_margin", "dev_kl", "score")}
            for entry in snapshots
        ],
    }
    emit("stage_j_selected", seed=seed, **selection)
    return {"history": history, "selection": selection}


@torch.inference_mode()
def sample_response(
    model: nn.Module,
    tokenizer: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    prefix: torch.Tensor | None,
    max_new_tokens: int,
    max_context_tokens: int,
    temperature: float,
    generator: torch.Generator,
) -> str:
    """Temperature sampling along the cached decode path.

    The mirror of ``runtime.generate_response`` with ``multinomial`` in place
    of ``argmax``; probabilities are computed on CPU in float32 so the stream
    drawn from ``generator`` is device-independent.
    """
    if temperature <= 0.0:
        raise ValueError("sampling temperature must be positive")
    prompt = render_prompt(
        tokenizer,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    if prompt.shape[1] > max_context_tokens:
        raise ValueError("generation prompt exceeds the context limit")
    device = model_input_device(model)
    prompt = prompt.to(device)
    current_ids: torch.Tensor | None = prompt
    current_embeds: torch.Tensor | None = None
    if prefix is not None:
        token_embeddings = model.get_input_embeddings()(prompt)
        current_embeds = insert_prefix_embeddings(
            token_embeddings,
            prefix,
            insertion=assistant_header_start(
                tokenizer,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            ),
        )
        current_ids = None
    stop = _stop_tokens(tokenizer)
    past_key_values = None
    tokens: list[int] = []
    for _ in range(max_new_tokens):
        output = model(
            input_ids=current_ids,
            inputs_embeds=current_embeds,
            past_key_values=past_key_values,
            use_cache=True,
        )
        values = output.logits[0, -1].float().cpu()
        probabilities = F.softmax(values / temperature, dim=-1)
        token = int(torch.multinomial(probabilities, 1, generator=generator))
        tokens.append(token)
        past_key_values = output.past_key_values
        current_ids = torch.tensor([[token]], dtype=torch.long, device=device)
        current_embeds = None
        if token in stop:
            break
    return tokenizer.decode(tokens, skip_special_tokens=True).strip()


def _pairs_for_axis(
    axis: str,
    split: str,
    *,
    case_limit: int | None = None,
) -> list[StylePair]:
    pairs = build_style_pairs((axis,), (split,))
    return pairs[:case_limit] if case_limit is not None else pairs


def _measure_generation(
    model: nn.Module,
    tokenizer: Any,
    *,
    pair: StylePair,
    axis: str,
    condition: str,
    system_prompt: str,
    prefix: torch.Tensor | None,
    decoding: str,
    seed: int,
    config: StageJConfig,
) -> dict[str, Any]:
    # Generation must not touch autograd: the prefix parameters carry grad,
    # and a grad-mode decode after the training phase retains per-step state
    # (see release_decode_transients). Detach here so every caller is safe.
    if prefix is not None:
        prefix = prefix.detach()
    if decoding == "greedy":
        with torch.inference_mode():
            text = generate_response(
                model,
                tokenizer,
                system_prompt=system_prompt,
                user_prompt=pair.prompt,
                prefix=prefix,
                max_new_tokens=config.max_new_tokens,
                max_context_tokens=config.max_context_tokens,
            )
    elif decoding == "sampled":
        text = sample_response(
            model,
            tokenizer,
            system_prompt=system_prompt,
            user_prompt=pair.prompt,
            prefix=prefix,
            max_new_tokens=config.max_new_tokens,
            max_context_tokens=config.max_context_tokens,
            temperature=config.sampled_temperature,
            generator=condition_generator(seed, condition, pair.pair_id),
        )
    else:
        raise ValueError(f"unknown decoding mode {decoding}")
    return {
        "text": text,
        "attribution": response_attribution(
            model,
            tokenizer,
            axis=axis,
            prompt=pair.prompt,
            response=text,
            max_seq_len=config.max_score_seq_len,
        ),
        "lexical": lexical_style_score(text, axis),
        "word_count": len(text.split()),
        "word_jaccard_with_core": word_jaccard(text, pair.neutral_response),
    }


# Trajectory conditions that reuse a prefix already generated by the signed
# audit: the zero state is the neutral center and the ±1 states are the signed
# poles, on the identical tensor path.
_TRAJECTORY_ALIASES = {
    "state_+0": "neutral_center",
    "state_+1": "prefix_positive",
    "state_-1": "prefix_negative",
}


def audit_stage_j(
    model: nn.Module,
    tokenizer: Any,
    bank: NeutralAnchoredPrefixBank,
    *,
    decoding: str,
    seed: int,
    config: StageJConfig,
    device: torch.device,
    cross_seed_cache: dict[tuple[str, str, str, str], dict[str, Any]],
    seed_cache: dict[tuple[str, str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    """The 168-generation audit under one decoding mode.

    Same condition set as the Stage I audit — off, neutral center, explicit
    poles, signed prefixes, wrong axis — so ``axis_response_metrics`` applies
    unchanged. Greedy prefix-free baselines are cached across seeds; sampled
    baselines depend on the run seed and are cached only within it.
    """
    rows: list[dict[str, Any]] = []
    axes = list(bank.axis_names)
    for axis_index, axis in enumerate(axes):
        wrong_axis = axes[(axis_index + 1) % len(axes)]
        for pair in _pairs_for_axis(axis, "test", case_limit=config.test_case_limit):
            cuda_gate(
                device,
                stage=f"Stage J {decoding} audit {pair.pair_id} preflight",
                minimum_free_gib=20.0 if device.type == "cuda" else 0.0,
            )
            conditions: list[tuple[str, str, torch.Tensor | None]] = [
                ("off", BASE_SYSTEM_PROMPT, None),
                ("neutral_center", BASE_SYSTEM_PROMPT, bank.neutral_prefix()),
                ("explicit_positive", style_system_prompt(axis, 1), None),
                ("explicit_negative", style_system_prompt(axis, -1), None),
                ("prefix_positive", BASE_SYSTEM_PROMPT, bank.prefix(axis, 1)),
                ("prefix_negative", BASE_SYSTEM_PROMPT, bank.prefix(axis, -1)),
                ("wrong_axis", BASE_SYSTEM_PROMPT, bank.prefix(wrong_axis, 1)),
            ]
            generated: dict[str, dict[str, Any]] = {}
            for condition, system_prompt, prefix in conditions:
                key = (decoding, axis, pair.pair_id, condition)
                if prefix is None and decoding == "greedy" and key in cross_seed_cache:
                    measured = copy.deepcopy(cross_seed_cache[key])
                elif key in seed_cache:
                    measured = copy.deepcopy(seed_cache[key])
                else:
                    measured = _measure_generation(
                        model,
                        tokenizer,
                        pair=pair,
                        axis=axis,
                        condition=condition,
                        system_prompt=system_prompt,
                        prefix=prefix,
                        decoding=decoding,
                        seed=seed,
                        config=config,
                    )
                    seed_cache[key] = copy.deepcopy(measured)
                    if prefix is None and decoding == "greedy":
                        cross_seed_cache[key] = copy.deepcopy(measured)
                    release_decode_transients(device)
                generated[condition] = measured
                emit(
                    "stage_j_generated",
                    decoding=decoding,
                    seed=seed,
                    pair_id=pair.pair_id,
                    axis=axis,
                    condition=condition,
                    attribution=measured["attribution"]["signed_attribution"],
                    words=measured["word_count"],
                    memory=memory_snapshot(device),
                )
            off = generated["off"]["text"]
            for measured in generated.values():
                measured["exactly_matches_off"] = measured["text"] == off
                measured["word_jaccard_with_off"] = word_jaccard(
                    measured["text"],
                    off,
                )
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "axis": axis,
                    "wrong_axis": wrong_axis,
                    "prompt": pair.prompt,
                    "responses": generated,
                }
            )
            cuda_gate(
                device,
                stage=f"Stage J {decoding} audit {pair.pair_id}",
                minimum_free_gib=20.0 if device.type == "cuda" else 0.0,
            )
    report = {
        "format": f"stage-j-{decoding}-responses-repro-v1",
        "created_at": time.time(),
        "seed": seed,
        "decoding": decoding,
        "axes": axes,
        "prefix_tokens": bank.prefix_tokens,
        "geometry": bank.geometry(),
        "rows": rows,
        "aggregate": aggregate_responses(rows),
        "memory": memory_snapshot(device),
    }
    report["metrics"] = _audit_metrics(report)
    return report


def trajectory_audit(
    model: nn.Module,
    tokenizer: Any,
    bank: NeutralAnchoredPrefixBank,
    *,
    seed: int,
    config: StageJConfig,
    device: torch.device,
    seed_cache: dict[tuple[str, str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Greedy decoding along the decay trajectory of each axis.

    Every state uses ``prefix = neutral + state × delta`` — the same tensor
    path, never prefix removal — so this measures the writer exactly where a
    temporal decay process would place it between a signed event and rest.
    """
    states = TRAJECTORY_STATES
    rows: list[dict[str, Any]] = []
    for axis in bank.axis_names:
        for pair in _pairs_for_axis(axis, "test", case_limit=config.test_case_limit):
            cuda_gate(
                device,
                stage=f"Stage J trajectory {pair.pair_id} preflight",
                minimum_free_gib=20.0 if device.type == "cuda" else 0.0,
            )
            generated: dict[str, dict[str, Any]] = {}
            for state in states:
                condition = f"state_{state:+g}"
                sign = 1 if state >= 0 else -1
                prefix = bank.prefix(axis, sign, strength=abs(state))
                alias = _TRAJECTORY_ALIASES.get(condition, condition)
                key = ("greedy", axis, pair.pair_id, alias)
                if key in seed_cache:
                    measured = copy.deepcopy(seed_cache[key])
                else:
                    measured = _measure_generation(
                        model,
                        tokenizer,
                        pair=pair,
                        axis=axis,
                        condition=condition,
                        system_prompt=BASE_SYSTEM_PROMPT,
                        prefix=prefix,
                        decoding="greedy",
                        seed=seed,
                        config=config,
                    )
                    seed_cache[key] = copy.deepcopy(measured)
                    release_decode_transients(device)
                generated[condition] = measured
                emit(
                    "stage_j_trajectory",
                    seed=seed,
                    pair_id=pair.pair_id,
                    axis=axis,
                    state=state,
                    attribution=measured["attribution"]["signed_attribution"],
                    memory=memory_snapshot(device),
                )
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "axis": axis,
                    "prompt": pair.prompt,
                    "responses": generated,
                }
            )
            cuda_gate(
                device,
                stage=f"Stage J trajectory {pair.pair_id}",
                minimum_free_gib=20.0 if device.type == "cuda" else 0.0,
            )
    report = {
        "format": "stage-j-trajectories-repro-v1",
        "created_at": time.time(),
        "seed": seed,
        "states": list(states),
        "axes": list(bank.axis_names),
        "rows": rows,
        "memory": memory_snapshot(device),
    }
    report["metrics"] = _trajectory_metrics(report)
    return report


def center_specific_prompts(report: Mapping[str, Any], axis: str) -> int:
    """Prompts where the signed span beats the center's own drift.

    The wrong-axis control asks whether the effect is axis-specific; this one
    asks whether it is larger than the shared center's departure from
    memory-off — the second comparison the Stage J registration demands.
    """
    count = 0
    for row in report["rows"]:
        if row["axis"] != axis:
            continue
        responses = row["responses"]
        positive = float(
            responses["prefix_positive"]["attribution"]["signed_attribution"]
        )
        negative = float(
            responses["prefix_negative"]["attribution"]["signed_attribution"]
        )
        center = float(
            responses["neutral_center"]["attribution"]["signed_attribution"]
        )
        off = float(responses["off"]["attribution"]["signed_attribution"])
        count += abs(positive - negative) > abs(center - off)
    return count


def _audit_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for axis in report["axes"]:
        try:
            values = axis_response_metrics(report, axis)
        except ValueError as error:
            metrics[axis] = {"invalid": str(error)}
            continue
        values["center_specific_prompts"] = center_specific_prompts(report, axis)
        metrics[axis] = values
    return metrics


def _trajectory_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for axis in report["axes"]:
        rows = [row for row in report["rows"] if row["axis"] == axis]
        if not rows:
            continue
        means: dict[str, float] = {}
        for state in report["states"]:
            condition = f"state_{state:+g}"
            values = [
                float(
                    row["responses"][condition]["attribution"]["signed_attribution"]
                )
                for row in rows
            ]
            means[f"{state:+g}"] = sum(values) / len(values)
        metrics[axis] = {
            "state_mean_attribution": means,
            "prompts": len(rows),
            **trajectory_gate(
                means,
                max_adjacent_inversions=STAGE_J_PREREG[
                    "trajectory_max_adjacent_inversions"
                ],
            ),
        }
    return metrics


def trajectory_gate(
    state_means: Mapping[str, float],
    *,
    max_adjacent_inversions: int,
) -> dict[str, Any]:
    """The frozen trajectory rule.

    Sign correctness is required at the moderate and full states — the region
    the Stage H nine-point sweep found reliable — and the seven-state mean
    curve may contain at most one adjacent inversion, which covers the
    low-dose wobble that sweep also found. The ±0.25 states are measured, not
    assumed linear.
    """
    by_state = {float(name): float(value) for name, value in state_means.items()}
    required = {-1.0, -0.5, 0.0, 0.5, 1.0}
    if not required <= set(by_state):
        raise ValueError("trajectory means are missing required states")
    center = by_state[0.0]
    sign_correct = (
        by_state[-1.0] < center
        and by_state[-0.5] < center
        and by_state[0.5] > center
        and by_state[1.0] > center
    )
    ordered = [by_state[state] for state in sorted(by_state)]
    inversions = sum(
        1 for left, right in zip(ordered, ordered[1:]) if right < left
    )
    return {
        "sign_correct": sign_correct,
        "adjacent_inversions": inversions,
        "trajectory_passed": sign_correct
        and inversions <= max_adjacent_inversions,
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    if not count:
        raise ValueError("cannot summarize an empty value list")
    middle = count // 2
    median = (
        ordered[middle]
        if count % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    mean = sum(ordered) / count
    variance = (
        sum((value - mean) ** 2 for value in ordered) / (count - 1)
        if count > 1
        else 0.0
    )
    return {
        "mean": mean,
        "median": median,
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "sample_stdev": variance**0.5,
    }


def consolidate_stage_j(
    seed_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not seed_records:
        raise ValueError("no seed records to consolidate")
    consolidated: dict[str, Any] = {
        "greedy": {},
        "sampled": {},
        "trajectory": {},
        "center_content": {},
        "invalid_axes": [],
    }
    for mode in ("greedy", "sampled"):
        for axis in STYLE_AXIS_NAMES:
            rows = [record[mode]["metrics"][axis] for record in seed_records]
            invalid = [row["invalid"] for row in rows if "invalid" in row]
            if invalid:
                consolidated["invalid_axes"].append(
                    {"mode": mode, "axis": axis, "reasons": invalid}
                )
                continue
            consolidated[mode][axis] = {
                "relative_span": _distribution(
                    [float(row["relative_span"]) for row in rows]
                ),
                "generated_span": _distribution(
                    [float(row["generated_span"]) for row in rows]
                ),
                "absolute_center_drift_fraction": _distribution(
                    [float(row["absolute_center_drift_fraction"]) for row in rows]
                ),
                "signed_prompts": sum(int(row["signed_prompts"]) for row in rows),
                "specific_prompts": sum(
                    int(row["specific_prompts"]) for row in rows
                ),
                "center_specific_prompts": sum(
                    int(row["center_specific_prompts"]) for row in rows
                ),
                "strict_center_prompts": sum(
                    int(row["strict_center_prompts"]) for row in rows
                ),
                "total_prompts": sum(int(row["prompts"]) for row in rows),
                "per_seed": rows,
            }
    states = [f"{state:+g}" for state in TRAJECTORY_STATES]
    for axis in STYLE_AXIS_NAMES:
        rows = [
            record["trajectory"]["metrics"][axis]
            for record in seed_records
            if axis in record["trajectory"]["metrics"]
        ]
        if not rows:
            continue
        pooled = {
            state: sum(float(row["state_mean_attribution"][state]) for row in rows)
            / len(rows)
            for state in states
        }
        consolidated["trajectory"][axis] = {
            "state_mean_attribution": pooled,
            "per_seed": rows,
            **trajectory_gate(
                pooled,
                max_adjacent_inversions=STAGE_J_PREREG[
                    "trajectory_max_adjacent_inversions"
                ],
            ),
        }
    consolidated["center_content"] = {
        "greedy_center_jaccard": _distribution(
            [float(record["greedy"]["center_jaccard"]) for record in seed_records]
        ),
        "greedy_center_exact_match_rate": _distribution(
            [
                float(record["greedy"]["center_exact_match_rate"])
                for record in seed_records
            ]
        ),
        "sampled_center_jaccard": _distribution(
            [float(record["sampled"]["center_jaccard"]) for record in seed_records]
        ),
    }
    return consolidated


def _count_gate(observed: int, total: int, minimum_of_24: int) -> bool:
    # Integer cross-multiplication keeps the of-24 registration exact for any
    # prompt total without floating-point edges.
    return observed * 24 >= minimum_of_24 * total


def stage_j_verdict(
    consolidated: Mapping[str, Any],
    *,
    corpus: Mapping[str, Any],
    profile: ModelProfile,
    suite: str,
    seeds: Sequence[int],
    prereg: Mapping[str, Any] = STAGE_J_PREREG,
) -> dict[str, Any]:
    """Apply the frozen rules and emit PASS, FAIL, or INVALID.

    PASS and FAIL are statements about the preregistered configuration only.
    Anything else — wrong corpus, wrong model pin, wrong seeds, a smoke suite,
    or a broken denominator — is INVALID, however the numbers look.
    """
    invalid: list[str] = []
    if corpus.get("sha256") != prereg["corpus_sha256"]:
        invalid.append("corpus hash differs from the preregistered pin")
    if (
        profile.model_id != prereg["model_id"]
        or profile.revision != prereg["model_revision"]
    ):
        invalid.append("model differs from the preregistered pin")
    if suite != "full":
        invalid.append("suite is not the preregistered full suite")
    if list(seeds) != list(prereg["seeds"]):
        invalid.append("seeds differ from the preregistered set")
    for entry in consolidated.get("invalid_axes", ()):
        invalid.append(
            f"{entry['mode']}/{entry['axis']} metrics invalid: {entry['reasons']}"
        )

    center_fidelity: dict[str, Any] = {"passed": True}
    signed_axes: dict[str, Any] = {"passed": True}
    trajectories: dict[str, Any] = {"passed": True}
    try:
        for mode in ("greedy", "sampled"):
            for axis in prereg["primary_axes"]:
                block = consolidated[mode][axis]
                drift = block["absolute_center_drift_fraction"]["median"]
                drift_passed = drift < prereg["center_drift_fraction_max"]
                center_fidelity[f"{mode}_{axis}_drift_median"] = drift
                center_fidelity[f"{mode}_{axis}_drift_passed"] = drift_passed
                center_fidelity["passed"] &= drift_passed

                total = int(block["total_prompts"])
                span = block["relative_span"]["median"]
                axis_gates = {
                    "relative_span_median": span,
                    "relative_span_passed": span
                    >= prereg["relative_span_min"][axis],
                    "signed_prompts": int(block["signed_prompts"]),
                    "signed_passed": _count_gate(
                        int(block["signed_prompts"]),
                        total,
                        prereg["signed_min_of_24"][axis],
                    ),
                    "specific_prompts": int(block["specific_prompts"]),
                    "specific_passed": _count_gate(
                        int(block["specific_prompts"]),
                        total,
                        prereg["specific_min_of_24"][axis],
                    ),
                    "center_specific_prompts": int(
                        block["center_specific_prompts"]
                    ),
                    "center_specific_passed": _count_gate(
                        int(block["center_specific_prompts"]),
                        total,
                        prereg["center_specific_min_of_24"][axis],
                    ),
                    "total_prompts": total,
                }
                axis_gates["passed"] = (
                    axis_gates["relative_span_passed"]
                    and axis_gates["signed_passed"]
                    and axis_gates["specific_passed"]
                    and axis_gates["center_specific_passed"]
                )
                signed_axes[f"{mode}_{axis}"] = axis_gates
                signed_axes["passed"] &= axis_gates["passed"]

        jaccard = consolidated["center_content"]["greedy_center_jaccard"]["median"]
        jaccard_passed = jaccard >= prereg["center_greedy_jaccard_min"]
        center_fidelity["greedy_center_jaccard_median"] = jaccard
        center_fidelity["greedy_center_jaccard_passed"] = jaccard_passed
        center_fidelity["passed"] &= jaccard_passed

        for axis in prereg["primary_axes"]:
            block = consolidated["trajectory"][axis]
            trajectories[axis] = {
                "sign_correct": block["sign_correct"],
                "adjacent_inversions": block["adjacent_inversions"],
                "passed": block["trajectory_passed"],
            }
            trajectories["passed"] &= block["trajectory_passed"]
    except (KeyError, TypeError) as error:
        invalid.append(f"consolidated metrics are incomplete: {error!r}")

    gates = {
        "center_fidelity": center_fidelity,
        "signed_axes": signed_axes,
        "trajectories": trajectories,
    }
    secondary = {
        axis: {
            mode: consolidated.get(mode, {}).get(axis)
            for mode in ("greedy", "sampled", "trajectory")
        }
        for axis in prereg["secondary_axes"]
    }
    if invalid:
        overall = "INVALID"
    elif all(gate["passed"] for gate in gates.values()):
        overall = "PASS"
    else:
        overall = "FAIL"
    return {
        "overall": overall,
        "invalid_reasons": invalid,
        "gates": gates,
        "secondary": secondary,
    }


def write_stage_j_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        f"# {report['format']}",
        "",
        f"- Seed: `{report['seed']}`",
        "",
    ]
    for row in report["rows"]:
        lines.extend(
            [
                f"## {row['pair_id']}",
                "",
                f"**Prompt:** {row['prompt']}",
            ]
        )
        for condition, response in row["responses"].items():
            lines.extend(
                [
                    "",
                    f"### {condition}",
                    "",
                    response["text"],
                    "",
                    (
                        f"`attribution="
                        f"{response['attribution']['signed_attribution']:.4f}`"
                    ),
                ]
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_stage_j_seed(
    model: nn.Module,
    tokenizer: Any,
    profile: ModelProfile,
    config: StageJConfig,
    *,
    seed: int,
    output_dir: Path,
    device: torch.device,
    cross_seed_cache: dict[tuple[str, str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    seed_dir = output_dir / f"seed-{seed}"
    seed_dir.mkdir(parents=True, exist_ok=False)
    # Axis-major fit inventory, matching the H/I confirmatory schedule.
    fit_pairs = [
        pair
        for axis in STYLE_AXIS_NAMES
        for pair in build_style_pairs((axis,), ("fit",))
    ]
    dev_pairs = [
        pair
        for axis in STYLE_AXIS_NAMES
        for pair in build_style_pairs((axis,), ("dev",))
    ]

    stage_h = SoftPrefixBank.initialize(
        model,
        tokenizer,
        axes=STYLE_AXIS_NAMES,
        prefix_tokens=8,
        device=model_input_device(model),
    )
    history_h = train_stage_h(
        model,
        tokenizer,
        stage_h,
        pairs=fit_pairs,
        steps=config.stage_h_steps,
        seed=seed,
        max_seq_len=config.max_train_seq_len,
        device=device,
    )
    stage_h_checkpoint = seed_dir / "stage-h-init-prefix.pt"
    atomic_torch_save(
        stage_h.payload(
            model_id=profile.model_id,
            model_revision=profile.revision,
            seed=seed,
            history=history_h,
        ),
        stage_h_checkpoint,
    )
    phase_cleanup(device)

    bank = NeutralAnchoredPrefixBank.from_stage_h(stage_h)
    del stage_h
    training = train_stage_j(
        model,
        tokenizer,
        bank,
        pairs=fit_pairs,
        dev_pairs=dev_pairs,
        config=config,
        seed=seed,
        device=device,
    )
    stage_j_checkpoint = seed_dir / "stage-j-prefix.pt"
    payload = bank.payload(
        model_id=profile.model_id,
        model_revision=profile.revision,
        seed=seed,
        history=training["history"],
        source_stage_h=str(stage_h_checkpoint.relative_to(output_dir)),
    )
    payload["settings"] = {
        "stage_h_steps": config.stage_h_steps,
        "pole_steps": config.pole_steps,
        "distill_steps": config.distill_steps,
        "dropout_rate": config.dropout_rate,
        "distill_new_tokens": config.distill_new_tokens,
        "learning_rate": config.learning_rate,
        "diversity_weight": 0.01,
        "anchor_weight": 0.001,
        "grad_clip": 1.0,
    }
    payload["selection"] = training["selection"]
    atomic_torch_save(payload, stage_j_checkpoint)
    phase_cleanup(device)

    seed_cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    record: dict[str, Any] = {
        "seed": seed,
        "stage_h_checkpoint": str(stage_h_checkpoint.relative_to(output_dir)),
        "stage_j_checkpoint": str(stage_j_checkpoint.relative_to(output_dir)),
        "selection": training["selection"],
    }
    for decoding in ("greedy", "sampled"):
        report = audit_stage_j(
            model,
            tokenizer,
            bank,
            decoding=decoding,
            seed=seed,
            config=config,
            device=device,
            cross_seed_cache=cross_seed_cache,
            seed_cache=seed_cache,
        )
        report_path = seed_dir / f"stage-j-{decoding}-responses.json"
        atomic_json_dump(report, report_path)
        write_stage_j_markdown(
            report,
            seed_dir / f"stage-j-{decoding}-responses.md",
        )
        center_rows = [
            row["responses"]["neutral_center"] for row in report["rows"]
        ]
        record[decoding] = {
            "responses": str(report_path.relative_to(output_dir)),
            "metrics": report["metrics"],
            "center_jaccard": sum(
                row["word_jaccard_with_off"] for row in center_rows
            )
            / len(center_rows),
            "center_exact_match_rate": sum(
                row["exactly_matches_off"] for row in center_rows
            )
            / len(center_rows),
        }
        phase_cleanup(device)

    trajectories = trajectory_audit(
        model,
        tokenizer,
        bank,
        seed=seed,
        config=config,
        device=device,
        seed_cache=seed_cache,
    )
    trajectory_path = seed_dir / "stage-j-trajectories.json"
    atomic_json_dump(trajectories, trajectory_path)
    write_stage_j_markdown(
        trajectories,
        seed_dir / "stage-j-trajectories.md",
    )
    record["trajectory"] = {
        "responses": str(trajectory_path.relative_to(output_dir)),
        "metrics": trajectories["metrics"],
    }
    record["memory"] = memory_snapshot(device)
    atomic_json_dump(record, seed_dir / "seed-summary.json")
    emit(
        "stage_j_seed_complete",
        seed=seed,
        greedy=record["greedy"]["metrics"],
        sampled=record["sampled"]["metrics"],
    )
    del bank
    phase_cleanup(device)
    return record


def reproduce_stage_j(
    *,
    suite: str,
    seeds: Sequence[int],
    output_dir: Path,
    requested_device: str,
) -> dict[str, Any]:
    if suite == "full":
        profile = FULL_PROFILE
        config = FULL_STAGE_J_CONFIG
    elif suite == "smoke":
        profile = SMOKE_PROFILE
        config = SMOKE_STAGE_J_CONFIG
        seeds = seeds[:1]
    else:
        raise ValueError(f"unknown suite {suite}")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be non-empty and unique")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(requested_device, profile)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", 0)
        # Smarty rail: cap the allocator below the physical card so a runaway
        # reservation aborts in Python instead of freezing the machine.
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(0.88, device=device.index)
    environment = environment_report(profile, device)
    atomic_json_dump(environment, output_dir / "environment.json")
    corpus = validate_style_data()
    model: nn.Module | None = None
    tokenizer: Any | None = None
    started = time.time()
    cross_seed_cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    try:
        model, tokenizer = load_model(profile, device=device)
        records = [
            run_stage_j_seed(
                model,
                tokenizer,
                profile,
                config,
                seed=seed,
                output_dir=output_dir,
                device=device,
                cross_seed_cache=cross_seed_cache,
            )
            for seed in seeds
        ]
        consolidated = consolidate_stage_j(records)
        verdict = stage_j_verdict(
            consolidated,
            corpus=corpus,
            profile=profile,
            suite=suite,
            seeds=seeds,
        )
        summary = {
            "format": "stage-j-onpolicy-distillation-repro-v1",
            "created_at": time.time(),
            "elapsed_seconds": time.time() - started,
            "scientific_target": profile.paper_target,
            "profile": asdict(profile),
            "config": asdict(config),
            "prereg": STAGE_J_PREREG,
            "seeds": list(seeds),
            "corpus": corpus,
            "environment": environment,
            "seed_records": records,
            "consolidated": consolidated,
            "verdict": verdict,
            "memory": memory_snapshot(device),
        }
        atomic_json_dump(summary, output_dir / "stage-j-summary.json")
        emit(
            "stage_j_complete",
            output=str(output_dir),
            verdict=verdict["overall"],
            invalid_reasons=verdict["invalid_reasons"],
            elapsed_seconds=summary["elapsed_seconds"],
        )
        return summary
    finally:
        release_model(device, model, tokenizer)
