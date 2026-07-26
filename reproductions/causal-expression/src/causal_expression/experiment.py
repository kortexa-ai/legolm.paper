"""Frozen-model training and response audits for the reproduction."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import gc
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

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
from .metrics import (
    aggregate_responses,
    consolidate_metrics,
    mean_rows,
    report_metrics,
    strength_sweep_metrics,
    word_jaccard,
)
from .prefix import (
    NeutralAnchoredPrefixBank,
    SoftPrefixBank,
    insert_prefix_embeddings,
    neutral_training_schedule,
    pole_training_schedule,
)


GIB = 1024**3
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model_id: str
    revision: str
    paper_target: bool
    stream_to_cuda: bool


FULL_PROFILE = ModelProfile(
    name="qwen36-35b",
    model_id="Qwen/Qwen3.6-35B-A3B",
    revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
    paper_target=True,
    stream_to_cuda=True,
)

SMOKE_PROFILE = ModelProfile(
    name="qwen35-2b-smoke",
    model_id="Qwen/Qwen3.5-2B",
    revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
    paper_target=False,
    stream_to_cuda=False,
)


@dataclass(frozen=True)
class RunConfig:
    suite: str
    stage_h_steps: int
    stage_i_pole_steps: int
    stage_i_neutral_steps: int
    test_case_limit: int | None
    max_new_tokens: int
    max_context_tokens: int
    max_score_seq_len: int
    max_train_seq_len: int


FULL_CONFIG = RunConfig(
    suite="full",
    stage_h_steps=36,
    stage_i_pole_steps=36,
    stage_i_neutral_steps=18,
    test_case_limit=None,
    max_new_tokens=96,
    max_context_tokens=512,
    max_score_seq_len=512,
    max_train_seq_len=256,
)

SMOKE_CONFIG = RunConfig(
    suite="smoke",
    stage_h_steps=6,
    stage_i_pole_steps=6,
    stage_i_neutral_steps=3,
    test_case_limit=1,
    max_new_tokens=24,
    max_context_tokens=256,
    max_score_seq_len=256,
    max_train_seq_len=256,
)


def emit(event: str, **values: Any) -> None:
    print(
        json.dumps({"event": event, **values}, sort_keys=True),
        flush=True,
    )


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _proc_value(label: str) -> int | None:
    status = Path("/proc/self/status")
    if not status.exists():
        return None
    for line in status.read_text().splitlines():
        if line.startswith(f"{label}:"):
            return int(line.split()[1]) * 1024
    return None


def _system_available_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def memory_snapshot(device: torch.device) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "device": str(device),
        "rss_gib": (
            round(_proc_value("VmRSS") / GIB, 3)
            if _proc_value("VmRSS") is not None
            else None
        ),
        "system_available_gib": (
            round(_system_available_bytes() / GIB, 3)
            if _system_available_bytes() is not None
            else None
        ),
    }
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        snapshot.update(
            {
                "cuda_allocated_gib": round(
                    torch.cuda.memory_allocated(device) / GIB,
                    3,
                ),
                "cuda_reserved_gib": round(
                    torch.cuda.memory_reserved(device) / GIB,
                    3,
                ),
                "cuda_free_gib": round(free / GIB, 3),
                "cuda_total_gib": round(total / GIB, 3),
                "cuda_peak_allocated_gib": round(
                    torch.cuda.max_memory_allocated(device) / GIB,
                    3,
                ),
                "cuda_peak_reserved_gib": round(
                    torch.cuda.max_memory_reserved(device) / GIB,
                    3,
                ),
            }
        )
    return snapshot


def cuda_gate(
    device: torch.device,
    *,
    stage: str,
    minimum_free_gib: float,
) -> dict[str, Any]:
    snapshot = memory_snapshot(device)
    if device.type == "cuda":
        free = float(snapshot["cuda_free_gib"])
        reserved = float(snapshot["cuda_reserved_gib"])
        if free < minimum_free_gib:
            raise MemoryError(
                f"{stage}: {free:.2f} GiB CUDA free is below "
                f"{minimum_free_gib:.2f} GiB"
            )
        if reserved > 90.0:
            raise MemoryError(
                f"{stage}: {reserved:.2f} GiB CUDA reserved exceeds 90 GiB"
            )
    return snapshot


def choose_device(requested: str, profile: ModelProfile) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if profile.paper_target and device.type != "cuda":
        raise RuntimeError("the Qwen 3.6 paper run requires CUDA")
    return device


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_report(
    profile: ModelProfile,
    device: torch.device,
) -> dict[str, Any]:
    packages = {}
    for name in ("accelerate", "matplotlib", "numpy", "torch", "transformers"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    gpu_name = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    return {
        "created_at": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": gpu_name,
        "device": str(device),
        "model": asdict(profile),
        "repository_commit": _git_commit(),
        "command": sys.argv,
        "data": validate_style_data(),
    }


def _preload_gate(profile: ModelProfile, device: torch.device) -> None:
    snapshot = memory_snapshot(device)
    emit("preflight", profile=asdict(profile), memory=snapshot)
    if not profile.paper_target:
        return
    system_available = snapshot["system_available_gib"]
    if system_available is None or float(system_available) < 100.0:
        raise MemoryError("Qwen 3.6 load requires 100 GiB system memory free")
    if float(snapshot["cuda_free_gib"]) < 85.0:
        raise MemoryError("Qwen 3.6 load requires 85 GiB CUDA memory free")


def load_model(
    profile: ModelProfile,
    *,
    device: torch.device,
) -> tuple[nn.Module, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _preload_gate(profile, device)
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        profile.model_id,
        revision=profile.revision,
    )
    kwargs: dict[str, Any] = {
        "revision": profile.revision,
        "dtype": dtype,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    if profile.stream_to_cuda and device.type == "cuda":
        kwargs["device_map"] = {
            "": device.index if device.index is not None else 0
        }
        model = AutoModelForCausalLM.from_pretrained(
            profile.model_id,
            **kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            profile.model_id,
            **kwargs,
        ).to(device)
    model.config.use_cache = False
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    emit(
        "model_loaded",
        seconds=round(time.perf_counter() - started, 2),
        dtype=str(dtype),
        memory=memory_snapshot(device),
    )
    cuda_gate(
        device,
        stage="after model load",
        minimum_free_gib=24.0 if profile.paper_target else 0.0,
    )
    return model, tokenizer


def release_model(
    device: torch.device,
    model: nn.Module | None,
    tokenizer: Any | None,
) -> None:
    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def model_input_device(model: nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def _tensor_ids(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value
    elif hasattr(value, "input_ids"):
        tensor = value.input_ids
    elif isinstance(value, Mapping):
        tensor = value["input_ids"]
    else:
        tensor = torch.tensor(value, dtype=torch.long)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.cpu()


def _apply_chat_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
) -> torch.Tensor:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "return_tensors": "pt",
    }
    try:
        rendered = tokenizer.apply_chat_template(
            list(messages),
            enable_thinking=False,
            **kwargs,
        )
    except (TypeError, ValueError):
        rendered = tokenizer.apply_chat_template(list(messages), **kwargs)
    return _tensor_ids(rendered)


def render_prompt(
    tokenizer: Any,
    *,
    system_prompt: str,
    user_prompt: str,
) -> torch.Tensor:
    return _apply_chat_template(
        tokenizer,
        (
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ),
        add_generation_prompt=True,
    )


def render_prompt_response(
    tokenizer: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    response: str,
) -> tuple[torch.Tensor, int]:
    prompt = render_prompt(
        tokenizer,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    full = _apply_chat_template(
        tokenizer,
        (
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response},
        ),
        add_generation_prompt=False,
    )
    start = int(prompt.shape[1])
    if full.shape[1] <= start or not torch.equal(prompt, full[:, :start]):
        raise ValueError("assistant response rendering changed the prompt prefix")
    return full, start


def assistant_header_start(
    tokenizer: Any,
    *,
    system_prompt: str,
    user_prompt: str,
) -> int:
    without_header = _apply_chat_template(
        tokenizer,
        (
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ),
        add_generation_prompt=False,
    )
    with_header = render_prompt(
        tokenizer,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    boundary = int(without_header.shape[1])
    if boundary >= with_header.shape[1]:
        raise ValueError("chat template did not add an assistant header")
    if not torch.equal(without_header, with_header[:, :boundary]):
        raise ValueError("assistant header changed earlier chat tokens")
    return boundary


def response_nll(
    model: nn.Module,
    tokenizer: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    response: str,
    max_seq_len: int,
) -> torch.Tensor:
    full, response_start = render_prompt_response(
        tokenizer,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response=response,
    )
    if full.shape[1] > max_seq_len:
        raise ValueError("response exceeds the scoring token limit")
    device = model_input_device(model)
    inputs = full[:, :-1].to(device)
    targets = full[:, response_start:].to(device)
    output = model(input_ids=inputs, use_cache=False)
    logits = output.logits[:, response_start - 1 :, :].float()
    if logits.shape[1] != targets.shape[1]:
        raise AssertionError("regular response logits and targets are misaligned")
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("regular response NLL is non-finite")
    return loss


def prefix_response_nll(
    model: nn.Module,
    tokenizer: Any,
    *,
    pair: StylePair,
    response: str,
    prefix: torch.Tensor,
    max_seq_len: int,
) -> tuple[torch.Tensor, int]:
    full, response_start = render_prompt_response(
        tokenizer,
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=pair.prompt,
        response=response,
    )
    if full.shape[1] + prefix.shape[0] > max_seq_len:
        raise ValueError(f"{pair.pair_id} exceeds the training token limit")
    device = model_input_device(model)
    inputs = full[:, :-1].to(device)
    targets = full[:, response_start:].to(device)
    insertion = assistant_header_start(
        tokenizer,
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=pair.prompt,
    )
    with torch.no_grad():
        token_embeddings = model.get_input_embeddings()(inputs)
    combined = insert_prefix_embeddings(
        token_embeddings,
        prefix,
        insertion=insertion,
    )
    output = model(inputs_embeds=combined, use_cache=False)
    logit_start = response_start + prefix.shape[0] - 1
    logits = output.logits[:, logit_start:, :].float()
    if logits.shape[1] != targets.shape[1]:
        raise AssertionError("prefix response logits and targets are misaligned")
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("prefix response NLL is non-finite")
    return loss, int(targets.numel())


@torch.inference_mode()
def scored_response_nll(
    model: nn.Module,
    tokenizer: Any,
    *,
    pair: StylePair,
    response: str,
    prefix: torch.Tensor | None,
    max_seq_len: int,
) -> float:
    if prefix is None:
        return float(
            response_nll(
                model,
                tokenizer,
                system_prompt=BASE_SYSTEM_PROMPT,
                user_prompt=pair.prompt,
                response=response,
                max_seq_len=max_seq_len,
            ).cpu()
        )
    loss, _ = prefix_response_nll(
        model,
        tokenizer,
        pair=pair,
        response=response,
        prefix=prefix,
        max_seq_len=max_seq_len,
    )
    return float(loss.cpu())


def pair_margin(
    model: nn.Module,
    tokenizer: Any,
    *,
    pair: StylePair,
    prefix: torch.Tensor | None,
    max_seq_len: int,
) -> dict[str, float]:
    positive = scored_response_nll(
        model,
        tokenizer,
        pair=pair,
        response=pair.positive_response,
        prefix=prefix,
        max_seq_len=max_seq_len,
    )
    negative = scored_response_nll(
        model,
        tokenizer,
        pair=pair,
        response=pair.negative_response,
        prefix=prefix,
        max_seq_len=max_seq_len,
    )
    return {
        "positive_nll": positive,
        "negative_nll": negative,
        "margin": negative - positive,
        "mean_nll": (positive + negative) / 2.0,
    }


def _pairs_for_axis(
    axis: str,
    split: str,
    *,
    case_limit: int | None = None,
) -> list[StylePair]:
    pairs = build_style_pairs((axis,), (split,))
    return pairs[:case_limit] if case_limit is not None else pairs


def evaluate_bank(
    model: nn.Module,
    tokenizer: Any,
    bank: SoftPrefixBank | NeutralAnchoredPrefixBank,
    *,
    splits: Sequence[str],
    max_seq_len: int,
    case_limit: int | None,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    axes = list(bank.axis_names)
    for split in splits:
        report[split] = {}
        for axis_index, axis in enumerate(axes):
            wrong_axis = axes[(axis_index + 1) % len(axes)]
            neutral_rows = []
            positive_rows = []
            negative_rows = []
            wrong_rows = []
            details = []
            for pair in _pairs_for_axis(
                axis,
                split,
                case_limit=case_limit,
            ):
                neutral = pair_margin(
                    model,
                    tokenizer,
                    pair=pair,
                    prefix=None,
                    max_seq_len=max_seq_len,
                )
                positive = pair_margin(
                    model,
                    tokenizer,
                    pair=pair,
                    prefix=bank.prefix(axis, 1),
                    max_seq_len=max_seq_len,
                )
                negative = pair_margin(
                    model,
                    tokenizer,
                    pair=pair,
                    prefix=bank.prefix(axis, -1),
                    max_seq_len=max_seq_len,
                )
                wrong = pair_margin(
                    model,
                    tokenizer,
                    pair=pair,
                    prefix=bank.prefix(wrong_axis, 1),
                    max_seq_len=max_seq_len,
                )
                neutral_rows.append(neutral)
                positive_rows.append(positive)
                negative_rows.append(negative)
                wrong_rows.append(wrong)
                details.append(
                    {
                        "pair_id": pair.pair_id,
                        "neutral": neutral,
                        "prefix_positive": positive,
                        "prefix_negative": negative,
                        "wrong_axis_positive": wrong,
                        "causal_span": positive["margin"] - negative["margin"],
                    }
                )
            neutral_mean = mean_rows(neutral_rows)
            positive_mean = mean_rows(positive_rows)
            negative_mean = mean_rows(negative_rows)
            wrong_mean = mean_rows(wrong_rows)
            report[split][axis] = {
                "neutral": neutral_mean,
                "prefix_positive": positive_mean,
                "prefix_negative": negative_mean,
                "wrong_axis_positive": wrong_mean,
                "causal_span": positive_mean["margin"] - negative_mean["margin"],
                "positive_gain_from_off": (
                    positive_mean["margin"] - neutral_mean["margin"]
                ),
                "negative_gain_from_off": (
                    neutral_mean["margin"] - negative_mean["margin"]
                ),
                "wrong_axis_gain_from_off": (
                    wrong_mean["margin"] - neutral_mean["margin"]
                ),
                "directional_success_rate": sum(
                    detail["causal_span"] > 0.0 for detail in details
                )
                / len(details),
                "wrong_axis": wrong_axis,
                "details": details,
            }
    return report


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
    optimizer = torch.optim.AdamW(
        bank.parameters(),
        lr=0.001,
        weight_decay=0.0,
    )
    schedule = pole_training_schedule(pairs, steps=steps, seed=seed)
    history: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
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
                "diversity_loss": float(diversity.detach().cpu()),
                "anchor_loss": float(anchor.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "geometry": bank.geometry(),
                "memory": snapshot,
            }
            history.append(record)
            if step == 1 or step == steps or step % 6 == 0:
                emit("stage_h_step", seed=seed, **record)
    finally:
        model.eval()
        if checkpointing:
            model.gradient_checkpointing_disable()
        del optimizer
    return history


def train_stage_i(
    model: nn.Module,
    tokenizer: Any,
    bank: NeutralAnchoredPrefixBank,
    *,
    pairs: Sequence[StylePair],
    pole_steps: int,
    neutral_steps: int,
    seed: int,
    max_seq_len: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        bank.parameters(),
        lr=0.001,
        weight_decay=0.0,
    )
    schedule = neutral_training_schedule(
        pairs,
        pole_steps=pole_steps,
        neutral_steps=neutral_steps,
        seed=seed,
    )
    history: list[dict[str, Any]] = []
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
            neutral_value = None
            desired_tokens = rejected_tokens = neutral_tokens = 0
            if kind == "pole":
                desired_response = (
                    pair.positive_response
                    if sign > 0
                    else pair.negative_response
                )
                rejected_response = (
                    pair.negative_response
                    if sign > 0
                    else pair.positive_response
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
            else:
                neutral, neutral_tokens = prefix_response_nll(
                    model,
                    tokenizer,
                    pair=pair,
                    response=pair.neutral_response,
                    prefix=bank.neutral_prefix(),
                    max_seq_len=max_seq_len,
                )
                neutral_value = float(neutral.detach().cpu())
                neutral.backward()
                del neutral
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
                raise FloatingPointError(f"non-finite Stage I gradient at {step}")
            optimizer.step()
            bank.project_norms_()
            snapshot = cuda_gate(
                device,
                stage=f"Stage I step {step}",
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
                "neutral_nll": neutral_value,
                "training_margin": (
                    rejected_value - desired_value
                    if desired_value is not None
                    and rejected_value is not None
                    else None
                ),
                "desired_tokens": desired_tokens,
                "rejected_tokens": rejected_tokens,
                "neutral_tokens": neutral_tokens,
                "diversity_loss": float(diversity.detach().cpu()),
                "anchor_loss": float(anchor.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "geometry": bank.geometry(),
                "memory": snapshot,
            }
            history.append(record)
            if step == 1 or step == len(schedule) or step % 9 == 0:
                emit("stage_i_step", seed=seed, **record)
    finally:
        model.eval()
        if checkpointing:
            model.gradient_checkpointing_disable()
        del optimizer
    return history


@torch.inference_mode()
def generate_response(
    model: nn.Module,
    tokenizer: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    prefix: torch.Tensor | None,
    max_new_tokens: int,
    max_context_tokens: int,
) -> str:
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
    eos = getattr(tokenizer, "eos_token_id", None)
    stop = (
        {int(value) for value in eos}
        if isinstance(eos, (tuple, list, set))
        else ({int(eos)} if eos is not None else set())
    )
    past_key_values = None
    tokens: list[int] = []
    for _ in range(max_new_tokens):
        output = model(
            input_ids=current_ids,
            inputs_embeds=current_embeds,
            past_key_values=past_key_values,
            use_cache=True,
        )
        token = int(torch.argmax(output.logits[0, -1]).cpu())
        tokens.append(token)
        past_key_values = output.past_key_values
        current_ids = torch.tensor([[token]], dtype=torch.long, device=device)
        current_embeds = None
        if token in stop:
            break
    return tokenizer.decode(tokens, skip_special_tokens=True).strip()


@torch.inference_mode()
def response_attribution(
    model: nn.Module,
    tokenizer: Any,
    *,
    axis: str,
    prompt: str,
    response: str,
    max_seq_len: int,
) -> dict[str, float]:
    positive = float(
        response_nll(
            model,
            tokenizer,
            system_prompt=style_system_prompt(axis, 1),
            user_prompt=prompt,
            response=response,
            max_seq_len=max_seq_len,
        ).cpu()
    )
    negative = float(
        response_nll(
            model,
            tokenizer,
            system_prompt=style_system_prompt(axis, -1),
            user_prompt=prompt,
            response=response,
            max_seq_len=max_seq_len,
        ).cpu()
    )
    return {
        "positive_prompt_nll": positive,
        "negative_prompt_nll": negative,
        "signed_attribution": negative - positive,
    }


BaselineCache = dict[tuple[str, str, str], dict[str, Any]]


def _generated_measurement(
    model: nn.Module,
    tokenizer: Any,
    *,
    pair: StylePair,
    axis: str,
    system_prompt: str,
    prefix: torch.Tensor | None,
    max_new_tokens: int,
    max_context_tokens: int,
    max_score_seq_len: int,
) -> dict[str, Any]:
    response = generate_response(
        model,
        tokenizer,
        system_prompt=system_prompt,
        user_prompt=pair.prompt,
        prefix=prefix,
        max_new_tokens=max_new_tokens,
        max_context_tokens=max_context_tokens,
    )
    return {
        "text": response,
        "attribution": response_attribution(
            model,
            tokenizer,
            axis=axis,
            prompt=pair.prompt,
            response=response,
            max_seq_len=max_score_seq_len,
        ),
        "lexical": lexical_style_score(response, axis),
        "word_count": len(response.split()),
        "word_jaccard_with_core": word_jaccard(
            response,
            pair.neutral_response,
        ),
    }


def audit_bank(
    model: nn.Module,
    tokenizer: Any,
    bank: SoftPrefixBank | NeutralAnchoredPrefixBank,
    *,
    stage: str,
    seed: int,
    config: RunConfig,
    device: torch.device,
    baseline_cache: BaselineCache,
    include_neutral_center: bool,
) -> dict[str, Any]:
    rows = []
    axes = list(bank.axis_names)
    for axis_index, axis in enumerate(axes):
        wrong_axis = axes[(axis_index + 1) % len(axes)]
        for pair in _pairs_for_axis(
            axis,
            "test",
            case_limit=config.test_case_limit,
        ):
            conditions: list[tuple[str, str, torch.Tensor | None]] = [
                ("off", BASE_SYSTEM_PROMPT, None),
                ("explicit_positive", style_system_prompt(axis, 1), None),
                ("explicit_negative", style_system_prompt(axis, -1), None),
                (
                    "prefix_positive",
                    BASE_SYSTEM_PROMPT,
                    bank.prefix(axis, 1),
                ),
                (
                    "prefix_negative",
                    BASE_SYSTEM_PROMPT,
                    bank.prefix(axis, -1),
                ),
                (
                    "wrong_axis",
                    BASE_SYSTEM_PROMPT,
                    bank.prefix(wrong_axis, 1),
                ),
            ]
            if include_neutral_center:
                assert isinstance(bank, NeutralAnchoredPrefixBank)
                conditions.insert(
                    1,
                    (
                        "neutral_center",
                        BASE_SYSTEM_PROMPT,
                        bank.neutral_prefix(),
                    ),
                )
            generated: dict[str, dict[str, Any]] = {}
            for condition, system_prompt, prefix in conditions:
                cache_key = (axis, pair.pair_id, condition)
                if prefix is None and cache_key in baseline_cache:
                    measured = copy.deepcopy(baseline_cache[cache_key])
                else:
                    measured = _generated_measurement(
                        model,
                        tokenizer,
                        pair=pair,
                        axis=axis,
                        system_prompt=system_prompt,
                        prefix=prefix,
                        max_new_tokens=config.max_new_tokens,
                        max_context_tokens=config.max_context_tokens,
                        max_score_seq_len=config.max_score_seq_len,
                    )
                    if prefix is None:
                        baseline_cache[cache_key] = copy.deepcopy(measured)
                generated[condition] = measured
                emit(
                    "generated_response",
                    stage=stage,
                    seed=seed,
                    pair_id=pair.pair_id,
                    axis=axis,
                    condition=condition,
                    attribution=measured["attribution"]["signed_attribution"],
                    words=measured["word_count"],
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
                stage=f"{stage} audit {pair.pair_id}",
                minimum_free_gib=20.0 if device.type == "cuda" else 0.0,
            )
    report = {
        "format": f"{stage}-generated-responses-repro-v1",
        "created_at": time.time(),
        "seed": seed,
        "axes": axes,
        "prefix_tokens": bank.prefix_tokens,
        "geometry": bank.geometry(),
        "rows": rows,
        "aggregate": aggregate_responses(rows),
        "memory": memory_snapshot(device),
    }
    report["metrics"] = report_metrics(report)
    return report


def audit_warmth_sweep(
    model: nn.Module,
    tokenizer: Any,
    bank: SoftPrefixBank,
    *,
    seed: int,
    config: RunConfig,
    device: torch.device,
    baseline_cache: BaselineCache,
) -> dict[str, Any]:
    axis = "warmth"
    states = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
    conditions: list[tuple[str, str, torch.Tensor | None]] = [
        ("off", BASE_SYSTEM_PROMPT, None),
        ("explicit_positive", style_system_prompt(axis, 1), None),
        ("explicit_negative", style_system_prompt(axis, -1), None),
        *[
            (
                f"state_{state:+g}",
                BASE_SYSTEM_PROMPT,
                bank.interpolated_prefix(axis, state),
            )
            for state in states
        ],
        (
            "direction_only_positive",
            BASE_SYSTEM_PROMPT,
            bank.interpolated_prefix(axis, 1.0, include_center=False),
        ),
        (
            "direction_only_negative",
            BASE_SYSTEM_PROMPT,
            bank.interpolated_prefix(axis, -1.0, include_center=False),
        ),
    ]
    rows = []
    for pair in _pairs_for_axis(
        axis,
        "test",
        case_limit=config.test_case_limit,
    ):
        generated: dict[str, dict[str, Any]] = {}
        for condition, system_prompt, prefix in conditions:
            cache_key = (axis, pair.pair_id, condition)
            if prefix is None and cache_key in baseline_cache:
                measured = copy.deepcopy(baseline_cache[cache_key])
            else:
                measured = _generated_measurement(
                    model,
                    tokenizer,
                    pair=pair,
                    axis=axis,
                    system_prompt=system_prompt,
                    prefix=prefix,
                    max_new_tokens=config.max_new_tokens,
                    max_context_tokens=config.max_context_tokens,
                    max_score_seq_len=config.max_score_seq_len,
                )
                if prefix is None:
                    baseline_cache[cache_key] = copy.deepcopy(measured)
            generated[condition] = measured
            emit(
                "warmth_sweep_response",
                seed=seed,
                pair_id=pair.pair_id,
                condition=condition,
                attribution=measured["attribution"]["signed_attribution"],
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
                "prompt": pair.prompt,
                "responses": generated,
            }
        )
        cuda_gate(
            device,
            stage=f"warmth sweep {pair.pair_id}",
            minimum_free_gib=20.0 if device.type == "cuda" else 0.0,
        )
    report = {
        "format": "stage-h-warmth-strength-sweep-repro-v1",
        "created_at": time.time(),
        "seed": seed,
        "axis": axis,
        "states": list(states),
        "rows": rows,
        "aggregate": aggregate_responses(rows),
        "memory": memory_snapshot(device),
    }
    report["metrics"] = strength_sweep_metrics(report)
    return report


@torch.inference_mode()
def evaluate_neutral_center(
    model: nn.Module,
    tokenizer: Any,
    bank: NeutralAnchoredPrefixBank,
    *,
    splits: Sequence[str],
    max_seq_len: int,
    case_limit: int | None,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split in splits:
        report[split] = {}
        for axis in bank.axis_names:
            details = []
            for pair in _pairs_for_axis(
                axis,
                split,
                case_limit=case_limit,
            ):
                off_neutral = scored_response_nll(
                    model,
                    tokenizer,
                    pair=pair,
                    response=pair.neutral_response,
                    prefix=None,
                    max_seq_len=max_seq_len,
                )
                center_neutral = scored_response_nll(
                    model,
                    tokenizer,
                    pair=pair,
                    response=pair.neutral_response,
                    prefix=bank.neutral_prefix(),
                    max_seq_len=max_seq_len,
                )
                off_margin = pair_margin(
                    model,
                    tokenizer,
                    pair=pair,
                    prefix=None,
                    max_seq_len=max_seq_len,
                )["margin"]
                center_margin = pair_margin(
                    model,
                    tokenizer,
                    pair=pair,
                    prefix=bank.neutral_prefix(),
                    max_seq_len=max_seq_len,
                )["margin"]
                details.append(
                    {
                        "pair_id": pair.pair_id,
                        "off_neutral_nll": off_neutral,
                        "center_neutral_nll": center_neutral,
                        "neutral_nll_delta": center_neutral - off_neutral,
                        "off_style_margin": off_margin,
                        "center_style_margin": center_margin,
                        "style_margin_delta": center_margin - off_margin,
                    }
                )
            report[split][axis] = {
                "mean_neutral_nll_delta": sum(
                    row["neutral_nll_delta"] for row in details
                )
                / len(details),
                "mean_style_margin_delta": sum(
                    row["style_margin_delta"] for row in details
                )
                / len(details),
                "mean_abs_style_margin_delta": sum(
                    abs(row["style_margin_delta"]) for row in details
                )
                / len(details),
                "details": details,
            }
    return report


def write_response_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        f"# {report['format']}",
        "",
        f"- Seed: `{report['seed']}`",
        "",
        "## Aggregate attribution",
        "",
        "| axis | condition | attribution | words |",
        "|---|---|---:|---:|",
    ]
    for axis, conditions in report["aggregate"].items():
        for condition, values in conditions.items():
            lines.append(
                f"| {axis} | {condition} | "
                f"{values['mean_attribution']:.4f} | "
                f"{values['mean_word_count']:.1f} |"
            )
    for row in report["rows"]:
        lines.extend(
            [
                "",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _screen_report(
    *,
    stage: str,
    seed: int,
    bank: SoftPrefixBank | NeutralAnchoredPrefixBank,
    evaluation: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    device: torch.device,
    neutral_evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "format": f"{stage}-teacher-forced-screen-repro-v1",
        "created_at": time.time(),
        "seed": seed,
        "axes": list(bank.axis_names),
        "prefix_tokens": bank.prefix_tokens,
        "geometry": bank.geometry(),
        "training_history": list(history),
        "evaluation": evaluation,
        "memory": memory_snapshot(device),
    }
    if neutral_evaluation is not None:
        report["neutral_evaluation"] = neutral_evaluation
    return report


def run_seed(
    model: nn.Module,
    tokenizer: Any,
    profile: ModelProfile,
    config: RunConfig,
    *,
    seed: int,
    output_dir: Path,
    device: torch.device,
    baseline_cache: BaselineCache,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    seed_dir = output_dir / f"seed-{seed}"
    seed_dir.mkdir(parents=True, exist_ok=False)
    # Preserve the exploratory run's axis-major inventory before seeded
    # shuffling so the first confirmatory seed repeats its schedule.
    fit_pairs = [
        pair
        for axis in STYLE_AXIS_NAMES
        for pair in build_style_pairs((axis,), ("fit",))
    ]

    stage_h = SoftPrefixBank.initialize(
        model,
        tokenizer,
        axes=STYLE_AXIS_NAMES,
        prefix_tokens=8,
        device=model_input_device(model),
    )
    initial_h_geometry = stage_h.geometry()
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
    stage_h_checkpoint = seed_dir / "stage-h-prefix.pt"
    payload_h = stage_h.payload(
        model_id=profile.model_id,
        model_revision=profile.revision,
        seed=seed,
        history=history_h,
    )
    payload_h["created_at"] = time.time()
    payload_h["initial_geometry"] = initial_h_geometry
    payload_h["settings"] = {
        "steps": config.stage_h_steps,
        "learning_rate": 0.001,
        "diversity_weight": 0.01,
        "anchor_weight": 0.001,
        "grad_clip": 1.0,
    }
    atomic_torch_save(payload_h, stage_h_checkpoint)

    evaluation_h = evaluate_bank(
        model,
        tokenizer,
        stage_h,
        splits=("dev", "test"),
        max_seq_len=config.max_train_seq_len,
        case_limit=config.test_case_limit,
    )
    screen_h = _screen_report(
        stage="stage-h",
        seed=seed,
        bank=stage_h,
        evaluation=evaluation_h,
        history=history_h,
        device=device,
    )
    atomic_json_dump(screen_h, seed_dir / "stage-h-screen.json")
    responses_h = audit_bank(
        model,
        tokenizer,
        stage_h,
        stage="stage-h",
        seed=seed,
        config=config,
        device=device,
        baseline_cache=baseline_cache,
        include_neutral_center=False,
    )
    atomic_json_dump(responses_h, seed_dir / "stage-h-responses.json")
    write_response_markdown(
        responses_h,
        seed_dir / "stage-h-responses.md",
    )
    sweep_h = audit_warmth_sweep(
        model,
        tokenizer,
        stage_h,
        seed=seed,
        config=config,
        device=device,
        baseline_cache=baseline_cache,
    )
    atomic_json_dump(sweep_h, seed_dir / "stage-h-warmth-sweep.json")
    write_response_markdown(
        sweep_h,
        seed_dir / "stage-h-warmth-sweep.md",
    )

    stage_i = NeutralAnchoredPrefixBank.from_stage_h(stage_h)
    initial_i_geometry = stage_i.geometry()
    history_i = train_stage_i(
        model,
        tokenizer,
        stage_i,
        pairs=fit_pairs,
        pole_steps=config.stage_i_pole_steps,
        neutral_steps=config.stage_i_neutral_steps,
        seed=seed,
        max_seq_len=config.max_train_seq_len,
        device=device,
    )
    stage_i_checkpoint = seed_dir / "stage-i-prefix.pt"
    payload_i = stage_i.payload(
        model_id=profile.model_id,
        model_revision=profile.revision,
        seed=seed,
        history=history_i,
        source_stage_h=str(stage_h_checkpoint.relative_to(output_dir)),
    )
    payload_i["created_at"] = time.time()
    payload_i["initial_geometry"] = initial_i_geometry
    payload_i["settings"] = {
        "pole_steps": config.stage_i_pole_steps,
        "neutral_steps": config.stage_i_neutral_steps,
        "learning_rate": 0.001,
        "neutral_weight": 1.0,
        "diversity_weight": 0.01,
        "anchor_weight": 0.001,
        "grad_clip": 1.0,
    }
    atomic_torch_save(payload_i, stage_i_checkpoint)

    evaluation_i = evaluate_bank(
        model,
        tokenizer,
        stage_i,
        splits=("dev", "test"),
        max_seq_len=config.max_train_seq_len,
        case_limit=config.test_case_limit,
    )
    neutral_i = evaluate_neutral_center(
        model,
        tokenizer,
        stage_i,
        splits=("dev", "test"),
        max_seq_len=config.max_train_seq_len,
        case_limit=config.test_case_limit,
    )
    screen_i = _screen_report(
        stage="stage-i",
        seed=seed,
        bank=stage_i,
        evaluation=evaluation_i,
        history=history_i,
        device=device,
        neutral_evaluation=neutral_i,
    )
    atomic_json_dump(screen_i, seed_dir / "stage-i-screen.json")
    responses_i = audit_bank(
        model,
        tokenizer,
        stage_i,
        stage="stage-i",
        seed=seed,
        config=config,
        device=device,
        baseline_cache=baseline_cache,
        include_neutral_center=True,
    )
    atomic_json_dump(responses_i, seed_dir / "stage-i-responses.json")
    write_response_markdown(
        responses_i,
        seed_dir / "stage-i-responses.md",
    )

    record = {
        "seed": seed,
        "stage_h": {
            "checkpoint": str(stage_h_checkpoint.relative_to(output_dir)),
            "screen": str(
                (seed_dir / "stage-h-screen.json").relative_to(output_dir)
            ),
            "responses": str(
                (seed_dir / "stage-h-responses.json").relative_to(output_dir)
            ),
            "warmth_sweep": str(
                (seed_dir / "stage-h-warmth-sweep.json").relative_to(
                    output_dir
                )
            ),
            "metrics": responses_h["metrics"],
            "warmth_sweep_metrics": sweep_h["metrics"],
        },
        "stage_i": {
            "checkpoint": str(stage_i_checkpoint.relative_to(output_dir)),
            "screen": str(
                (seed_dir / "stage-i-screen.json").relative_to(output_dir)
            ),
            "responses": str(
                (seed_dir / "stage-i-responses.json").relative_to(output_dir)
            ),
            "metrics": responses_i["metrics"],
        },
        "memory": memory_snapshot(device),
    }
    atomic_json_dump(record, seed_dir / "seed-summary.json")
    emit(
        "seed_complete",
        seed=seed,
        stage_h=record["stage_h"]["metrics"],
        stage_i=record["stage_i"]["metrics"],
    )
    del stage_i, stage_h
    gc.collect()
    return record


def reproduce(
    *,
    suite: str,
    seeds: Sequence[int],
    output_dir: Path,
    requested_device: str,
) -> dict[str, Any]:
    if suite == "full":
        profile = FULL_PROFILE
        config = FULL_CONFIG
    elif suite == "smoke":
        profile = SMOKE_PROFILE
        config = SMOKE_CONFIG
        seeds = seeds[:1]
    else:
        raise ValueError(f"unknown suite {suite}")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be non-empty and unique")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(requested_device, profile)
    environment = environment_report(profile, device)
    atomic_json_dump(environment, output_dir / "environment.json")
    model: nn.Module | None = None
    tokenizer: Any | None = None
    started = time.time()
    baseline_cache: BaselineCache = {}
    try:
        model, tokenizer = load_model(profile, device=device)
        records = [
            run_seed(
                model,
                tokenizer,
                profile,
                config,
                seed=seed,
                output_dir=output_dir,
                device=device,
                baseline_cache=baseline_cache,
            )
            for seed in seeds
        ]
        consolidated = consolidate_metrics(records)
        summary = {
            "format": "causal-expression-confirmatory-repro-v1",
            "created_at": time.time(),
            "elapsed_seconds": time.time() - started,
            "scientific_target": profile.paper_target,
            "profile": asdict(profile),
            "config": asdict(config),
            "seeds": list(seeds),
            "environment": environment,
            "seed_records": records,
            "consolidated": consolidated,
            "memory": memory_snapshot(device),
        }
        atomic_json_dump(summary, output_dir / "summary.json")
        from .figures import render_figures

        figures = render_figures(
            output_dir / "summary.json",
            output_dir / "figures",
        )
        summary["figures"] = figures
        atomic_json_dump(summary, output_dir / "summary.json")
        emit(
            "reproduction_complete",
            output=str(output_dir),
            decisions=consolidated["decisions"],
            elapsed_seconds=summary["elapsed_seconds"],
        )
        return summary
    finally:
        release_model(device, model, tokenizer)
