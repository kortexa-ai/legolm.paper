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
from .prefix import insert_prefix_embeddings


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


def bounded_context(prompt: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    """Reject an over-long prompt rather than truncating it.

    Silent truncation would drop the assistant frontier, which is exactly the
    position every measurement in this study reads, so an over-long prompt is a
    corpus bug and is raised as one.
    """
    if prompt.shape[1] > max_seq_len:
        raise ValueError(
            f"prompt is {prompt.shape[1]} tokens, over the {max_seq_len}-token limit"
        )
    return prompt


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
