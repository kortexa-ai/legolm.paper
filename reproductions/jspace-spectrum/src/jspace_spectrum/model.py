"""Model loading, targeted lens fitting, calibration, and token measurement."""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
import gc
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from .data import AXES, AxisSpec, EvalCase


GIB = 1024**3


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model_id: str
    revision: str
    source_layer: int
    target_layer: int
    trace_layers: tuple[int, ...]
    paper_target: bool
    minimum_cuda_free_gib: float
    minimum_system_available_gib: float


QWEN36_35B = ModelProfile(
    name="qwen36-35b",
    model_id="Qwen/Qwen3.6-35B-A3B",
    revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
    source_layer=35,
    target_layer=39,
    trace_layers=(3, 7, 11, 15, 19, 23, 27, 31, 35, 39),
    paper_target=True,
    minimum_cuda_free_gib=85.0,
    minimum_system_available_gib=100.0,
)


QWEN36_27B = ModelProfile(
    name="qwen36-27b",
    model_id="Qwen/Qwen3.6-27B",
    revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    source_layer=55,
    target_layer=63,
    trace_layers=(3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63),
    paper_target=True,
    minimum_cuda_free_gib=85.0,
    minimum_system_available_gib=100.0,
)


SMOKE_PROFILE = ModelProfile(
    name="qwen35-2b-smoke",
    model_id="Qwen/Qwen3.5-2B",
    revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
    source_layer=19,
    target_layer=23,
    trace_layers=(3, 7, 11, 15, 19, 23),
    paper_target=False,
    minimum_cuda_free_gib=4.0,
    minimum_system_available_gib=8.0,
)


PROFILES = {
    profile.name: profile for profile in (QWEN36_35B, QWEN36_27B, SMOKE_PROFILE)
}


def emit(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


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
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except (ImportError, AttributeError):
        return None


def memory_snapshot(device: torch.device) -> dict[str, Any]:
    rss = _proc_value("VmRSS")
    available = _system_available_bytes()
    snapshot: dict[str, Any] = {
        "device": str(device),
        "rss_gib": round(rss / GIB, 3) if rss is not None else None,
        "system_available_gib": (
            round(available / GIB, 3) if available is not None else None
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
                "cuda_peak_allocated_gib": round(
                    torch.cuda.max_memory_allocated(device) / GIB,
                    3,
                ),
                "cuda_peak_reserved_gib": round(
                    torch.cuda.max_memory_reserved(device) / GIB,
                    3,
                ),
                "cuda_free_gib": round(free / GIB, 3),
                "cuda_total_gib": round(total / GIB, 3),
            }
        )
    elif device.type == "mps" and torch.backends.mps.is_available():
        snapshot.update(
            {
                "mps_allocated_gib": round(
                    torch.mps.current_allocated_memory() / GIB,
                    3,
                ),
                "mps_driver_gib": round(
                    torch.mps.driver_allocated_memory() / GIB,
                    3,
                ),
            }
        )
    return snapshot


def preload_gate(profile: ModelProfile, device: torch.device) -> dict[str, Any]:
    snapshot = memory_snapshot(device)
    emit("preflight", profile=asdict(profile), memory=snapshot)
    available = snapshot["system_available_gib"]
    if (
        available is not None
        and float(available) < profile.minimum_system_available_gib
    ):
        raise MemoryError(
            f"{profile.name}: {available} GiB system memory available is below "
            f"{profile.minimum_system_available_gib:.1f} GiB"
        )
    if device.type == "cuda":
        free = float(snapshot["cuda_free_gib"])
        if free < profile.minimum_cuda_free_gib:
            raise MemoryError(
                f"{profile.name}: {free:.2f} GiB CUDA free is below "
                f"{profile.minimum_cuda_free_gib:.1f} GiB"
            )
    elif profile.paper_target:
        raise RuntimeError(f"{profile.name} requires CUDA")
    return snapshot


def runtime_gate(
    device: torch.device,
    *,
    stage: str,
    minimum_cuda_headroom_gib: float = 20.0,
    maximum_cuda_fraction: float = 0.90,
) -> dict[str, Any]:
    snapshot = memory_snapshot(device)
    if device.type == "cuda":
        free = float(snapshot["cuda_free_gib"])
        total = float(snapshot["cuda_total_gib"])
        if free < minimum_cuda_headroom_gib:
            raise MemoryError(
                f"{stage}: {free:.2f} GiB CUDA free is below "
                f"{minimum_cuda_headroom_gib:.1f} GiB"
            )
        if total > 0 and 1.0 - free / total > maximum_cuda_fraction:
            raise MemoryError(f"{stage}: CUDA use exceeds {maximum_cuda_fraction:.0%}")
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
        raise RuntimeError("paper models require CUDA")
    return device


def load_model(
    profile: ModelProfile,
    *,
    device: torch.device,
) -> tuple[Any, Any, Any]:
    import jlens
    from transformers import AutoModelForCausalLM, AutoTokenizer

    preload_gate(profile, device)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        profile.model_id,
        revision=profile.revision,
    )
    kwargs: dict[str, Any] = {
        "revision": profile.revision,
        "dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    if device.type == "cuda":
        kwargs["device_map"] = {"": device.index or 0}
        model = AutoModelForCausalLM.from_pretrained(profile.model_id, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            profile.model_id,
            **kwargs,
        ).to(device)
    model.eval()
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    lens_model = jlens.from_hf(model, tokenizer)
    if int(lens_model.n_layers) <= profile.target_layer:
        raise ValueError(
            f"{profile.name} exposes {lens_model.n_layers} layers, but the "
            f"profile requests target layer {profile.target_layer}"
        )
    snapshot = runtime_gate(
        device,
        stage=f"{profile.name} after model load",
        minimum_cuda_headroom_gib=(20.0 if profile.paper_target else 0.5),
    )
    emit(
        "model_loaded",
        profile=profile.name,
        seconds=round(time.perf_counter() - started, 3),
        layers=int(lens_model.n_layers),
        hidden_size=int(lens_model.d_model),
        memory=snapshot,
    )
    return model, tokenizer, lens_model


def release_model(model: Any, tokenizer: Any, lens_model: Any) -> None:
    del lens_model, tokenizer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def resolve_axis_token_ids(
    tokenizer: Any,
    axes: Sequence[AxisSpec] = AXES,
) -> dict[str, dict[str, list[int]]]:
    resolved: dict[str, dict[str, list[int]]] = {}
    for axis in axes:
        resolved[axis.name] = {}
        for pole, words in (
            ("positive", axis.positive_words),
            ("negative", axis.negative_words),
        ):
            token_ids: list[int] = []
            for word in words:
                for variant in (
                    word,
                    f" {word}",
                    word.capitalize(),
                    f" {word.capitalize()}",
                ):
                    ids = tokenizer.encode(variant, add_special_tokens=False)
                    if len(ids) == 1:
                        token_ids.append(int(ids[0]))
            unique = list(dict.fromkeys(token_ids))
            if len(unique) < 2:
                raise ValueError(f"{axis.name}/{pole} resolved to {len(unique)} tokens")
            resolved[axis.name][pole] = unique
    return resolved


def _score_columns(
    axis_token_ids: Mapping[str, Mapping[str, Sequence[int]]],
    axis_names: Sequence[str],
) -> tuple[list[int], dict[str, tuple[list[int], list[int]]]]:
    all_ids: list[int] = []
    for axis in axis_names:
        all_ids.extend(axis_token_ids[axis]["positive"])
        all_ids.extend(axis_token_ids[axis]["negative"])
    all_ids = list(dict.fromkeys(all_ids))
    columns = {token_id: index for index, token_id in enumerate(all_ids)}
    by_axis = {
        axis: (
            [columns[token_id] for token_id in axis_token_ids[axis]["positive"]],
            [columns[token_id] for token_id in axis_token_ids[axis]["negative"]],
        )
        for axis in axis_names
    }
    return all_ids, by_axis


def fit_targeted_lens(
    lens_model: Any,
    *,
    profile: ModelProfile,
    prompts: Sequence[str],
    axis_token_ids: Mapping[str, Mapping[str, Sequence[int]]],
    axes: Sequence[AxisSpec] = AXES,
    max_seq_len: int = 96,
    skip_first: int = 8,
    fit_state: Path | None = None,
    runtime_headroom_gib: float = 20.0,
) -> dict[str, Any]:
    from jlens.fitting import valid_position_mask
    from jlens.hooks import ActivationRecorder

    axis_names = tuple(axis.name for axis in axes)
    token_ids, score_columns = _score_columns(axis_token_ids, axis_names)
    direction_sum = torch.zeros(
        len(axis_names),
        int(lens_model.d_model),
        dtype=torch.float32,
    )
    n_done = 0
    next_index = 0
    if fit_state is not None and fit_state.exists():
        saved = torch.load(fit_state, map_location="cpu", weights_only=True)
        expected = {
            "profile": profile.name,
            "source_layer": profile.source_layer,
            "target_layer": profile.target_layer,
            "axis_names": list(axis_names),
            "hidden_size": int(lens_model.d_model),
        }
        for key, value in expected.items():
            if saved.get(key) != value:
                raise ValueError(
                    f"fit state {key}={saved.get(key)!r}; expected {value!r}"
                )
        direction_sum = saved["direction_sum"].float()
        n_done = int(saved["n_done"])
        next_index = int(saved["next_index"])

    device = lens_model.input_device
    for prompt_index, prompt in enumerate(prompts):
        if prompt_index < next_index:
            continue
        started = time.perf_counter()
        input_ids = lens_model.encode(prompt, max_length=max_seq_len)
        seq_len = int(input_ids.shape[1])
        positions = (
            valid_position_mask(
                seq_len,
                skip_first=skip_first,
            )
            .nonzero(as_tuple=True)[0]
            .to(device)
        )
        with (
            ActivationRecorder(
                lens_model.layers,
                at=[profile.source_layer, profile.target_layer],
                start_graph_at=profile.source_layer,
            ) as recorder,
            torch.enable_grad(),
        ):
            lens_model.forward(input_ids)
            target_activation = recorder.activations[profile.target_layer]
            source_activation = recorder.activations[profile.source_layer]
            normed = lens_model._final_norm(  # noqa: SLF001
                target_activation.to(lens_model._lm_head.weight.dtype)  # noqa: SLF001
            )
            token_index = torch.tensor(
                token_ids,
                dtype=torch.long,
                device=lens_model._lm_head.weight.device,  # noqa: SLF001
            )
            selected_weight = lens_model._lm_head.weight.index_select(  # noqa: SLF001
                0,
                token_index,
            )
            logits = F.linear(
                normed.to(selected_weight.dtype),
                selected_weight,
            ).float()
            for axis_index, axis_name in enumerate(axis_names):
                positive, negative = score_columns[axis_name]
                score = (
                    logits[:, positions, :][:, :, positive].mean()
                    - logits[:, positions, :][:, :, negative].mean()
                )
                (gradient,) = torch.autograd.grad(
                    score,
                    (source_activation,),
                    retain_graph=axis_index < len(axis_names) - 1,
                )
                direction_sum[axis_index] += (
                    gradient[0, positions, :].float().mean(dim=0).cpu()
                )

        n_done += 1
        next_index = prompt_index + 1
        if fit_state is not None:
            atomic_torch_save(
                {
                    "profile": profile.name,
                    "source_layer": profile.source_layer,
                    "target_layer": profile.target_layer,
                    "axis_names": list(axis_names),
                    "hidden_size": int(lens_model.d_model),
                    "direction_sum": direction_sum,
                    "n_done": n_done,
                    "next_index": next_index,
                },
                fit_state,
            )
        snapshot = runtime_gate(
            device,
            stage=f"{profile.name} lens prompt {next_index}",
            minimum_cuda_headroom_gib=(
                runtime_headroom_gib if profile.paper_target else 0.5
            ),
        )
        emit(
            "lens_prompt",
            profile=profile.name,
            index=next_index,
            total=len(prompts),
            sequence_tokens=seq_len,
            seconds=round(time.perf_counter() - started, 3),
            memory=snapshot,
        )
        del (
            input_ids,
            target_activation,
            source_activation,
            normed,
            token_index,
            selected_weight,
            logits,
            score,
            gradient,
            recorder,
        )
        gc.collect()

    if n_done != len(prompts):
        raise ValueError(f"fit completed {n_done}/{len(prompts)} prompts")
    return {
        "format": "targeted-affect-lens-v1",
        "profile": asdict(profile),
        "axis_names": list(axis_names),
        "axis_token_ids": {
            axis: {
                pole: [int(token_id) for token_id in ids] for pole, ids in poles.items()
            }
            for axis, poles in axis_token_ids.items()
        },
        "fit_prompt_count": n_done,
        "max_seq_len": max_seq_len,
        "skip_first": skip_first,
        "read_directions": direction_sum / n_done,
    }


@contextlib.contextmanager
def capture_layer_activations(
    layers: Sequence[torch.nn.Module],
    layer_indices: Iterable[int],
):
    activations: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(index: int):
        def hook(_module: Any, _args: Any, output: Any) -> None:
            tensor = output if torch.is_tensor(output) else output[0]
            activations[index] = tensor.detach()

        return hook

    try:
        for index in sorted(set(layer_indices)):
            handles.append(layers[index].register_forward_hook(make_hook(index)))
        yield activations
    finally:
        for handle in handles:
            handle.remove()


def _tensor_ids(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value
    elif hasattr(value, "input_ids"):
        tensor = value.input_ids
    else:
        tensor = torch.tensor(value, dtype=torch.long)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.cpu()


def render_messages(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
) -> torch.Tensor:
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt",
        enable_thinking=False,
    )
    return _tensor_ids(rendered)


def inserted_token_positions(
    empty_turn_ids: torch.Tensor,
    content_turn_ids: torch.Tensor,
) -> list[int]:
    if empty_turn_ids.ndim != 2 or content_turn_ids.ndim != 2:
        raise ValueError("chat token tensors must be rank two")
    if empty_turn_ids.shape[0] != 1 or content_turn_ids.shape[0] != 1:
        raise ValueError("chat token tensors must have batch size one")
    empty = empty_turn_ids[0]
    content = content_turn_ids[0]
    shared = min(len(empty), len(content))
    prefix = 0
    while prefix < shared and int(empty[prefix]) == int(content[prefix]):
        prefix += 1
    suffix_limit = min(len(empty) - prefix, len(content) - prefix)
    suffix = 0
    while suffix < suffix_limit and int(empty[len(empty) - suffix - 1]) == int(
        content[len(content) - suffix - 1]
    ):
        suffix += 1
    stop = len(content) - suffix
    if stop <= prefix:
        raise ValueError("chat template did not insert user-content tokens")
    return list(range(prefix, stop))


def render_user_content(
    tokenizer: Any,
    *,
    system_prompt: str,
    user_text: str,
) -> tuple[torch.Tensor, list[int]]:
    before = [{"role": "system", "content": system_prompt}]
    empty = [*before, {"role": "user", "content": ""}]
    populated = [*before, {"role": "user", "content": user_text}]
    empty_ids = render_messages(
        tokenizer,
        empty,
        add_generation_prompt=False,
    )
    content_ids = render_messages(
        tokenizer,
        populated,
        add_generation_prompt=False,
    )
    context_ids = render_messages(
        tokenizer,
        populated,
        add_generation_prompt=True,
    )
    positions = inserted_token_positions(empty_ids, content_ids)
    if not torch.equal(content_ids, context_ids[:, : content_ids.shape[1]]):
        raise ValueError("generation prompt changed existing chat tokens")
    return context_ids, positions


def token_text(tokenizer: Any, token_id: int) -> str:
    return tokenizer.convert_ids_to_tokens(int(token_id))


def display_token(text: str) -> str:
    replacements = {
        "Ġ": "▁",
        "▁": "▁",
        "Ċ": "↵",
        "Ŀ": "↵",
    }
    rendered = text
    for source, target in replacements.items():
        rendered = rendered.replace(source, target)
    return rendered


@torch.inference_mode()
def raw_case_coordinates(
    lens_model: Any,
    tokenizer: Any,
    *,
    system_prompt: str,
    user_text: str,
    read_matrix: torch.Tensor,
    trace_layers: Sequence[int],
    max_seq_len: int,
) -> tuple[torch.Tensor, list[int], dict[int, torch.Tensor]]:
    input_ids, positions = render_user_content(
        tokenizer,
        system_prompt=system_prompt,
        user_text=user_text,
    )
    if input_ids.shape[1] > max_seq_len:
        raise ValueError(
            f"rendered input has {input_ids.shape[1]} tokens; maximum is {max_seq_len}"
        )
    with capture_layer_activations(
        lens_model.layers,
        trace_layers,
    ) as activations:
        output = lens_model.forward(input_ids.to(lens_model.input_device))
    del output
    selected_positions = torch.tensor(
        positions,
        dtype=torch.long,
        device=lens_model.input_device,
    )
    rows = read_matrix.to(
        device=lens_model.input_device,
        dtype=torch.float32,
    )
    raw = {}
    for layer in trace_layers:
        hidden = activations[int(layer)][0].index_select(
            0,
            selected_positions,
        )
        raw[int(layer)] = (hidden.float() @ rows.T).cpu()
    return input_ids, positions, raw


def fit_calibration(
    lens_model: Any,
    tokenizer: Any,
    *,
    system_prompts: Mapping[str, str],
    messages: Sequence[str],
    read_matrix: torch.Tensor,
    trace_layers: Sequence[int],
    max_seq_len: int,
    profile: ModelProfile,
) -> dict[str, dict[int, dict[str, torch.Tensor]]]:
    calibration: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    device = lens_model.input_device
    for system_index, (system_id, system_prompt) in enumerate(
        system_prompts.items(),
        start=1,
    ):
        samples: dict[int, list[torch.Tensor]] = {
            int(layer): [] for layer in trace_layers
        }
        for message_index, message in enumerate(messages, start=1):
            _ids, _positions, raw = raw_case_coordinates(
                lens_model,
                tokenizer,
                system_prompt=system_prompt,
                user_text=message,
                read_matrix=read_matrix,
                trace_layers=trace_layers,
                max_seq_len=max_seq_len,
            )
            for layer in trace_layers:
                samples[int(layer)].append(raw[int(layer)].mean(dim=0))
            if message_index % 8 == 0 or message_index == len(messages):
                emit(
                    "calibration_progress",
                    profile=profile.name,
                    system=system_id,
                    system_index=system_index,
                    systems=len(system_prompts),
                    message=message_index,
                    messages=len(messages),
                )
        calibration[system_id] = {}
        for layer in trace_layers:
            values = torch.stack(samples[int(layer)]).float()
            calibration[system_id][int(layer)] = {
                "center": values.mean(dim=0),
                "scale": values.std(dim=0, unbiased=False).clamp_min(1e-6),
            }
        runtime_gate(
            device,
            stage=f"{profile.name} calibration {system_id}",
            minimum_cuda_headroom_gib=(20.0 if profile.paper_target else 0.5),
        )
    return calibration


def _rounded(values: torch.Tensor, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values.tolist()]


@torch.inference_mode()
def measure_case(
    lens_model: Any,
    tokenizer: Any,
    *,
    case: EvalCase,
    system_id: str,
    system_prompt: str,
    read_matrix: torch.Tensor,
    calibration: Mapping[int, Mapping[str, torch.Tensor]],
    profile: ModelProfile,
    max_seq_len: int,
) -> dict[str, Any]:
    input_ids, positions, raw = raw_case_coordinates(
        lens_model,
        tokenizer,
        system_prompt=system_prompt,
        user_text=case.text,
        read_matrix=read_matrix,
        trace_layers=profile.trace_layers,
        max_seq_len=max_seq_len,
    )
    standardized = {
        layer: (
            (values.float() - calibration[layer]["center"].float().unsqueeze(0))
            / calibration[layer]["scale"].float().unsqueeze(0)
        )
        for layer, values in raw.items()
    }
    source = standardized[profile.source_layer]
    tokens = []
    for token_index, absolute_position in enumerate(positions):
        raw_text = token_text(tokenizer, int(input_ids[0, absolute_position]))
        tokens.append(
            {
                "index": token_index,
                "position": int(absolute_position),
                "token_id": int(input_ids[0, absolute_position]),
                "text": raw_text,
                "label": display_token(raw_text),
                "coordinates": _rounded(source[token_index]),
                "coordinates_by_layer": {
                    str(layer): _rounded(values[token_index])
                    for layer, values in standardized.items()
                },
            }
        )
    result = {
        "id": f"{case.case_id}--{system_id}",
        "case_id": case.case_id,
        "kind": case.kind,
        "group": case.group,
        "text": case.text,
        "variant": case.variant,
        "axis": case.axis,
        "pole": case.pole,
        "subset": case.subset,
        "system": system_id,
        "context_tokens": int(input_ids.shape[1]),
        "user_tokens": len(positions),
        "utterance": _rounded(source.mean(dim=0)),
        "frontier": _rounded(source[-1]),
        "utterance_by_layer": {
            str(layer): _rounded(values.mean(dim=0))
            for layer, values in standardized.items()
        },
        "frontier_by_layer": {
            str(layer): _rounded(values[-1]) for layer, values in standardized.items()
        },
        "tokens": tokens,
    }
    return result


def serializable_lens(
    lens: Mapping[str, Any],
    calibration: Mapping[str, Mapping[int, Mapping[str, torch.Tensor]]],
) -> dict[str, Any]:
    return {
        **{key: value for key, value in lens.items() if key != "read_directions"},
        "read_directions": lens["read_directions"].cpu(),
        "calibration": {
            system: {
                int(layer): {
                    "center": values["center"].cpu(),
                    "scale": values["scale"].cpu(),
                }
                for layer, values in layers.items()
            }
            for system, layers in calibration.items()
        },
    }


def _git_commit(repository: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_report(
    profile: ModelProfile,
    device: torch.device,
    *,
    repository: Path,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    packages = {}
    for name in (
        "accelerate",
        "jlens",
        "matplotlib",
        "numpy",
        "torch",
        "transformers",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "created_at": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
        "device": str(device),
        "model": asdict(profile),
        "repository_commit": _git_commit(repository),
        "command": list(command if command is not None else sys.argv),
    }
