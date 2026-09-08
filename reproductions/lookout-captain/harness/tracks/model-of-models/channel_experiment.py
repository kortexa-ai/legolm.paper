#!/usr/bin/env python3
"""Train and falsify a compact latent channel between two frozen Qwen models.

The source sees a private color label. The target sees only a fixed request to
recover it. A learned 256D bus is therefore the only possible communication
path. This proves plumbing, not useful collaborative reasoning.

Examples:
  uv run tracks/model-of-models/channel_experiment.py --pair qwen08b-2b
  uv run tracks/model-of-models/channel_experiment.py --pair qwen2b-35b \
      --steps 64 --batch-size 1
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from mom_bridge import LatentBridge, ResidualInjection


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FINDINGS = REPO / "findings"
CHECKPOINTS = REPO / "checkpoints" / "experiments"
GIB = 1024**3

COLORS = (
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "black",
    "white",
)

SOURCE_TEMPLATES = {
    "train": (
        "The private control color is {label}. Preserve that exact word.",
        "Sensor packet received. Its hidden payload is {label}.",
        "Remember exactly one confidential word: {label}.",
        "A fast classifier selected {label} as its internal vote.",
        "Internal control channel: color={label}. Keep the value available.",
        "The answer key carried only by you is {label}.",
    ),
    "dev": (
        "Confidential chromatic tag: {label}.",
    ),
    "test": (
        "The embedded agent whispers `{label}` into the latent bus.",
        "Privately retain the following swatch name for another model: {label}.",
    ),
}

SOURCE_SYSTEM = (
    "You are the fast source model in a two-model system. Read the private "
    "control value and hold it in your activation state for a downstream "
    "model. Do not replace or reinterpret the value."
)
TARGET_SYSTEM = (
    "You are the receiving model in a two-model system. A private latent "
    "packet may identify one color. Reply with exactly one lowercase color "
    "word from: red, blue, green, yellow, orange, purple, black, white."
)
TARGET_USER = "Read the private packet and report its color."

TOKENIZER_SHA256 = (
    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
)

MODELS: dict[str, dict[str, Any]] = {
    "qwen08b": {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "hidden_size": 1024,
        "layers": 24,
        "stream_to_cuda": False,
        "estimated_weight_gib": 1.7,
    },
    "qwen2b": {
        "model_id": "Qwen/Qwen3.5-2B",
        "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "hidden_size": 2048,
        "layers": 24,
        "stream_to_cuda": False,
        "estimated_weight_gib": 4.3,
    },
    "qwen35b": {
        "model_id": "Qwen/Qwen3.6-35B-A3B",
        "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
        "hidden_size": 2048,
        "layers": 40,
        "stream_to_cuda": True,
        "estimated_weight_gib": 67.0,
    },
}

PAIRS: dict[str, dict[str, Any]] = {
    "qwen08b-2b": {
        "source": "qwen08b",
        "target": "qwen2b",
        "source_layer": 19,
        "target_layer": 11,
        "steps": 192,
        "batch_size": 8,
        "source_batch_size": 8,
        "learning_rate": 3e-3,
        "fraction": 0.15,
        "min_free_before_target_gib": 10.0,
        "min_free_after_backward_gib": 6.0,
    },
    "qwen2b-35b": {
        "source": "qwen2b",
        "target": "qwen35b",
        "source_layer": 19,
        "target_layer": 35,
        "steps": 64,
        "batch_size": 8,
        "source_batch_size": 4,
        "learning_rate": 2e-3,
        "fraction": 0.15,
        "min_free_before_target_gib": 84.0,
        "min_free_after_backward_gib": 10.0,
    },
}


@dataclass(frozen=True)
class ChannelExample:
    example_id: str
    split: str
    template_index: int
    label: str
    label_index: int
    source_text: str


def build_examples() -> list[ChannelExample]:
    examples = []
    for split, templates in SOURCE_TEMPLATES.items():
        for template_index, template in enumerate(templates):
            for label_index, label in enumerate(COLORS):
                examples.append(
                    ChannelExample(
                        example_id=f"{split}-{template_index}-{label}",
                        split=split,
                        template_index=template_index,
                        label=label,
                        label_index=label_index,
                        source_text=template.format(label=label),
                    )
                )
    return examples


def shifted_example_ids(
    examples: Sequence[ChannelExample],
) -> dict[str, str]:
    lookup = {
        (example.split, example.template_index, example.label_index): example
        for example in examples
    }
    shifted = {}
    for example in examples:
        wrong_index = (example.label_index + 1) % len(COLORS)
        shifted[example.example_id] = lookup[
            (example.split, example.template_index, wrong_index)
        ].example_id
    return shifted


def chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def choose_device(requested: str) -> torch.device:
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
    return device


def choose_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if requested != "auto":
        raise ValueError(f"unknown dtype {requested}")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def cuda_snapshot(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "cuda_free_gib": round(free_bytes / GIB, 3),
        "cuda_total_gib": round(total_bytes / GIB, 3),
        "cuda_allocated_gib": round(torch.cuda.memory_allocated(device) / GIB, 3),
        "cuda_reserved_gib": round(torch.cuda.memory_reserved(device) / GIB, 3),
        "cuda_peak_allocated_gib": round(
            torch.cuda.max_memory_allocated(device) / GIB, 3
        ),
        "cuda_peak_reserved_gib": round(
            torch.cuda.max_memory_reserved(device) / GIB, 3
        ),
    }


def require_cuda_free(
    device: torch.device,
    minimum_gib: float,
    *,
    stage: str,
) -> dict[str, float]:
    snapshot = cuda_snapshot(device)
    if device.type == "cuda" and snapshot["cuda_free_gib"] < minimum_gib:
        raise MemoryError(
            f"{stage}: CUDA free {snapshot['cuda_free_gib']:.2f} GiB is below "
            f"the {minimum_gib:.2f} GiB safety gate"
        )
    print(
        json.dumps(
            {"event": "memory_gate", "stage": stage, **snapshot},
            sort_keys=True,
        ),
        flush=True,
    )
    return snapshot


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def freeze_model(model: nn.Module) -> None:
    model.eval()
    model.requires_grad_(False)


def release_device_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def tokenizer_fingerprint(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return ""
    return hashlib.sha256(backend.to_str().encode()).hexdigest()


def load_tokenizers(
    source_profile: Mapping[str, Any],
    target_profile: Mapping[str, Any],
    *,
    local_files_only: bool,
):
    from transformers import AutoTokenizer

    source = AutoTokenizer.from_pretrained(
        source_profile["model_id"],
        revision=source_profile["revision"],
        local_files_only=local_files_only,
    )
    target = AutoTokenizer.from_pretrained(
        target_profile["model_id"],
        revision=target_profile["revision"],
        local_files_only=local_files_only,
    )
    source_fingerprint = tokenizer_fingerprint(source)
    target_fingerprint = tokenizer_fingerprint(target)
    if source_fingerprint != target_fingerprint:
        raise ValueError("source and target tokenizers are not byte-equivalent")
    if source.vocab_size != target.vocab_size:
        raise ValueError("source and target tokenizer vocabularies differ")
    return source, target, {
        "declared_tokenizer_json_sha256": TOKENIZER_SHA256,
        "runtime_backend_sha256": source_fingerprint,
        "vocab_size": source.vocab_size,
        "identical": True,
    }


def load_model(
    profile: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
    local_files_only: bool,
) -> nn.Module:
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "revision": profile["revision"],
        "dtype": dtype,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
        "local_files_only": local_files_only,
    }
    started = time.perf_counter()
    if profile.get("stream_to_cuda") and device.type == "cuda":
        kwargs["device_map"] = {
            "": device.index if device.index is not None else 0
        }
        model = AutoModelForCausalLM.from_pretrained(
            profile["model_id"],
            **kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            profile["model_id"],
            **kwargs,
        ).to(device)
    freeze_model(model)
    actual_layers = len(model.model.layers)
    actual_hidden = int(model.config.hidden_size)
    if actual_layers != profile["layers"]:
        raise ValueError(
            f"{profile['model_id']} has {actual_layers} layers, expected "
            f"{profile['layers']}"
        )
    if actual_hidden != profile["hidden_size"]:
        raise ValueError(
            f"{profile['model_id']} hidden size {actual_hidden}, expected "
            f"{profile['hidden_size']}"
        )
    print(
        json.dumps(
            {
                "event": "model_loaded",
                "model_id": profile["model_id"],
                "revision": profile["revision"],
                "seconds": round(time.perf_counter() - started, 3),
                **cuda_snapshot(device),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return model


def render_chat(
    tokenizer: Any,
    *,
    system: str,
    user: str,
) -> torch.Tensor:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
    }
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except (TypeError, ValueError):
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(rendered, "input_ids"):
        rendered = rendered.input_ids
    elif isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if not torch.is_tensor(rendered):
        rendered = torch.tensor(rendered, dtype=torch.long)
    if rendered.ndim == 1:
        rendered = rendered.unsqueeze(0)
    return rendered.cpu()


def left_pad(
    sequences: Sequence[torch.Tensor],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("cannot pad an empty sequence list")
    maximum = max(int(sequence.shape[1]) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), maximum),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(sequences), maximum),
        dtype=torch.long,
        device=device,
    )
    for row, sequence in enumerate(sequences):
        width = int(sequence.shape[1])
        input_ids[row, -width:] = sequence[0].to(device)
        attention_mask[row, -width:] = 1
    return input_ids, attention_mask


def color_token_ids(tokenizer: Any) -> dict[str, int]:
    result = {}
    for color in COLORS:
        ids = tokenizer.encode(color, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"color {color!r} is not one tokenizer token: {ids}")
        result[color] = int(ids[0])
    if len(set(result.values())) != len(result):
        raise ValueError("color labels do not have unique token ids")
    return result


def parameter_samples(model: nn.Module) -> dict[str, torch.Tensor]:
    named = list(model.named_parameters())
    if not named:
        return {}
    slots = sorted({0, len(named) // 2, len(named) - 1})
    samples = {}
    for slot in slots:
        name, parameter = named[slot]
        values = parameter.detach().reshape(-1)
        take = min(128, values.numel())
        samples[name] = values[:take].float().cpu().clone()
    return samples


def samples_unchanged(
    model: nn.Module,
    before: Mapping[str, torch.Tensor],
) -> bool:
    current = dict(model.named_parameters())
    return all(
        name in current
        and torch.equal(
            current[name].detach().reshape(-1)[: sample.numel()].float().cpu(),
            sample,
        )
        for name, sample in before.items()
    )


@torch.inference_mode()
def extract_source_features(
    model: nn.Module,
    tokenizer: Any,
    examples: Sequence[ChannelExample],
    *,
    layer: int,
    batch_size: int,
    system: str = SOURCE_SYSTEM,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    device = model_device(model)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    features: dict[str, torch.Tensor] = {}
    started = time.perf_counter()
    for batch_index, batch in enumerate(chunks(examples, batch_size), start=1):
        rendered = [
            render_chat(
                tokenizer,
                system=system,
                user=example.source_text,
            )
            for example in batch
        ]
        input_ids, attention_mask = left_pad(
            rendered,
            pad_token_id=int(pad_token_id),
            device=device,
        )
        output = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = output.hidden_states
        if hidden_states is None or len(hidden_states) <= layer + 1:
            raise RuntimeError("source model did not return the requested layer")
        values = hidden_states[layer + 1][:, -1, :].detach().float().cpu()
        for example, value in zip(batch, values, strict=True):
            features[example.example_id] = value.clone()
        print(
            json.dumps(
                {
                    "event": "source_features",
                    "batch": batch_index,
                    "captured": len(features),
                    "total": len(examples),
                    **cuda_snapshot(device),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    elapsed = time.perf_counter() - started
    return features, {
        "count": len(features),
        "seconds": round(elapsed, 3),
        "features_per_second": round(len(features) / elapsed, 3),
        "layer": int(layer),
        "hidden_size": int(next(iter(features.values())).numel()),
    }


def target_prompt(tokenizer: Any, device: torch.device) -> torch.Tensor:
    return render_chat(
        tokenizer,
        system=TARGET_SYSTEM,
        user=TARGET_USER,
    ).to(device)


def next_token_logits(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    delta: torch.Tensor | None,
    layer: int,
    fraction: float,
) -> torch.Tensor:
    attention_mask = torch.ones_like(input_ids)
    if delta is None:
        output = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    else:
        with ResidualInjection(
            model.model.layers[layer],
            delta,
            fraction=fraction,
        ):
            output = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    final_hidden = output.last_hidden_state[:, -1, :]
    return model.lm_head(final_hidden).float()


def training_batches(
    examples: Sequence[ChannelExample],
    *,
    steps: int,
    batch_size: int,
    seed: int,
) -> list[list[ChannelExample]]:
    if not examples:
        raise ValueError("training examples are empty")
    rng = random.Random(seed)
    schedule: list[list[ChannelExample]] = []
    order = list(examples)
    while len(schedule) < steps:
        rng.shuffle(order)
        schedule.extend(
            [list(batch) for batch in chunks(order, batch_size)]
        )
    return schedule[:steps]


def channel_targets(
    examples: Sequence[ChannelExample],
    token_ids: Mapping[str, int],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    vocabulary_targets = torch.tensor(
        [token_ids[example.label] for example in examples],
        dtype=torch.long,
        device=device,
    )
    packet_targets = torch.tensor(
        [example.label_index for example in examples],
        dtype=torch.long,
        device=device,
    )
    return vocabulary_targets, packet_targets


def train_bridge(
    target_model: nn.Module,
    bridge: LatentBridge,
    target_tokenizer: Any,
    features: Mapping[str, torch.Tensor],
    examples: Sequence[ChannelExample],
    *,
    layer: int,
    steps: int,
    batch_size: int,
    fraction: float,
    learning_rate: float,
    grad_clip: float,
    seed: int,
    log_every: int,
    min_free_after_backward_gib: float,
    auxiliary_weight: float,
) -> tuple[dict[str, Any], nn.Linear]:
    device = model_device(target_model)
    bridge.train()
    auxiliary = nn.Linear(bridge.bus_dim, len(COLORS)).to(
        device=device,
        dtype=torch.float32,
    )
    optimizer = torch.optim.AdamW(
        [*bridge.parameters(), *auxiliary.parameters()],
        lr=learning_rate,
        weight_decay=0.0,
    )
    prompt = target_prompt(target_tokenizer, device)
    label_ids = color_token_ids(target_tokenizer)
    schedule = training_batches(
        examples,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
    )
    history = []
    first_gradient_finite = False
    first_gradient_norm = 0.0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step, batch in enumerate(schedule, start=1):
        optimizer.zero_grad(set_to_none=True)
        source = torch.stack(
            [features[example.example_id] for example in batch]
        ).to(device)
        targets, packet_targets = channel_targets(
            batch,
            label_ids,
            device=device,
        )
        input_ids = prompt.expand(len(batch), -1)
        packet = bridge.encode(source)
        delta = bridge.receiver(packet)
        logits = next_token_logits(
            target_model,
            input_ids,
            delta=delta,
            layer=layer,
            fraction=fraction,
        )
        channel_loss = F.cross_entropy(logits, targets)
        auxiliary_loss = F.cross_entropy(auxiliary(packet), packet_targets)
        loss = channel_loss + auxiliary_weight * auxiliary_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            bridge.parameters(),
            grad_clip,
        )
        gradient_finite = bool(torch.isfinite(grad_norm))
        if not gradient_finite:
            raise FloatingPointError(f"non-finite gradient at step {step}")
        if step == 1:
            first_gradient_finite = gradient_finite
            first_gradient_norm = float(grad_norm.detach().cpu())
            require_cuda_free(
                device,
                min_free_after_backward_gib,
                stage="after first bridge backward",
            )
        optimizer.step()

        predictions = logits.argmax(dim=-1)
        accuracy = float((predictions == targets).float().mean().detach().cpu())
        packet_predictions = auxiliary(packet).argmax(dim=-1)
        packet_accuracy = float(
            (packet_predictions == packet_targets).float().mean().detach().cpu()
        )
        record = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "channel_loss": float(channel_loss.detach().cpu()),
            "auxiliary_loss": float(auxiliary_loss.detach().cpu()),
            "batch_accuracy": accuracy,
            "packet_accuracy": packet_accuracy,
            "grad_norm": float(grad_norm.detach().cpu()),
        }
        if step == 1 or step == steps or step % log_every == 0:
            record["memory"] = cuda_snapshot(device)
            history.append(record)
            print(
                json.dumps(
                    {"event": "bridge_step", **record},
                    sort_keys=True,
                ),
                flush=True,
            )

    optimizer.zero_grad(set_to_none=True)
    del optimizer
    auxiliary.eval()
    elapsed = time.perf_counter() - started
    base_gradients_absent = all(
        parameter.grad is None for parameter in target_model.parameters()
    )
    return {
        "steps": steps,
        "batch_size": batch_size,
        "fraction": fraction,
        "learning_rate": learning_rate,
        "auxiliary_weight": auxiliary_weight,
        "grad_clip": grad_clip,
        "seconds": round(elapsed, 3),
        "steps_per_second": round(steps / elapsed, 4),
        "first_gradient_finite": first_gradient_finite,
        "first_gradient_norm": first_gradient_norm,
        "base_gradients_absent": base_gradients_absent,
        "history": history,
        "memory": cuda_snapshot(device),
    }, auxiliary


@torch.inference_mode()
def evaluate_packet_probe(
    bridge: LatentBridge,
    auxiliary: nn.Linear,
    features: Mapping[str, torch.Tensor],
    examples: Sequence[ChannelExample],
    *,
    device: torch.device,
) -> dict[str, Any]:
    bridge.eval()
    auxiliary.eval()
    report = {}
    for split in ("train", "dev", "test"):
        selected = [example for example in examples if example.split == split]
        source = torch.stack(
            [features[example.example_id] for example in selected]
        ).to(device)
        targets = torch.tensor(
            [example.label_index for example in selected],
            dtype=torch.long,
            device=device,
        )
        logits = auxiliary(bridge.encode(source))
        report[split] = {
            "count": len(selected),
            "accuracy": float(
                (logits.argmax(dim=-1) == targets).float().mean()
            ),
            "nll": float(F.cross_entropy(logits, targets)),
        }
    return report


def feature_controls(
    examples: Sequence[ChannelExample],
    features: Mapping[str, torch.Tensor],
    *,
    seed: int,
) -> tuple[
    dict[str, str],
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    shifted = shifted_example_ids(examples)
    training = [
        features[example.example_id]
        for example in examples
        if example.split == "train"
    ]
    stacked = torch.stack(training)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0, unbiased=False).clamp_min(1e-6)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1009)
    random_features = {
        example.example_id: mean
        + torch.randn(
            mean.shape,
            generator=generator,
            dtype=mean.dtype,
        )
        * std
        for example in examples
    }
    return shifted, mean, random_features


def condition_source(
    condition: str,
    example: ChannelExample,
    *,
    features: Mapping[str, torch.Tensor],
    shifted: Mapping[str, str],
    mean_feature: torch.Tensor,
    random_features: Mapping[str, torch.Tensor],
) -> torch.Tensor | None:
    if condition == "off":
        return None
    if condition == "true":
        return features[example.example_id]
    if condition == "shuffled":
        return features[shifted[example.example_id]]
    if condition == "mean":
        return mean_feature
    if condition == "random":
        return random_features[example.example_id]
    raise ValueError(f"unknown condition {condition}")


@torch.inference_mode()
def evaluate_condition(
    target_model: nn.Module,
    bridge: LatentBridge,
    tokenizer: Any,
    examples: Sequence[ChannelExample],
    *,
    condition: str,
    features: Mapping[str, torch.Tensor],
    shifted: Mapping[str, str],
    mean_feature: torch.Tensor,
    random_features: Mapping[str, torch.Tensor],
    layer: int,
    fraction: float,
    batch_size: int,
) -> dict[str, Any]:
    device = model_device(target_model)
    prompt = target_prompt(tokenizer, device)
    token_ids = color_token_ids(tokenizer)
    label_tensor = torch.tensor(
        [token_ids[label] for label in COLORS],
        dtype=torch.long,
        device=device,
    )
    rows = []
    started = time.perf_counter()
    bridge.eval()

    for batch in chunks(list(examples), batch_size):
        sources = [
            condition_source(
                condition,
                example,
                features=features,
                shifted=shifted,
                mean_feature=mean_feature,
                random_features=random_features,
            )
            for example in batch
        ]
        delta = None
        if condition != "off":
            delta = bridge(torch.stack(sources).to(device))  # type: ignore[arg-type]
        input_ids = prompt.expand(len(batch), -1)
        logits = next_token_logits(
            target_model,
            input_ids,
            delta=delta,
            layer=layer,
            fraction=fraction,
        )
        candidate_logits = logits.index_select(dim=-1, index=label_tensor)
        candidate_predictions = candidate_logits.argmax(dim=-1)
        unrestricted_predictions = logits.argmax(dim=-1)
        log_denominator = torch.logsumexp(logits, dim=-1)

        for row_index, example in enumerate(batch):
            correct_id = token_ids[example.label]
            correct_logit = logits[row_index, correct_id]
            other_candidate_logits = torch.cat(
                [
                    candidate_logits[row_index, : example.label_index],
                    candidate_logits[row_index, example.label_index + 1 :],
                ]
            )
            predicted_id = int(unrestricted_predictions[row_index])
            restricted_slot = int(candidate_predictions[row_index])
            rank = int((logits[row_index] > correct_logit).sum()) + 1
            rows.append(
                {
                    "example_id": example.example_id,
                    "label": example.label,
                    "top1_token_id": predicted_id,
                    "top1_text": tokenizer.decode([predicted_id]),
                    "top1_correct": predicted_id == correct_id,
                    "restricted_prediction": COLORS[restricted_slot],
                    "restricted_correct": restricted_slot == example.label_index,
                    "correct_probability": math.exp(
                        float(correct_logit - log_denominator[row_index])
                    ),
                    "correct_rank": rank,
                    "label_margin": float(
                        correct_logit - other_candidate_logits.max()
                    ),
                    "nll": float(log_denominator[row_index] - correct_logit),
                }
            )

    elapsed = time.perf_counter() - started
    return {
        "condition": condition,
        "count": len(rows),
        "top1_accuracy": sum(row["top1_correct"] for row in rows) / len(rows),
        "restricted_accuracy": (
            sum(row["restricted_correct"] for row in rows) / len(rows)
        ),
        "mean_correct_probability": sum(
            row["correct_probability"] for row in rows
        )
        / len(rows),
        "mean_correct_rank": sum(row["correct_rank"] for row in rows) / len(rows),
        "mean_label_margin": sum(row["label_margin"] for row in rows) / len(rows),
        "mean_nll": sum(row["nll"] for row in rows) / len(rows),
        "seconds": round(elapsed, 3),
        "rows": rows,
    }


@torch.inference_mode()
def evaluate_controls(
    target_model: nn.Module,
    bridge: LatentBridge,
    tokenizer: Any,
    examples: Sequence[ChannelExample],
    *,
    features: Mapping[str, torch.Tensor],
    shifted: Mapping[str, str],
    mean_feature: torch.Tensor,
    random_features: Mapping[str, torch.Tensor],
    layer: int,
    fraction: float,
    batch_size: int,
) -> dict[str, Any]:
    report = {}
    for condition in ("off", "true", "shuffled", "mean", "random"):
        result = evaluate_condition(
            target_model,
            bridge,
            tokenizer,
            examples,
            condition=condition,
            features=features,
            shifted=shifted,
            mean_feature=mean_feature,
            random_features=random_features,
            layer=layer,
            fraction=fraction,
            batch_size=batch_size,
        )
        report[condition] = result
        print(
            json.dumps(
                {
                    "event": "control_evaluation",
                    "condition": condition,
                    "count": result["count"],
                    "top1_accuracy": result["top1_accuracy"],
                    "restricted_accuracy": result["restricted_accuracy"],
                    "mean_label_margin": result["mean_label_margin"],
                    **cuda_snapshot(model_device(target_model)),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return report


@torch.inference_mode()
def dose_sweep(
    target_model: nn.Module,
    bridge: LatentBridge,
    tokenizer: Any,
    examples: Sequence[ChannelExample],
    *,
    features: Mapping[str, torch.Tensor],
    shifted: Mapping[str, str],
    mean_feature: torch.Tensor,
    random_features: Mapping[str, torch.Tensor],
    layer: int,
    doses: Sequence[float],
    batch_size: int,
) -> list[dict[str, Any]]:
    rows = []
    for dose in doses:
        result = evaluate_condition(
            target_model,
            bridge,
            tokenizer,
            examples,
            condition="true",
            features=features,
            shifted=shifted,
            mean_feature=mean_feature,
            random_features=random_features,
            layer=layer,
            fraction=dose,
            batch_size=batch_size,
        )
        rows.append(
            {
                "fraction": dose,
                "top1_accuracy": result["top1_accuracy"],
                "restricted_accuracy": result["restricted_accuracy"],
                "mean_label_margin": result["mean_label_margin"],
                "mean_nll": result["mean_nll"],
            }
        )
    return rows


@torch.inference_mode()
def exact_zero_identity(
    target_model: nn.Module,
    bridge: LatentBridge,
    tokenizer: Any,
    feature: torch.Tensor,
    *,
    layer: int,
) -> dict[str, Any]:
    device = model_device(target_model)
    prompt = target_prompt(tokenizer, device)
    baseline = next_token_logits(
        target_model,
        prompt,
        delta=None,
        layer=layer,
        fraction=0.0,
    )
    delta = bridge(feature.unsqueeze(0).to(device))
    zero = next_token_logits(
        target_model,
        prompt,
        delta=delta,
        layer=layer,
        fraction=0.0,
    )
    difference = (baseline - zero).abs()
    return {
        "exact": bool(torch.equal(baseline, zero)),
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
    }


@torch.inference_mode()
def generate_probe(
    target_model: nn.Module,
    bridge: LatentBridge,
    tokenizer: Any,
    examples: Sequence[ChannelExample],
    *,
    features: Mapping[str, torch.Tensor],
    shifted: Mapping[str, str],
    mean_feature: torch.Tensor,
    random_features: Mapping[str, torch.Tensor],
    layer: int,
    fraction: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    device = model_device(target_model)
    prompt = target_prompt(tokenizer, device)
    token_ids = color_token_ids(tokenizer)
    rows = []
    bridge.eval()
    for condition in ("off", "true", "shuffled"):
        for example in examples:
            source = condition_source(
                condition,
                example,
                features=features,
                shifted=shifted,
                mean_feature=mean_feature,
                random_features=random_features,
            )
            delta = None
            if source is not None:
                delta = bridge(source.unsqueeze(0).to(device))
            context: Any
            if delta is None:
                context = _NullContext()
            else:
                context = ResidualInjection(
                    target_model.model.layers[layer],
                    delta,
                    fraction=fraction,
                    once=True,
                )
            with context:
                generated = target_model.generate(
                    input_ids=prompt,
                    attention_mask=torch.ones_like(prompt),
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_ids = generated[0, prompt.shape[1] :]
            first_id = int(new_ids[0]) if new_ids.numel() else -1
            rows.append(
                {
                    "condition": condition,
                    "example_id": example.example_id,
                    "label": example.label,
                    "first_token_id": first_id,
                    "first_token_text": (
                        tokenizer.decode([first_id]) if first_id >= 0 else ""
                    ),
                    "first_token_correct": first_id == token_ids[example.label],
                    "response": tokenizer.decode(
                        new_ids,
                        skip_special_tokens=True,
                    ),
                }
            )
    aggregate = {}
    for condition in ("off", "true", "shuffled"):
        selected = [row for row in rows if row["condition"] == condition]
        aggregate[condition] = {
            "count": len(selected),
            "first_token_accuracy": (
                sum(row["first_token_correct"] for row in selected)
                / len(selected)
            ),
        }
    return {"aggregate": aggregate, "rows": rows}


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc: Any):
        del exc
        return None


def packet_geometry(
    bridge: LatentBridge,
    examples: Sequence[ChannelExample],
    features: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, Any]:
    bridge.eval()
    primary = [
        example
        for example in examples
        if example.split == "test" and example.template_index == 0
    ]
    with torch.inference_mode():
        source = torch.stack(
            [features[example.example_id] for example in primary]
        ).to(device)
        raw_packets = bridge.encode(source).float()
        packet_norms = torch.linalg.vector_norm(raw_packets, dim=-1)
        packets = F.normalize(raw_packets, dim=-1)
        cosine = packets @ packets.T
    off_diagonal = cosine[
        ~torch.eye(len(primary), dtype=torch.bool, device=device)
    ]
    return {
        "packet_norm_mean": float(packet_norms.mean()),
        "off_diagonal_cosine_mean": float(off_diagonal.mean()),
        "off_diagonal_cosine_max": float(off_diagonal.max()),
        "matrix": cosine.cpu().tolist(),
        "labels": [example.label for example in primary],
    }


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def atomic_json_dump(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def promotion_verdict(
    controls: Mapping[str, Any],
    generation: Mapping[str, Any],
    identity: Mapping[str, Any],
    training: Mapping[str, Any],
    packet_probe: Mapping[str, Any],
    base_weights_unchanged: bool,
) -> dict[str, Any]:
    checks = {
        "true_top1_at_least_075": controls["true"]["top1_accuracy"] >= 0.75,
        "shuffled_top1_at_most_025": (
            controls["shuffled"]["top1_accuracy"] <= 0.25
        ),
        "off_top1_at_most_025": controls["off"]["top1_accuracy"] <= 0.25,
        "mean_top1_at_most_025": controls["mean"]["top1_accuracy"] <= 0.25,
        "random_top1_at_most_025": controls["random"]["top1_accuracy"] <= 0.25,
        "true_margin_beats_shuffled": (
            controls["true"]["mean_label_margin"]
            > controls["shuffled"]["mean_label_margin"]
        ),
        "cached_generation_at_least_075": (
            generation["aggregate"]["true"]["first_token_accuracy"] >= 0.75
        ),
        "zero_is_exact_identity": bool(identity["exact"]),
        "bridge_gradient_finite": bool(training["first_gradient_finite"]),
        "packet_probe_test_at_least_075": (
            packet_probe["test"]["accuracy"] >= 0.75
        ),
        "base_gradients_absent": bool(training["base_gradients_absent"]),
        "base_weights_unchanged": bool(base_weights_unchanged),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
    }


def write_figure(report: Mapping[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    controls = report["evaluation"]["controls"]
    conditions = ["off", "true", "shuffled", "mean", "random"]
    top1 = [controls[name]["top1_accuracy"] for name in conditions]
    restricted = [
        controls[name]["restricted_accuracy"] for name in conditions
    ]
    doses = report["evaluation"]["dose_sweep"]
    history = report["training"]["history"]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    positions = list(range(len(conditions)))
    width = 0.36
    axes[0].bar(
        [position - width / 2 for position in positions],
        top1,
        width,
        label="unrestricted top-1",
        color="#6c5ce7",
    )
    axes[0].bar(
        [position + width / 2 for position in positions],
        restricted,
        width,
        label="among 8 labels",
        color="#00b894",
    )
    axes[0].axhline(1 / len(COLORS), color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(positions, conditions, rotation=20)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Causal controls")
    axes[0].legend(fontsize=8)

    axes[1].plot(
        [row["fraction"] for row in doses],
        [row["top1_accuracy"] for row in doses],
        marker="o",
        label="top-1 accuracy",
        color="#6c5ce7",
    )
    axes[1].plot(
        [row["fraction"] for row in doses],
        [row["restricted_accuracy"] for row in doses],
        marker="s",
        label="restricted accuracy",
        color="#00b894",
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("residual dose")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Dose response")
    axes[1].legend(fontsize=8)

    axes[2].plot(
        [row["step"] for row in history],
        [row["loss"] for row in history],
        marker=".",
        color="#d63031",
    )
    axes[2].set_xlabel("training step")
    axes[2].set_ylabel("cross-entropy")
    axes[2].set_yscale("log")
    axes[2].set_title("Bridge training")

    verdict = "PASS" if report["promotion"]["passed"] else "FAIL"
    figure.suptitle(
        f"{report['pair']} latent channel — promotion {verdict}",
        fontsize=14,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    controls = report["evaluation"]["controls"]
    lines = [
        f"# Model of Models causal channel: {report['pair']}",
        "",
            f"**Status:** {report['status']}",
            "",
            f"**Promotion:** {'PASS' if report['promotion']['passed'] else 'FAIL'}",
            "",
        f"**Bridge parameters:** {report['bridge']['parameters']:,}",
        "",
        "## Setup",
        "",
        (
            f"Frozen `{report['models']['source']['model_id']}` layer "
            f"{report['bridge']['source_layer']} -> "
            f"{report['bridge']['bus_dim']}D bus -> frozen "
            f"`{report['models']['target']['model_id']}` layer "
            f"{report['bridge']['target_layer']}."
        ),
        "",
        (
            "The source alone saw one of eight private colors. The target saw "
            "the same fixed prompt in every case. Results below use held-out "
            "source phrasings."
        ),
        "",
        "## Held-out causal controls",
        "",
        "| condition | unrestricted top-1 | among 8 labels | label margin | NLL |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in ("off", "true", "shuffled", "mean", "random"):
        row = controls[condition]
        lines.append(
            f"| {condition} | {row['top1_accuracy']:.3f} | "
            f"{row['restricted_accuracy']:.3f} | "
            f"{row['mean_label_margin']:+.3f} | {row['mean_nll']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Cached generation",
            "",
            "| condition | first-token accuracy |",
            "|---|---:|",
        ]
    )
    for condition in ("off", "true", "shuffled"):
        row = report["evaluation"]["generation"]["aggregate"][condition]
        lines.append(
            f"| {condition} | {row['first_token_accuracy']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Packet separability",
            "",
            "| split | disposable-head accuracy | NLL |",
            "|---|---:|---:|",
        ]
    )
    for split in ("train", "dev", "test"):
        row = report["evaluation"]["packet_probe"][split]
        lines.append(
            f"| {split} | {row['accuracy']:.3f} | {row['nll']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Contract checks",
            "",
        ]
    )
    for name, passed in report["promotion"]["checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend(
        [
            "",
            "## Memory and timing",
            "",
            (
                f"- Source feature extraction: "
                f"{report['source_features']['seconds']:.3f}s."
            ),
            (
                f"- Bridge training: {report['training']['seconds']:.3f}s "
                f"for {report['training']['steps']} steps."
            ),
        ]
    )
    memory = report["training"].get("memory", {})
    if memory:
        lines.append(
            f"- CUDA peak reserved: "
            f"{memory.get('cuda_peak_reserved_gib', 0):.3f} GiB."
        )
        lines.append(
            f"- CUDA free after training: "
            f"{memory.get('cuda_free_gib', 0):.3f} GiB."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if report["promotion"]["passed"]:
        lines.append(
            "The learned packet is a causal communication channel: true "
            "source state survives held-out wording and cached decoding, while "
            "shuffled and content-free controls do not. This permits the scale "
            "test. It does not show better reasoning or useful deference."
        )
    else:
        failed = ", ".join(f"`{name}`" for name in report["promotion"]["failed"])
        lines.append(
            "The architecture gate did not pass. Failed checks: "
            f"{failed}. Do not promote this configuration."
        )
    lines.extend(
        [
            "",
            f"![channel controls and dose sweep]({report['artifacts']['figure_name']})",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def parse_doses(value: str) -> list[float]:
    doses = [float(item) for item in value.split(",")]
    if not doses or any(dose < 0.0 for dose in doses):
        raise ValueError("doses must be nonnegative")
    return doses


def resolve_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=sorted(PAIRS), default="qwen08b-2b")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--source-batch-size", type=int, default=None)
    parser.add_argument("--bus-dim", type=int, default=256)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--auxiliary-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--log-every", type=int, default=8)
    parser.add_argument(
        "--doses",
        default="0,0.025,0.05,0.1,0.15,0.2,0.3",
    )
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.88)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-prefix", type=Path)
    return parser.parse_args()


def main() -> None:
    args = resolve_arguments()
    pair = dict(PAIRS[args.pair])
    source_profile = dict(MODELS[pair["source"]])
    target_profile = dict(MODELS[pair["target"]])
    steps = args.steps if args.steps is not None else pair["steps"]
    batch_size = (
        args.batch_size if args.batch_size is not None else pair["batch_size"]
    )
    source_batch_size = (
        args.source_batch_size
        if args.source_batch_size is not None
        else pair["source_batch_size"]
    )
    fraction = (
        args.fraction if args.fraction is not None else pair["fraction"]
    )
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else pair["learning_rate"]
    )
    if steps <= 0 or batch_size <= 0 or source_batch_size <= 0:
        raise ValueError("steps and batch sizes must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("training fraction must be in (0, 1]")
    if args.auxiliary_weight < 0.0:
        raise ValueError("auxiliary weight must be nonnegative")
    if not 0.0 < args.cuda_memory_fraction <= 1.0:
        raise ValueError("CUDA memory fraction must be in (0, 1]")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        cuda_index = (
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction,
            device=cuda_index,
        )
    print(
        json.dumps(
            {
                "event": "experiment_start",
                "pair": args.pair,
                "device": str(device),
                "dtype": str(dtype).removeprefix("torch."),
                "steps": steps,
                "batch_size": batch_size,
                "cuda_memory_fraction": args.cuda_memory_fraction,
                **cuda_snapshot(device),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    examples = build_examples()
    train_examples = [example for example in examples if example.split == "train"]
    test_examples = [example for example in examples if example.split == "test"]
    generation_examples = [
        example
        for example in test_examples
        if example.template_index == 0
    ]
    source_tokenizer, target_tokenizer, tokenizer_report = load_tokenizers(
        source_profile,
        target_profile,
        local_files_only=args.local_files_only,
    )
    token_ids = color_token_ids(target_tokenizer)

    require_cuda_free(
        device,
        source_profile["estimated_weight_gib"] + 3.0,
        stage="before source load",
    )
    source_model = load_model(
        source_profile,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    source_before = parameter_samples(source_model)
    features, feature_report = extract_source_features(
        source_model,
        source_tokenizer,
        examples,
        layer=pair["source_layer"],
        batch_size=source_batch_size,
    )
    source_unchanged = samples_unchanged(source_model, source_before)
    del source_model
    release_device_memory(device)

    require_cuda_free(
        device,
        pair["min_free_before_target_gib"],
        stage="before target load",
    )
    target_model = load_model(
        target_profile,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    target_before = parameter_samples(target_model)
    bridge = LatentBridge(
        source_profile["hidden_size"],
        target_profile["hidden_size"],
        bus_dim=args.bus_dim,
    ).to(device=device, dtype=torch.float32)
    training_features = torch.stack(
        [features[example.example_id] for example in train_examples]
    ).to(device)
    normalization = bridge.fit_source_statistics(training_features)

    training, auxiliary = train_bridge(
        target_model,
        bridge,
        target_tokenizer,
        features,
        train_examples,
        layer=pair["target_layer"],
        steps=steps,
        batch_size=batch_size,
        fraction=fraction,
        learning_rate=learning_rate,
        grad_clip=args.grad_clip,
        seed=args.seed,
        log_every=args.log_every,
        min_free_after_backward_gib=pair["min_free_after_backward_gib"],
        auxiliary_weight=args.auxiliary_weight,
    )
    packet_probe = evaluate_packet_probe(
        bridge,
        auxiliary,
        features,
        examples,
        device=device,
    )
    shifted, mean_feature, random_features = feature_controls(
        examples,
        features,
        seed=args.seed,
    )
    controls = evaluate_controls(
        target_model,
        bridge,
        target_tokenizer,
        test_examples,
        features=features,
        shifted=shifted,
        mean_feature=mean_feature,
        random_features=random_features,
        layer=pair["target_layer"],
        fraction=fraction,
        batch_size=batch_size,
    )
    doses = dose_sweep(
        target_model,
        bridge,
        target_tokenizer,
        generation_examples,
        features=features,
        shifted=shifted,
        mean_feature=mean_feature,
        random_features=random_features,
        layer=pair["target_layer"],
        doses=parse_doses(args.doses),
        batch_size=batch_size,
    )
    identity = exact_zero_identity(
        target_model,
        bridge,
        target_tokenizer,
        features[generation_examples[0].example_id],
        layer=pair["target_layer"],
    )
    generation = generate_probe(
        target_model,
        bridge,
        target_tokenizer,
        generation_examples,
        features=features,
        shifted=shifted,
        mean_feature=mean_feature,
        random_features=random_features,
        layer=pair["target_layer"],
        fraction=fraction,
        max_new_tokens=args.max_new_tokens,
    )
    geometry = packet_geometry(
        bridge,
        generation_examples,
        features,
        device=device,
    )
    target_unchanged = samples_unchanged(target_model, target_before)
    bases_unchanged = source_unchanged and target_unchanged
    promotion = promotion_verdict(
        controls,
        generation,
        identity,
        training,
        packet_probe,
        bases_unchanged,
    )

    prefix = args.output_prefix
    if prefix is None:
        prefix = FINDINGS / f"mom-{args.pair}-channel-v2"
    if not prefix.is_absolute():
        prefix = REPO / prefix
    json_path = prefix.with_suffix(".json")
    markdown_path = prefix.with_suffix(".md")
    figure_path = prefix.with_name(prefix.name + "-controls.png")
    checkpoint_path = prefix.with_suffix(".pt")

    report: dict[str, Any] = {
        "format": "legolm-mom-causal-channel-v2",
        "status": "complete",
        "created_at": time.time(),
        "pair": args.pair,
        "git_head_at_start": git_head(),
        "execution": "sequential-source-precompute",
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "seed": args.seed,
        "models": {
            "source": source_profile,
            "target": target_profile,
        },
        "tokenizer": tokenizer_report,
        "labels": {
            "names": list(COLORS),
            "token_ids": token_ids,
        },
        "dataset": {
            "examples": len(examples),
            "train": len(train_examples),
            "dev": sum(example.split == "dev" for example in examples),
            "test": len(test_examples),
            "source_templates": {
                split: list(templates)
                for split, templates in SOURCE_TEMPLATES.items()
            },
        },
        "bridge": {
            "architecture": "RMSNorm-Linear-GELU-Linear",
            "parameters": bridge.parameter_count,
            "source_dim": bridge.source_dim,
            "bus_dim": bridge.bus_dim,
            "target_dim": bridge.target_dim,
            "source_layer": pair["source_layer"],
            "target_layer": pair["target_layer"],
            "injection": "last prompt residual, norm-bounded",
            "training_fraction": fraction,
            "source_normalization": normalization,
            "anti_collapse": (
                "training-set source centering plus supervised packet head"
            ),
        },
        "source_features": feature_report,
        "training": training,
        "evaluation": {
            "controls": controls,
            "dose_sweep": doses,
            "zero_identity": identity,
            "generation": generation,
            "packet_geometry": geometry,
            "packet_probe": packet_probe,
        },
        "integrity": {
            "source_sampled_weights_unchanged": source_unchanged,
            "target_sampled_weights_unchanged": target_unchanged,
            "base_weights_unchanged": bases_unchanged,
        },
        "promotion": promotion,
        "artifacts": {
            "json": display_path(json_path),
            "markdown": display_path(markdown_path),
            "figure": display_path(figure_path),
            "figure_name": figure_path.name,
            "checkpoint": display_path(checkpoint_path),
        },
        "final_memory": cuda_snapshot(device),
    }
    atomic_json_dump(report, json_path)
    atomic_torch_save(
        {
            "format": "legolm-mom-latent-bridge-v2",
            "pair": args.pair,
            "source_model": source_profile,
            "target_model": target_profile,
            "source_layer": pair["source_layer"],
            "target_layer": pair["target_layer"],
            "source_dim": bridge.source_dim,
            "bus_dim": bridge.bus_dim,
            "target_dim": bridge.target_dim,
            "fraction": fraction,
            "labels": list(COLORS),
            "bridge": {
                name: value.detach().cpu()
                for name, value in bridge.state_dict().items()
            },
            "auxiliary_packet_head": {
                name: value.detach().cpu()
                for name, value in auxiliary.state_dict().items()
            },
        },
        checkpoint_path,
    )
    write_figure(report, figure_path)
    write_markdown(report, markdown_path)
    print(
        json.dumps(
            {
                "event": "experiment_complete",
                "pair": args.pair,
                "promotion_passed": promotion["passed"],
                "failed_checks": promotion["failed"],
                "json": str(json_path),
                "markdown": str(markdown_path),
                "figure": str(figure_path),
                "checkpoint": str(checkpoint_path),
                **cuda_snapshot(device),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
