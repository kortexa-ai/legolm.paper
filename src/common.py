from __future__ import annotations

import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
from .runtime_data import EVAL_TOKENS, MAX_SEQ_LEN, Tokenizer, get_token_bytes, make_dataloader
from .runtime_hyper import apply_hypernet_weights
from .runtime_lm import QwenConfig, QwenModel
from .runtime_lora import LoRALinear, apply_lora, get_lora_params

DEFAULT_MINI_CHECKPOINT = REPO_ROOT / "checkpoints" / "experiments" / "mini-base.pt"
DEFAULT_VISION_PERCEIVER = REPO_ROOT / "checkpoints" / "experiments" / "vision-perceiver.pt"
DEFAULT_AUDIO_ENCODER = REPO_ROOT / "checkpoints" / "experiments" / "esc50-512d.pt"
DEFAULT_IMU_ENCODER = REPO_ROOT / "checkpoints" / "encoders" / "imu.pt"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def assert_materialized_asset(path: str | Path) -> Path:
    """Fail early when a required asset is missing or still a Git LFS pointer."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing required asset: {path}")
    with open(path, "rb") as handle:
        prefix = handle.read(len(LFS_POINTER_PREFIX))
    if prefix == LFS_POINTER_PREFIX:
        raise RuntimeError(
            f"Required asset is still a Git LFS pointer: {path}\n"
            "Run `git lfs pull` from the repository root, then retry."
        )
    return path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> tuple[torch.device, str]:
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"


def autocast_for(device_type: str):
    if device_type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def load_lm(checkpoint_path: str | Path = DEFAULT_MINI_CHECKPOINT):
    if isinstance(checkpoint_path, str) and checkpoint_path.startswith("hf:"):
        from .runtime_lfm import load_hf_lm

        return load_hf_lm(checkpoint_path[len("hf:"):])
    checkpoint_path = assert_materialized_asset(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = QwenConfig(**ckpt["config"])
    model = QwenModel(config)
    model.load_state_dict(ckpt["model"], strict=False)
    return model, config, ckpt


def freeze_non_lora(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if "lora_a" not in name and "lora_b" not in name:
            param.requires_grad_(False)


def total_lora_dim(model: torch.nn.Module) -> int:
    return sum(
        module.lora_a.numel() + module.lora_b.numel()
        for module in model.modules()
        if isinstance(module, LoRALinear)
    )


def evaluate_bpb(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    batch_size: int,
    eval_tokens: int | None = None,
) -> float:
    device = next(model.parameters()).device
    token_bytes = get_token_bytes(device=device)
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = max(1, (eval_tokens or EVAL_TOKENS) // (batch_size * MAX_SEQ_LEN))
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction="none").view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)


def sensor_limit_for(modality: str, requested: int | None) -> int:
    if requested is not None:
        return requested
    return 50 if modality == "vision" else 200


def cycle_batch(data: torch.Tensor, batch_size: int, offset: int) -> tuple[torch.Tensor, int]:
    if len(data) == 0:
        raise ValueError("Cannot cycle an empty tensor")
    if batch_size <= len(data):
        end = offset + batch_size
        if end <= len(data):
            return data[offset:end], end % len(data)
    indices = [(offset + i) % len(data) for i in range(batch_size)]
    return data[indices], (offset + batch_size) % len(data)
