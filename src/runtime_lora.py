from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low-rank adaptor wrapping a frozen linear layer."""

    def __init__(self, linear, rank, alpha=None):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha or rank
        in_features = linear.in_features
        out_features = linear.out_features

        self.lora_a = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(rank, out_features))

        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x):
        base = self.linear(x)
        a = self._hyper_a if hasattr(self, "_hyper_a") else self.lora_a
        b = self._hyper_b if hasattr(self, "_hyper_b") else self.lora_b
        lora = (x @ a @ b) * (self.alpha / self.rank)
        return base + lora


def apply_lora(model, rank, target="attn", alpha=None):
    """
    Replace selected linear layers with LoRA-wrapped versions.
    """
    lora_modules = []
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    if hasattr(raw, "apply_lora_targets"):
        return raw.apply_lora_targets(rank, target, alpha)

    for block in raw.transformer.h:
        if target in ("attn", "all"):
            block.attn.c_q = LoRALinear(block.attn.c_q, rank, alpha)
            block.attn.c_k = LoRALinear(block.attn.c_k, rank, alpha)
            block.attn.c_v = LoRALinear(block.attn.c_v, rank, alpha)
            block.attn.c_proj = LoRALinear(block.attn.c_proj, rank, alpha)
            lora_modules.extend([block.attn.c_q, block.attn.c_k, block.attn.c_v, block.attn.c_proj])
        if target in ("ffn", "all"):
            block.mlp.gate_proj = LoRALinear(block.mlp.gate_proj, rank, alpha)
            block.mlp.up_proj = LoRALinear(block.mlp.up_proj, rank, alpha)
            block.mlp.down_proj = LoRALinear(block.mlp.down_proj, rank, alpha)
            lora_modules.extend([block.mlp.gate_proj, block.mlp.up_proj, block.mlp.down_proj])

    return lora_modules


def get_lora_params(model):
    """Return LoRA shell parameters only."""
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    params = []
    for module in raw.modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora_a, module.lora_b])
    return params
