from __future__ import annotations

from .runtime_lora import LoRALinear


def apply_hypernet_weights(model, weight_vector, rank, target, alpha=None):
    """
    Apply a flat hypernetwork weight vector to LoRA modules while preserving autograd.
    """
    alpha = alpha or rank
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    offset = 0
    for module in raw.modules():
        if isinstance(module, LoRALinear):
            a_size = module.lora_a.numel()
            b_size = module.lora_b.numel()
            module._hyper_a = weight_vector[offset : offset + a_size].view_as(module.lora_a)
            offset += a_size
            module._hyper_b = weight_vector[offset : offset + b_size].view_as(module.lora_b)
            offset += b_size
    assert offset == weight_vector.numel(), f"Weight vector size mismatch: {offset} vs {weight_vector.numel()}"
