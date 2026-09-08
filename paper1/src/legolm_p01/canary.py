"""Bounded numerical acceptance checks before scientific trajectories."""
from __future__ import annotations

import gc
import time
from contextlib import nullcontext

import torch

from .hosts import load_host, Bridge, Prefix
from .provenance import write_json, source_identity
from .text import label_rows


def precision(device):
    # BF16 LFM canary batch-versus-scalar losses differed by 0.365 nats.
    # The corrected study uses FP32; no tolerance is widened to accept BF16.
    return nullcontext()


def memory(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return {"free_gib": free/2**30, "total_gib": total/2**30,
                "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30}
    return {}


def run(name, asset_root, cache, output, device):
    torch.manual_seed(42)
    host = load_host(name, asset_root, cache, device)
    x,y,mask = label_rows(host.tokens, "The activity is: ", ["walking", "standing", "walking_upstairs"])
    x,y,mask = x.to(device),y.to(device),mask.to(device)
    with torch.no_grad(), precision(device):
        host.inject(enabled=False)
        native = host.logits(x)
        host.inject(torch.zeros(host.dimension, device=device))
        torch.testing.assert_close(host.logits(x), native, atol=1e-5, rtol=1e-5)
        empty = torch.zeros((len(x), 0, host.hidden), device=device)
        torch.testing.assert_close(host.logits(x, empty), native, atol=2e-5, rtol=2e-5)
    bridge = Bridge(256, host.dimension).to(device)
    features = torch.randn(3, 256, device=device)
    with precision(device):
        host.inject(bridge(features[:1]))
        first_loss = host.losses(x[:1], y[:1])[y[:1] >= 0].mean()
    first_loss.backward()
    batch1_memory = memory(device)
    if batch1_memory and batch1_memory["free_gib"] < 10:
        raise MemoryError("Batch-1 first backward leaves less than 10 GiB free")
    bridge.zero_grad(set_to_none=True)
    with precision(device):
        generated = bridge(features)
        host.inject(generated)
        batched = host.losses(x,y)
        individual = []
        for i in range(3):
            host.inject(generated[i])
            individual.append(host.losses(x[i:i+1],y[i:i+1]))
        # This FP32 tolerance is fixed before scientific outcomes are inspected.
        torch.testing.assert_close(batched, torch.cat(individual), atol=2e-4, rtol=2e-4)
        loss = (batched * mask).sum() / mask.sum()
    loss.backward()
    assert bridge.mlp[-1].weight.grad is not None and bridge.mlp[-1].weight.grad.norm() > 0
    assert all(p.grad is None for p in host.model.parameters())
    first_backward = memory(device)
    if first_backward and first_backward["free_gib"] < 10:
        raise MemoryError("First backward leaves less than 10 GiB free")
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=.001, weight_decay=.01)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    host.inject(enabled=False)
    prefix = Prefix(256,host.hidden).to(device)
    with precision(device):
        p = prefix(features[:1])
        losses = host.losses(x[:1],y[:1],p)
        assert losses.shape == y[:1].shape
        loss = losses[y[:1] >= 0].mean()
    loss.backward()
    assert prefix.proj[0].weight.grad.norm() > 0
    assert all(p.grad is None for p in host.model.parameters())
    result = {"host": host.identity, "checks": ["zero-LoRA", "zero-length-prefix", "candidate-batch-parity", "bridge-gradient", "prefix-gradient", "frozen-host", "matching-prefix-targets"],
              "batch1_first_backward": batch1_memory, "batch3_backward": first_backward, "final_memory": memory(device), "source": source_identity(),
              "bridge_parameters": sum(p.numel() for p in bridge.parameters()), "prefix_parameters": sum(p.numel() for p in prefix.parameters())}
    write_json(output, result)
    return result
