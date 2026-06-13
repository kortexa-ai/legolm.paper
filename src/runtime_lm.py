from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QwenConfig:
    sequence_len: int = 1024
    vocab_size: int = 32768
    n_layer: int = 24
    n_head: int = 8
    n_kv_head: int = 2
    n_embd: int = 1024
    intermediate_size: int = 3584
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    attn_output_gate: bool = True
    tie_word_embeddings: bool = True
    window_pattern: str = "SSSL"


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        assert self.n_head % self.n_kv_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_head * self.head_dim, self.n_embd, bias=False)

        if config.attn_output_gate:
            self.output_gate = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        else:
            self.output_gate = None

    def forward(self, x, cos_sin):
        bsz, seqlen, _ = x.size()
        q = self.c_q(x).view(bsz, seqlen, self.n_head, self.head_dim)
        k = self.c_k(x).view(bsz, seqlen, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(bsz, seqlen, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if self.n_kv_head < self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        if self.output_gate is not None:
            gate = torch.sigmoid(self.output_gate(x))
            y = y * gate

        return self.c_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.ln_2 = nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos_sin):
        x = x + self.attn(self.ln_1(x), cos_sin)
        x = x + self.mlp(self.ln_2(x))
        return x


class QwenModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
                "ln_f": nn.RMSNorm(config.n_embd, eps=config.rms_norm_eps),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.wte.weight

        self.rotary_seq_len = config.sequence_len * 2
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, config.head_dim, config.rope_theta
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, theta, device=None):
        if device is None:
            device = "cpu"
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (theta ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        return cos[None, :, None, :], sin[None, :, None, :]

    def forward(self, idx, targets=None, reduction="mean"):
        _, seqlen = idx.size()
        assert seqlen <= self.cos.size(1)
        cos_sin = self.cos[:, :seqlen], self.sin[:, :seqlen]

        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x, cos_sin)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x).float()
        softcap = 30
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=reduction,
            )
        return logits
