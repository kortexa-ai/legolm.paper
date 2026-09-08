"""Frozen hosts, explicit adaptation targets, and per-run tokenizers."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .legacy.runtime_lm import QwenConfig, QwenModel
from .provenance import file_hash, digest

HOSTS = {
    "mini": {"model_id": "mini-base", "revision": None},
    "lfm230m": {"model_id": "LiquidAI/LFM2.5-230M", "revision": "40cb2ad3b3044d5a41eee083a6103c8b523afa45"},
    "qwen08": {"model_id": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17"},
}


class Tokens:
    def __init__(self, host: str, cache: Path):
        self.host = host
        self.hf = None
        if host == "mini":
            with (cache / "tokenizer" / "tokenizer.pkl").open("rb") as f:
                self.enc = pickle.load(f)
            self.bos = self.enc.encode_single_token("<|reserved_0|>")
            self.eos = self.bos
            self.widths = torch.load(cache / "tokenizer" / "token_bytes.pt", map_location="cpu", weights_only=True).long()
            self.identity = {"host": host, "tokenizer_sha256": file_hash(cache / "tokenizer" / "tokenizer.pkl"),
                             "bytes_sha256": file_hash(cache / "tokenizer" / "token_bytes.pt")}
        else:
            from transformers import AutoTokenizer, AutoConfig
            spec = HOSTS[host]
            self.hf = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec["revision"])
            config = AutoConfig.from_pretrained(spec["model_id"], revision=spec["revision"])
            config = getattr(config, "text_config", config)
            self.bos = config.bos_token_id
            self.eos = config.eos_token_id
            if host == "qwen08" and self.bos is not None:
                raise ValueError("The registered Qwen checkpoint has no BOS")
            added = self.hf.added_tokens_decoder
            values = []
            for i in range(config.vocab_size):
                if i in added:
                    values.append(0 if added[i].special else len(added[i].content.encode("utf-8")))
                else:
                    token = self.hf.convert_ids_to_tokens(i)
                    # These two registered tokenizers use the byte-level alphabet.
                    values.append(0 if token is None else len(token))
            self.widths = torch.tensor(values, dtype=torch.long)
            self.identity = {**spec, "tokenizer_sha256": digest(json.loads(self.hf.backend_tokenizer.to_str())),
                             "bos": self.bos, "eos": self.eos}
        for sample in ["Hello world", " leading space ", "Много букви", "emoji 🦝\n", "python <think> <|im_start|>"]:
            ids = self.encode(sample)
            if self.decode(ids) != sample or self.widths[ids].sum().item() != len(sample.encode("utf-8")):
                raise ValueError(f"Tokenizer byte round-trip failed: {sample!r}")

    def encode(self, text: str, *, start: bool = False) -> list[int]:
        ids = self.enc.encode_ordinary(text) if self.hf is None else self.hf(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]
        if start and self.bos is not None:
            ids = [self.bos] + ids
        return ids

    def decode(self, ids) -> str:
        ids = list(ids)
        return self.enc.decode(ids) if self.hf is None else self.hf.decode(ids, skip_special_tokens=False)

    def task_encode(self, text: str) -> list[int]:
        return self.encode(text, start=self.host != "mini")


class LoRA(nn.Module):
    """The original A(in,r) B(r,out) convention, also accepting sample batches."""
    def __init__(self, base: nn.Linear, rank: int = 4):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = rank
        self.a = nn.Parameter(torch.randn(base.in_features, rank) * 0.01)
        self.b = nn.Parameter(torch.zeros(rank, base.out_features))
        self.generated = None
        self.parts = None
        self.enabled = True
        self.batch_repeat = 1
        self.base.requires_grad_(False)

    def forward(self, x):
        result = self.base(x)
        if not self.enabled:
            return result
        pairs = self.parts if self.parts is not None else [self.generated or (self.a, self.b)]
        for a, b in pairs:
            if a.ndim == 3:
                if self.batch_repeat > 1:
                    a, b = a.repeat_interleave(self.batch_repeat, dim=0), b.repeat_interleave(self.batch_repeat, dim=0)
                if a.shape[0] == 1 and x.shape[0] != 1:
                    a, b = a.expand(x.shape[0], -1, -1), b.expand(x.shape[0], -1, -1)
                value = torch.bmm(torch.bmm(x, a), b)
            else:
                value = x @ a @ b
            result = result + value * (self.alpha / self.rank)
        return result


class Bridge(nn.Module):
    def __init__(self, feature_dim: int, output_dim: int):
        super().__init__()
        self.context = nn.Parameter(torch.randn(64) * 0.02)
        self.proj = nn.Sequential(nn.Linear(feature_dim, 64), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, output_dim))
        nn.init.normal_(self.mlp[-1].weight, std=0.001)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features):
        if features.ndim == 1:
            features = features[None]
        return self.mlp(torch.cat((self.proj(features), self.context[None].expand(len(features), -1)), dim=1))


class Prefix(nn.Module):
    def __init__(self, feature_dim: int, hidden: int, count: int = 8):
        super().__init__()
        self.count, self.hidden = count, hidden
        self.proj = nn.Sequential(nn.Linear(feature_dim, hidden * count), nn.GELU())

    def forward(self, features):
        return self.proj(features).view(len(features), self.count, self.hidden)


@dataclass
class Host:
    name: str
    model: nn.Module
    hidden: int
    tokens: Tokens
    identity: dict
    targets: list

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dimension(self):
        return sum(m.a.numel() + m.b.numel() for _, m in self.targets)

    def target_manifest(self):
        offset, rows = 0, []
        for name, m in self.targets:
            count = m.a.numel() + m.b.numel()
            rows.append({"module": name, "a_shape": list(m.a.shape), "b_shape": list(m.b.shape),
                         "offset": offset, "count": count, "rank": m.rank, "alpha": m.alpha})
            offset += count
        return rows

    def inject(self, vector=None, *, enabled=True, components=None, mean=False, repeat=1):
        if vector is not None and vector.shape[-1] != self.dimension:
            raise ValueError("Generated vector does not match the explicit host target map")
        offset = 0
        for _, m in self.targets:
            m.enabled, m.generated, m.parts = enabled, None, None
            m.batch_repeat = repeat
            na, nb = m.a.numel(), m.b.numel()
            def split(v):
                leading = tuple(v.shape[:-1])
                a = v[..., offset:offset+na].reshape(*leading, *m.a.shape)
                b = v[..., offset+na:offset+na+nb].reshape(*leading, *m.b.shape)
                return a, b
            if vector is not None:
                m.generated = split(vector)
            if components is not None:
                m.parts = []
                for component in components:
                    a, b = split(component)
                    m.parts.append((a, b / len(components) if mean else b))
            offset += na + nb

    def logits(self, ids, prefix=None):
        if self.name == "mini":
            if prefix is None:
                return self.model(ids)
            n = ids.shape[1] + prefix.shape[1]
            x = torch.cat((prefix, self.model.transformer.wte(ids)), dim=1)
            for block in self.model.transformer.h:
                x = block(x, (self.model.cos[:, :n], self.model.sin[:, :n]))
            logits = self.model.lm_head(self.model.transformer.ln_f(x)).float()
            return 30 * torch.tanh(logits / 30)
        kwargs = {"use_cache": False}
        if prefix is None:
            kwargs["input_ids"] = ids
        else:
            kwargs["inputs_embeds"] = torch.cat((prefix.to(self.model.get_input_embeddings().weight.dtype), self.model.get_input_embeddings()(ids)), dim=1)
        return self.model(**kwargs).logits.float()

    def losses(self, x, y, prefix=None):
        logits = self.logits(x, prefix)
        if prefix is not None:
            logits = logits[:, prefix.shape[1]:]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none", ignore_index=-1).view_as(y)


def load_host(name: str, asset_root: Path, cache: Path, device: str, dtype: str = "float32") -> Host:
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    tokens = Tokens(name, cache)
    spec = HOSTS[name]
    if name == "mini":
        path = asset_root / "checkpoints/experiments/mini-base.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = QwenConfig(**checkpoint["config"])
        model = QwenModel(cfg)
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        if missing or unexpected:
            raise ValueError({"missing": missing, "unexpected": unexpected})
        hidden = cfg.n_embd
        identity = {**spec, "checkpoint_sha256": file_hash(path), "config": checkpoint["config"]}
        selections = [(f"transformer.h.{i}.{part}.{key}", getattr(getattr(layer, part), key))
                      for i, layer in enumerate(model.transformer.h)
                      for part, keys in [("attn", ["c_q", "c_k", "c_v", "c_proj"]), ("mlp", ["gate_proj", "up_proj", "down_proj"])] for key in keys]
    else:
        from transformers import AutoModelForCausalLM, Qwen3_5ForCausalLM
        loader = Qwen3_5ForCausalLM if name == "qwen08" else AutoModelForCausalLM
        model, info = loader.from_pretrained(spec["model_id"], revision=spec["revision"], dtype=getattr(torch, dtype), output_loading_info=True)
        if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
            raise ValueError({k: info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")})
        config = getattr(model.config, "text_config", model.config)
        hidden = config.hidden_size
        identity = {**spec, "config": config.to_dict(), "unused_checkpoint_keys": sorted(info.get("unexpected_keys", []))}
        selections = []
        for i, layer in enumerate(model.model.layers):
            if getattr(layer, "self_attn", None) is not None:
                for key in ["q_proj", "k_proj", "v_proj", "o_proj" if name == "qwen08" else "out_proj"]:
                    selections.append((f"model.layers.{i}.self_attn.{key}", getattr(layer.self_attn, key)))
            mlp_name = "mlp" if name == "qwen08" else "feed_forward"
            mlp = getattr(layer, mlp_name)
            for key in (["gate_proj", "up_proj", "down_proj"] if name == "qwen08" else ["w1", "w2", "w3"]):
                selections.append((f"model.layers.{i}.{mlp_name}.{key}", getattr(mlp, key)))
    model.requires_grad_(False)
    identity["frozen_parameters"] = sum(p.numel() for p in model.parameters())
    targets = []
    for path, module in selections:
        parent_path, key = path.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
        replacement = LoRA(module)
        setattr(parent, key, replacement)
        targets.append((path, replacement))
    model.to(device).eval()
    host = Host(name, model, hidden, tokens, identity, targets)
    if name == "mini" and host.dimension != 327680:
        raise ValueError(f"Mini target dimension changed: {host.dimension}")
    if name == "lfm230m" and (host.dimension != 774144 or len(targets) != 66):
        raise ValueError(f"LFM target map changed: {host.dimension}, {len(targets)}")
    if name == "qwen08" and (len(targets) != 96 or any("linear_attn" in p or "vision" in p for p, _ in targets)):
        raise ValueError("Qwen must target 24 full-attention projections and 72 MLP projections")
    identity.update({"dtype": dtype, "autocast": False, "tf32": False, "tokenizer": tokens.identity, "targets": host.target_manifest(), "lora_dim": host.dimension})
    return host
