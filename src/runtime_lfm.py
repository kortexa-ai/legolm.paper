from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import runtime_data
from .runtime_lora import LoRALinear

DEFAULT_LFM_MODEL = "LiquidAI/LFM2.5-230M"

# Adapter for running the paper suite against an HF LFM2-family base model
# (checkpoint spec "hf:<model-id>"). Contract mirrored from the mini runner:
# forward(idx, targets, reduction) returns cross-entropy with ignore_index=-1,
# LoRA targets follow the paper's semantics mapped onto LFM2's hybrid stack
# (target "attn" = q/k/v/out projections on the attention layers, "ffn" =
# w1/w2/w3 on every layer's SwiGLU MLP, "all" = both). Short-conv projections
# are left unadapted: the mini model has no analog and the paper's "all" means
# attention + MLP linears. No logit softcap: that is an architectural detail
# of the mini checkpoint, not of LFM.


def _model_slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def tokenizer_dir_for(model_id: str) -> str:
    return os.path.join(runtime_data.CACHE_DIR, f"tokenizer-{_model_slug(model_id)}")


def _bytelevel_char_widths() -> dict[str, int]:
    """Inverse of the GPT-2 ByteLevel byte-to-unicode table: every alphabet
    char stands for exactly one raw byte."""
    visible = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = [chr(b) for b in visible]
    shift = 0
    for b in range(256):
        if b not in visible:
            chars.append(chr(256 + shift))
            shift += 1
    return {ch: 1 for ch in chars}


class HFTokenizerAdapter:
    """tiktoken-surface adapter around an HF fast tokenizer.

    encode_ordinary mirrors tiktoken semantics for true specials: special-token
    strings inside raw text are tokenized as plain text (split_special_tokens=
    True), never as control ids. NON-special added tokens ('python', '<think>',
    ...) are still matched — that is faithful to how the model reads ordinary
    text, and token_bytes assigns them their real UTF-8 length. The paper
    runner requests its BOS by the literal name "<|reserved_0|>"; that role
    maps onto this tokenizer's own BOS token.
    """

    def __init__(self, hf_tokenizer, vocab_size: int):
        self.hf = hf_tokenizer
        self.n_vocab = vocab_size

    def encode_ordinary(self, text: str) -> list[int]:
        return self.hf(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]

    def encode_ordinary_batch(self, texts: list[str], num_threads: int = 8) -> list[list[int]]:
        return self.hf(list(texts), add_special_tokens=False, split_special_tokens=True)["input_ids"]

    def encode_single_token(self, name: str) -> int:
        if name == "<|reserved_0|>":
            name = self.hf.bos_token
        token_id = self.hf.convert_tokens_to_ids(name)
        if token_id is None:
            raise KeyError(name)
        return token_id

    def decode(self, ids) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self.hf.decode(ids, skip_special_tokens=False)

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        widths = _bytelevel_char_widths()
        added_token = self.hf.added_tokens_decoder.get(token_id)
        if added_token is not None:
            return b"" if added_token.special else added_token.content.encode("utf-8")
        token = self.hf.convert_ids_to_tokens(token_id)
        if token is None:
            return b""
        assert all(ch in widths for ch in token), f"Non-byte-level token {token_id}: {token!r}"
        return self.hf.decode([token_id]).encode("utf-8")


def _token_bytes_for_hf(hf_tokenizer, vocab_size: int) -> torch.Tensor:
    """Raw byte length per token id, sized to the model vocab. True special
    tokens and padded ids count as 0 bytes (mirroring the mini tokenizer);
    NON-special added tokens (LFM has e.g. 'python', '<think>') are matched
    in ordinary text by the fast tokenizer, so they carry their real UTF-8
    length or the BPB denominator silently drops those positions."""
    widths = _bytelevel_char_widths()
    added = hf_tokenizer.added_tokens_decoder
    token_bytes = []
    for token_id in range(vocab_size):
        added_token = added.get(token_id)
        if added_token is not None:
            token_bytes.append(0 if added_token.special else len(added_token.content.encode("utf-8")))
            continue
        token = hf_tokenizer.convert_ids_to_tokens(token_id)
        if token is None:
            token_bytes.append(0)
            continue
        assert all(ch in widths for ch in token), f"Non-byte-level token {token_id}: {token!r}"
        token_bytes.append(len(token))
    return torch.tensor(token_bytes, dtype=torch.int32)


def _validate_token_bytes(adapter: HFTokenizerAdapter, token_bytes: torch.Tensor) -> None:
    """The BPB denominator is only valid if token byte lengths reconstruct the
    exact UTF-8 length of any encoded text."""
    samples = [
        "Hello world",
        " leading space and trailing ",
        "Много букви на кирилица, за проверка.",
        "emoji 🦝 and\nnewlines\n\ttabs",
        'literal specials <|im_start|> stay plain text',
        # Non-special added tokens are matched in ordinary text and must carry
        # their real byte length (python is literally in LFM's added vocab).
        "I love python code, even micropython, and <think> tags with <|tool_call_start|> markers.",
    ]
    for text in samples:
        ids = adapter.encode_ordinary(text)
        total = sum(int(token_bytes[i]) for i in ids)
        expected = len(text.encode("utf-8"))
        if total != expected:
            raise RuntimeError(
                f"token_bytes mismatch for {text!r}: {total} vs utf-8 {expected}"
            )


def ensure_hf_tokenizer(model_id: str = DEFAULT_LFM_MODEL) -> str:
    tokenizer_dir = tokenizer_dir_for(model_id)
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_pkl = os.path.join(tokenizer_dir, "tokenizer.pkl")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        return tokenizer_dir

    from transformers import AutoConfig, AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
    vocab_size = AutoConfig.from_pretrained(model_id).vocab_size
    adapter = HFTokenizerAdapter(hf_tokenizer, vocab_size)
    token_bytes = _token_bytes_for_hf(hf_tokenizer, vocab_size)
    _validate_token_bytes(adapter, token_bytes)

    with open(tokenizer_pkl, "wb") as handle:
        pickle.dump(adapter, handle)
    torch.save(token_bytes, token_bytes_path)
    return tokenizer_dir


def configure_tokenizer(model_id: str = DEFAULT_LFM_MODEL) -> None:
    """Route runtime_data's tokenizer/token_bytes lookups to this model's assets."""
    runtime_data.ACTIVE_TOKENIZER_DIR = ensure_hf_tokenizer(model_id)


@dataclass
class LFMConfigShim:
    """Duck-types the QwenConfig fields the experiment code reads."""

    model_id: str
    n_embd: int
    vocab_size: int
    sequence_len: int
    n_layer: int


class LFMModel(nn.Module):
    """HF Lfm2ForCausalLM behind the paper runner's model contract."""

    def __init__(self, hf_model):
        super().__init__()
        self.hf = hf_model

    def forward(self, idx, targets=None, reduction="mean"):
        logits = self.hf(input_ids=idx, use_cache=False).logits.float()
        if targets is None:
            return logits
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=reduction,
        )

    def forward_with_prefix(self, prefix_tokens: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        text_embeds = self.hf.get_input_embeddings()(text_ids)
        combined = torch.cat([prefix_tokens.to(text_embeds.dtype), text_embeds], dim=1)
        return self.hf(inputs_embeds=combined, use_cache=False).logits.float()

    def apply_lora_targets(self, rank: int, target: str, alpha=None) -> list[LoRALinear]:
        lora_modules = []
        for layer in self.hf.model.layers:
            if target in ("attn", "all") and hasattr(layer, "self_attn"):
                attn = layer.self_attn
                for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    setattr(attn, name, LoRALinear(getattr(attn, name), rank, alpha))
                    lora_modules.append(getattr(attn, name))
            if target in ("ffn", "all"):
                feed_forward = layer.feed_forward
                for name in ("w1", "w2", "w3"):
                    setattr(feed_forward, name, LoRALinear(getattr(feed_forward, name), rank, alpha))
                    lora_modules.append(getattr(feed_forward, name))
        return lora_modules


def load_hf_lm(model_id: str = DEFAULT_LFM_MODEL):
    from transformers import AutoModelForCausalLM

    configure_tokenizer(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    hf_model.eval()
    config = hf_model.config
    shim = LFMConfigShim(
        model_id=model_id,
        n_embd=config.hidden_size,
        vocab_size=config.vocab_size,
        sequence_len=config.max_position_embeddings,
        n_layer=config.num_hidden_layers,
    )
    return LFMModel(hf_model), shim, None
