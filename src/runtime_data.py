from __future__ import annotations

import math
import os
import pickle

import pyarrow.parquet as pq
import torch

MAX_SEQ_LEN = int(os.environ.get("AUTORESEARCH_SEQ_LEN", 128))
EVAL_TOKENS = int(os.environ.get("AUTORESEARCH_EVAL_TOKENS", str(2 * 524288)))
DEFAULT_NUM_TRAIN_SHARDS = int(os.environ.get("AUTORESEARCH_NUM_TRAIN_SHARDS", 3))

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
# Re-pointed at load time when an hf:<model-id> base model is configured.
ACTIVE_TOKENIZER_DIR = TOKENIZER_DIR
MAX_SHARD = 6542
VAL_SHARD = MAX_SHARD
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"


def configured_train_shard_ids() -> list[int]:
    raw = os.environ.get("AUTORESEARCH_TRAIN_SHARDS")
    if raw:
        ids = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            shard_id = int(part)
            if shard_id < 0 or shard_id >= VAL_SHARD:
                raise ValueError(f"Invalid train shard id: {shard_id}")
            ids.append(shard_id)
        if not ids:
            raise ValueError("AUTORESEARCH_TRAIN_SHARDS was set but no shard ids were parsed")
        return sorted(set(ids))

    if DEFAULT_NUM_TRAIN_SHARDS < 1 or DEFAULT_NUM_TRAIN_SHARDS >= VAL_SHARD:
        raise ValueError(f"Invalid AUTORESEARCH_NUM_TRAIN_SHARDS: {DEFAULT_NUM_TRAIN_SHARDS}")
    return list(range(DEFAULT_NUM_TRAIN_SHARDS))


def required_parquet_filenames() -> list[str]:
    train_files = [f"shard_{shard_id:05d}.parquet" for shard_id in configured_train_shard_ids()]
    return train_files + [VAL_FILENAME]


class Tokenizer:
    """Minimal tokenizer wrapper around the prepared BPE assets."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token("<|reserved_0|>")

    @classmethod
    def from_directory(cls, tokenizer_dir=None):
        tokenizer_dir = tokenizer_dir or ACTIVE_TOKENIZER_DIR
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)

    def get_bos_token_id(self):
        return self.bos_token_id


def get_token_bytes(device="cpu"):
    path = os.path.join(ACTIVE_TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def list_parquet_files():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    required = required_parquet_filenames()
    missing = [name for name in required if name not in files]
    if missing:
        raise FileNotFoundError(
            "Missing required parquet shards in "
            f"{DATA_DIR}: {', '.join(missing)}. Run ./setup.sh or adjust "
            "AUTORESEARCH_TRAIN_SHARDS/AUTORESEARCH_NUM_TRAIN_SHARDS."
        )
    return [os.path.join(DATA_DIR, f) for f in files if f in set(required)]


def _document_batches(split, tokenizer_batch_size=128):
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run ./setup.sh first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
        assert len(parquet_paths) > 0, "No training shards found."
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column("text").to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i : i + tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, batch_size, seq_len, split, buffer_size=1000):
    """
    BOS-aligned best-fit packing dataloader copied into the paper runner.
    """
    assert split in ["train", "val"]
    row_capacity = seq_len + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    row_buffer = torch.empty((batch_size, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * batch_size * seq_len, dtype=torch.long, pin_memory=True)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    gpu_buffer = torch.empty(2 * batch_size * seq_len, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[: batch_size * seq_len].view(batch_size, seq_len)
    cpu_targets = cpu_buffer[batch_size * seq_len :].view(batch_size, seq_len)
    inputs = gpu_buffer[: batch_size * seq_len].view(batch_size, seq_len)
    targets = gpu_buffer[batch_size * seq_len :].view(batch_size, seq_len)

    while True:
        for row_idx in range(batch_size):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos
                best_idx = -1
                best_len = 0
                for idx, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = idx
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos : pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda idx: len(doc_buffer[idx]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos : pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch
