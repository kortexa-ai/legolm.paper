from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import pyarrow.parquet as pq
import requests
import rustbpe
import tiktoken
import torch

from .audio_snapshot import load_esc50
from .common import (
    DEFAULT_AUDIO_ENCODER,
    DEFAULT_IMU_ENCODER,
    DEFAULT_MINI_CHECKPOINT,
    DEFAULT_VISION_PERCEIVER,
    REPO_ROOT,
    assert_materialized_asset,
)
from .imu_snapshot import download_uci_har
from .runtime_data import (
    CACHE_DIR,
    DATA_DIR,
    TOKENIZER_DIR,
    VAL_FILENAME,
    configured_train_shard_ids,
    required_parquet_filenames,
)

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
VOCAB_SIZE = 8192
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SETUP_STATE_PATH = Path(CACHE_DIR) / "paper_setup.json"


def _required_repo_paths() -> list[Path]:
    return [
        DEFAULT_MINI_CHECKPOINT,
        DEFAULT_VISION_PERCEIVER,
        DEFAULT_AUDIO_ENCODER,
        DEFAULT_IMU_ENCODER,
        REPO_ROOT / "data" / "vision" / "dataset.json",
    ]


def _validate_repo_assets() -> None:
    missing = [str(path) for path in _required_repo_paths() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required repo assets:\n" + "\n".join(f"- {path}" for path in missing)
        )
    for path in _required_repo_paths():
        assert_materialized_asset(path)


def _download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    if destination.exists():
        return
    for attempt in range(1, 6):
        try:
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                with open(tmp_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            handle.write(chunk)
            os.replace(tmp_path, destination)
            return
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == 5:
                raise
            time.sleep(2**attempt)


def ensure_text_shards() -> list[Path]:
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    required = required_parquet_filenames()
    ready = []
    for filename in required:
        target = data_dir / filename
        if not target.exists():
            _download_file(f"{BASE_URL}/{filename}", target)
        ready.append(target)
    return ready


def _text_iterator(parquet_paths: list[Path], max_chars: int = 1_000_000_000, doc_cap: int = 10_000):
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for row_group_idx in range(pf.num_row_groups):
            row_group = pf.read_row_group(row_group_idx)
            for text in row_group.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def _token_bytes_tensor(enc) -> torch.Tensor:
    """True byte length per token, straight from the BPE byte vocabulary.

    Decoding a single token to str and re-encoding miscounts tokens that are not
    valid UTF-8 on their own (they decode to U+FFFD, 3 bytes); reading the raw
    token bytes avoids that. Specials count as 0 bytes.
    """
    special_ids = {enc.encode_single_token(name) for name in SPECIAL_TOKENS}
    token_bytes = []
    for token_id in range(enc.n_vocab):
        if token_id in special_ids:
            token_bytes.append(0)
        else:
            token_bytes.append(len(enc.decode_single_token_bytes(token_id)))
    return torch.tensor(token_bytes, dtype=torch.int32)


def _ensure_token_bytes_current(tokenizer_pkl: Path, token_bytes_path: Path) -> None:
    """Rebuild token_bytes.pt when a cached copy disagrees with the tokenizer."""
    with open(tokenizer_pkl, "rb") as handle:
        enc = pickle.load(handle)
    expected = _token_bytes_tensor(enc)
    if token_bytes_path.exists():
        existing = torch.load(token_bytes_path)
        if torch.equal(existing, expected):
            return
        print(f"Rebuilding stale {token_bytes_path}")
    torch.save(expected, token_bytes_path)


def ensure_tokenizer() -> None:
    tokenizer_dir = Path(TOKENIZER_DIR)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_pkl = tokenizer_dir / "tokenizer.pkl"
    token_bytes_path = tokenizer_dir / "token_bytes.pt"
    if tokenizer_pkl.exists():
        _ensure_token_bytes_current(tokenizer_pkl, token_bytes_path)
        return

    train_paths = [
        Path(DATA_DIR) / f"shard_{shard_id:05d}.parquet"
        for shard_id in configured_train_shard_ids()
    ]
    if not train_paths:
        raise RuntimeError("No train shards configured for tokenizer bootstrap")

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(_text_iterator(train_paths), vocab_size_no_special, pattern=SPLIT_PATTERN)

    mergeable_ranks = {bytes(key): value for key, value in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + index for index, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=tokenizer.get_pattern(),
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as handle:
        pickle.dump(enc, handle)

    torch.save(_token_bytes_tensor(enc), token_bytes_path)


def ensure_audio_cache() -> None:
    load_esc50()


def ensure_imu_cache() -> None:
    download_uci_har()


def write_setup_state() -> None:
    payload = {
        "train_shard_ids": configured_train_shard_ids(),
        "val_shard": int(VAL_FILENAME.removeprefix("shard_").removesuffix(".parquet")),
        "required_parquet_filenames": required_parquet_filenames(),
        "tokenizer_dir": TOKENIZER_DIR,
        "repo_root": str(REPO_ROOT),
    }
    SETUP_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SETUP_STATE_PATH, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap paper repo cache assets")
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip downloading/extracting ESC-50",
    )
    parser.add_argument(
        "--skip-imu",
        action="store_true",
        help="Skip downloading/extracting UCI HAR",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_repo_assets()
    ensure_text_shards()
    ensure_tokenizer()
    if not args.skip_audio:
        ensure_audio_cache()
    if not args.skip_imu:
        ensure_imu_cache()
    write_setup_state()
    print(f"Bootstrap complete. Cache rooted at {CACHE_DIR}")


if __name__ == "__main__":
    main()
