"""Materialized text rows shared by every interface within a host."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import pyarrow.parquet as pq

from .hosts import Tokens
from .provenance import file_hash, write_json, digest


def documents(cache, split):
    names = ["shard_06542.parquet"] if split == "val" else [f"shard_{i:05d}.parquet" for i in range(3)]
    while True:
        for name in names:
            pf = pq.ParquetFile(cache / "data" / name)
            for batch in pf.iter_batches(batch_size=128, columns=["text"]):
                yield batch.column("text").to_pylist()


def packed_rows(tokens: Tokens, cache: Path, split: str, count: int, length: int = 128):
    """Original best-fit packing with an explicit native document-start convention."""
    source = documents(cache, split)
    buffer = []
    rows = []
    while len(rows) < count:
        row = []
        while len(row) < length + 1:
            while len(buffer) < 1000:
                batch = [tokens.encode(t, start=True) for t in next(source)]
                buffer.extend(t for t in batch if t)
            remaining = length + 1 - len(row)
            best, best_len = -1, 0
            for i, doc in enumerate(buffer):
                if best_len < len(doc) <= remaining:
                    best, best_len = i, len(doc)
            if best >= 0:
                row.extend(buffer.pop(best))
            else:
                shortest = min(range(len(buffer)), key=lambda i: len(buffer[i]))
                row.extend(buffer.pop(shortest)[:remaining])
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long)


def prepare_text(host: str, cache: Path, output: Path, train_rows=24000, val_rows=1024):
    tokens = Tokens(host, cache)
    identity = {"host": host, "tokenizer": tokens.identity, "train_rows": train_rows, "val_rows": val_rows, "text_length": 128,
                "shards": {name: file_hash(cache / "data" / name) for name in ["shard_00000.parquet", "shard_00001.parquet", "shard_00002.parquet", "shard_06542.parquet"]}}
    path = output / f"text-{host}-{digest(identity)[:16]}.pt"
    output.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        result = {"identity": identity, "train": packed_rows(tokens, cache, "train", train_rows),
                  "val": packed_rows(tokens, cache, "val", val_rows)}
        torch.save(result, path)
    write_json(output / f"text-{host}.json", {**identity, "file": path.name, "sha256": file_hash(path)})
    return path


def label_rows(tokens, prompt, categories):
    """Natural prompt+label tokenization, including tokens crossing the boundary."""
    p = tokens.task_encode(prompt)
    rows, masks = [], []
    for category in categories:
        ids = tokens.task_encode(prompt + category)
        boundary = 0
        while boundary < min(len(p), len(ids)) and p[boundary] == ids[boundary]:
            boundary += 1
        if boundary < 1 or boundary >= len(ids):
            raise ValueError(f"Invalid scored label boundary: {prompt!r} + {category!r}")
        rows.append(ids)
        masks.append([i >= boundary for i in range(1, len(ids))])
    width = max(len(r) for r in rows) - 1
    x = torch.zeros((len(rows), width), dtype=torch.long)
    y = torch.full_like(x, -1)
    label_mask = torch.zeros_like(x, dtype=torch.bool)
    for i, (row, mask) in enumerate(zip(rows, masks)):
        x[i, :len(row)-1] = torch.tensor(row[:-1])
        y[i, :len(row)-1] = torch.tensor(row[1:])
        label_mask[i, :len(mask)] = torch.tensor(mask)
    return x, y, label_mask


def prepare_common_text(cache, output, hosts=("mini","lfm230m","qwen08"), count=512):
    tokenizers={name:Tokens(name,cache) for name in hosts}
    pf=pq.ParquetFile(cache/"data/shard_06542.parquet")
    examples=[]
    seen=set()
    for batch in pf.iter_batches(batch_size=128,columns=["text"]):
        for text in batch.column("text").to_pylist():
            document_id=digest(text)
            if document_id in seen:
                continue
            seen.add(document_id)
            if len(text.encode("utf-8"))<64:
                continue
            span=text[:128]
            while len(span)>32 and any(len(t.encode(span,start=True))>129 for t in tokenizers.values()):
                span=span[:-1]
            views={name:t.encode(span,start=True) for name,t in tokenizers.items()}
            boundaries={}
            for name,ids in views.items():
                widths=tokenizers[name].widths[ids]
                if int(widths.sum())!=len(span.encode("utf-8")):
                    raise ValueError("Common text must round-trip to identical bytes")
                boundaries[name]=set(widths.cumsum(0).tolist())
            shared=set.intersection(*boundaries.values())
            choices=sorted(b for b in shared if 16<=b<=48 and b<len(span.encode("utf-8"))-16)
            if not choices:
                continue
            choices=[b for b in choices if span.encode("utf-8")[:b].decode("utf-8",errors="ignore").encode("utf-8")==span.encode("utf-8")[:b]]
            if not choices:
                continue
            boundary=choices[0]
            example={"id":f"climbmix-common-{len(examples):04d}","document_id":document_id,"text":span,"context_bytes":boundary,
                     "target_hex":span.encode("utf-8")[boundary:].hex(),"views":{}}
            for name,ids in views.items():
                widths=tokenizers[name].widths[ids]
                # Predict target token i from prefix up to token i-1.
                ends=widths.cumsum(0)
                mask=[bool(ends[i]>boundary) for i in range(1,len(ids))]
                assert int(widths[1:][mask].sum())==len(bytes.fromhex(example["target_hex"]))
                example["views"][name]={"x":ids[:-1],"y":ids[1:],"mask":mask}
            examples.append(example)
            if len(examples)==count:
                manifest={"corpus_sha256":file_hash(cache/"data/shard_06542.parquet"),"tokenizers":{n:t.identity for n,t in tokenizers.items()},"examples":examples}
                write_json(output/"common-text.json",manifest)
                return manifest
    raise ValueError("Not enough shared raw text spans")
