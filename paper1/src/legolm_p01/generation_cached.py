"""Identical generation batches may be reused only with fixed effective inputs.

Keep the registered training/template implementation untouched. This evaluator
retains its ordering, batch composition, and every sensor record. Native-chat
decoding explicitly stops at the tokenizer's chat-end token as well as model EOS.
Only an entire previously decoded prompt batch can be reused, and only when
the host's effective weights do not depend on the supplied sensor values.
"""
from __future__ import annotations

from copy import deepcopy

import torch

from . import generation as g


def chat_stop_ids(host):
    """The pinned Qwen model EOS and native assistant-end token differ."""
    token = host.tokens.hf.eos_token_id
    if host.name != "qwen08" or not isinstance(token, int):
        raise ValueError("Native-chat evaluation requires the pinned Qwen chat-end token")
    configured = host.model.generation_config.eos_token_id
    stops = [] if configured is None else (list(configured) if isinstance(configured, (list, tuple)) else [configured])
    return list(dict.fromkeys([*stops, token]))


def decode(host, adaptation, kind, features, prompts, *, native_chat, intervention):
    if not native_chat:
        return g.decode(host, adaptation, kind, features, prompts, native_chat=False, intervention=intervention)
    original = host.model.generation_config
    config = deepcopy(original)
    config.eos_token_id = chat_stop_ids(host)
    config._from_model_config = False
    # Keep the registered training module intact. Only this evaluation call
    # supplies the corrected stopping policy; restore even after decoder errors.
    host.model.generation_config = config
    try:
        return g.decode(host, adaptation, kind, features, prompts, native_chat=True, intervention=intervention)
    finally:
        host.model.generation_config = original


@torch.no_grad()
def evaluate_generation(host, adaptation, kind, dataset, constant, *, objective,
                        intervention="correct", batch_size=8, oracle_labels=None):
    indices = g.panel_indices(dataset)
    selected, mapping = g.intervene(dataset["test"]["features"], dataset["test"]["labels"], intervention, 700044, constant.cpu())
    if objective == "G":
        cases = g.evaluation_cases(dataset, indices)
    else:
        cases = [{"sensor_index": i, "family": "label", "template_split": "original-ranking", "template": 0,
                  "prompt": dataset["prompt"], "target": g.category_text(dataset["categories"][dataset["test"]["labels"][i]]),
                  "label": dataset["test"]["labels"][i]} for i in indices]
    fixed = kind in ("none", "static") or intervention in ("off", "constant")
    cache = {}
    items = []
    decoded = reused = 0
    for start in range(0, len(cases), batch_size):
        group = cases[start:start+batch_size]
        prompts = [c["prompt"] for c in group]
        if oracle_labels is not None:
            prompts = ["The sensor category is " + g.category_text(dataset["categories"][oracle_labels[c["sensor_index"]]]) + ". " + c["prompt"] for c in group]
        key = tuple(prompts)
        if fixed and key in cache:
            responses = cache[key]
            reused += 1
        else:
            responses = decode(host, adaptation, kind, selected[[c["sensor_index"] for c in group]], prompts,
                               native_chat=objective == "G", intervention=intervention)
            if len(responses) != len(group):
                raise ValueError("Decoder omitted a registered generation case")
            decoded += 1
            if fixed:
                cache[key] = responses
        for c, p, response in zip(group, prompts, responses):
            i = c["sensor_index"]
            items.append({**c, "id": dataset["test"]["ids"][i], "group": dataset["test"]["groups"][i], "actual_prompt": p,
                          "sensor_id": None if mapping is None else dataset["test"]["ids"][mapping[i]],
                          "response": response, **g.score(response, c, dataset)})
    summaries = {}
    for family in sorted({i["family"] for i in items}):
        for split in sorted({i["template_split"] for i in items}):
            rows = [i for i in items if i["family"] == family and i["template_split"] == split]
            if rows:
                summaries[family + "/" + split] = {"outputs": len(rows), "sensor_units": len({r["id"] for r in rows}),
                    **{key: sum(r[key] for r in rows)/len(rows) for key in ("semantic_correct", "format_compliant", "joint_success")}}
    return {"summary": summaries, "items": items, "panel_ids": [dataset["test"]["ids"][i] for i in indices],
            "decoding": {"greedy": True, "max_new_tokens": 64, "native_chat": objective == "G", "enable_thinking": False,
                         **({"eos_token_ids": chat_stop_ids(host)} if objective == "G" else {})},
            "intervention": intervention, "execution": {"batch_size": batch_size, "decoded_batches": decoded, "reused_identical_batches": reused,
                "scope": "Same complete prompt batch and fixed effective weights; all sensor cases retained. No reuse across checkpoints or evaluations."}}
