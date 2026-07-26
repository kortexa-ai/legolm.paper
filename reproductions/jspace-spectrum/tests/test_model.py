import torch

from jspace_spectrum.model import (
    PROFILES,
    inserted_token_positions,
)


def test_inserted_token_span_excludes_template_prefix_and_suffix() -> None:
    empty = torch.tensor([[10, 11, 12, 90, 91]])
    content = torch.tensor([[10, 11, 40, 41, 42, 12, 90, 91]])
    assert inserted_token_positions(empty, content) == [2, 3, 4]


def test_paper_profiles_use_full_attention_layers() -> None:
    moe = PROFILES["qwen36-35b"]
    dense = PROFILES["qwen36-27b"]
    assert moe.source_layer in moe.trace_layers
    assert moe.target_layer in moe.trace_layers
    assert dense.source_layer in dense.trace_layers
    assert dense.target_layer in dense.trace_layers
    assert all((layer + 1) % 4 == 0 for layer in moe.trace_layers)
    assert all((layer + 1) % 4 == 0 for layer in dense.trace_layers)
