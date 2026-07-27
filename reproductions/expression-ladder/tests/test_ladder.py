"""Tests that need no model weights."""

from __future__ import annotations

import pytest
import torch

from expression_ladder.data import build_style_pairs, validate_style_data
from expression_ladder.ladder import (
    DirectionSteering,
    SteeringSpec,
    direction_geometry,
    distribute_budget,
    hidden_tensor,
    replace_hidden,
)
from expression_ladder.runtime import bounded_context


def test_corpus_hash_is_frozen():
    report = validate_style_data()
    assert report["sha256"] == (
        "85f6e3599474ad33c37092b583b21ccfc3175fd85b168c65b735c2dd75332346"
    )
    assert (report["fit_cases"], report["dev_cases"], report["test_cases"]) == (6, 2, 8)


def test_pairs_carry_matched_responses():
    pairs = build_style_pairs(("warmth",), ("test",))
    assert pairs
    for pair in pairs:
        assert pair.positive_response != pair.negative_response
        assert pair.split == "test"


def test_budget_splits_in_quadrature():
    # Six sites at a total of 0.16 should each carry 0.16/sqrt(6).
    assert distribute_budget(0.16, 6) == pytest.approx(0.16 / 6**0.5)
    assert distribute_budget(0.16, 1) == pytest.approx(0.16)
    with pytest.raises(ValueError):
        distribute_budget(0.16, 0)


def test_bounded_context_raises_rather_than_truncating():
    prompt = torch.zeros((1, 40), dtype=torch.long)
    assert bounded_context(prompt, 40).shape == (1, 40)
    with pytest.raises(ValueError):
        bounded_context(prompt, 39)


def test_hidden_tensor_roundtrip_preserves_tuple_tail():
    tensor = torch.zeros((1, 2, 3))
    packed = (tensor, "cache")
    assert hidden_tensor(packed) is tensor
    replaced = replace_hidden(packed, torch.ones((1, 2, 3)))
    assert replaced[1] == "cache"
    assert torch.equal(replaced[0], torch.ones((1, 2, 3)))


def test_steering_scales_with_activation_norm():
    """The delta must be a fixed fraction of the activation it is added to."""
    module = torch.nn.Identity()
    direction = torch.tensor([1.0, 0.0, 0.0])
    steering = DirectionSteering(
        module, direction, fraction=0.1, sign=1, window=0, scoring_start=0
    )
    activation = torch.tensor([[[3.0, 4.0, 0.0]]])  # norm 5
    out = steering._replace(activation, activation, slice(0, 1))
    assert out[0, 0, 0].item() == pytest.approx(3.0 + 0.5)


def test_steering_rejects_a_zero_direction():
    with pytest.raises(ValueError):
        DirectionSteering(
            torch.nn.Identity(),
            torch.zeros(3),
            fraction=0.1,
            sign=1,
            window=0,
            scoring_start=0,
        )


def test_spec_name_is_stable():
    spec = SteeringSpec("warmth", 35, "residual", 0.02, -1, 8)
    assert spec.name == "warmth-residual-l35-f0.02-w8-negative"


def test_direction_geometry_reports_cross_axis_cosine():
    directions = {
        "a": {"residual": {0: torch.tensor([1.0, 0.0])}},
        "b": {"residual": {0: torch.tensor([0.0, 1.0])}},
    }
    report = direction_geometry(
        directions, axes=("a", "b"), components=("residual",), layers=(0,)
    )
    assert report["residual-layer-0"]["mean_abs_cross_cosine"] == pytest.approx(0.0, abs=1e-6)
