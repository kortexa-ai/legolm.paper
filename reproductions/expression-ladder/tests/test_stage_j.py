from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from expression_ladder.cli import parse_seeds
from expression_ladder.data import STYLE_AXIS_NAMES, build_style_pairs
from expression_ladder.metrics import axis_response_metrics
from expression_ladder.prefix import NeutralAnchoredPrefixBank
from expression_ladder.runtime import FULL_PROFILE, SMOKE_PROFILE
from expression_ladder.stage_j import (
    FULL_STAGE_J_CONFIG,
    STAGE_J_PREREG,
    StageJConfig,
    _measure_generation,
    center_specific_prompts,
    consolidate_stage_j,
    distill_training_schedule,
    distillation_kl,
    sample_continuation_ids,
    sample_response,
    stage_j_verdict,
    token_kl,
    train_stage_j,
    trajectory_gate,
)


class FakeTokenizer:
    eos_token_id = 2

    @staticmethod
    def _content_ids(text: str) -> list[int]:
        return [10 + (ord(character) % 40) for character in text]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_tensors,
        enable_thinking=False,
    ):
        assert tokenize and return_tensors == "pt"
        assert enable_thinking is False
        role_ids = {"system": 3, "user": 4, "assistant": 5}
        ids = [1]
        for message in messages:
            ids.append(role_ids[message["role"]])
            ids.extend(self._content_ids(message["content"]))
            ids.append(6)
        if add_generation_prompt:
            ids.append(role_ids["assistant"])
        return torch.tensor([ids], dtype=torch.long)

    def decode(self, ids, skip_special_tokens=True):
        kept = [
            str(int(value))
            for value in ids
            if not (skip_special_tokens and int(value) == self.eos_token_id)
        ]
        return " ".join(kept)


class TinyCausalModel(nn.Module):
    def __init__(self, vocab: int = 64, hidden: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.output = nn.Linear(hidden, vocab, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        use_cache=False,
        past_key_values=None,
        **_kwargs,
    ):
        hidden = (
            self.embedding(input_ids)
            if inputs_embeds is None
            else inputs_embeds
        )
        # Causal cumulative mean: every position sees its predecessors, so an
        # inserted prefix actually shifts downstream logits.
        steps = torch.arange(
            1,
            hidden.shape[1] + 1,
            device=hidden.device,
        ).reshape(1, -1, 1)
        mixed = torch.cumsum(hidden, dim=1) / steps
        return SimpleNamespace(logits=self.output(mixed), past_key_values=None)


def _frozen_tiny_model() -> TinyCausalModel:
    model = TinyCausalModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class SpyCausalModel(TinyCausalModel):
    """Records, per forward, whether inference mode was on and whether the
    inputs carried gradient — the two conditions the 35B OOM proved matter."""

    def __init__(self) -> None:
        super().__init__()
        self.forward_modes: list[tuple[bool, bool]] = []

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        self.forward_modes.append(
            (
                torch.is_inference_mode_enabled(),
                bool(inputs_embeds is not None and inputs_embeds.requires_grad),
            )
        )
        return super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


def _frozen_spy_model() -> SpyCausalModel:
    model = SpyCausalModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _tiny_bank() -> NeutralAnchoredPrefixBank:
    return NeutralAnchoredPrefixBank(
        STYLE_AXIS_NAMES,
        torch.randn(2, 8) * 0.1,
        torch.randn(3, 2, 8) * 0.1,
        max_token_norm=10.0,
    )


def _tiny_config(pole_steps: int, distill_steps: int) -> StageJConfig:
    return StageJConfig(
        suite="smoke",
        stage_h_steps=1,
        pole_steps=pole_steps,
        distill_steps=distill_steps,
        dropout_rate=0.0,
        distill_new_tokens=2,
        learning_rate=0.05,
        sampled_temperature=0.8,
        snapshot_count=1,
        test_case_limit=1,
        max_new_tokens=4,
        max_context_tokens=4096,
        max_score_seq_len=4096,
        max_train_seq_len=4096,
    )


def test_prereg_pins_model_data_thresholds_and_config():
    assert STAGE_J_PREREG["model_id"] == FULL_PROFILE.model_id
    assert STAGE_J_PREREG["model_revision"] == FULL_PROFILE.revision
    assert STAGE_J_PREREG["corpus_sha256"] == (
        "85f6e3599474ad33c37092b583b21ccfc3175fd85b168c65b735c2dd75332346"
    )
    assert STAGE_J_PREREG["primary_axes"] == ["warmth", "patience"]
    assert STAGE_J_PREREG["secondary_axes"] == ["goodwill"]
    assert STAGE_J_PREREG["center_drift_fraction_max"] == 0.10
    assert STAGE_J_PREREG["relative_span_min"] == {
        "warmth": 0.40,
        "patience": 0.20,
    }
    assert FULL_STAGE_J_CONFIG.pole_steps == 48
    assert FULL_STAGE_J_CONFIG.distill_steps == 18
    assert FULL_STAGE_J_CONFIG.dropout_rate == 0.25
    assert FULL_STAGE_J_CONFIG.learning_rate == 0.001
    assert (
        FULL_STAGE_J_CONFIG.sampled_temperature
        == STAGE_J_PREREG["sampled_temperature"]
    )
    assert len(STAGE_J_PREREG["seeds"]) == 3
    assert parse_seeds("1,2,3") == (1, 2, 3)


def test_distill_schedule_is_deterministic_and_boosts_negative_coverage():
    pairs = build_style_pairs(STYLE_AXIS_NAMES, ("fit",))
    first = distill_training_schedule(
        pairs,
        pole_steps=48,
        distill_steps=18,
        dropout_rate=0.0,
        seed=20260801,
    )
    second = distill_training_schedule(
        pairs,
        pole_steps=48,
        distill_steps=18,
        dropout_rate=0.0,
        seed=20260801,
    )
    assert [
        (kind, pair.pair_id, sign) for kind, pair, sign in first
    ] == [(kind, pair.pair_id, sign) for kind, pair, sign in second]
    assert len(first) == 66
    pole = [(pair.axis, sign) for kind, pair, sign in first if kind == "pole"]
    assert len(pole) == 48
    # One exact epoch over the inventory: negatives doubled for patience and
    # goodwill, warmth untouched.
    assert pole.count(("warmth", -1)) == 6
    assert pole.count(("patience", -1)) == 12
    assert pole.count(("goodwill", -1)) == 12
    assert pole.count(("warmth", 1)) == 6
    assert pole.count(("patience", 1)) == 6


def test_prefix_dropout_converts_pole_events_to_zero_state_distillation():
    pairs = build_style_pairs(STYLE_AXIS_NAMES, ("fit",))
    events = distill_training_schedule(
        pairs,
        pole_steps=48,
        distill_steps=18,
        dropout_rate=0.5,
        seed=20260801,
    )
    assert len(events) == 66
    distill = [event for event in events if event[0] == "distill"]
    assert len(distill) > 18
    assert all(sign == 0 for _kind, _pair, sign in distill)
    without_dropout = distill_training_schedule(
        pairs,
        pole_steps=48,
        distill_steps=18,
        dropout_rate=0.0,
        seed=20260801,
    )
    assert sum(event[0] == "distill" for event in without_dropout) == 18


def test_token_kl_matches_manual_value_and_vanishes_on_identity():
    logits = torch.tensor([[[0.0, 1.0, -1.0]]])
    torch.testing.assert_close(
        token_kl(logits, logits),
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0.0,
    )
    other = torch.tensor([[[1.0, 0.0, 0.5]]])
    teacher_log = torch.log_softmax(logits.float(), dim=-1)
    student_log = torch.log_softmax(other.float(), dim=-1)
    manual = torch.sum(
        teacher_log.exp() * (teacher_log - student_log),
        dim=-1,
    ).mean()
    torch.testing.assert_close(token_kl(logits, other), manual)
    assert float(token_kl(logits, other)) > 0.0
    assert float(token_kl(logits, other)) != float(token_kl(other, logits))


def test_distillation_kl_aligns_positions_and_reaches_the_prefix_gradient():
    model = _frozen_tiny_model()
    tokenizer = FakeTokenizer()
    pair = build_style_pairs(("warmth",), ("fit",))[0]
    prefix = nn.Parameter(torch.randn(2, 8))
    continuation = torch.tensor([11, 12, 13], dtype=torch.long)
    loss, positions = distillation_kl(
        model,
        tokenizer,
        user_prompt=pair.prompt,
        prefix=prefix,
        continuation=continuation,
        max_seq_len=4096,
    )
    assert positions == continuation.numel() + 1
    assert float(loss.detach()) >= 0.0
    loss.backward()
    assert prefix.grad is not None
    assert bool(torch.isfinite(prefix.grad).all())
    frontier_only, frontier_positions = distillation_kl(
        model,
        tokenizer,
        user_prompt=pair.prompt,
        prefix=prefix.detach(),
        continuation=torch.empty(0, dtype=torch.long),
        max_seq_len=4096,
    )
    assert frontier_positions == 1
    assert float(frontier_only) >= 0.0


def test_training_updates_only_the_scheduled_parameters():
    tokenizer = FakeTokenizer()
    pairs = build_style_pairs(STYLE_AXIS_NAMES, ("fit",))[:3]
    dev_pairs = build_style_pairs(STYLE_AXIS_NAMES, ("dev",))

    model = _frozen_tiny_model()
    bank = _tiny_bank()
    before_neutral = bank.neutral.detach().clone()
    before_delta = bank.delta.detach().clone()
    train_stage_j(
        model,
        tokenizer,
        bank,
        pairs=pairs,
        dev_pairs=dev_pairs,
        config=_tiny_config(pole_steps=2, distill_steps=0),
        seed=20260801,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(bank.neutral.detach(), before_neutral)
    assert not torch.allclose(bank.delta.detach(), before_delta)

    model = _frozen_tiny_model()
    bank = _tiny_bank()
    before_neutral = bank.neutral.detach().clone()
    before_delta = bank.delta.detach().clone()
    result = train_stage_j(
        model,
        tokenizer,
        bank,
        pairs=pairs,
        dev_pairs=dev_pairs,
        config=_tiny_config(pole_steps=0, distill_steps=2),
        seed=20260801,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(bank.delta.detach(), before_delta)
    assert not torch.allclose(bank.neutral.detach(), before_neutral)
    assert result["selection"]["selected_step"] in {
        entry["step"] for entry in result["selection"]["snapshots"]
    }
    kinds = {record["kind"] for record in result["history"]}
    assert kinds == {"distill"}


def test_sampled_generation_is_seeded_and_respects_stop_tokens():
    model = _frozen_tiny_model()
    tokenizer = FakeTokenizer()
    shared = dict(
        system_prompt="system",
        user_prompt="user",
        prefix=None,
        max_new_tokens=6,
        max_context_tokens=4096,
        temperature=0.8,
    )
    first = sample_response(
        model,
        tokenizer,
        generator=torch.Generator().manual_seed(7),
        **shared,
    )
    second = sample_response(
        model,
        tokenizer,
        generator=torch.Generator().manual_seed(7),
        **shared,
    )
    assert first == second

    with torch.no_grad():
        model.output.weight.zero_()
        model.output.weight[tokenizer.eos_token_id, :] = 25.0
        model.embedding.weight.fill_(0.1)
    stopped = sample_response(
        model,
        tokenizer,
        generator=torch.Generator().manual_seed(7),
        **shared,
    )
    assert stopped == ""


def test_generation_paths_run_inference_mode_without_grad_inputs():
    tokenizer = FakeTokenizer()
    pair = build_style_pairs(("warmth",), ("test",))[0]
    prefix = nn.Parameter(torch.randn(2, 8))
    config = _tiny_config(pole_steps=1, distill_steps=1)
    for decoding in ("greedy", "sampled"):
        model = _frozen_spy_model()
        measured = _measure_generation(
            model,
            tokenizer,
            pair=pair,
            axis="warmth",
            condition="prefix_positive",
            system_prompt="system",
            prefix=prefix,
            decoding=decoding,
            seed=7,
            config=config,
        )
        assert isinstance(measured["text"], str)
        assert model.forward_modes
        assert all(inference for inference, _grad in model.forward_modes)
        assert all(not grad for _inference, grad in model.forward_modes)
        assert prefix.grad is None

    model = _frozen_spy_model()
    continuation = sample_continuation_ids(
        model,
        tokenizer,
        user_prompt=pair.prompt,
        prefix=prefix,
        new_tokens=2,
        temperature=1.0,
        generator=torch.Generator().manual_seed(3),
        max_seq_len=4096,
    )
    assert model.forward_modes
    assert all(inference for inference, _grad in model.forward_modes)
    assert all(not grad for _inference, grad in model.forward_modes)
    assert not continuation.requires_grad


def _response(value: float) -> dict:
    return {
        "attribution": {"signed_attribution": value},
        "lexical": {"signed_score": 0.0},
        "exactly_matches_off": False,
        "word_jaccard_with_off": 0.7,
        "word_jaccard_with_core": 0.5,
        "word_count": 20,
        "text": "response",
    }


def _report() -> dict:
    rows = []
    aggregate = {}
    for axis in STYLE_AXIS_NAMES:
        responses = {
            "off": _response(0.0),
            "neutral_center": _response(0.05),
            "explicit_positive": _response(1.0),
            "explicit_negative": _response(-1.0),
            "prefix_positive": _response(0.7),
            "prefix_negative": _response(-0.5),
            "wrong_axis": _response(0.1),
        }
        rows.append({"axis": axis, "responses": responses})
        aggregate[axis] = {
            condition: {
                "mean_attribution": response["attribution"]["signed_attribution"]
            }
            for condition, response in responses.items()
        }
    return {
        "axes": list(STYLE_AXIS_NAMES),
        "rows": rows,
        "aggregate": aggregate,
    }


def test_center_specificity_counts_prompts_beating_center_drift():
    report = _report()
    assert center_specific_prompts(report, "warmth") == 1
    report["rows"][0]["responses"]["neutral_center"]["attribution"][
        "signed_attribution"
    ] = 2.0
    assert center_specific_prompts(report, "warmth") == 0


def _seed_record(seed: int, *, patience_specific: int = 7) -> dict:
    metrics = {}
    for axis in STYLE_AXIS_NAMES:
        row = axis_response_metrics(_report(), axis)
        row["prompts"] = 8
        row["signed_prompts"] = 8
        row["specific_prompts"] = (
            patience_specific if axis == "patience" else 8
        )
        row["center_specific_prompts"] = 8
        row["strict_center_prompts"] = 8
        row["relative_span"] = 0.45 if axis == "warmth" else 0.25
        row["absolute_center_drift_fraction"] = 0.04
        metrics[axis] = row
    states = {
        "-1": -0.4,
        "-0.5": -0.2,
        "-0.25": -0.05,
        "+0": 0.01,
        "+0.25": 0.08,
        "+0.5": 0.2,
        "+1": 0.4,
    }
    trajectory = {
        axis: {
            "state_mean_attribution": dict(states),
            "prompts": 8,
            **trajectory_gate(states, max_adjacent_inversions=1),
        }
        for axis in STYLE_AXIS_NAMES
    }
    return {
        "seed": seed,
        "greedy": {
            "metrics": metrics,
            "center_jaccard": 0.75,
            "center_exact_match_rate": 0.25,
        },
        "sampled": {
            "metrics": metrics,
            "center_jaccard": 0.55,
            "center_exact_match_rate": 0.0,
        },
        "trajectory": {"metrics": trajectory},
    }


def test_gate_arithmetic_applies_frozen_primary_thresholds():
    records = [_seed_record(seed) for seed in STAGE_J_PREREG["seeds"]]
    consolidated = consolidate_stage_j(records)
    corpus = {"sha256": STAGE_J_PREREG["corpus_sha256"]}
    verdict = stage_j_verdict(
        consolidated,
        corpus=corpus,
        profile=FULL_PROFILE,
        suite="full",
        seeds=STAGE_J_PREREG["seeds"],
    )
    assert verdict["overall"] == "PASS"
    assert verdict["gates"]["center_fidelity"]["passed"]
    assert verdict["gates"]["signed_axes"]["greedy_patience"]["passed"]
    assert verdict["gates"]["trajectories"]["passed"]
    assert "goodwill" in verdict["secondary"]

    weak_patience = [
        _seed_record(seed, patience_specific=4)
        for seed in STAGE_J_PREREG["seeds"]
    ]
    failed = stage_j_verdict(
        consolidate_stage_j(weak_patience),
        corpus=corpus,
        profile=FULL_PROFILE,
        suite="full",
        seeds=STAGE_J_PREREG["seeds"],
    )
    assert failed["overall"] == "FAIL"
    assert not failed["gates"]["signed_axes"]["greedy_patience"][
        "specific_passed"
    ]

    weak_goodwill = [_seed_record(seed) for seed in STAGE_J_PREREG["seeds"]]
    for record in weak_goodwill:
        for mode in ("greedy", "sampled"):
            record[mode]["metrics"]["goodwill"]["relative_span"] = 0.01
            record[mode]["metrics"]["goodwill"]["signed_prompts"] = 2
    still_passing = stage_j_verdict(
        consolidate_stage_j(weak_goodwill),
        corpus=corpus,
        profile=FULL_PROFILE,
        suite="full",
        seeds=STAGE_J_PREREG["seeds"],
    )
    assert still_passing["overall"] == "PASS"


def test_verdict_is_invalid_off_the_preregistered_configuration():
    records = [_seed_record(seed) for seed in STAGE_J_PREREG["seeds"]]
    consolidated = consolidate_stage_j(records)
    good_corpus = {"sha256": STAGE_J_PREREG["corpus_sha256"]}

    bad_corpus = stage_j_verdict(
        consolidated,
        corpus={"sha256": "0" * 64},
        profile=FULL_PROFILE,
        suite="full",
        seeds=STAGE_J_PREREG["seeds"],
    )
    assert bad_corpus["overall"] == "INVALID"

    smoke = stage_j_verdict(
        consolidated,
        corpus=good_corpus,
        profile=SMOKE_PROFILE,
        suite="smoke",
        seeds=STAGE_J_PREREG["seeds"][:1],
    )
    assert smoke["overall"] == "INVALID"
    assert any("model" in reason for reason in smoke["invalid_reasons"])

    broken = [_seed_record(seed) for seed in STAGE_J_PREREG["seeds"]]
    broken[0]["sampled"]["metrics"]["patience"] = {
        "invalid": "patience explicit span must be positive"
    }
    denominator = stage_j_verdict(
        consolidate_stage_j(broken),
        corpus=good_corpus,
        profile=FULL_PROFILE,
        suite="full",
        seeds=STAGE_J_PREREG["seeds"],
    )
    assert denominator["overall"] == "INVALID"


def test_trajectory_gate_orders_states_and_tolerates_one_inversion():
    clean = {
        "-1": -0.4,
        "-0.5": -0.2,
        "-0.25": -0.05,
        "+0": 0.0,
        "+0.25": 0.05,
        "+0.5": 0.2,
        "+1": 0.4,
    }
    assert trajectory_gate(clean, max_adjacent_inversions=1)[
        "trajectory_passed"
    ]

    wobble = dict(clean)
    wobble["-0.25"] = -0.25
    result = trajectory_gate(wobble, max_adjacent_inversions=1)
    assert result["adjacent_inversions"] == 1
    assert result["trajectory_passed"]

    double_wobble = dict(wobble)
    double_wobble["+0.25"] = -0.02
    assert not trajectory_gate(double_wobble, max_adjacent_inversions=1)[
        "trajectory_passed"
    ]

    sign_flip = dict(clean)
    sign_flip["-0.5"] = 0.1
    assert not trajectory_gate(sign_flip, max_adjacent_inversions=1)[
        "sign_correct"
    ]
