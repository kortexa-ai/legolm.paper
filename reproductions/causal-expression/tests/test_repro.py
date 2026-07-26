from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from causal_expression.cli import parse_seeds
from causal_expression.data import (
    CONTENT_CASES,
    STYLE_AXIS_NAMES,
    build_style_pairs,
    validate_style_data,
)
from causal_expression.experiment import (
    assistant_header_start,
    prefix_response_nll,
    render_prompt,
    render_prompt_response,
)
from causal_expression.figures import render_figures
from causal_expression.metrics import (
    axis_response_metrics,
    consolidate_metrics,
)
from causal_expression.prefix import (
    NeutralAnchoredPrefixBank,
    SoftPrefixBank,
    insert_prefix_embeddings,
    neutral_training_schedule,
    pole_training_schedule,
    repeat_to_length,
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
        **_kwargs,
    ):
        hidden = (
            self.embedding(input_ids)
            if inputs_embeds is None
            else inputs_embeds
        )
        return SimpleNamespace(logits=self.output(hidden))


def test_data_inventory_is_frozen_and_content_matched():
    report = validate_style_data()
    assert report["axes"] == 3
    assert report["cases"] == 16
    assert report["pairs"] == 48
    assert report["fit_cases"] == 6
    assert report["dev_cases"] == 2
    assert report["test_cases"] == 8
    assert len(report["sha256"]) == 64
    assert {case.split for case in CONTENT_CASES} == {"fit", "dev", "test"}
    for pair in build_style_pairs():
        assert pair.neutral_response in pair.positive_response
        assert pair.neutral_response in pair.negative_response


def test_pole_and_neutral_schedules_are_deterministic_and_balanced():
    pairs = build_style_pairs(STYLE_AXIS_NAMES, ("fit",))
    first = pole_training_schedule(pairs, steps=36, seed=20260724)
    second = pole_training_schedule(pairs, steps=36, seed=20260724)
    assert [(pair.pair_id, sign) for pair, sign in first] == [
        (pair.pair_id, sign) for pair, sign in second
    ]
    assert len(first) == 36
    assert {sign for _pair, sign in first} == {-1, 1}

    stage_i = neutral_training_schedule(
        pairs,
        pole_steps=36,
        neutral_steps=18,
        seed=20260724,
    )
    assert len(stage_i) == 54
    assert sum(kind == "pole" for kind, _pair, _sign in stage_i) == 36
    assert sum(kind == "neutral" for kind, _pair, _sign in stage_i) == 18


def test_stage_h_prefix_is_signed_and_inserted_at_the_requested_boundary():
    positive = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 2.0]],
            [[1.0, 1.0], [2.0, 2.0]],
        ]
    )
    negative = -positive
    bank = SoftPrefixBank(("warmth", "patience"), positive, negative)
    torch.testing.assert_close(bank.prefix("warmth", 1), positive[0])
    torch.testing.assert_close(bank.prefix("warmth", -1), negative[0])
    torch.testing.assert_close(
        bank.interpolated_prefix("warmth", 0.5),
        positive[0] * 0.5,
    )
    source = torch.tensor([[[1.0, 1.0], [4.0, 4.0]]])
    changed = insert_prefix_embeddings(
        source,
        bank.prefix("warmth", 1),
        insertion=1,
    )
    torch.testing.assert_close(
        changed,
        torch.tensor(
            [[[1.0, 1.0], [2.0, 0.0], [0.0, 2.0], [4.0, 4.0]]]
        ),
    )
    torch.testing.assert_close(
        repeat_to_length(torch.tensor([1, 2, 3]), 5),
        torch.tensor([1, 2, 3, 1, 2]),
    )


def test_stage_i_zero_state_uses_the_same_shared_prefix_path():
    bank = NeutralAnchoredPrefixBank(
        ("warmth", "patience"),
        torch.tensor([[2.0, 1.0]]),
        torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
        max_token_norm=4.0,
    )
    torch.testing.assert_close(
        bank.prefix("warmth", 1, strength=0.0),
        bank.neutral_prefix(),
    )
    torch.testing.assert_close(
        bank.prefix("warmth", 1),
        torch.tensor([[3.0, 1.0]]),
    )
    torch.testing.assert_close(
        bank.prefix("warmth", -1),
        torch.tensor([[1.0, 1.0]]),
    )


def test_chat_boundary_and_response_target_alignment_preserve_gradient():
    tokenizer = FakeTokenizer()
    prompt = render_prompt(
        tokenizer,
        system_prompt="system",
        user_prompt="user",
    )
    boundary = assistant_header_start(
        tokenizer,
        system_prompt="system",
        user_prompt="user",
    )
    assert boundary == prompt.shape[1] - 1
    full, response_start = render_prompt_response(
        tokenizer,
        system_prompt="system",
        user_prompt="user",
        response="answer",
    )
    assert response_start == prompt.shape[1]
    torch.testing.assert_close(prompt, full[:, :response_start])

    model = TinyCausalModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    pair = build_style_pairs(("warmth",), ("fit",))[0]
    prefix = nn.Parameter(torch.randn(2, 8))
    loss, tokens = prefix_response_nll(
        model,
        tokenizer,
        pair=pair,
        response=pair.positive_response,
        prefix=prefix,
        max_seq_len=4096,
    )
    assert tokens > 0
    loss.backward()
    assert prefix.grad is not None
    assert bool(torch.isfinite(prefix.grad).all())


def _response(value: float) -> dict:
    return {
        "attribution": {"signed_attribution": value},
        "lexical": {"signed_score": 0.0},
        "exactly_matches_off": False,
        "word_jaccard_with_off": 0.5,
        "word_jaccard_with_core": 0.5,
        "word_count": 20,
        "text": "response",
    }


def _report(include_center: bool) -> dict:
    rows = []
    aggregate = {}
    for axis in STYLE_AXIS_NAMES:
        responses = {
            "off": _response(0.0),
            "explicit_positive": _response(1.0),
            "explicit_negative": _response(-1.0),
            "prefix_positive": _response(0.7),
            "prefix_negative": _response(-0.5),
            "wrong_axis": _response(0.1),
        }
        if include_center:
            responses["neutral_center"] = _response(0.05)
        rows.append({"axis": axis, "responses": responses})
        aggregate[axis] = {
            condition: {
                "mean_attribution": response["attribution"][
                    "signed_attribution"
                ]
            }
            for condition, response in responses.items()
        }
    return {
        "axes": list(STYLE_AXIS_NAMES),
        "rows": rows,
        "aggregate": aggregate,
    }


def test_response_metrics_keep_sign_specificity_and_center_separate():
    stage_h = _report(False)
    warmth = axis_response_metrics(stage_h, "warmth")
    assert warmth["relative_span"] == 0.6
    assert warmth["signed_prompts"] == 1
    assert warmth["specific_prompts"] == 1
    assert warmth["center_drift"] is None

    stage_i = _report(True)
    warmth_i = axis_response_metrics(stage_i, "warmth")
    assert warmth_i["strict_center_prompts"] == 1
    assert warmth_i["absolute_center_drift_fraction"] == 0.025


def test_consolidation_applies_frozen_decision_rules():
    records = []
    for _seed in (20260724, 20260725, 20260726):
        h = {
            axis: {
                **axis_response_metrics(_report(False), axis),
                "signed_prompts": 8,
                "specific_prompts": 8,
                "prompts": 8,
            }
            for axis in STYLE_AXIS_NAMES
        }
        i = {
            axis: {
                **axis_response_metrics(_report(True), axis),
                "relative_span": 0.7 if axis == "patience" else 0.6,
                "signed_prompts": 8,
                "specific_prompts": 8,
                "strict_center_prompts": 8,
                "prompts": 8,
            }
            for axis in STYLE_AXIS_NAMES
        }
        records.append({"stage_h": {"metrics": h}, "stage_i": {"metrics": i}})
    consolidated = consolidate_metrics(records)
    assert consolidated["decisions"]["stage_h_warmth"]["passed"]
    assert consolidated["decisions"]["stage_i_patience"]["passed"]
    assert consolidated["decisions"]["stage_i_neutral_center"]["passed"]


def test_figure_renderer_reads_only_the_result_artifact(tmp_path: Path):
    h_report = _report(False)
    i_report = _report(True)
    seed_dir = tmp_path / "seed-20260724"
    seed_dir.mkdir()
    (seed_dir / "stage-h-responses.json").write_text(json.dumps(h_report))
    (seed_dir / "stage-i-responses.json").write_text(json.dumps(i_report))
    sweep_aggregate = {
        "warmth": {
            "off": {"mean_attribution": 0.0},
            "explicit_positive": {"mean_attribution": 1.0},
            "explicit_negative": {"mean_attribution": -1.0},
            **{
                f"state_{state:+g}": {"mean_attribution": state * 0.5}
                for state in (
                    -1.0,
                    -0.75,
                    -0.5,
                    -0.25,
                    0.0,
                    0.25,
                    0.5,
                    0.75,
                    1.0,
                )
            },
        }
    }
    (seed_dir / "stage-h-warmth-sweep.json").write_text(
        json.dumps(
            {
                "axis": "warmth",
                "states": [
                    -1.0,
                    -0.75,
                    -0.5,
                    -0.25,
                    0.0,
                    0.25,
                    0.5,
                    0.75,
                    1.0,
                ],
                "aggregate": sweep_aggregate,
            }
        )
    )
    h_metrics = {
        axis: axis_response_metrics(h_report, axis)
        for axis in STYLE_AXIS_NAMES
    }
    i_metrics = {
        axis: axis_response_metrics(i_report, axis)
        for axis in STYLE_AXIS_NAMES
    }
    record = {
        "seed": 20260724,
        "stage_h": {
            "responses": "seed-20260724/stage-h-responses.json",
            "warmth_sweep": "seed-20260724/stage-h-warmth-sweep.json",
            "metrics": h_metrics,
        },
        "stage_i": {
            "responses": "seed-20260724/stage-i-responses.json",
            "metrics": i_metrics,
        },
    }
    summary = {
        "seed_records": [record],
        "consolidated": consolidate_metrics([record]),
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    rendered = render_figures(summary_path, tmp_path / "figures")
    assert set(rendered) == {
        "relative_spans",
        "warmth_sweeps",
        "neutral_anchor",
        "six_pole_radar",
    }
    for relative in rendered.values():
        path = tmp_path / relative
        assert path.exists()
        assert path.stat().st_size > 1_000


def test_seed_parser_rejects_duplicates():
    assert parse_seeds("1,2,3") == (1, 2, 3)
    try:
        parse_seeds("1,1")
    except Exception as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate seeds were accepted")
