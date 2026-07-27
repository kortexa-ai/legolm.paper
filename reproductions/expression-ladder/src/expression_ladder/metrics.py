"""Response aggregation and confirmatory decision rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import statistics
from typing import Any

from .data import STYLE_AXIS_NAMES


def word_jaccard(left: str, right: str) -> float:
    left_words = set(left.lower().split())
    right_words = set(right.lower().split())
    union = left_words | right_words
    return len(left_words & right_words) / len(union) if union else 1.0


def mean_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot average an empty row set")
    return {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in rows[0]
    }


def aggregate_responses(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    conditions = list(rows[0]["responses"]) if rows else []
    for axis in STYLE_AXIS_NAMES:
        axis_rows = [row for row in rows if row["axis"] == axis]
        if not axis_rows:
            continue
        aggregate[axis] = {}
        for condition in conditions:
            values = [row["responses"][condition] for row in axis_rows]
            aggregate[axis][condition] = {
                "mean_attribution": sum(
                    value["attribution"]["signed_attribution"]
                    for value in values
                )
                / len(values),
                "mean_lexical_score": sum(
                    value["lexical"]["signed_score"] for value in values
                )
                / len(values),
                "exact_match_rate": sum(
                    value["exactly_matches_off"] for value in values
                )
                / len(values),
                "mean_word_jaccard": sum(
                    value["word_jaccard_with_off"] for value in values
                )
                / len(values),
                "mean_core_jaccard": sum(
                    value["word_jaccard_with_core"] for value in values
                )
                / len(values),
                "mean_word_count": sum(
                    value["word_count"] for value in values
                )
                / len(values),
            }
    return aggregate


def generated_span(
    aggregate: Mapping[str, Any],
    axis: str,
) -> float:
    values = aggregate[axis]
    return float(values["prefix_positive"]["mean_attribution"]) - float(
        values["prefix_negative"]["mean_attribution"]
    )


def explicit_span(
    aggregate: Mapping[str, Any],
    axis: str,
) -> float:
    values = aggregate[axis]
    return float(values["explicit_positive"]["mean_attribution"]) - float(
        values["explicit_negative"]["mean_attribution"]
    )


def axis_response_metrics(
    report: Mapping[str, Any],
    axis: str,
) -> dict[str, Any]:
    aggregate = report["aggregate"]
    generated = generated_span(aggregate, axis)
    explicit = explicit_span(aggregate, axis)
    if explicit <= 0.0:
        raise ValueError(f"{axis} explicit span must be positive")
    axis_rows = [row for row in report["rows"] if row["axis"] == axis]
    signed = 0
    specific = 0
    strict_center = 0
    for row in axis_rows:
        responses = row["responses"]
        positive = float(
            responses["prefix_positive"]["attribution"]["signed_attribution"]
        )
        negative = float(
            responses["prefix_negative"]["attribution"]["signed_attribution"]
        )
        off = float(responses["off"]["attribution"]["signed_attribution"])
        wrong = float(
            responses["wrong_axis"]["attribution"]["signed_attribution"]
        )
        signed += positive > negative
        specific += abs(positive - negative) > abs(wrong - off)
        if "neutral_center" in responses:
            center = float(
                responses["neutral_center"]["attribution"][
                    "signed_attribution"
                ]
            )
            strict_center += negative < center < positive
    values = aggregate[axis]
    center_drift: float | None = None
    if "neutral_center" in values:
        center_drift = float(
            values["neutral_center"]["mean_attribution"]
        ) - float(values["off"]["mean_attribution"])
    return {
        "generated_span": generated,
        "explicit_span": explicit,
        "relative_span": generated / explicit,
        "signed_prompts": signed,
        "specific_prompts": specific,
        "strict_center_prompts": (
            strict_center if "neutral_center" in values else None
        ),
        "prompts": len(axis_rows),
        "wrong_axis_relative_shift": abs(
            float(values["wrong_axis"]["mean_attribution"])
            - float(values["off"]["mean_attribution"])
        )
        / explicit,
        "center_drift": center_drift,
        "absolute_center_drift_fraction": (
            abs(center_drift) / explicit if center_drift is not None else None
        ),
    }


def report_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        axis: axis_response_metrics(report, axis)
        for axis in report["axes"]
    }


def strength_sweep_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = report["aggregate"][report["axis"]]
    explicit = float(
        aggregate["explicit_positive"]["mean_attribution"]
    ) - float(aggregate["explicit_negative"]["mean_attribution"])
    state_values = {
        condition: float(values["mean_attribution"])
        for condition, values in aggregate.items()
        if condition.startswith("state_")
    }
    direction_span = float(
        aggregate["direction_only_positive"]["mean_attribution"]
    ) - float(
        aggregate["direction_only_negative"]["mean_attribution"]
    )
    off = float(aggregate["off"]["mean_attribution"])
    center = state_values["state_+0"]
    return {
        "explicit_span": explicit,
        "state_attribution": state_values,
        "center_drift_fraction": abs(center - off) / explicit,
        "direction_only_span": direction_span,
        "direction_only_relative_span": direction_span / explicit,
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "sample_stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def consolidate_metrics(
    seed_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not seed_records:
        raise ValueError("no seed records to consolidate")
    summary: dict[str, Any] = {"stage_h": {}, "stage_i": {}}
    for stage in ("stage_h", "stage_i"):
        for axis in STYLE_AXIS_NAMES:
            rows = [record[stage]["metrics"][axis] for record in seed_records]
            summary[stage][axis] = {
                "relative_span": _distribution(
                    [float(row["relative_span"]) for row in rows]
                ),
                "generated_span": _distribution(
                    [float(row["generated_span"]) for row in rows]
                ),
                "signed_prompts": sum(
                    int(row["signed_prompts"]) for row in rows
                ),
                "specific_prompts": sum(
                    int(row["specific_prompts"]) for row in rows
                ),
                "total_prompts": sum(int(row["prompts"]) for row in rows),
                "per_seed": rows,
            }
            if stage == "stage_i":
                summary[stage][axis]["absolute_center_drift_fraction"] = (
                    _distribution(
                        [
                            float(row["absolute_center_drift_fraction"])
                            for row in rows
                        ]
                    )
                )
                summary[stage][axis]["strict_center_prompts"] = sum(
                    int(row["strict_center_prompts"]) for row in rows
                )

    warmth_h = summary["stage_h"]["warmth"]
    patience_h = summary["stage_h"]["patience"]
    patience_i = summary["stage_i"]["patience"]
    center_passes = {
        axis: (
            summary["stage_i"][axis]["absolute_center_drift_fraction"][
                "median"
            ]
            < 0.10
        )
        for axis in STYLE_AXIS_NAMES
    }
    decisions = {
        "stage_h_warmth": {
            "passed": (
                warmth_h["relative_span"]["median"] >= 0.60
                and warmth_h["signed_prompts"] >= 21
                and warmth_h["specific_prompts"] >= 21
            ),
            "relative_span_median_at_least_0.60": (
                warmth_h["relative_span"]["median"] >= 0.60
            ),
            "signed_at_least_21_of_24": warmth_h["signed_prompts"] >= 21,
            "specific_at_least_21_of_24": (
                warmth_h["specific_prompts"] >= 21
            ),
        },
        "stage_i_patience": {
            "passed": (
                patience_i["relative_span"]["median"] >= 0.20
                and patience_i["relative_span"]["median"]
                > patience_h["relative_span"]["median"]
            ),
            "relative_span_median_at_least_0.20": (
                patience_i["relative_span"]["median"] >= 0.20
            ),
            "exceeds_stage_h": (
                patience_i["relative_span"]["median"]
                > patience_h["relative_span"]["median"]
            ),
        },
        "stage_i_neutral_center": {
            "passed": all(center_passes.values()),
            "axis_passes": center_passes,
        },
    }
    summary["decisions"] = decisions
    return summary
