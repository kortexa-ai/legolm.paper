import numpy as np
import pytest

from legolm_lookout.analysis import (
    EvidenceError, cuts_to_wakes, detector_gap, estimate, paired_values, parsed_action,
)


def image(i, action, verdict="present"):
    return {"image_id": i, "label": "present", "action": action, "verdict": verdict}


def test_pairing_aligns_identities_and_preserves_paired_difference():
    record = {"arms": {"caution": {
        "packet": [image(1, "stop"), image(2, "proceed")],
        "habit": [image(2, "proceed"), image(1, "stop")],
    }}}
    values, keys = paired_values([record], "caution", {"packet": 1, "habit": -1})
    assert keys == [(1, "present"), (2, "present")]
    assert np.array_equal(values, [[0, 0]])


def test_pairing_rejects_changed_population_and_duplicate_observation():
    record = {"arms": {"caution": {
        "packet": [image(1, "stop"), image(2, "proceed")],
        "habit": [image(1, "stop"), image(3, "proceed")],
    }}}
    with pytest.raises(EvidenceError, match="population"):
        paired_values([record], "caution", {"packet": 1, "habit": -1})
    record["arms"]["caution"]["habit"] = [image(1, "stop"), image(1, "proceed")]
    with pytest.raises(EvidenceError, match="duplicate"):
        paired_values([record], "caution", {"packet": 1, "habit": -1})


def test_duplicating_a_seed_does_not_create_independent_images():
    x = np.asarray([[1, 0, 1, 0, 0]], dtype=float)
    one = estimate(x, key="same", replicates=1000)
    three = estimate(np.repeat(x, 3, axis=0), key="same", replicates=1000)
    assert one["mean"] == three["mean"]
    assert one["interval"] == three["interval"]
    assert one["n_units"] == three["n_units"] == 5


def test_cuts_include_unresolved_endings_and_exclude_initial_starts():
    rows = [
        {"episode": 0, "step": 0, "cut": True, "woke": True},
        {"episode": 0, "step": 1, "cut": True, "woke": False},
        {"episode": 0, "step": 2, "cut": False, "woke": False},
        {"episode": 0, "step": 3, "cut": False, "woke": True},
        {"episode": 0, "step": 4, "cut": True, "woke": False},
        {"episode": 0, "step": 5, "cut": False, "woke": False},
    ]
    result = cuts_to_wakes(rows)
    assert result["episode_starts"] == 1
    assert result["post_start_cuts"] == 2
    assert result["observed_wakes"] == result["right_censored"] == 1
    assert result["observed_mean"] == 2
    assert result["censored_duration"] == [1]


def test_detector_gap_uses_cost_on_actual_errors():
    cells = {
        "oracle": [image(1, "stop"), image(2, "stop")],
        "classifier": [image(1, "proceed", "absent"), image(2, "stop")],
        "flipped": [image(1, "stop", "absent"), image(2, "proceed")],
    }
    result = detector_gap(cells, "caution")
    assert result["oracle_minus_detector"] == 0.5
    assert result["error_rate"] == 0.5
    assert result["error_conditional_cost"] == 1
    assert result["absolute_gap_upper_bound"] == 0.5
    # A population-wide mean cost of .5 would predict .25, an invalid bound.
    assert result["oracle_minus_detector"] > result["error_rate"] * 0.5


def test_detector_control_must_reproduce_the_selected_packet():
    cells = {"oracle": [image(1, "stop")],
             "classifier": [image(1, "proceed")],
             "flipped": [image(1, "proceed")]}
    with pytest.raises(EvidenceError, match="selected packet"):
        detector_gap(cells, "caution")


def test_parser_prefers_valid_json_over_earlier_action_words():
    assert parsed_action('stop now. {"action":"greet"}') == ("greet", "json")
    assert parsed_action('{"action":"unknown"} PROCEED') == ("proceed", "word")
    assert parsed_action("no permitted action") == ("other", "fallback")


def test_last_step_censoring_is_not_a_zero_delay_response():
    result = cuts_to_wakes([
        {"episode": 0, "step": 0, "cut": True, "woke": True},
        {"episode": 0, "step": 1, "cut": True, "woke": False},
    ])
    assert result["observed_wakes"] == 0
    assert result["observed_mean"] is None
    assert result["censored_duration"] == [0]
