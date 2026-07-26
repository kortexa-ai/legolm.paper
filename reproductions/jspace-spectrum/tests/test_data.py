from jspace_spectrum.data import (
    ATLAS_FAMILIES,
    AXES,
    build_cases,
    validate_data,
)
from jspace_spectrum.experiment import suite_config


def test_frozen_inventory_is_complete_and_disjoint() -> None:
    report = validate_data()
    assert report["sha256"] == (
        "a9b52273f23fcff7845e1d3e49bdfc7f34d8262ed1814be2800650348329deb4"
    )
    assert report["axes"] == 12
    assert report["landmark_cases"] == 96
    assert report["atlas_families"] == 21
    assert report["atlas_cases"] == 126
    assert report["evaluation_passes_per_model"] == 666
    assert report["pole_word_collisions"] == {}


def test_case_shapes_match_method_contract() -> None:
    cases = build_cases()
    assert len(AXES) == 12
    assert all(len(axis.positive_words) == 6 for axis in AXES)
    assert all(len(axis.negative_words) == 6 for axis in AXES)
    assert all(len(rows) == 6 for rows in ATLAS_FAMILIES.values())
    assert len({case.case_id for case in cases}) == len(cases)


def test_smoke_suite_keeps_every_axis_and_required_atlas_groups() -> None:
    config = suite_config("smoke")
    landmark_pairs = {
        (case.axis, case.pole) for case in config["cases"] if case.kind == "landmark"
    }
    assert len(landmark_pairs) == 24
    assert sum(case.group == "meh" for case in config["cases"]) == 6
    assert sum(case.group == "neutral" for case in config["cases"]) == 6
    assert len(config["system_prompts"]) == 2
