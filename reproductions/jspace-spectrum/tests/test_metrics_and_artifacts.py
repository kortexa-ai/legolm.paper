from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jspace_spectrum.data import AXES, SYSTEM_PROMPTS, build_cases
from jspace_spectrum.figures import render_figures
from jspace_spectrum.metrics import (
    clustered_bootstrap,
    compare_models,
    summarize_measurements,
)
from jspace_spectrum.model import QWEN36_35B
from jspace_spectrum.viewer import render_viewer, summarize_groups


AXIS_NAMES = [axis.name for axis in AXES]


def synthetic_rows() -> list[dict]:
    rows = []
    trace_layers = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
    atlas_index = {
        group: index
        for index, group in enumerate(
            sorted({case.group for case in build_cases() if case.kind == "atlas"})
        )
    }
    for case in build_cases():
        for system_index, system in enumerate(SYSTEM_PROMPTS):
            vector = np.zeros(len(AXIS_NAMES), dtype=float)
            if case.kind == "landmark":
                index = AXIS_NAMES.index(case.axis)
                vector[index] = 4.0 if case.pole == "positive" else -4.0
            elif case.group == "meh":
                vector[AXIS_NAMES.index("engagement")] = -5.0
                vector[AXIS_NAMES.index("care")] = -1.0
            elif case.group != "neutral":
                index = atlas_index[case.group] % len(AXIS_NAMES)
                vector[index] = 1.5 + atlas_index[case.group] / 50
                vector[(index + 3) % len(AXIS_NAMES)] = -0.75
            vector += system_index * 0.001
            token_values = [vector * 0.9, vector * 1.1]
            by_layer = {
                str(layer): (vector * ((layer + 1) / 40)).round(6).tolist()
                for layer in trace_layers
            }
            token_by_layer = [
                {
                    str(layer): (values * ((layer + 1) / 40)).round(6).tolist()
                    for layer in trace_layers
                }
                for values in token_values
            ]
            rows.append(
                {
                    "id": f"{case.case_id}--{system}",
                    "case_id": case.case_id,
                    "kind": case.kind,
                    "group": case.group,
                    "text": case.text,
                    "variant": case.variant,
                    "axis": case.axis,
                    "pole": case.pole,
                    "subset": case.subset,
                    "system": system,
                    "context_tokens": 12,
                    "user_tokens": 2,
                    "utterance": vector.round(6).tolist(),
                    "frontier": token_values[-1].round(6).tolist(),
                    "utterance_by_layer": by_layer,
                    "frontier_by_layer": {
                        layer: token_by_layer[-1][layer] for layer in by_layer
                    },
                    "tokens": [
                        {
                            "index": index,
                            "position": 5 + index,
                            "token_id": 100 + index,
                            "text": text,
                            "label": text,
                            "coordinates": values.round(6).tolist(),
                            "coordinates_by_layer": token_by_layer[index],
                        }
                        for index, (text, values) in enumerate(
                            zip(("▁synthetic", "▁case"), token_values, strict=True)
                        )
                    ],
                }
            )
    return rows


def synthetic_summary() -> tuple[dict, list[dict]]:
    rows = synthetic_rows()
    metrics = summarize_measurements(
        rows,
        axis_names=AXIS_NAMES,
        trace_layers=QWEN36_35B.trace_layers,
        bootstrap_samples=50,
    )
    summary = {
        "format": "jspace-spectrum-experiment-v1",
        "model": {
            "name": QWEN36_35B.name,
            "source_layer": QWEN36_35B.source_layer,
            "target_layer": QWEN36_35B.target_layer,
        },
        "metrics": metrics,
    }
    return summary, rows


def test_clustered_bootstrap_is_deterministic() -> None:
    rows = [
        {"case_id": "a", "utterance": [1.0, -1.0]},
        {"case_id": "a", "utterance": [1.2, -0.8]},
        {"case_id": "b", "utterance": [2.0, -2.0]},
        {"case_id": "b", "utterance": [2.2, -1.8]},
    ]
    assert clustered_bootstrap(rows, samples=40, seed=7) == clustered_bootstrap(
        rows,
        samples=40,
        seed=7,
    )


def test_frozen_decisions_pass_on_separable_synthetic_data() -> None:
    summary, _rows = synthetic_summary()
    assert summary["metrics"]["decisions"] == {
        "heldout_orientation": True,
        "meh_non_null": True,
        "meh_boredom_adjacent": True,
        "template_stability": True,
    }


def test_figures_and_viewer_regenerate_from_saved_artifacts(tmp_path: Path) -> None:
    summary, rows = synthetic_summary()
    outputs = render_figures(summary, tmp_path / "figures")
    assert len(outputs) == 4
    for output in outputs:
        assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert output.stat().st_size > 10_000

    recording = {
        "format": "jspace-spectrum-recording-v1",
        "status": "complete",
        "model": {
            "id": QWEN36_35B.model_id,
            "device": "cuda",
        },
        "lens": {
            "source_layer": QWEN36_35B.source_layer,
            "trace_layers": list(QWEN36_35B.trace_layers),
        },
        "axis_names": AXIS_NAMES,
        "axis_poles": [
            {
                "axis": axis.name,
                "positive": axis.positive_label,
                "negative": axis.negative_label,
            }
            for axis in AXES
        ],
        "cases": rows,
        "groups": summarize_groups(rows),
    }
    output = render_viewer(recording, tmp_path / "viewer.html")
    text = output.read_text()
    assert "__JSPACE_RECORDING__" not in text
    assert "Where is “meh”?" in text
    assert "\\u003c" not in json.dumps(recording)
    assert len(text) > 50_000


def test_model_comparison_reports_family_alignment() -> None:
    summary, _rows = synthetic_summary()
    comparison = compare_models(summary["metrics"], summary["metrics"])
    assert comparison["family_cosine_median"] == 1.0
    assert comparison["family_sign_agreement_mean"] == 1.0
    assert comparison["meh"]["cosine"] == 1.0
