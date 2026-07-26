"""Paper figures rendered only from the consolidated reproduction artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .data import STYLE_AXIS_NAMES
from .metrics import explicit_span


COLORS = {
    "stage_h": "#2563eb",
    "stage_i": "#16a34a",
    "warmth": "#2563eb",
    "patience": "#7c3aed",
    "goodwill": "#16a34a",
    "wrong": "#f59e0b",
    "neutral": "#94a3b8",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _seed_report(
    summary_path: Path,
    record: Mapping[str, Any],
    stage: str,
) -> dict[str, Any]:
    return _read(summary_path.parent / record[stage]["responses"])


def _save(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")


def plot_relative_spans(
    summary: Mapping[str, Any],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, panel = plt.subplots(figsize=(9.4, 5.2))
    x = np.arange(len(STYLE_AXIS_NAMES))
    width = 0.34
    for stage_index, stage in enumerate(("stage_h", "stage_i")):
        positions = x + (stage_index - 0.5) * width
        medians = [
            100.0
            * summary["consolidated"][stage][axis]["relative_span"]["median"]
            for axis in STYLE_AXIS_NAMES
        ]
        panel.bar(
            positions,
            medians,
            width,
            color=COLORS[stage],
            alpha=0.82,
            label="Stage H: axis center ± direction"
            if stage == "stage_h"
            else "Stage I: shared center + direction",
        )
        for axis_index, axis in enumerate(STYLE_AXIS_NAMES):
            values = [
                100.0 * float(row["relative_span"])
                for row in summary["consolidated"][stage][axis]["per_seed"]
            ]
            offsets = np.linspace(-0.055, 0.055, len(values))
            panel.scatter(
                positions[axis_index] + offsets,
                values,
                color="#111827",
                marker="o" if stage == "stage_h" else "D",
                s=24,
                zorder=4,
            )
            panel.text(
                positions[axis_index],
                medians[axis_index]
                + (2.0 if medians[axis_index] >= 0 else -3.5),
                f"{medians[axis_index]:.1f}%",
                ha="center",
                va="bottom" if medians[axis_index] >= 0 else "top",
            )
    panel.axhline(0.0, color="#475569", linewidth=1.0)
    panel.set_xticks(x, [axis.capitalize() for axis in STYLE_AXIS_NAMES])
    panel.set_ylabel("generated span as % of explicit-prompt span")
    panel.set_title("Fresh-prefix response control across seeds")
    panel.grid(axis="y", alpha=0.2)
    panel.legend(frameon=False, loc="upper right")
    figure.tight_layout()
    _save(figure, output)
    plt.close(figure)


def plot_warmth_sweeps(
    summary_path: Path,
    summary: Mapping[str, Any],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, panel = plt.subplots(figsize=(9.4, 5.2))
    states = None
    normalized_rows = []
    for record in summary["seed_records"]:
        sweep_path = summary_path.parent / record["stage_h"]["warmth_sweep"]
        sweep = _read(sweep_path)
        aggregate = sweep["aggregate"]["warmth"]
        off = float(aggregate["off"]["mean_attribution"])
        explicit = explicit_span(sweep["aggregate"], "warmth")
        states = [float(state) for state in sweep["states"]]
        values = [
            100.0
            * (
                float(
                    aggregate[f"state_{state:+g}"]["mean_attribution"]
                )
                - off
            )
            / explicit
            for state in states
        ]
        normalized_rows.append(values)
        panel.plot(
            states,
            values,
            color=COLORS["warmth"],
            alpha=0.35,
            linewidth=1.5,
            marker="o",
            markersize=3,
            label=f"seed {record['seed']}",
        )
    assert states is not None
    medians = np.median(np.asarray(normalized_rows), axis=0)
    panel.plot(
        states,
        medians,
        color="#111827",
        linewidth=2.8,
        marker="o",
        label="median",
        zorder=4,
    )
    panel.axhline(0.0, color="#475569", linewidth=1.0)
    panel.axvline(0.0, color="#94a3b8", linewidth=1.0)
    panel.set_xticks(states)
    panel.set_xlabel("signed warmth state")
    panel.set_ylabel("shift from regular model (% of explicit span)")
    panel.set_title("Stage H warmth is tested as a trajectory")
    panel.grid(alpha=0.2)
    panel.legend(frameon=False, ncol=2)
    figure.tight_layout()
    _save(figure, output)
    plt.close(figure)


def plot_neutral_anchor(
    summary_path: Path,
    summary: Mapping[str, Any],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, panels = plt.subplots(
        1,
        len(STYLE_AXIS_NAMES),
        figsize=(12.6, 4.7),
        sharey=True,
    )
    positions = np.arange(3)
    for panel, axis in zip(panels, STYLE_AXIS_NAMES, strict=True):
        seed_values = []
        for record in summary["seed_records"]:
            report = _seed_report(summary_path, record, "stage_i")
            aggregate = report["aggregate"][axis]
            off = float(aggregate["off"]["mean_attribution"])
            explicit = explicit_span(report["aggregate"], axis)
            seed_values.append(
                [
                    100.0
                    * (
                        float(
                            aggregate["prefix_negative"]["mean_attribution"]
                        )
                        - off
                    )
                    / explicit,
                    100.0
                    * (
                        float(
                            aggregate["neutral_center"]["mean_attribution"]
                        )
                        - off
                    )
                    / explicit,
                    100.0
                    * (
                        float(
                            aggregate["prefix_positive"]["mean_attribution"]
                        )
                        - off
                    )
                    / explicit,
                ]
            )
        values = np.asarray(seed_values)
        medians = np.median(values, axis=0)
        panel.plot(
            positions,
            medians,
            color=COLORS[axis],
            linewidth=2.5,
            marker="o",
        )
        for seed_index, row in enumerate(values):
            panel.scatter(
                positions
                + (seed_index - (len(values) - 1) / 2.0) * 0.035,
                row,
                color="#111827",
                s=20,
                alpha=0.72,
                zorder=4,
            )
        panel.axhline(0.0, color="#475569", linewidth=1.0)
        panel.set_xticks(
            positions,
            ("negative", "center", "positive"),
            rotation=15,
        )
        panel.set_title(axis.capitalize())
        panel.grid(axis="y", alpha=0.2)
    panels[0].set_ylabel("shift from regular model (% of explicit span)")
    figure.suptitle("Stage I uses one continuous prefix path")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    _save(figure, output)
    plt.close(figure)


def plot_six_pole_radar(
    summary_path: Path,
    summary: Mapping[str, Any],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ("warm", "patient", "goodwill", "cold / hostile", "annoyed", "resentful")
    reaches = []
    for record in summary["seed_records"]:
        report = _seed_report(summary_path, record, "stage_h")
        aggregate = report["aggregate"]
        positive = []
        negative = []
        for axis in STYLE_AXIS_NAMES:
            values = aggregate[axis]
            off = float(values["off"]["mean_attribution"])
            explicit_positive = (
                float(values["explicit_positive"]["mean_attribution"]) - off
            )
            explicit_negative = (
                off - float(values["explicit_negative"]["mean_attribution"])
            )
            positive.append(
                (
                    float(values["prefix_positive"]["mean_attribution"]) - off
                )
                / explicit_positive
                if explicit_positive > 0
                else 0.0
            )
            negative.append(
                (
                    off
                    - float(values["prefix_negative"]["mean_attribution"])
                )
                / explicit_negative
                if explicit_negative > 0
                else 0.0
            )
        reaches.append([*positive, *negative])
    median = np.median(np.asarray(reaches), axis=0)
    values = np.clip(median, 0.0, 1.25)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    closed_angles = np.append(angles, angles[0])
    closed_values = np.append(values, values[0])
    figure = plt.figure(figsize=(7.2, 6.2))
    panel = figure.add_subplot(111, polar=True)
    panel.set_theta_offset(np.pi / 2)
    panel.set_theta_direction(-1)
    panel.plot(
        closed_angles,
        closed_values,
        color=COLORS["stage_h"],
        linewidth=2.5,
        marker="o",
        label="learned prefix",
    )
    panel.fill(closed_angles, closed_values, color=COLORS["stage_h"], alpha=0.15)
    panel.scatter(
        [0.0],
        [0.0],
        color="#111827",
        marker="s",
        s=34,
        zorder=5,
        label="regular model (0% shift)",
    )
    panel.plot(
        closed_angles,
        np.ones_like(closed_angles),
        color="#64748b",
        linewidth=1.5,
        linestyle="--",
        label="explicit style prompt (100%)",
    )
    panel.set_xticks(angles, labels)
    panel.set_ylim(0.0, 1.25)
    panel.set_yticks((0.25, 0.5, 0.75, 1.0), ("25%", "50%", "75%", "100%"))
    panel.set_title("Stage H median pole reach")
    panel.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.18, -0.08))
    figure.tight_layout()
    _save(figure, output)
    plt.close(figure)


def render_figures(summary_path: Path, output_dir: Path) -> dict[str, str]:
    summary = _read(summary_path)
    paths = {
        "relative_spans": output_dir / "relative-spans.png",
        "warmth_sweeps": output_dir / "warmth-sweeps.png",
        "neutral_anchor": output_dir / "neutral-anchor.png",
        "six_pole_radar": output_dir / "six-pole-radar.png",
    }
    plot_relative_spans(summary, paths["relative_spans"])
    plot_warmth_sweeps(summary_path, summary, paths["warmth_sweeps"])
    plot_neutral_anchor(summary_path, summary, paths["neutral_anchor"])
    plot_six_pole_radar(summary_path, summary, paths["six_pole_radar"])
    return {
        name: str(path.relative_to(summary_path.parent))
        for name, path in paths.items()
    }
