"""Artifact-driven paper figures."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402

from .data import AXES


INK = "#17202a"
MUTED = "#667085"
GRID = "#d0d5dd"
BLUE = "#2762a8"
TEAL = "#16837a"
CORAL = "#d05a47"
GOLD = "#b27b16"
BACKGROUND = "#fbfaf7"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
        }
    )


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.16)
    plt.close(figure)


def _metrics(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("summary does not contain metrics")
    return metrics


def plot_landmark_separation(
    summary: Mapping[str, Any],
    path: Path,
) -> None:
    metrics = _metrics(summary)
    rows = metrics["landmarks"]["separations"]
    labels = [row["axis"].replace("_", " ") for row in rows]
    means = np.asarray([row["target_mean"] for row in rows], dtype=float)
    lower = np.asarray([row["target_lower"] for row in rows], dtype=float)
    upper = np.asarray([row["target_upper"] for row in rows], dtype=float)
    order = np.arange(len(rows))[::-1]
    colors = [TEAL if value > 0 else CORAL for value in means]

    figure, axis = plt.subplots(figsize=(8.8, 5.8))
    axis.axvline(0, color=INK, linewidth=0.9, alpha=0.65)
    axis.barh(
        order,
        means,
        xerr=np.vstack((means - lower, upper - means)),
        color=colors,
        alpha=0.88,
        edgecolor="none",
        error_kw={"elinewidth": 1, "ecolor": INK, "capsize": 2},
    )
    axis.set_yticks(order, labels)
    axis.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)
    axis.set_xlabel("held-out positive minus negative mean (calibrated units)")
    passed = metrics["landmarks"]["orientation_pass"]
    positive = metrics["landmarks"]["positive_target_lowers"]
    axis.set_title("Held-out landmarks orient the twelve fitted directions", loc="left")
    axis.text(
        0,
        0.985,
        f"{positive}/12 lower intervals above zero · decision: "
        f"{'pass' if passed else 'fail'}",
        transform=axis.transAxes,
        color=MUTED,
        va="top",
    )
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    _save(figure, path)


def _closed(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.concatenate((array, array[:1]))


def plot_meh_radar(summary: Mapping[str, Any], path: Path) -> None:
    metrics = _metrics(summary)
    names = list(metrics["axis_names"])
    meh = np.asarray(metrics["meh"]["mean"]["mean"], dtype=float)
    neutral = np.asarray(metrics["meh"]["neutral"]["mean"], dtype=float)
    delta = meh - neutral
    maximum = max(4.0, float(np.max(np.abs(np.concatenate((meh, neutral))))))
    limit = math.ceil(maximum)
    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    closed_angles = np.concatenate((angles, angles[:1]))
    labels = [f"{axis.positive_label}\n{axis.negative_label}" for axis in AXES]

    figure = plt.figure(figsize=(12.4, 6.7))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.25, 0.9),
        wspace=0.32,
        left=0.055,
        right=0.98,
        bottom=0.08,
        top=0.79,
    )
    radar = figure.add_subplot(grid[0, 0], projection="polar")
    radar.set_theta_offset(np.pi / 2)
    radar.set_theta_direction(-1)
    radar.set_xticks(angles, labels)
    radar.tick_params(axis="x", pad=10, labelsize=8.2)
    radar.set_ylim(0, 2 * limit)
    ticks = np.linspace(-limit, limit, 5)
    radar.set_yticks(ticks + limit, [f"{value:+.0f}" for value in ticks])
    radar.set_rlabel_position(10)
    radar.grid(color=GRID, linewidth=0.7, alpha=0.8)
    radar.spines["polar"].set_color(GRID)
    radar.plot(
        closed_angles,
        _closed(meh + limit),
        color=CORAL,
        linewidth=2.3,
        label="meh",
    )
    radar.fill(
        closed_angles,
        _closed(meh + limit),
        color=CORAL,
        alpha=0.14,
    )
    radar.plot(
        closed_angles,
        _closed(neutral + limit),
        color=BLUE,
        linewidth=1.8,
        label="neutral",
    )
    radar.plot(
        closed_angles,
        np.full(len(names) + 1, limit),
        color=INK,
        linewidth=0.9,
        linestyle=(0, (3, 3)),
        alpha=0.75,
    )
    radar.legend(
        loc="upper right",
        bbox_to_anchor=(1.18, 1.08),
        frameon=False,
    )
    radar.set_title("The signed coordinate of “meh”", pad=27)

    bars = figure.add_subplot(grid[0, 1])
    y = np.arange(len(names))[::-1]
    colors = [CORAL if value < 0 else TEAL for value in delta]
    bars.barh(y, delta, color=colors, alpha=0.88)
    bars.axvline(0, color=INK, linewidth=0.9)
    delta_bound = max(1.0, float(np.max(np.abs(delta))) * 1.12)
    bars.set_xlim(-delta_bound, delta_bound)
    bars.set_yticks(y, [name.replace("_", " ") for name in names])
    bars.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.7)
    bars.set_axisbelow(True)
    bars.set_xlabel("meh minus neutral")
    nearest = metrics["meh"]["nearest_landmarks"][0]
    bars.set_title("Contrast with neutral utterances", loc="left")
    bars.text(
        0,
        0.985,
        f"nearest landmark: {nearest['group']} · distance {nearest['euclidean']:.2f}",
        transform=bars.transAxes,
        color=MUTED,
        va="top",
    )
    for spine in ("top", "right", "left"):
        bars.spines[spine].set_visible(False)
    figure.suptitle(
        "A short shrug occupies a reproducible region of the fitted spectrum",
        x=0.045,
        y=0.97,
        ha="left",
        fontsize=15,
        color=INK,
    )
    _save(figure, path)


def plot_atlas_heatmap(summary: Mapping[str, Any], path: Path) -> None:
    metrics = _metrics(summary)
    families = metrics["atlas"]["families"]
    names = list(metrics["axis_names"])
    values = np.asarray([row["mean"] for row in families], dtype=float)
    bound = max(1.0, float(np.quantile(np.abs(values), 0.97)))

    figure, axis = plt.subplots(figsize=(11.2, 7.7))
    image = axis.imshow(
        values,
        aspect="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
        interpolation="nearest",
    )
    axis.set_xticks(
        np.arange(len(names)),
        [name.replace("_", " ") for name in names],
        rotation=42,
        ha="right",
    )
    axis.set_yticks(
        np.arange(len(families)),
        [row["group"] for row in families],
    )
    axis.set_title("Social-term atlas across the twelve signed directions", loc="left")
    axis.text(
        0,
        0.985,
        "family centroids; blue is negative, red is positive",
        transform=axis.transAxes,
        color=MUTED,
        va="top",
    )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.84, pad=0.02)
    colorbar.set_label("neutral-calibrated units")
    axis.set_xticks(
        np.arange(-0.5, len(names), 1),
        minor=True,
    )
    axis.set_yticks(
        np.arange(-0.5, len(families), 1),
        minor=True,
    )
    axis.grid(which="minor", color=BACKGROUND, linewidth=0.8)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    _save(figure, path)


def plot_meh_depth(summary: Mapping[str, Any], path: Path) -> None:
    metrics = _metrics(summary)
    names = list(metrics["axis_names"])
    depth_rows = metrics["depth"]["layers"]
    layers = np.asarray([row["layer"] for row in depth_rows], dtype=int)
    deltas = np.asarray(
        [row["meh_minus_neutral"] for row in depth_rows],
        dtype=float,
    )
    selected = ("engagement", "care", "patience", "playfulness")
    colors = (CORAL, BLUE, GOLD, TEAL)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.8, 6.6),
        sharex=True,
        constrained_layout=True,
    )
    for plot, name, color in zip(axes.flat, selected, colors, strict=True):
        index = names.index(name)
        values = deltas[:, index]
        plot.plot(layers, values, color=color, marker="o", linewidth=2)
        plot.axhline(0, color=INK, linewidth=0.8, alpha=0.65)
        plot.axvline(
            summary["model"]["source_layer"],
            color=MUTED,
            linewidth=0.8,
            linestyle=(0, (3, 3)),
        )
        plot.fill_between(layers, 0, values, color=color, alpha=0.11)
        plot.grid(color=GRID, linewidth=0.7, alpha=0.7)
        plot.set_title(name.replace("_", " "), loc="left")
        plot.set_ylabel("meh minus neutral")
        for spine in ("top", "right"):
            plot.spines[spine].set_visible(False)
    for plot in axes[-1]:
        plot.set_xlabel("residual layer")
    figure.suptitle(
        "The “meh” contrast develops through residual depth",
        x=0.01,
        ha="left",
        fontsize=15,
        color=INK,
    )
    _save(figure, path)


def render_figures(summary: Mapping[str, Any], output_dir: Path) -> list[Path]:
    _style()
    outputs = [
        output_dir / "landmark-separation.png",
        output_dir / "meh-radar.png",
        output_dir / "atlas-heatmap.png",
        output_dir / "meh-depth.png",
    ]
    plot_landmark_separation(summary, outputs[0])
    plot_meh_radar(summary, outputs[1])
    plot_atlas_heatmap(summary, outputs[2])
    plot_meh_depth(summary, outputs[3])
    return outputs
