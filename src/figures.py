from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#666666",
    "light_gray": "#E8E8E8",
    "dark": "#222222",
}

W, H = 1800, 1100
MARGIN = 140
PLOT_TOP = 210
PLOT_BOTTOM = 850
PLOT_LEFT = 190
PLOT_RIGHT = 1660


def load_summary(path: Path) -> dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], content: str, size: int = 34, fill: str = PALETTE["dark"], bold: bool = False, anchor: str | None = None) -> None:
    draw.text(xy, content, fill=fill, font=font(size, bold), anchor=anchor)


def result(payload: dict[str, Any], name: str) -> dict[str, Any] | None:
    entry = payload.get("results", {}).get(name)
    return entry.get("result") if entry else None


def improvement(payload: dict[str, Any], name: str) -> float | None:
    item = result(payload, name)
    if not item:
        return None
    value = item.get("improvement")
    return float(value) if value is not None else None


def blank(title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    text(draw, (MARGIN, 80), title, 52, bold=True)
    if subtitle:
        text(draw, (MARGIN, 145), subtitle, 28, fill=PALETTE["gray"])
    return img, draw


def axes(draw: ImageDraw.ImageDraw, ymin: float, ymax: float, ylabel: str) -> None:
    draw.line((PLOT_LEFT, PLOT_TOP, PLOT_LEFT, PLOT_BOTTOM), fill=PALETTE["dark"], width=4)
    draw.line((PLOT_LEFT, PLOT_BOTTOM, PLOT_RIGHT, PLOT_BOTTOM), fill=PALETTE["dark"], width=4)
    for i in range(6):
        frac = i / 5
        y = int(PLOT_BOTTOM - frac * (PLOT_BOTTOM - PLOT_TOP))
        val = ymin + frac * (ymax - ymin)
        draw.line((PLOT_LEFT - 10, y, PLOT_RIGHT, y), fill=PALETTE["light_gray"], width=2)
        text(draw, (PLOT_LEFT - 20, y), f"{val:.4f}", 24, fill=PALETTE["gray"], anchor="rm")
    text(draw, (55, (PLOT_TOP + PLOT_BOTTOM) // 2), ylabel, 28, fill=PALETTE["gray"], anchor="mm")


def bar_chart(path: Path, title: str, subtitle: str, labels: list[str], values: list[float], colors: list[str], ylabel: str = "BPB improvement ↑", zero: bool = True) -> None:
    img, draw = blank(title, subtitle)
    finite = [v for v in values if v is not None]
    ymin = min(0.0 if zero else min(finite), min(finite))
    ymax = max(finite) * 1.15 if max(finite) > 0 else max(finite) * 0.85
    if ymax == ymin:
        ymax = ymin + 1
    axes(draw, ymin, ymax, ylabel)
    span = ymax - ymin
    n = len(values)
    slot = (PLOT_RIGHT - PLOT_LEFT) / max(1, n)
    zero_y = int(PLOT_BOTTOM - ((0 - ymin) / span) * (PLOT_BOTTOM - PLOT_TOP))
    if ymin < 0 < ymax:
        draw.line((PLOT_LEFT, zero_y, PLOT_RIGHT, zero_y), fill=PALETTE["dark"], width=2)
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        cx = int(PLOT_LEFT + slot * (i + 0.5))
        bw = int(slot * 0.58)
        y = int(PLOT_BOTTOM - ((value - ymin) / span) * (PLOT_BOTTOM - PLOT_TOP))
        top, bottom = min(y, zero_y), max(y, zero_y)
        draw.rounded_rectangle((cx - bw // 2, top, cx + bw // 2, bottom), radius=10, fill=color)
        text(draw, (cx, top - 12 if value >= 0 else bottom + 12), f"{value:+.4f}", 25, bold=True, anchor="ms" if value >= 0 else "mt")
        for j, part in enumerate(label.split("\n")):
            text(draw, (cx, PLOT_BOTTOM + 45 + j * 32), part, 25, anchor="mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, dpi=(300, 300))


def line_chart(path: Path, title: str, subtitle: str, xs: list[float], ys: list[float], ylabel: str, xlabel: str) -> None:
    img, draw = blank(title, subtitle)
    ymin, ymax = min(ys), max(ys)
    pad = (ymax - ymin) * 0.15 or 1.0
    ymin -= pad
    ymax += pad
    axes(draw, ymin, ymax, ylabel)
    xmin, xmax = min(xs), max(xs)
    points = []
    for x, yv in zip(xs, ys):
        px = int(PLOT_LEFT + ((x - xmin) / (xmax - xmin or 1)) * (PLOT_RIGHT - PLOT_LEFT))
        py = int(PLOT_BOTTOM - ((yv - ymin) / (ymax - ymin)) * (PLOT_BOTTOM - PLOT_TOP))
        points.append((px, py))
    if len(points) > 1:
        draw.line(points, fill=PALETTE["blue"], width=6)
    for x, yv, pt in zip(xs, ys, points):
        draw.ellipse((pt[0] - 13, pt[1] - 13, pt[0] + 13, pt[1] + 13), fill=PALETTE["orange"], outline=PALETTE["dark"], width=3)
        text(draw, (pt[0], pt[1] - 26), f"{yv:.3f}", 23, anchor="ms", bold=True)
        text(draw, (pt[0], PLOT_BOTTOM + 45), f"{x:g}", 25, anchor="mm")
    text(draw, ((PLOT_LEFT + PLOT_RIGHT) // 2, PLOT_BOTTOM + 95), xlabel, 28, fill=PALETTE["gray"], anchor="mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, dpi=(300, 300))


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _has_results(payload: dict[str, Any], prefix: str, modalities: list[str]) -> bool:
    return any(improvement(payload, f"{prefix}-{m}") is not None for m in modalities)


def main_benchmark(payload: dict[str, Any], out: Path) -> None:
    modalities = ["audio", "vision", "imu"]
    labels = ["Static\nLoRA", "Conditional\nbridge", "Shuffled\nfeatures", "Random\nfeatures"]
    names = ["static-lora", "bridge", "shuffled", "random"]
    colors = [PALETTE["gray"], PALETTE["blue"], PALETTE["green"], PALETTE["orange"]]
    if _has_results(payload, "constant", modalities):
        labels.append("Constant\nfeatures")
        names.append("constant")
        colors.append(PALETTE["purple"])
    values = []
    for prefix in names:
        vals = [improvement(payload, f"{prefix}-{m}") for m in modalities]
        values.append(mean([v for v in vals if v is not None]))
    bar_chart(out / "paper_main_benchmark.png", "Bridge and baseline BPB improvements", "Mean across audio, vision, and IMU; regenerated from summary.json", labels, values, colors)


def feature_controls(payload: dict[str, Any], out: Path) -> None:
    modalities = ["audio", "vision", "imu"]
    labels = ["True\nfeatures", "Shuffled\nfeatures", "Random\nfeatures", "Static\nLoRA"]
    prefixes = ["bridge", "shuffled", "random", "static-lora"]
    colors = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"], PALETTE["gray"]]
    if _has_results(payload, "constant", modalities):
        labels.insert(3, "Constant\nfeatures")
        prefixes.insert(3, "constant")
        colors.insert(3, PALETTE["purple"])
    values = [mean([v for v in [improvement(payload, f"{p}-{m}") for m in modalities] if v is not None]) for p in prefixes]
    bar_chart(out / "paper_feature_controls.png", "Feature controls separate capacity from conditioning", "BPB alone is not sufficient evidence of semantic conditioning", labels, values, colors)


def composition(payload: dict[str, Any], out: Path) -> None:
    entries = [("V", "compose-V"), ("A", "compose-A"), ("I", "compose-I"), ("VA", "compose-VA"), ("VI", "compose-VI"), ("AI", "compose-AI"), ("VAI", "compose-VAI")]
    labels, values = [], []
    for label, name in entries:
        val = improvement(payload, name)
        if val is not None:
            labels.append(label)
            values.append(val)
    bar_chart(out / "paper_composition.png", "Composition of independently trained bridges", "Conditioned additive merge; labels denote active modalities", labels, values, [PALETTE["blue"], PALETTE["sky"], PALETTE["green"], PALETTE["orange"], PALETTE["purple"], PALETTE["red"], PALETTE["gray"]][: len(labels)])


def diversity(payload: dict[str, Any], out: Path) -> None:
    xs, ys = [], []
    for name, entry in payload.get("results", {}).items():
        if not name.startswith("diversity-imu-l"):
            continue
        res = entry["result"]
        xs.append(float(res["diversity_weight"]))
        probe = res.get("heldout_probe") or {}
        ys.append(float(probe.get("cross_input_cosine_mean", res.get("last_diversity", 0.0))))
    pairs = sorted(zip(xs, ys))
    if pairs:
        xs, ys = map(list, zip(*pairs))
        line_chart(out / "paper_diversity_ablation.png", "Diversity regularization reduces weight collapse", "Lower cross-input cosine means more input-dependent generated LoRA weights", xs, ys, "Cross-input cosine ↓", "Diversity weight λ")


def reproducibility(payload: dict[str, Any], out: Path) -> None:
    labels, means, stds = [], [], []
    for modality in ["audio", "vision", "imu"]:
        res = result(payload, f"repro-{modality}")
        if res:
            labels.append(modality.capitalize())
            means.append(float(res["mean_improvement"]))
            stds.append(float(res["std_improvement"]))
    img, draw = blank("Reproducibility across random seeds", "Mean ± std BPB improvement across seeds 42, 1042, and 2042")
    ymax = max([m + s for m, s in zip(means, stds)] + [1e-6]) * 1.25
    axes(draw, 0.0, ymax, "BPB improvement ↑")
    slot = (PLOT_RIGHT - PLOT_LEFT) / max(1, len(means))
    for i, (label, val, sd) in enumerate(zip(labels, means, stds)):
        cx = int(PLOT_LEFT + slot * (i + 0.5))
        bw = int(slot * 0.45)
        y = int(PLOT_BOTTOM - (val / ymax) * (PLOT_BOTTOM - PLOT_TOP))
        err_top = int(PLOT_BOTTOM - ((val + sd) / ymax) * (PLOT_BOTTOM - PLOT_TOP))
        err_bot = int(PLOT_BOTTOM - ((max(0, val - sd)) / ymax) * (PLOT_BOTTOM - PLOT_TOP))
        draw.rounded_rectangle((cx - bw // 2, y, cx + bw // 2, PLOT_BOTTOM), radius=10, fill=[PALETTE["blue"], PALETTE["orange"], PALETTE["green"]][i])
        draw.line((cx, err_top, cx, err_bot), fill=PALETTE["dark"], width=5)
        draw.line((cx - 35, err_top, cx + 35, err_top), fill=PALETTE["dark"], width=5)
        draw.line((cx - 35, err_bot, cx + 35, err_bot), fill=PALETTE["dark"], width=5)
        text(draw, (cx, y - 18), f"{val:.4f} ± {sd:.4f}", 25, bold=True, anchor="ms")
        text(draw, (cx, PLOT_BOTTOM + 50), label, 28, anchor="mm")
    out.mkdir(parents=True, exist_ok=True)
    img.save(out / "paper_reproducibility.png", dpi=(300, 300))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate paper figures from paper-reproduce summary.json")
    parser.add_argument("summary", type=Path, help="Path to paper-reproduce summary.json")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_summary(args.summary)
    main_benchmark(payload, args.output_dir)
    feature_controls(payload, args.output_dir)
    composition(payload, args.output_dir)
    diversity(payload, args.output_dir)
    reproducibility(payload, args.output_dir)
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
