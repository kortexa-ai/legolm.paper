"""Render the ladder summary as the comparison the paper is about.

One table, two columns: what each stage scores when measured in the space it
was optimized in, and what it scores on text the model actually produced.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _pct(value: float, denominator: float) -> str:
    if not denominator:
        return "n/a"
    return f"{value / denominator * 100:.1f}%"


def render(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    explicit = summary["stage_a"]["explicit_span"]

    lines.append("Stage A — explicit-prompt upper bound (the denominator)")
    for axis in sorted(explicit):
        lines.append(f"  {axis:9s} {explicit[axis]:+.4f}")

    lines.append("")
    lines.append("Stages C/D — static writer: teacher-forced vs generated")
    lines.append(f"  {'axis':9s} {'teacher':>9s} {'generated':>10s} {'gen/A':>7s} {'signed':>8s} {'specific':>9s}")
    best = summary["stage_c"]["best"]
    audit = summary["stage_d"]["summary"]
    for axis in sorted(best):
        row = audit.get(axis, {})
        n = row.get("n", 0)
        lines.append(
            f"  {axis:9s} {best[axis]['span']:+9.4f} {row.get('generated_span', float('nan')):+10.4f} "
            f"{_pct(row.get('generated_span', 0.0), explicit[axis]):>7s} "
            f"{str(row.get('signed', 0)) + '/' + str(n):>8s} {str(row.get('specific', 0)) + '/' + str(n):>9s}"
        )

    lines.append("")
    stage_f = summary["stage_f"]
    grouped = defaultdict(list)
    for record in stage_f["records"]:
        grouped[record["axis"]].append(record)
    lines.append(
        f"Stage F — contextual, {stage_f['sites']} sites at {stage_f['per_site_fraction']:.4f} each"
    )
    for axis in sorted(grouped):
        rows = grouped[axis]
        teacher = sum(r["teacher_span"] for r in rows) / len(rows)
        generated = sum(r["generated_span"] for r in rows) / len(rows)
        lines.append(
            f"  {axis:9s} {teacher:+9.4f} {generated:+10.4f} {_pct(generated, explicit[axis]):>7s}"
        )

    lines.append("")
    stage_g = summary["stage_g"]
    history = stage_g["loss_history"]
    lines.append(
        f"Stage G — trained residual writer (loss {history[0]:+.3f} -> {history[-1]:+.3f})"
    )
    lines.append(f"  cross-axis cosine: {stage_g['geometry']}")
    for axis, row in sorted(stage_g["audit"]["summary"].items()):
        n = row["n"]
        lines.append(
            f"  {axis:9s} generated {row['generated_span']:+.4f} "
            f"({_pct(row['generated_span'], explicit[axis])} of A) "
            f"signed {row['signed']}/{n} specific {row['specific']}/{n}"
        )

    peak = summary.get("memory_peak", {}).get("cuda_peak_reserved_gib")
    if peak is not None:
        lines.append("")
        lines.append(f"peak CUDA reserved: {peak} GiB")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a ladder run")
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    print(render(json.loads(args.summary.read_text())))


if __name__ == "__main__":
    main()
