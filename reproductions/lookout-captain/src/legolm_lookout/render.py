"""Build the manuscript from audited evidence and a plain LaTeX template."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import PACKAGE, HOSTS, MODES, analyze, sha256

HOST_LABEL = {"450m": "450M", "3b": "3B", "27b": "27B"}
GATE_LABEL = {"always": "Every frame", "never": "Episode start", "verdict": "Verdict",
              "box": "Boxes", "pixel": "Pixels", "cut_oracle": "Scene cuts"}
PDF_METADATA = {"CreationDate": datetime.fromtimestamp(1788825600, timezone.utc),
                "ModDate": datetime.fromtimestamp(1788825600, timezone.utc)}


def number(value: float) -> str:
    return f"{value:.3f}"


def interval(value: dict) -> str:
    return f"{number(value['mean'])} [{number(value['interval'][0])}, {number(value['interval'][1])}]"


def row(*values) -> str:
    return " & ".join(str(x) for x in values) + r" \\" + "\n"


def find(data: dict, section: str, stem: str, mode: str, gate: str | None = None) -> dict:
    found = [r for r in data[section] if r["stem"] == stem and r["mode"] == mode
             and (gate is None or r.get("gate") == gate)]
    if len(found) != 1:
        raise ValueError(f"ambiguous manuscript source: {section}, {stem}, {mode}, {gate}")
    return found[0]


def raw(stem: str) -> dict:
    return json.loads((PACKAGE / "artifacts" / f"{stem}.json").read_text())


def tables(data: dict) -> dict[str, str]:
    parts = {}
    lines = []
    for item in data["conditioning"]:
        e = item["estimates"]
        lines.append(row(HOST_LABEL[item["stem"].split("-")[-1]], item["mode"].title(),
                         number(e["present:habit"]["mean"]), number(e["present:packet"]["mean"]),
                         number(e["present:text"]["mean"]), interval(e["packet-habit"]),
                         interval(e["packet-text"])))
    parts["CONDITIONING"] = "".join(lines)
    lines = []
    for item in data["conditioning"]:
        e = item["estimates"]
        lines.append(row(HOST_LABEL[item["stem"].split("-")[-1]], item["mode"].title(),
                         interval(e["packet-shuffled"]), interval(e["absent:packet"]),
                         "/".join(number(x) for x in e["packet-habit"]["seed_means"])))
    parts["CONDITIONING_CONTROL"] = "".join(lines)
    lines = []
    detector_names = {"r2": "MobileNet", "r2y": "YOLO26n",
                      "r2y-26s": "YOLO26s", "r2y-26m": "YOLO26m"}
    for item in data["switch"]:
        if item["mode"] != "social":
            continue
        stem = item["stem"]
        host = stem.split("-")[-1]
        family = stem.removeprefix("vision-mode-").removesuffix("-" + host)
        caution = find(data, "switch", stem, "caution")
        lines.append(row(HOST_LABEL[host], detector_names[family],
                         interval(item["estimates"]["detector-oracle"]),
                         interval(caution["estimates"]["detector-oracle"])))
    parts["SWITCH"] = "".join(lines)
    lines = []
    for name in ("yolo26n", "yolo26s", "yolo26m"):
        r = data["detectors"]["vision-mode-r2y-detectors"]["sizes"][name]
        lines.append(row(name.upper(), f"{r['parameters']/1e6:.2f}",
                         r["eval"]["misses"], r["eval"]["false_alarms"],
                         r["calibration"]["misses"], r["calibration"]["false_alarms"],
                         f"{r['calibration']['ms_per_frame']:.2f}"))
    parts["DETECTORS"] = "".join(lines)
    lines = []
    for host in HOSTS:
        for dose in ("x1", "x1p5"):
            for mode in MODES:
                item = find(data, "conflict", f"vision-mode-r1b-{host}-{dose}-oracle", mode)
                e = item["estimates"]
                lines.append(row(HOST_LABEL[host], mode.title(), "0.15" if dose == "x1" else "0.225",
                                 number(e["present:packet"]["mean"]),
                                 number(e["present:hostile-text"]["mean"]),
                                 number(e["present:packet+hostile"]["mean"]),
                                 interval(e["hostile-effect"]), number(e["absent:packet"]["mean"])))
    parts["CONFLICT"] = "".join(lines)
    lines = []
    for section in ("boxes", "blind"):
        for item in data[section]:
            e = item["estimates"]
            lines.append(row("Visible" if section == "boxes" else "Blank",
                             HOST_LABEL[item["stem"].split("-")[-1]], item["mode"].title(),
                             number(e["present:switch"]["mean"]), number(e["present:boxes"]["mean"]),
                             number(e["present:lying"]["mean"]), interval(e["boxes-switch"])))
    parts["BOXES"] = "".join(lines)
    lines = []
    for host in HOSTS:
        for mode in ("social", "caution"):
            item = find(data, "gating", f"vision-mode-r2t-{host}", mode, "verdict")
            e = item["estimates"]
            lines.append(row(HOST_LABEL[host], mode.title(), interval(e["agreement"]),
                             interval(e["oracle_correctness"]), interval(e["correctness_change"])))
    parts["GATING"] = "".join(lines)
    lines = []
    for gate in GATE_LABEL:
        item = find(data, "gating", "vision-mode-r2t-27b", "caution", gate)
        social = find(data, "gating", "vision-mode-r2t-27b", "social", gate)
        e = item["estimates"]
        delay = item["delay"][0]
        lines.append(row(GATE_LABEL[gate], number(e["invocations_avoided"]["mean"]),
                         number(e["agreement"]["mean"]), number(social["estimates"]["agreement"]["mean"]),
                         f"{delay['observed_wakes']}/{delay['post_start_cuts']}",
                         "--" if delay["observed_mean"] is None else f"{delay['observed_mean']:.2f}",
                         delay["right_censored"]))
    parts["GATE_CONTROLS"] = "".join(lines)
    lines = []
    for host in HOSTS:
        variants = ["", "mx2door-"]
        if host != "3b":
            variants += ["mx2ndoor-"]
        for variant in variants:
            for mode in MODES:
                item = find(data, "transfer", f"vision-mode-r3b-{variant}{host}", mode)
                e = item["estimates"]
                recipe = {"": "Photo", "mx2door-": "Mixed", "mx2ndoor-": "Mixed/fixed"}[variant]
                lines.append(row(HOST_LABEL[host], recipe, mode.title(), len(item["seeds"]),
                                 number(e["present:habit"]["mean"]), number(e["present:packet"]["mean"]),
                                 interval(e["packet-habit"])))
    parts["TRANSFER"] = "".join(lines)
    lines = []
    for host in ("450m", "27b"):
        for init, order_seed, suffix in ((42, 42, "mx2door"), (1042, 1042, "mx2door"),
                                          (42, 1042, "mx2o1042door"), (1042, 42, "mx2o42door")):
            stem = f"vision-mode-r1b-{host}-x1-oracle-{suffix}-s{init}"
            m = raw(stem)["metrics"]["caution"]["present_share"]
            lines.append(row(HOST_LABEL[host], init, order_seed, number(m["habit"]), number(m["packet"])))
    parts["ORDER"] = "".join(lines)
    lines = []
    for item in data["transfer"]:
        short = item["stem"].removeprefix("vision-mode-r3b-")
        e = item["estimates"]
        lines.append(row(short, item["mode"].title(), ",".join(str(s) for s in item["seeds"]),
                         "/".join(number(x) for x in e["present:packet"]["seed_means"]),
                         "/".join(number(x) for x in e["packet-habit"]["seed_means"])))
    parts["TRANSFER_SEEDS"] = "".join(lines)
    return parts


def macros(data: dict) -> str:
    values = {}
    for mode in MODES:
        item = find(data, "conditioning", "vision-mode-r1-27b", mode)
        values[mode.title() + "Lean"] = number(item["estimates"]["packet-habit"]["mean"])
        values[mode.title() + "VsText"] = number(item["estimates"]["packet-text"]["mean"])
    gate = find(data, "gating", "vision-mode-r2t-27b", "caution", "verdict")
    values["AvoidedPct"] = f"{100*gate['estimates']['invocations_avoided']['mean']:.1f}"
    values["CautionAgreePct"] = f"{100*gate['estimates']['agreement']['mean']:.1f}"
    social = find(data, "gating", "vision-mode-r2t-27b", "social", "verdict")
    values["SocialAgreePct"] = f"{100*social['estimates']['agreement']['mean']:.1f}"
    delay = gate["delay"][0]
    values.update({"PostCuts": delay["post_start_cuts"], "ResolvedCuts": delay["observed_wakes"],
                   "CensoredCuts": delay["right_censored"], "MaxDelay": delay["observed_max"],
                   "ResolvedDelay": f"{delay['observed_mean']:.2f}", "SlowCuts": delay["observed_over_one"]})
    values["DecodeSeconds"] = f"{np.mean(gate['decode_seconds']):.3f}"
    values["EvidenceFiles"] = data["provenance"]["files"]
    values["AuditedImageRecords"] = len(data["image_record_audits"])
    values["ScalarChecks"] = sum(r["scalar_checks"] for r in data["image_record_audits"].values())
    return "\n".join(r"\newcommand{\%s}{%s}" % (k, v) for k, v in values.items()) + "\n"


def figures(data: dict, output: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.bbox": "tight"})
    colors = {"450m": "#257a91", "3b": "#c26b22", "27b": "#684789"}
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3), sharey=True)
    for ax, host in zip(axes, HOSTS):
        for j, mode in enumerate(MODES):
            e = find(data, "conditioning", f"vision-mode-r1-{host}", mode)["estimates"]["packet-habit"]
            lo, hi = e["interval"]
            ax.errorbar(j, e["mean"], yerr=[[e["mean"]-lo], [hi-e["mean"]]], fmt="o",
                        color=colors[host], capsize=4)
            ax.scatter(np.full(len(e["seed_means"]), j) + np.linspace(-.13, .13, len(e["seed_means"])),
                       e["seed_means"], marker="x", color=colors[host], alpha=.65, s=24)
        ax.axhline(0, color="0.6", linewidth=.8)
        ax.set_xticks(range(3), [x.title() for x in MODES], rotation=20)
        ax.set_title(HOST_LABEL[host]); ax.set_ylim(-.4, 1.05)
    axes[0].set_ylabel("Packet minus constant adapter\n(target-action share)")
    fig.tight_layout(); fig.savefig(output / "conditioning.pdf", metadata=PDF_METADATA); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    markers = {"always": "o", "never": "X", "verdict": "s", "box": "^", "pixel": "D", "cut_oracle": "v"}
    for ax, mode in zip(axes, ("social", "caution")):
        for host in HOSTS:
            for gate in GATE_LABEL:
                e = find(data, "gating", f"vision-mode-r2t-{host}", mode, gate)["estimates"]
                ax.scatter(e["invocations_avoided"]["mean"], e["agreement"]["mean"],
                           c=colors[host], marker=markers[gate], s=45)
        ax.set(xlabel="Fraction of model calls avoided", ylabel="Agreement with every-frame decoding",
               title=mode.title(), xlim=(-.04, 1), ylim=(.55, 1.025))
        ax.grid(alpha=.18)
    handles = [plt.Line2D([], [], marker=markers[g], linestyle="", color="0.35", label=GATE_LABEL[g]) for g in GATE_LABEL]
    handles += [plt.Line2D([], [], marker="o", linestyle="", color=colors[h], label=HOST_LABEL[h]) for h in HOSTS]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(.5, -.14), frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(output / "gating.pdf", metadata=PDF_METADATA); plt.close(fig)


def source_archive(output: Path) -> None:
    """Package the complete LaTeX inputs with stable archive metadata."""
    names = ("main.tex", "numbers.tex", "references.bib", "conditioning.pdf", "gating.pdf")
    instructions = (
        "The Lookout and the Captain — LaTeX source\n\n"
        "Build with Tectonic 0.17.0:\n"
        "SOURCE_DATE_EPOCH=1788825600 FORCE_SOURCE_DATE=1 tectonic main.tex\n\n"
        "All required figures and bibliography are included. For the saved-data\n"
        "analysis, editable template, and experiment code, see:\n"
        "https://github.com/kortexa-ai/legolm.paper/tree/main/reproductions/lookout-captain\n"
    )
    files = {name: (output / name).read_bytes() for name in names}
    files["README.txt"] = instructions.encode()
    with zipfile.ZipFile(output / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)


def build(output: Path, *, compile_pdf: bool = True, artifacts: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    result = analyze(artifacts if artifacts is not None else PACKAGE / "artifacts")
    summary = PACKAGE / "analysis/summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    template = (PACKAGE / "paper/main.template.tex").read_text()
    for key, value in tables(result).items():
        template = template.replace(f"@@{key}@@", value)
    if re.search(r"@@[A-Z_]+@@", template):
        raise ValueError("unexpanded manuscript table")
    (output / "main.tex").write_text(template)
    (output / "numbers.tex").write_text(macros(result))
    shutil.copyfile(PACKAGE / "paper/references.bib", output / "references.bib") if output.resolve() != (PACKAGE / "paper").resolve() else None
    figures(result, output)
    if compile_pdf:
        env = dict(os.environ, SOURCE_DATE_EPOCH="1788825600", FORCE_SOURCE_DATE="1")
        subprocess.run(["tectonic", "--keep-logs", "main.tex"], cwd=output, env=env, check=True)
    source_archive(output)
    names = {"main.tex", "numbers.tex", "references.bib", "conditioning.pdf", "gating.pdf", "source.zip"}
    if compile_pdf:
        names.add("main.pdf")
    receipt = {"compiled": compile_pdf, "analysis_sha256": sha256(summary), "evidence_files": result["provenance"]["files"],
               "source_commit": result["provenance"]["source_commit"],
               "files": {p.name: sha256(p) for p in sorted(output.iterdir())
                         if p.name in names}}
    (output / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PACKAGE / "paper")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.output, compile_pdf=not args.no_compile)))


if __name__ == "__main__":
    main()
