from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from .render_arxiv import REPO_ROOT, fmt, load_summary, r, stats

# Replication-note renderer: every number is read from four suite artifacts
# (mini standard + 10x, LFM standard + 10x), mirroring the paper-1 discipline.


def task_number(payload: dict[str, Any], modality: str, condition: str, field: str):
    res = r(payload, f"task-{modality}")
    if not res:
        return None
    item = res.get("results", {}).get(condition)
    return item.get(field) if item else None


def diversity_rows(payload: dict[str, Any]) -> list[tuple[float, float, float]]:
    rows = []
    for name, entry in payload.get("results", {}).items():
        if name.startswith("diversity-imu-l"):
            res = entry["result"]
            probe = res.get("heldout_probe") or {}
            rows.append((float(res["diversity_weight"]), res.get("improvement"), probe.get("cross_input_cosine_mean")))
    return sorted(rows)


def diversity_cell(payload: dict[str, Any], lam: float, index: int):
    for row in diversity_rows(payload):
        if abs(row[0] - lam) < 1e-9:
            return row[index]
    return None


def replication_table(mini: dict[str, Any], lfm: dict[str, Any]) -> str:
    sm, sl = stats(mini), stats(lfm)
    rows = [
        ("Static LoRA", "static"),
        ("Conditional bridge", "bridge"),
        ("Shuffled features", "shuffled"),
        ("Random features", "random"),
        ("Constant features", "constant"),
        ("Prefix tuning", "prefix"),
    ]
    lines = []
    for label, key in rows:
        lines.append(f"{label} & {fmt(sm[key], sign=True)} & {fmt(sl[key], sign=True)} " + r"\\")
    return "\n".join(lines)


def task_table(mini: dict[str, Any], lfm: dict[str, Any], mini10: dict[str, Any], lfm10: dict[str, Any]) -> str:
    lines = []
    for modality, chance in (("imu", "0.17"), ("audio", "0.02")):
        for label, payloads in (
            ("standard", (mini, lfm)),
            (r"10$\times$", (mini10, lfm10)),
        ):
            m, l = payloads
            lines.append(
                f"{modality.upper()} & {label} & {chance} & "
                f"{fmt(task_number(m, modality, 'true', 'rank1'), 2)} & "
                f"{fmt(task_number(m, modality, 'no_bridge', 'rank1'), 2)} & "
                f"{fmt(task_number(l, modality, 'true', 'rank1'), 2)} & "
                f"{fmt(task_number(l, modality, 'no_bridge', 'rank1'), 2)} " + r"\\"
            )
    return "\n".join(lines)


def diversity_table(mini: dict[str, Any], lfm: dict[str, Any], mini10: dict[str, Any], lfm10: dict[str, Any]) -> str:
    lines = []
    for lam in (0.0, 0.01, 0.05, 0.1, 0.2):
        cells = []
        for payload in (mini, mini10, lfm, lfm10):
            cells.append(fmt(diversity_cell(payload, lam, 2), 2))
        lines.append(f"{lam:.2f} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def baseline_of(payload: dict[str, Any]):
    for name in ("bridge-imu", "bridge-audio", "static-lora-imu"):
        res = r(payload, name)
        if res and res.get("baseline") is not None:
            return float(res["baseline"])
    return None


def write_tex(mini, mini10, lfm, lfm10, out_dir: Path) -> Path:
    sm, sl, sm10, sl10 = stats(mini), stats(lfm), stats(mini10), stats(lfm10)
    base_m, base_l = baseline_of(mini), baseline_of(lfm)
    imu_m, imu_m10 = task_number(mini, "imu", "true", "rank1"), task_number(mini10, "imu", "true", "rank1")
    imu_l, imu_l10 = task_number(lfm, "imu", "true", "rank1"), task_number(lfm10, "imu", "true", "rank1")
    aud_m, aud_m10 = task_number(mini, "audio", "true", "rank1"), task_number(mini10, "audio", "true", "rank1")
    aud_l, aud_l10 = task_number(lfm, "audio", "true", "rank1"), task_number(lfm10, "audio", "true", "rank1")
    div_ratio_l = diversity_cell(lfm, 0.2, 1) / diversity_cell(lfm, 0.0, 1)
    lfm_bridge = r(lfm, "bridge-imu") or {}

    tex = rf"""
\ifdefined\XeTeXrevision\else\pdfoutput=1\fi
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{microtype}}
\hypersetup{{
  colorlinks=true,
  linkcolor=blue!60!black,
  citecolor=blue!60!black,
  urlcolor=blue!60!black,
  pdftitle={{Conditional LoRA Bridges at 230M: What Replicates, What Softens, and What Inverts}},
  pdfauthor={{Franci Penov}},
}}
\title{{Conditional LoRA Bridges at 230M:\\What Replicates, What Softens, and What Inverts}}
\author{{Franci Penov \\ kortexa.ai \\ \texttt{{francip@kortexa.ai}}}}
\date{{2026-07-03}}

\begin{{document}}
\maketitle

\begin{{abstract}}
Our controlled study of conditional LoRA bridges \cite{{penov2026bridges}} established its findings on a purpose-trained 33.6M-parameter model, leaving open which conclusions are properties of the mechanism and which are artifacts of scale. We port the unchanged experiment suite to a pretrained 230M-parameter model (LiquidAI LFM~2.5) and rerun it at both the standard and the $10\times$ training budget. The methodological core replicates exactly: bits-per-byte cannot detect sensor conditioning (the capacity-matched constant-feature control matches the true-feature bridge, {fmt(sl['constant'], sign=True)} vs {fmt(sl['bridge'], sign=True)}), while task-aligned probes detect it decisively ({fmt(imu_l, 2)} rank-1 on six-way IMU, {fmt(aud_l, 2)} on fifty-way audio, all controls at chance). Three scale-dependent results reverse or soften. Prefix tuning flips sign, from {fmt(sm['prefix'], sign=True)} BPB on the small model to {fmt(sl['prefix'], sign=True)} at 230M, while remaining the weakest adapter. The diversity--BPB tradeoff softens: the strongest penalty retains {div_ratio_l * 100:.0f}\% of the unregularized BPB gain where the small model retained none. Most strikingly, the long-budget collapse inverts: extended unregularized task training destroyed IMU conditioning on the small model (rank-1 {fmt(imu_m, 2)} to {fmt(imu_m10, 2)}, chance) but strengthens it at 230M ({fmt(imu_l, 2)} to {fmt(imu_l10, 2)}) --- even as weight collapse under pure text loss persists at both scales. At the two scales tested, capacity and pretraining are allies of task-conditioned weight-space bridges, and single-scale claims about the mechanism's failure modes should be read as lower bounds.
\end{{abstract}}

\section{{Motivation}}
The conditional-LoRA-bridges study \cite{{penov2026bridges}} reported one sharp methodological result --- aggregate language-modeling metrics cannot certify sensor conditioning --- and one mechanism result, hypernetwork weight collapse, with facets that include degradation from token-space (prefix) conditioning and a budget-dependent destruction of task-level conditioning under prolonged unregularized training. All of it was measured on a single 33.6M-parameter model trained from scratch. This note answers the obvious question: which of those findings survive contact with a real pretrained model?

\section{{Porting the harness}}
We add a checkpoint adapter (\texttt{{hf:<model-id>}}) to the published harness so that every experiment --- baselines, controls, probes, budgets, seeds --- runs unchanged against a HuggingFace causal LM; here LiquidAI LFM~2.5 230M \cite{{liquidai2025lfm2}}, a 14-layer hybrid short-convolution/attention model with a 65{{,}}536-token vocabulary. Three porting details matter for fidelity:
\begin{{itemize}}
  \item \textbf{{LoRA targets.}} The paper's target set ``all'' (every attention and MLP linear) maps to q/k/v/out projections on the six attention layers and the SwiGLU projections on all fourteen layers; short-convolution projections are left unadapted. The generated LoRA vector has $D = 774{{,}}144$ dimensions at rank~4.
  \item \textbf{{Byte-exact BPB across tokenizers.}} Bits-per-byte is tokenizer-independent only if the per-token byte table reconstructs the exact UTF-8 length of any encoded text. We build and validate such a table for the model's byte-level BPE, including its non-special added tokens, so BPB values are comparable across the two vocabularies.
  \item \textbf{{Sequence-start sensitivity.}} LFM~2.5 is strongly BOS-dependent (7.13 nats/token without BOS vs.\ 2.70 with, on the same text). The packing dataloader already places BOS at every row start; the task probes prepend BOS for \texttt{{hf:}} models so the probe measures conditioning, not an out-of-distribution artifact. The published small-model protocol is unchanged and reproduces bit-identically.
\end{{itemize}}
The frozen 230M model evaluates at {fmt(base_l, 4)} BPB on the paper's validation stream (the small model: {fmt(base_m, 4)}).

\section{{What replicates}}
Table~\ref{{tab:bpb}} gives the BPB decomposition at the standard budget. The paper's central methodological claim transfers unchanged: the conditional bridge ({fmt(sl['bridge'], sign=True)}) is indistinguishable from the capacity-matched constant-feature control ({fmt(sl['constant'], sign=True)}), static LoRA is the best pure-text adapter ({fmt(sl['static'], sign=True)}), and three-seed spreads are below $10^{{-3}}$. BPB still cannot see conditioning; the task probes still can (Table~\ref{{tab:task}}): true features reach {fmt(imu_l, 2)} rank-1 on six-way IMU and {fmt(aud_l, 2)} on fifty-way audio while shuffled, random, and no-bridge controls sit at chance --- numerically close to the small model's {fmt(imu_m, 2)} and {fmt(aud_m, 2)}. Composition of independently trained bridges remains sub-additive at both scales, and unregularized text-loss training still collapses generated weights toward input-independence (cross-input cosine {fmt(diversity_cell(lfm, 0.0, 2), 2)} at $\lambda=0$).

\begin{{table}}[t]
\centering
\caption{{BPB improvement over the frozen baseline (mean across audio, vision, IMU; standard budget; seed-42 protocol as in the original paper). The decomposition --- bridge $\approx$ constant control $<$ static LoRA --- replicates; prefix tuning flips sign.}}
\label{{tab:bpb}}
\begin{{tabular}}{{lrr}}
\toprule
Method & 33.6M (paper) & 230M (this note) \\
\midrule
{replication_table(mini, lfm)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[t]
\centering
\caption{{Task-probe rank-1 accuracy for true features vs.\ the no-bridge reference, at both budgets. The small model's IMU probe collapses to chance at the $10\times$ budget; the 230M model's strengthens.}}
\label{{tab:task}}
\begin{{tabular}}{{llccccc}}
\toprule
 & & & \multicolumn{{2}}{{c}}{{33.6M}} & \multicolumn{{2}}{{c}}{{230M}} \\
Probe & Budget & Chance & true & none & true & none \\
\midrule
{task_table(mini, lfm, mini10, lfm10)}
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{What softens}}
\paragraph{{Prefix tuning flips sign.}}
On the small model, feature-conditioned prefix tuning strictly degraded BPB ({fmt(sm['prefix'], sign=True)}); at 230M it improves it ({fmt(sl['prefix'], sign=True)}) --- yet remains the weakest adapter by a wide margin against the $\sim${fmt(sl['bridge'], sign=True)} weight-space rows. The original claim is therefore correctly scoped to small models trained without prefixes, while the paper's ordering --- weight-space conditioning dominates token-space conditioning --- holds at both scales.

\paragraph{{The diversity--BPB tradeoff softens.}}
On the small model, enforcing input-dependent weights at $\lambda=0.20$ erased the BPB gain entirely; at 230M the same penalty retains {div_ratio_l * 100:.0f}\% of it ({fmt(diversity_cell(lfm, 0.2, 1), sign=True)} of {fmt(diversity_cell(lfm, 0.0, 1), sign=True)}). The non-monotonicity the paper reports at the largest tested weight (cross-input cosine reversing at $\lambda=0.20$) appears at the standard budget at 230M as well ({fmt(diversity_cell(lfm, 0.1, 2), 2)} at $\lambda=0.10$ vs.\ {fmt(diversity_cell(lfm, 0.2, 2), 2)} at $\lambda=0.20$), but at the $10\times$ budget the reversal disappears and cross-input cosine falls to {fmt(diversity_cell(lfm10, 0.2, 2), 2)} at the largest weight (Table~\ref{{tab:div}}). The penalty required for input-dependence grows with both budget and scale: on the small model $\lambda=0.01$ loses its effect at the long budget, while at 230M penalties below $\lambda=0.10$ never produce input-dependent weights at either budget.

\begin{{table}}[t]
\centering
\caption{{Cross-input cosine of generated weights (IMU; lower = more input-dependent) across the diversity sweep, at both scales and budgets.}}
\label{{tab:div}}
\begin{{tabular}}{{rcccc}}
\toprule
$\lambda$ & 33.6M std & 33.6M $10\times$ & 230M std & 230M $10\times$ \\
\midrule
{diversity_table(mini, lfm, mini10, lfm10)}
\bottomrule
\end{{tabular}}
\end{{table}}

\section{{What inverts}}
The sharpest warning in the original paper was budget-dependent: at $10\times$ training, the small model's unregularized IMU bridge became behaviorally indistinguishable from no bridge at all (rank-1 {fmt(imu_m10, 2)}, equal to chance), having scored {fmt(imu_m, 2)} at the standard budget. At 230M the same protocol \emph{{strengthens}} conditioning: IMU rank-1 rises from {fmt(imu_l, 2)} to {fmt(imu_l10, 2)} (top-2 {fmt(task_number(lfm10, 'imu', 'true', 'top2'), 2)}), and audio rises from {fmt(aud_l, 2)} to {fmt(aud_l10, 2)} against a 0.02 chance level, with every control still at chance. Prolonged task-aligned training is an ally of conditioning at this scale, not a failure mode. The weight-space collapse under pure text loss still occurs at 230M (Table~\ref{{tab:div}}, $\lambda=0$); what changes is that label-paired training at scale maintains input dependence without regularization over the budgets tested.

\section{{Implications}}
Three practical updates for anyone building on the paper. First, the evaluation discipline (capacity-matched controls, task-aligned probes) transfers verbatim and remains necessary: BPB is exactly as blind at 230M as at 33.6M. Second, the mechanism's failure modes shift in the favorable direction from the 33.6M model to the 230M one --- the task-conditioning collapse threshold moves outward and the cost of enforcing input-dependent weights shrinks. Two caveats bound that statement: the comparison confounds scale with pretraining (the small model was trained from scratch, LFM~2.5 is pretrained), and it concerns task-conditioned training only --- collapse under pure text loss is unchanged at 230M. Even so, claims about weight-space conditioning's fragility measured at one scale should be read as lower bounds. Third, token-space and weight-space conditioning both improve with scale, but the gap between them persists; weight-space modulation remains the stronger interface across every configuration tested.

\section*{{Reproducibility}}
The port is additive: the published harness reproduces the original paper bit-identically, and the same runner produces every number here via \texttt{{--checkpoint hf:LiquidAI/LFM2.5-230M}}. Artifacts: the standard-budget suite (35 experiments) and the $10\times$ suite (33 of 35; the two remaining seed-stability repetitions were not run) ship alongside this note as \texttt{{summary-lfm.json}} and \texttt{{summary-lfm-scaling.json}} in the code repository, with the original paper's artifacts unchanged. Code: \url{{https://github.com/kortexa-ai/legolm.paper}}.

\begin{{thebibliography}}{{9}}
\bibitem{{penov2026bridges}} F.~Penov. Conditional LoRA Bridges for Modular Sensor Adaptation of Frozen Small Language Models. Preprint, 2026. \url{{https://github.com/kortexa-ai/legolm.paper}}.
\bibitem{{liquidai2025lfm2}} Liquid~AI. LFM2.5-230M. Model release, 2025--2026. \url{{https://huggingface.co/LiquidAI/LFM2.5-230M}}.
\end{{thebibliography}}

\end{{document}}
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "main.tex"
    target.write_text(tex.strip() + "\n")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the 230M replication note from four suite artifacts")
    parser.add_argument("--mini", type=Path, default=REPO_ROOT / "arxiv" / "summary.json")
    parser.add_argument("--mini-scaling", type=Path, default=REPO_ROOT / "arxiv" / "summary-scaling.json")
    parser.add_argument("--lfm", type=Path, default=REPO_ROOT / "results" / "lfm230m-standard-20260701" / "summary.json")
    parser.add_argument("--lfm-scaling", type=Path, default=REPO_ROOT / "results" / "lfm230m-scaling-20260701" / "summary.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "note")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    tex = write_tex(
        load_summary(args.mini),
        load_summary(args.mini_scaling),
        load_summary(args.lfm),
        load_summary(args.lfm_scaling),
        args.output_dir,
    )
    print(f"Wrote {tex}")
    import shutil

    shutil.copy2(args.lfm, args.output_dir / "summary-lfm.json")
    shutil.copy2(args.lfm_scaling, args.output_dir / "summary-lfm-scaling.json")
    if args.compile:
        subprocess.run(["tectonic", "-X", "compile", "main.tex", "--keep-logs"], cwd=args.output_dir, check=True)


if __name__ == "__main__":
    main()
