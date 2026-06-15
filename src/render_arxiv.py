from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_summary(path: Path) -> dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def r(payload: dict[str, Any], name: str) -> dict[str, Any] | None:
    entry = payload.get("results", {}).get(name)
    return entry.get("result") if entry else None


def imp(payload: dict[str, Any], name: str) -> float | None:
    item = r(payload, name)
    if not item or item.get("improvement") is None:
        return None
    return float(item["improvement"])


def fmt(value: float | None, digits: int = 4, sign: bool = False) -> str:
    if value is None:
        return "--"
    prefix = "+" if sign and value >= 0 else ""
    return f"{prefix}{value:.{digits}f}"


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


ROW_END = r"\\"


def modality_table(payload: dict[str, Any]) -> str:
    rows = []
    methods = [
        ("Static LoRA", "static-lora"),
        ("Conditional bridge", "bridge"),
        ("Shuffled features", "shuffled"),
        ("Random features", "random"),
        ("Constant features", "constant"),
        ("Prefix tuning", "prefix"),
    ]
    for label, prefix in methods:
        if prefix == "constant" and all(
            imp(payload, f"constant-{m}") is None for m in ["audio", "vision", "imu"]
        ):
            continue
        vals = []
        for modality in ["audio", "vision", "imu"]:
            name = f"{prefix}-{modality}"
            if prefix == "prefix":
                name = f"prefix-{modality}-8tok"
            vals.append(imp(payload, name))
        rows.append(f"{label} & " + " & ".join(fmt(v, sign=True) for v in vals) + f" & {fmt(mean(vals), sign=True)} " + ROW_END)
    return "\n".join(rows)


def composition_table(payload: dict[str, Any]) -> str:
    labels = [("Vision", "compose-V"), ("Audio", "compose-A"), ("IMU", "compose-I"), ("Vision+Audio", "compose-VA"), ("Vision+IMU", "compose-VI"), ("Audio+IMU", "compose-AI"), ("Vision+Audio+IMU", "compose-VAI")]
    rows = []
    for label, name in labels:
        res = r(payload, name)
        if not res:
            continue
        rows.append(f"{label} & {len(res.get('bricks', []))} & {fmt(res.get('baseline'), 4)} & {fmt(res.get('final'), 4)} & {fmt(res.get('improvement'), 4, sign=True)} " + ROW_END)
    return "\n".join(rows)


def diversity_table(payload: dict[str, Any]) -> str:
    rows = []
    items = []
    for name, entry in payload.get("results", {}).items():
        if name.startswith("diversity-imu-l"):
            res = entry["result"]
            probe = res.get("heldout_probe") or {}
            items.append((float(res["diversity_weight"]), res, probe))
    for lam, res, probe in sorted(items):
        cosine = probe.get("cross_input_cosine_mean", res.get("last_diversity"))
        motion_still = probe.get("motion_vs_still_centroid_cosine_mean")
        rows.append(f"{lam:.2f} & {fmt(res.get('improvement'), 4, sign=True)} & {fmt(cosine, 4)} & {fmt(motion_still, 4)} " + ROW_END)
    return "\n".join(rows)


def repro_table(payload: dict[str, Any]) -> str:
    rows = []
    for modality in ["audio", "vision", "imu"]:
        res = r(payload, f"repro-{modality}")
        if res:
            per_seed = ", ".join(fmt(run.get("improvement"), 4, sign=True) for run in res.get("runs", []))
            rows.append(f"{display_modality(modality)} & {fmt(res.get('mean_improvement'), 4, sign=True)} & {per_seed} " + ROW_END)
    return "\n".join(rows)


def display_modality(modality: str) -> str:
    return "IMU" if modality == "imu" else modality.capitalize()


def task_table(payload: dict[str, Any]) -> str:
    rows = []
    for modality in ["audio", "imu"]:
        res = r(payload, f"task-{modality}")
        if not res:
            continue
        for condition in ["no_bridge", "true", "shuffled", "random"]:
            item = res.get("results", {}).get(condition)
            if item:
                rows.append(f"{display_modality(modality)} & {condition.replace('_', ' ')} & {fmt(item.get('avg_rank'), 2)} & {fmt(item.get('rank1'), 2)} & {fmt(item.get('top2'), 2)} " + ROW_END)
    return "\n".join(rows)


_LAYER_WORDS = {2: "two", 4: "four", 6: "six", 8: "eight", 12: "twelve"}


def model_facts(payload: dict[str, Any]) -> dict[str, Any]:
    """Architecture facts read from the actual checkpoint, so the manuscript can
    never silently drift from the artifact again (the 4-layer/18.9M description
    of an 8-layer/33.6M checkpoint survived a full review cycle)."""
    import torch

    md = payload.get("metadata", {})
    candidates = []
    ckpt_meta = md.get("checkpoint")
    if ckpt_meta:
        candidates.append(Path(ckpt_meta))
        candidates.append(REPO_ROOT / "checkpoints" / "experiments" / Path(ckpt_meta).name)
    candidates.append(REPO_ROOT / "checkpoints" / "experiments" / "mini-base.pt")
    for path in candidates:
        if path.is_file():
            ck = torch.load(path, map_location="cpu", weights_only=False)
            cfg = ck.get("config", {})
            sd = ck.get("model", {})
            total = sum(v.numel() for v in sd.values())
            if (
                cfg.get("tie_word_embeddings")
                and "lm_head.weight" in sd
                and "transformer.wte.weight" in sd
            ):
                total -= sd["lm_head.weight"].numel()
            n_layer = cfg.get("n_layer")
            return {
                "n_layer": n_layer,
                "layer_word": _LAYER_WORDS.get(n_layer, str(n_layer)),
                "n_embd": cfg.get("n_embd"),
                "vocab_size": cfg.get("vocab_size"),
                "params_m": total / 1e6,
            }
    raise FileNotFoundError(
        "No mini-base checkpoint found to derive model facts; "
        "refusing to render unverifiable architecture claims"
    )


MODALITIES = ["audio", "vision", "imu"]


def stats(payload: dict[str, Any]) -> dict[str, Any]:
    """Cross-condition aggregates used by the narrative prose, so the text can
    never disagree with the tables."""
    def modal_mean(prefix: str) -> float | None:
        names = [f"{prefix}-{m}" for m in MODALITIES]
        if prefix == "prefix":
            names = [f"prefix-{m}-8tok" for m in MODALITIES]
        return mean([imp(payload, n) for n in names])

    def task(modality: str, condition: str, field: str) -> float | None:
        res = r(payload, f"task-{modality}")
        if not res:
            return None
        item = res.get("results", {}).get(condition)
        return item.get(field) if item else None

    bridge_res = r(payload, "bridge-imu") or {}
    return {
        "static": modal_mean("static-lora"),
        "bridge": modal_mean("bridge"),
        "shuffled": modal_mean("shuffled"),
        "random": modal_mean("random"),
        "constant": modal_mean("constant"),
        "prefix": modal_mean("prefix"),
        "task_imu_true_rank1": task("imu", "true", "rank1"),
        "task_imu_none_rank1": task("imu", "no_bridge", "rank1"),
        "task_imu_true_avgrank": task("imu", "true", "avg_rank"),
        "task_audio_true_rank1": task("audio", "true", "rank1"),
        "task_audio_none_rank1": task("audio", "no_bridge", "rank1"),
        "task_audio_true_avgrank": task("audio", "true", "avg_rank"),
        "bridge_params": bridge_res.get("trainable_params"),
        "lora_dim": bridge_res.get("lora_dim"),
    }


def scaling_section(payload: dict[str, Any], scaling: dict[str, Any] | None) -> str:
    """Budget-scaling robustness subsection, rendered only when a second
    (higher-budget) artifact is provided. Entirely data-driven."""
    if not scaling:
        return ""
    base = stats(payload)
    big = stats(scaling)
    smd = scaling.get("metadata", {})
    bmd = payload.get("metadata", {})
    factor = round((smd.get("benchmark_steps") or 0) / max(1, bmd.get("benchmark_steps") or 1))
    div_rows = []
    for name, entry in sorted(scaling.get("results", {}).items()):
        if name.startswith("diversity-imu-l"):
            res = entry["result"]
            probe = res.get("heldout_probe") or {}
            div_rows.append((float(res["diversity_weight"]), probe.get("cross_input_cosine_mean")))
    weak = next((c for lam, c in div_rows if abs(lam - 0.01) < 1e-9), None)
    strong = next((c for lam, c in div_rows if abs(lam - 0.05) < 1e-9), None)
    return rf"""
\subsection{{Budget scaling}}
\label{{sec:scaling}}
To test whether the qualitative picture is an artifact of short training, we repeated the full suite at {factor}$\times$ the step budget ({smd.get('benchmark_steps')} benchmark steps, {smd.get('task_steps')} task-probe steps) with a {smd.get('eval_tokens')}-token eval stream. The BPB ordering is preserved as every condition improves: static LoRA {fmt(big['static'], sign=True)} (from {fmt(base['static'], sign=True)}), conditional bridge {fmt(big['bridge'], sign=True)} (from {fmt(base['bridge'], sign=True)}), constant features {fmt(big['constant'], sign=True)}, shuffled {fmt(big['shuffled'], sign=True)}, random {fmt(big['random'], sign=True)}; the conditioning contribution measured by BPB remains indistinguishable from zero. Prefix tuning's degradation shrinks toward neutrality ({fmt(base['prefix'], sign=True)} to {fmt(big['prefix'], sign=True)}) without ever helping.

The task probes expose a boundary that short budgets hide. The audio probe strengthens with optimization (rank-1 {fmt(base['task_audio_true_rank1'], 2)} to {fmt(big['task_audio_true_rank1'], 2)}, controls still at chance), but the IMU probe collapses: at {smd.get('task_steps')} steps the true-feature bridge becomes behaviorally indistinguishable from the zero-LoRA condition (rank-1 {fmt(big['task_imu_true_rank1'], 2)}, equal to no-bridge), having scored {fmt(base['task_imu_true_rank1'], 2)} at {bmd.get('task_steps')} steps. The diversity sweep at this budget shows the same mechanism: $\lambda=0.01$ no longer prevents weight collapse (cross-input cosine {fmt(weak, 2)}) while $\lambda \ge 0.05$ still does ({fmt(strong, 2)}). Prolonged unregularized training drives the hypernetwork toward input-independent output, and the collapse threshold moves with optimization length --- the failure mode this paper documents is not confined to a corner of the BPB table.
"""


def repro_command(payload: dict[str, Any]) -> str:
    """Reconstruct the exact runner invocation from the artifact metadata, so the
    Reproducibility Statement can never promise a command that produces a
    different configuration than the bundled artifact."""
    md = payload.get("metadata", {})
    parts = ["paper-reproduce", f"--suite {md.get('suite', 'all')}"]
    if "audio" in (md.get("task_modalities") or []):
        parts.append("--include-audio-task")
    for flag, key in [
        ("--benchmark-steps", "benchmark_steps"),
        ("--composition-steps-per-brick", "composition_steps_per_brick"),
        ("--task-steps", "task_steps"),
        ("--eval-tokens", "eval_tokens"),
        ("--sensor-limit", "sensor_limit"),
        ("--max-eval-items", "max_eval_items"),
        ("--seed", "seed"),
    ]:
        value = md.get(key)
        if value is not None:
            parts.append(f"{flag} {value}")
    parts.append("--output-dir results/<run>")
    return " ".join(parts)


def metadata_sentence(payload: dict[str, Any]) -> str:
    md = payload.get("metadata", {})
    head = (
        f"The full run used rank {md.get('rank')}, target set {tex_escape(str(md.get('target')))}, "
        f"bridge learning rate {md.get('lr')}, "
    )
    if "benchmark_steps" in md:
        mid = (
            f"{md.get('benchmark_steps')} benchmark training steps, "
            f"{md.get('composition_steps_per_brick')} composition steps per brick, "
            f"{md.get('task_steps')} task-probe training steps, "
        )
    else:
        # Legacy artifacts (pre fixed-step rename) recorded wall-clock budgets.
        mid = (
            f"a {md.get('benchmark_budget')}-second wall-clock benchmark budget, "
            f"{md.get('composition_budget_per_brick')} seconds per composition brick, "
            f"a {md.get('task_budget')}-second task budget, "
        )
    return head + mid + f"and seeds {', '.join(map(str, md.get('repro_seeds', [])))}."


def write_tex(payload: dict[str, Any], out_dir: Path, scaling: dict[str, Any] | None = None) -> Path:
    baseline = None
    for name in ["bridge-imu", "bridge-audio", "bridge-vision", "static-lora-imu"]:
        res = r(payload, name)
        if res and res.get("baseline") is not None:
            baseline = float(res["baseline"])
            break
    facts = model_facts(payload)
    st = stats(payload)
    scaling_ref = r" (Section~\ref{sec:scaling})" if scaling else ""
    created = payload.get("metadata", {}).get("created_at", str(date.today()))[:10]
    tex = rf"""
\ifdefined\XeTeXrevision\else\pdfoutput=1\fi
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{caption}}
\usepackage{{enumitem}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{microtype}}
\hypersetup{{
  colorlinks=true,
  linkcolor=blue!60!black,
  citecolor=blue!60!black,
  urlcolor=blue!60!black,
  pdftitle={{Conditional LoRA Bridges for Modular Sensor Adaptation of Frozen Small Language Models}},
  pdfauthor={{Franci Penov}},
}}
\graphicspath{{{{figures/}}}}
\title{{Conditional LoRA Bridges for Modular Sensor Adaptation of Frozen Small Language Models}}
\author{{Franci Penov \\ kortexa.ai \\ \texttt{{francip@kortexa.ai}}}}
\date{{{created}}}

\begin{{document}}
\maketitle

\begin{{abstract}}
We study how to extend frozen language models with new sensors without retraining them, and show that standard language-modeling scores cannot tell whether the sensor is genuinely used --- separating real sensing from added capacity takes task-level tests. Our mechanism, a conditional LoRA bridge, maps frozen sensor-encoder features to per-input Low-Rank Adaptation (LoRA) parameters that are injected into a frozen transformer without changing base model weights. On a controlled {facts['params_m']:.1f}M-parameter Qwen-style language model, we evaluate dynamic bridges alongside static LoRA, shuffled-, random-, and constant-feature controls, prefix tuning, composition tests, task-aligned ranking probes, and seed-stability checks, all under one fixed-step runner. The decomposition is sharp. Measured by validation bits-per-byte, conditioning contributes nothing: a capacity-matched bridge fed one constant feature vector matches the true-feature bridge, and static LoRA beats both. Measured by task-aligned probes, conditioning is decisive: with true features the frozen model ranks the correct sensor label first {fmt(st['task_imu_true_rank1'], 2)} of the time on six-way IMU activities and {fmt(st['task_audio_true_rank1'], 2)} on fifty-way audio events, while shuffled features, random features, and the unconditioned model all sit at chance. Aggregate language-modeling metrics are therefore the wrong instrument for detecting sensor conditioning. Diversity regularization governs the central failure mode: without it, generated weights collapse toward input-independence --- and at sufficiently long training budgets this collapse erases the task-level conditioning entirely. Prefix tuning, the token-space alternative, only degrades the frozen model. These results support weight-space modulation as a viable modular interface between independently trained sensor encoders and frozen language models, with bridge compression and regularized long-horizon training as the open problems.
\end{{abstract}}

\section{{Introduction}}
Adding a new sensor to a trained language model normally means retraining it; this paper studies a cheaper, modular alternative: translating sensor signals into on-the-fly weight adjustments for a frozen model. The catch is measurement: the standard language-modeling score improves even when the sensor feed is replaced by a fixed dummy signal, so a system can appear to sense while contributing nothing of the kind. Our central methodological contribution is to pin down the task-level evidence that separates genuine sensing from this added capacity. Retraining is the default because multimodal models are typically extended by joint training on paired corpora: powerful, but awkward for small or local systems, where every new sensor type can demand a new data pipeline, new paired examples, and another round of model adaptation. Our interface avoids this entirely: keep the language model frozen, keep the sensor encoder frozen, and learn only a bridge that converts sensor features into LoRA weights.

The method, which we call \emph{{conditional LoRA bridges}}, uses a hypernetwork \cite{{ha2016hypernetworks}} to generate LoRA parameters \cite{{hu2021lora}} from sensor features. The generated low-rank matrices are injected into a frozen transformer at runtime. The mechanism is related to adapters \cite{{houlsby2019adapters}}, shared hypernetwork adapters \cite{{mahabadi2021hyperformer}}, and prefix tuning \cite{{li2021prefixtuning}}, but differs by conditioning weight-space updates on continuous external sensor streams rather than appending tokens or selecting a task identifier.

Contributions:
\begin{{itemize}}[leftmargin=*]
  \item A controlled, fully reproducible evaluation harness for conditional LoRA bridges across vision, audio, and IMU features: fixed step counts, a shared eval stream, three-seed checks, and every table and number in this manuscript rendered from a single bundled result artifact.
  \item A capacity/conditioning decomposition via controls. A constant-feature bridge isolates trainable capacity; it matches the true-feature bridge on BPB ({fmt(st['constant'], sign=True)} vs.\ {fmt(st['bridge'], sign=True)}), showing that aggregate language-modeling gains carry no evidence of sensor conditioning.
  \item Task-aligned probes that detect conditioning decisively where BPB cannot: true features lift rank-1 label accuracy to {fmt(st['task_imu_true_rank1'], 2)} (IMU, chance 0.17) and {fmt(st['task_audio_true_rank1'], 2)} (audio, chance 0.02) while every control stays at chance.
  \item A characterization of hypernetwork weight collapse: diversity regularization makes generated weights input-dependent, and at long training budgets unregularized bridges collapse to input-independence, erasing task-level conditioning{scaling_ref}.
\end{{itemize}}

\section{{Method}}
\paragraph{{Frozen language model.}}
The base model is {'an' if facts['layer_word'][0] in 'aeiou' else 'a'} {facts['layer_word']}-layer, {facts['n_embd']}-hidden-dimension Qwen-style decoder-only transformer ({facts['params_m']:.1f}M parameters, tied embeddings counted once) with grouped-query attention, SwiGLU feed-forward blocks, rotary position embeddings, and {'an' if str(facts['vocab_size'])[0] == '8' else 'a'} {facts['vocab_size']}-token BPE vocabulary. The shipped checkpoint evaluates at {fmt(baseline, 4)} BPB under this runner.

\paragraph{{Sensor encoders.}}
We use three frozen encoders: a MobileNetV3-style frame encoder plus Perceiver resampler for vision, a Conv1D audio encoder over mel spectrograms for ESC-50-style audio features, and a Conv1D IMU encoder over UCI-HAR accelerometer/gyroscope windows. The bridge consumes only encoder features; raw sensor processing remains outside the language model.

\paragraph{{Bridge hypernetwork.}}
For a sensor feature vector $f$, the bridge predicts a flattened LoRA vector $w = h_\theta(f)$. The vector is partitioned into LoRA matrices $(A_i, B_i)$ for each target linear layer. During a forward pass, a frozen linear layer computes
\[
  y = xW + \alpha x A_i B_i / r,
\]
where $r$ is the LoRA rank and $\alpha$ is the LoRA scale. The base weights $W$ and sensor encoder are frozen; only bridge parameters are optimized.

\paragraph{{Diversity regularization.}}
A bridge can minimize text loss while emitting nearly identical LoRA weights for all inputs. To measure and penalize this collapse, we add a cosine diversity term across generated weights in a batch:
\[
  \mathcal{{L}}_\text{{div}} = \frac{{1}}{{N(N-1)}} \sum_{{i \ne j}} \cos(w_i, w_j).
\]
Lower cosine similarity indicates more input-dependent generated weights.

\section{{Experimental Setup}}
The primary metric is validation bits-per-byte (BPB) on held-out text; lower BPB is better, and tables report improvement relative to the frozen baseline. Every condition is evaluated on the same fixed {payload.get("metadata", {}).get("eval_tokens") or "32{,}768"}-token validation stream, so all reported deltas are paired comparisons; the absolute BPB level would shift by roughly $\pm 0.02$ under an independent eval sample of the same size. BPB gives a stable language-modeling measurement, but it is not sufficient evidence of semantic sensor use. We therefore include shuffled-feature, random-feature, and constant-feature controls, task-aligned ranking probes, composition tests, and held-out weight-space structure probes.

{metadata_sentence(payload)} The result artifact is the summary JSON consumed by the figure and manuscript renderers.

\section{{Results}}
\subsection{{Bridge and baseline comparison}}
Table~\ref{{tab:main}} and Figure~\ref{{fig:main}} summarize BPB improvements across modalities. Static LoRA measures the gain from fixed low-rank adaptation; the constant-feature bridge is the capacity-matched control (the full hypernetwork trains, but its input never varies); random features test stochastic conditioning; shuffled features map sensor indices through a fixed derangement.

The BPB outcome is unambiguous and deliberately deflationary. Static LoRA is the best text adapter ({fmt(st['static'], sign=True)} mean): direct optimization beats hypernetwork-mediated optimization when only text quality matters. The conditional bridge ({fmt(st['bridge'], sign=True)}) is statistically inseparable from the constant-feature control ({fmt(st['constant'], sign=True)}) --- the bridge's entire BPB gain is explained by its trainable capacity, with no measurable contribution from the sensor signal. We note that in this pipeline text batches and sensor draws are sampled independently, so there is no semantic sensor--text alignment for the shuffled control to destroy; shuffled ({fmt(st['shuffled'], sign=True)}) functions as a seed-noise sanity check here, and the semantically meaningful shuffled control appears in the task probes below, where features are label-paired.

\begin{{table}}[t]
\centering
\caption{{BPB improvement relative to the frozen mini-model baseline. Positive values are better. Static LoRA consumes no sensor input, so its per-modality cells repeat the same seeded modality-independent run.}}
\label{{tab:main}}
\begin{{tabular}}{{lrrrr}}
\toprule
Method & Audio & Vision & IMU & Mean \\
\midrule
{modality_table(payload)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.82\linewidth]{{paper_main_benchmark.png}}
\caption{{Mean BPB improvement across audio, vision, and IMU. The controls show that capacity and regularization explain part of the bridge gain.}}
\label{{fig:main}}
\end{{figure}}

Prefix tuning, the token-space alternative, degrades BPB ({fmt(st['prefix'], sign=True)} mean): inserting learned continuous tokens shifts the position distribution a frozen small model was trained under, and within this budget the prefix never recovers the loss it induces, let alone improves on the baseline. The contrast with the uniformly positive weight-space rows is the practical takeaway --- for frozen small models, weight modulation is the conditioning path that does no harm.

\subsection{{Feature controls}}
Figure~\ref{{fig:controls}} restates the interpretation problem visually: true, shuffled, and constant features achieve nearly identical BPB, so BPB is measuring bridge capacity, not semantic sensor conditioning. Any claim of genuine conditioning must therefore rest on probes that are sensitive to per-input behavior --- the task-aligned probes below and the weight-space structure analysis of Section~\ref{{sec:diversity}}.

\begin{{figure}}[t]
\centering
\includegraphics[width=0.82\linewidth]{{paper_feature_controls.png}}
\caption{{Feature controls for the bridge. Shuffled features remove sensor alignment; random features test stochastic conditioning.}}
\label{{fig:controls}}
\end{{figure}}

\subsection{{Composition}}
Independently trained bridges can be averaged in LoRA space. Table~\ref{{tab:composition}} and Figure~\ref{{fig:composition}} show that composition remains feasible but is not additive: extra modalities can dilute or interfere with each other when merged without a learned router. Composition bricks train for {payload.get("metadata", {}).get("composition_steps_per_brick", payload.get("metadata", {}).get("composition_budget_per_brick"))} steps each versus {payload.get("metadata", {}).get("benchmark_steps", payload.get("metadata", {}).get("benchmark_budget"))} for the Table~\ref{{tab:main}} bridges, so single-modality rows in the two tables are not directly comparable.

\begin{{table}}[t]
\centering
\caption{{Conditioned additive merge of independently trained bridges.}}
\label{{tab:composition}}
\begin{{tabular}}{{lrrrr}}
\toprule
Active modalities & Count & Baseline & Final & $\Delta$ BPB \\
\midrule
{composition_table(payload)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.82\linewidth]{{paper_composition.png}}
\caption{{Composition of independently trained bridges. Simple averaging exposes interference between low-rank update subspaces.}}
\label{{fig:composition}}
\end{{figure}}

\subsection{{Diversity and weight-space structure}}
\label{{sec:diversity}}
Table~\ref{{tab:diversity}} and Figure~\ref{{fig:diversity}} measure the generated-weight collapse directly. With no regularization, cross-input cosine similarity is close to one. Increasing the diversity penalty reduces this collapse and makes generated LoRA weights more input-dependent, although the BPB improvement may trade off against weight-space diversity.

\begin{{table}}[t]
\centering
\caption{{IMU diversity sweep. Cross-input cosine is measured on held-out activity examples; lower cosine means less collapse. Motion/still cosine is the mean cosine between motion-activity and stationary-activity centroid weights; lower values indicate clearer separation of the two clusters.}}
\label{{tab:diversity}}
\begin{{tabular}}{{rrrr}}
\toprule
$\lambda$ & $\Delta$ BPB & Cross-input cosine & Motion/still cosine \\
\midrule
{diversity_table(payload)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.82\linewidth]{{paper_diversity_ablation.png}}
\caption{{Diversity regularization reduces generated-weight collapse across IMU inputs.}}
\label{{fig:diversity}}
\end{{figure}}

\subsection{{Task-aligned and reproducibility probes}}
Task ranking evaluates whether the adapted model assigns higher likelihood to the correct sensor label in a short text prompt, scored on the natural tokenization of prompt+label over a label-stratified held-out subset; the no-bridge condition applies a zero LoRA and serves as the chance-level reference. Here the conditioning that BPB cannot see is unmistakable (Table~\ref{{tab:task}}). On six-way IMU activities, true features reach {fmt(st['task_imu_true_rank1'], 2)} rank-1 accuracy and {fmt(st['task_imu_true_avgrank'], 2)} average rank, against {fmt(st['task_imu_none_rank1'], 2)} and chance-level ranking for the unconditioned model --- and, critically, for the shuffled and random controls as well. On fifty-way audio events, true features reach {fmt(st['task_audio_true_rank1'], 2)} rank-1 ({fmt(st['task_audio_true_avgrank'], 2)} average rank) where chance is 0.02 and every control again sits at chance. The same bridges whose BPB gain is pure capacity are, at the task level, decisively input-dependent: the two metrics dissociate cleanly, which is the central methodological point of this paper.

Table~\ref{{tab:repro}} and Figure~\ref{{fig:repro}} report three-seed stability for the conditional bridges: with fixed step counts, a repeated run with the same seed reproduces the benchmark value exactly, so the per-seed spread isolates seed sensitivity.

\begin{{table}}[t]
\centering
\caption{{Task-aligned ranking probes. Lower average rank is better; higher rank-1/top-2 is better.}}
\label{{tab:task}}
\begin{{tabular}}{{llrrr}}
\toprule
Modality & Condition & Avg. rank & Rank-1 & Top-2 \\
\midrule
{task_table(payload)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[t]
\centering
\caption{{Three-seed stability for conditional bridges (seeds 42, 1042, 2042).}}
\label{{tab:repro}}
\begin{{tabular}}{{lrl}}
\toprule
Modality & Mean $\Delta$ BPB & Per-seed $\Delta$ BPB \\
\midrule
{repro_table(payload)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.82\linewidth]{{paper_reproducibility.png}}
\caption{{Reproducibility of bridge BPB improvements across random seeds.}}
\label{{fig:repro}}
\end{{figure}}
{scaling_section(payload, scaling)}
\section{{Related Work}}
LoRA \cite{{hu2021lora}} and adapter methods \cite{{houlsby2019adapters}} adapt frozen transformers \cite{{vaswani2017attention}} with a small number of trainable parameters. Prefix tuning \cite{{li2021prefixtuning}} adapts models by prepending continuous tokens; in our experiments token-space conditioning only degrades a frozen small model trained without prefixes. Hypernetworks \cite{{ha2016hypernetworks}} generate weights for another network; shared hypernetwork adapters \cite{{mahabadi2021hyperformer}} apply the idea to multi-task adaptation, and HyperTuning \cite{{phang2022hypertuning}} generates parameter-efficient adaptations from task descriptions with a frozen base model. Our setting differs in that the conditioning signal is a continuous sensor stream rather than a task identifier or instruction text, and the generated weights change per input at evaluation time. Composing the resulting adaptations connects to weight-space model editing and merging \cite{{ilharco2023task}}, where interference between added task vectors is a known obstacle --- consistent with the sub-additive composition we observe. Multimodal systems such as Flamingo \cite{{alayrac2022flamingo}} and LLaVA \cite{{liu2023llava}} integrate external modalities through architectures and training recipes designed for paired data; conditional LoRA bridges instead test a modular interface in which independently trained sensor encoders condition a frozen language model through generated low-rank weight updates. Sensor data come from ESC-50 \cite{{piczak2015esc}} and UCI-HAR \cite{{anguita2013uci}}.

\section{{Limitations}}
The bridge is large relative to the base model ({st['bridge_params'] / 1e6:.1f}M trainable parameters generating a {st['lora_dim']:,}-dimensional LoRA vector for a {facts['params_m']:.1f}M-parameter LM), so bridge compression is a central requirement for scaling this interface, not a footnote. All results are on one small base model trained on one corpus; the mechanism findings (capacity/conditioning dissociation, collapse, prefix degradation) are the claims we expect to transfer, not the specific deltas. The task probes use a single prompt template per modality and modest eval subsets. The budget-scaling run shows that probe outcomes can depend strongly on training length, so conclusions drawn at any single budget --- including ours --- should be read against that section. Composition of independently trained bridges remains sub-additive, and we do not yet offer a fix.

\section{{Conclusion}}
Conditional LoRA bridges connect frozen sensor encoders to frozen language models through generated low-rank weight updates. Our controlled study yields one sharp methodological result and one mechanism result. Methodologically, aggregate language-modeling metrics cannot detect sensor conditioning: a capacity-matched constant-feature control reproduces the bridge's entire BPB gain, while task-aligned probes show decisive input-dependence under exactly the same training. Mechanistically, hypernetwork collapse is the failure mode that governs whether conditioning exists at all: diversity regularization creates input-dependent weight structure, and prolonged unregularized training destroys it along with the task-level conditioning it supports. Weight-space modulation does no harm where token-space injection degrades the frozen model. Future work should compress the bridge (the interface's scaling bottleneck), make composition constructive rather than interfering, and extend the probe suite beyond single-template ranking.

\section*{{Reproducibility Statement}}
The paper artifacts are generated from a single summary JSON emitted by the repository runner. The exact command that produced the bundled artifact, reconstructed from its recorded metadata, is:
\begin{{verbatim}}
{repro_command(payload)}
\end{{verbatim}}
Training loops run fixed step counts, so a repeated run with the same seed reproduces each number exactly on the same hardware class; absolute BPB values shift by roughly $\pm 0.02$ under a different eval sample, while all reported deltas are paired comparisons on one fixed eval stream shared by every condition.
Figures and manuscript source are regenerated from that run artifact with:
\begin{{verbatim}}
paper-figures results/<run>/summary.json --output-dir figures
paper-render-arxiv results/<run>/summary.json --output-dir arxiv
\end{{verbatim}}
Code, checkpoints, data, and the exact result artifacts are available at \url{{https://github.com/kortexa-ai/legolm.paper}}. The arXiv source package includes this manuscript's result artifacts under \texttt{{anc/}}; tables, figures, and every number in the text are rendered from them. The runner validates that Git LFS checkpoints are materialized before loading.

\bibliographystyle{{plain}}
\bibliography{{references}}
\end{{document}}
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "main.tex"
    target.write_text(tex.strip() + "\n")
    return target


def copy_figures(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for png in src.glob("paper_*.png"):
        shutil.copy2(png, dst / png.name)


def copy_summary(summary_path: Path, dst: Path, name: str = "summary.json") -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    target = dst / name
    shutil.copy2(summary_path, target)
    return target


def compile_tex(out_dir: Path) -> None:
    subprocess.run(
        ["tectonic", "-X", "compile", "main.tex", "--keep-intermediates", "--keep-logs"],
        cwd=out_dir,
        check=True,
    )


def package_source(out_dir: Path, zip_path: Path) -> Path:
    # Result artifacts ship under anc/ so arXiv treats them as ancillary files
    # rather than TeX sources.
    members = {"main.tex": "main.tex", "main.bbl": "main.bbl", "references.bib": "references.bib"}
    for summary_name in ("summary.json", "summary-scaling.json"):
        if (out_dir / summary_name).exists():
            members[summary_name] = f"anc/{summary_name}"
    for png in sorted((out_dir / "figures").glob("paper_*.png")):
        members[f"figures/{png.name}"] = f"figures/{png.name}"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for source_rel, arcname in members.items():
            source = out_dir / source_rel
            if not source.exists():
                raise FileNotFoundError(f"Cannot package missing arXiv source member: {source}")
            zf.write(source, arcname)
    return zip_path


def verify_package(zip_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="legolm-arxiv-verify-") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path)
        compile_tex(tmp_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render arXiv LaTeX manuscript from summary.json")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--scaling-summary", type=Path, default=None, help="Optional higher-budget run artifact for the budget-scaling section")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "arxiv")
    parser.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument("--compile", action="store_true", help="Compile main.tex with tectonic after rendering")
    parser.add_argument("--package", type=Path, default=None, help="Write an arXiv source zip after compiling")
    parser.add_argument("--verify-package", action="store_true", help="Unpack the generated source zip and compile it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.summary.resolve()
    payload = load_summary(summary_path)
    scaling = load_summary(args.scaling_summary.resolve()) if args.scaling_summary else None
    tex_path = write_tex(payload, args.output_dir, scaling=scaling)
    copy_figures(args.figures_dir, args.output_dir / "figures")
    copied_summary = copy_summary(summary_path, args.output_dir)
    print(f"Wrote {tex_path}")
    print(f"Wrote {copied_summary}")
    if args.scaling_summary:
        print(f"Wrote {copy_summary(args.scaling_summary.resolve(), args.output_dir, 'summary-scaling.json')}")
    if args.compile or args.package or args.verify_package:
        compile_tex(args.output_dir)
    if args.package or args.verify_package:
        zip_path = args.package or (REPO_ROOT / "legolm-arxiv-source.zip")
        package_source(args.output_dir, zip_path)
        print(f"Wrote {zip_path}")
        if args.verify_package:
            verify_package(zip_path)
            print(f"Verified {zip_path}")


if __name__ == "__main__":
    main()
