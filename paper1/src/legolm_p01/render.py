"""Render the manuscript from verified evidence, with no hand-entered results."""
from __future__ import annotations

import csv
from datetime import date as calendar_date
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

import numpy as np

from .artifacts import PROMPTS, load, verify
from .aggregate import TRAJECTORY_SEEDS
from .provenance import file_hash, write_json

HOSTS = ("mini", "lfm230m", "qwen08")
MODALITIES = ("imu", "audio", "vision")
HOST_NAMES = {"mini": "Mini", "lfm230m": "LFM 230M", "qwen08": "Qwen 0.8B"}
MOD_NAMES = {"imu": "IMU", "audio": "Audio", "vision": "Vision"}
CONTROLS = (("bridge", "true", "True"), ("bridge", "constant", "Constant"),
            ("bridge", "shuffled", "Shuffled"), ("bridge", "random", "Random"),
            ("static", "true", "Static"))
COLORS = ("#087e8b", "#b26c21", "#9055a2", "#757575", "#253d65")


def tex(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(value))


def number(value, digits=3):
    if value is None:
        return "--"
    if value != 0 and abs(value) < .5 * 10 ** -digits:
        mantissa, exponent = f"{value:.1e}".split("e")
        return "$" + mantissa + r"\times10^{" + str(int(exponent)) + "}$"
    return f"{value if value else 0:.{digits}f}"


def estimate(value, *, interval=True, digits=3):
    if value is None:
        return "--"
    result = number(value["mean"], digits)
    if interval and "ci95" in value:
        result += " [" + ", ".join(number(v, digits) for v in value["ci95"]) + "]"
    return result


def table(headers, rows, caption, label, *, long=False, columns=None):
    columns = columns or "@{}" + "l" * len(headers) + "@{}"
    body = " & ".join(headers) + r" \\" + "\n\\midrule\n"
    body += "\n".join(" & ".join(map(str, row)) + r" \\" for row in rows)
    if long:
        return ("{\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{longtable}{" + columns + "}\n"
                + "\\caption{" + caption + "}\\label{" + label + r"}\\" + "\n\\toprule\n"
                + " & ".join(headers) + r" \\" + "\n\\midrule\\endfirsthead\n\\toprule\n"
                + " & ".join(headers) + r" \\" + "\n\\midrule\\endhead\n"
                + "\n".join(" & ".join(map(str, row)) + r" \\" for row in rows)
                + "\n\\bottomrule\n\\end{longtable}}\\par\n")
    return ("\\begin{table}[!htbp]\\centering\\small\n\\setlength{\\tabcolsep}{4pt}\n"
            + "\\caption{" + caption + "}\\label{" + label + "}\n"
            + "\\begin{tabular}{" + columns + "}\n\\toprule\n" + body
            + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\\par\n")


class Report:
    def __init__(self, summary, output, *, draft=False, evidence=None):
        self.summary, self.output, self.draft = summary, Path(output), draft
        self.evidence = Path(evidence) if evidence else None
        self.fragments = self.output / "generated"
        self.figures = self.output / "figures"
        self.fragments.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        self.by_id = {r["condition_id"]: r for r in summary["conditions"]}

    def write(self, name, content):
        normalized = "\n".join(line.rstrip() for line in content.splitlines()).rstrip() + "\n"
        (self.fragments / (name + ".tex")).write_text(normalized)

    def failures(self):
        failures = self.summary["coverage"].get("numerical_failures", [])
        self.failure_appendix = ""
        if not failures:
            self.write("failure-results", "No numerical training failures are recorded" + (" in this incomplete draft." if self.draft else "."))
            return
        rows = []
        for item in sorted(failures, key=lambda f: json.dumps(f["recipe"], sort_keys=True)):
            r, detail = item["recipe"], item["failure"]
            first, last = detail["step_range"]
            step = str(first) if first == last else f"{first}--{last}"
            rows.append([HOST_NAMES[r["host"]] + "/" + r["cohort"], MOD_NAMES[r["modality"]],
                         tex(r["kind"] + "/" + r["feature_mode"]), str(r["seed"]), f'{r["diversity"]:g}',
                         step, ", ".join(map(str, item["unavailable_checkpoints"]))])
        self.failure_appendix = table(["Host/cohort", "Task", "Interface", "Seed", "$\\lambda$", "Failure step", "Unavailable steps"],
            rows, "Numerical failures under the fixed recipe. A range is bounded by the original execution log; "
            "the separate diagnostic replay is not a replacement trial. Saved checkpoints before failure remain in the evidence.",
            "tab:numerical-failures", long=True)
        total = self.summary["coverage"].get("registered_trajectories", 411)
        self.write("failure-results", str(len(failures)) + " of the " + str(total) + " declared trials terminated with "
            "nonfinite loss or gradient norms before an optimizer update (Table~\\ref{tab:numerical-failures}). "
            "These are fixed-recipe outcomes. Earlier checkpoints do not replace unavailable endpoints, and seeds are not retried "
            "until they succeed. An aggregate requires its complete declared seed set: a dash denotes an unavailable estimate, "
            "not an average over surviving seeds. All available per-seed measurements remain in the evidence.")

    def find(self, host, modality, measurement, *, cohort="U", kind="bridge", mode="true",
             penalty=0., step=None, intervention="correct", prompt=None, trajectory=False, **settings):
        if step is None:
            step = 0 if kind == "none" else (3000 if cohort == "U" else 6000)
        if kind == "none" and cohort == "P":
            cohort = "U"
        if measurement == "ranking":
            settings.update(prompt=prompt or PROMPTS[modality][0], intervention=intervention)
        elif measurement == "native_bpb":
            settings.setdefault("rows", 1024 if step > (300 if cohort == "U" else 600) or kind == "none" else 256)
        elif measurement == "generation":
            settings.setdefault("objective", "P" if cohort in ("U", "P") else "G")
            settings["intervention"] = intervention
        condition = {"host": host, "modality": modality, "cohort": cohort, "kind": kind,
                     "feature_mode": mode, "diversity": penalty, "step": step,
                     "measurement": measurement, "settings": settings}
        collection = "trajectory_conditions" if trajectory else "conditions"
        if self.draft and collection not in self.summary:
            collection = "conditions"
        found = [r for r in self.summary.get(collection, []) if r["condition"] == condition]
        if len(found) == 1:
            return found[0]
        unavailable = [r for r in self.summary.get("unavailable_" + collection, []) if r["condition"] == condition]
        if not found and len(unavailable) == 1 and unavailable[0]["reason"] == "numerical_failure":
            return None
        if not found and self.draft:
            return None
        raise ValueError(f"Missing or ambiguous manuscript condition: {condition}")

    def stat(self, host, modality, measurement, *keys, **kwargs):
        row = self.find(host, modality, measurement, **kwargs)
        value = row["statistics"] if row else None
        for key in keys:
            value = value[key] if value is not None else None
        return value

    def contrast(self, first, second, *keys):
        if first is None or second is None:
            return None
        found = [r for r in self.summary["contrasts"]
                 if r["first"] == first["condition_id"] and r["comparison"] == second["condition_id"]]
        if len(found) != 1:
            if self.draft and not found:
                return None
            raise ValueError("Missing or repeated paired manuscript contrast")
        value = found[0]["statistics"]
        for key in keys:
            value = value[key]
        return value

    def host_table(self):
        rows = []
        for host in HOSTS:
            identity = self.summary["hosts"].get(host)
            if identity is None:
                if self.draft:
                    continue
                raise ValueError("Missing host identity")
            trained = [r["trainable_parameters"] for r in self.summary["training"]
                       if r["recipe"]["host"] == host and r["recipe"]["kind"] == "bridge"
                       and r["recipe"]["modality"] == "imu"]
            if len(set(trained)) > 1:
                raise ValueError("Bridge parameter count differs across otherwise matched recipes")
            count = trained[0] if trained else None
            rows.append([HOST_NAMES[host], number(identity["frozen_parameters"] / 1e6, 2),
                         f'{identity["lora_dim"]:,}', str(len(identity["targets"])),
                         number(count / 1e6, 2) if count else "--",
                         number(count / identity["frozen_parameters"], 2) if count else "--"])
        self.write("hosts", table(["Host", "Host M", "$D$", "Targets", "Bridge M", "Bridge/host"], rows,
            "Instantiated language-host and IMU-bridge sizes. M denotes million parameters. "
            "Audio and vision change the small input projection; their exact counts are in the resource artifact. "
            "The native Qwen vision tower is excluded from these counts.", "tab:hosts"))

    def bpb(self):
        rows, capacity_deltas, static_deltas, prefix_deltas = [], [], [], []
        for host in HOSTS:
            for modality in MODALITIES:
                base = self.stat(host, modality, "native_bpb", "bpb", kind="none")
                true = self.find(host, modality, "native_bpb")
                constant = self.find(host, modality, "native_bpb", mode="constant")
                static = self.find(host, modality, "native_bpb", kind="static")
                delta = self.contrast(true, constant, "bpb")
                rows.append([HOST_NAMES[host], MOD_NAMES[modality], estimate(base, interval=False),
                             estimate(true["statistics"]["bpb"] if true else None),
                             estimate(constant["statistics"]["bpb"] if constant else None, interval=False),
                             estimate(static["statistics"]["bpb"] if static else None, interval=False)])
                if delta:
                    capacity_deltas.append(delta)
                delta = self.contrast(true, static, "bpb")
                if delta:
                    static_deltas.append(delta)
                prefix = self.find(host, modality, "native_bpb", kind="prefix")
                frozen = self.find(host, modality, "native_bpb", kind="none")
                delta = self.contrast(prefix, frozen, "bpb")
                if delta:
                    prefix_deltas.append(delta)
        self.write("bpb", table(["Host", "Task", "Frozen", "True [95\\% interval]", "Const.", "Static"], rows,
            "Unpaired native BPB at 3000 steps (1024 shared rows per host). Lower is better. "
            "True-feature intervals include seeds and text units; complete control intervals and both budgets are in the appendix.", "tab:bpb"))
        capacity_text = "The true-minus-constant comparison is unavailable for the registered seed set. "
        if capacity_deltas:
            means = [d["mean"] for d in capacity_deltas]
            spans_zero = sum(d["ci95"][0] <= 0 <= d["ci95"][1] for d in capacity_deltas)
            capacity_text = ("Across " + str(len(means)) + " host/modality pairs, the true-minus-constant BPB "
                             "differences range from " + number(min(means), 4) + " to " + number(max(means), 4)
                             + "; " + str(spans_zero) + " of their " + str(len(means)) + " intervals include zero. ")
        static_text = "Static adapters provide a sensor-free text-adaptation reference. "
        if static_deltas:
            below = sum(d["ci95"][1] < 0 for d in static_deltas)
            above = sum(d["ci95"][0] > 0 for d in static_deltas)
            static_text += ("For true-minus-static BPB, " + str(above) + " intervals are above zero, "
                            + str(below) + " below zero, and " + str(len(static_deltas) - above - below)
                            + " include zero. Negative differences favor the bridge; the ordering is not uniform. ")
        self.write("text-results", "Table~\\ref{tab:bpb} reports unpaired adaptation at 3000 steps. "
                   + capacity_text + "These within-host comparisons test whether variable features are needed under "
                   "independent text/sensor sampling. Intervals spanning zero do not establish equivalence, and this "
                   "experiment does not rule out a likelihood benefit from conditioning on paired sensor/text data. "
                   + static_text + "Figure~\\ref{fig:bpb} includes every training corruption at both budgets; "
                   "the machine-readable summary retains the paired intervals, and the appendix reports the "
                   "separate common-byte panel.")
        rows = []
        for host in HOSTS:
            for modality in MODALITIES:
                values = [self.stat(host, modality, "native_bpb", "bpb", kind="prefix", step=step)
                          for step in (0, 300, 3000)]
                rows.append([HOST_NAMES[host], MOD_NAMES[modality], *[estimate(v) for v in values]])
        self.write("prefix", table(["Host", "Task", "Initialized", "300 steps", "3000 steps"], rows,
            "Eight generated input-prefix embeddings: native BPB at initialization and both budgets. "
            "Initialization and 300 steps use 256 rows; 3000 steps use 1024. Compare each with its matching frozen panel, "
            "not across different panels. Each score includes the same 128 text positions as its reference.", "tab:prefix"))
        prefix_text = "The trained-prefix comparison is unavailable for the registered seed set. "
        if prefix_deltas:
            means = [d["mean"] for d in prefix_deltas]
            below = sum(d["ci95"][1] < 0 for d in prefix_deltas)
            prefix_text = ("At 3000 U steps, prefix-minus-frozen BPB differences range from "
                           + number(min(means), 4) + " to " + number(max(means), 4) + "; "
                           + str(below) + " of " + str(len(means)) + " paired intervals are below zero. ")
        self.write("prefix-results", prefix_text + "Table~\\ref{tab:prefix} also shows initialization and the short budget. "
                   "The corrected comparison uses the same 128 scored text positions for prefix and reference. "
                   "These results concern eight input-prefix embeddings under the U objective; they do not establish "
                   "sensor use or compare prefix conditioning with the separately trained P and G bridges.")

    def tasks(self):
        rows, contrasts, vision = [], [], []
        for host in HOSTS:
            for modality in MODALITIES:
                if host == "lfm230m" and modality == "vision":
                    continue
                true = self.find(host, modality, "ranking", cohort="P")
                values = [self.stat(host, modality, "ranking", "rank1", "macro", cohort="P", kind=kind, mode=mode)
                          for kind, mode in (("none", "true"), ("static", "true"), ("bridge", "constant"), ("bridge", "shuffled"), ("bridge", "random"))]
                rows.append([HOST_NAMES[host], MOD_NAMES[modality],
                             *[estimate(v, interval=False) for v in values],
                             estimate(true["statistics"]["rank1"]["macro"] if true else None)])
                shuffled = self.find(host, modality, "ranking", cohort="P", intervention="shuffle-0")
                delta = self.contrast(true, shuffled, "rank1", "macro")
                if delta:
                    contrasts.append(HOST_NAMES[host] + "/" + MOD_NAMES[modality] + " " + estimate(delta))
                if modality == "vision" and true:
                    different = self.find(host, modality, "ranking", cohort="P", intervention="different_class")
                    delta = self.contrast(true, different, "rank1", "macro")
                    vision.append(HOST_NAMES[host] + " rank-1 accuracy "
                                  + estimate(true["statistics"]["rank1"]["macro"])
                                  + ", correct-minus-different-class difference " + estimate(delta))
        self.write("tasks", table(["Host", "Task", "Frozen", "Static", "Const.", "Shuf.", "Rand.", "True [95\\%]"], rows,
            "Full-population macro rank-1 accuracy at 6000 P steps, with no diversity penalty. "
            "Uniform-choice references are 1/6 (IMU), 1/50 (audio), and 1/10 (vision). "
            "Mini/Qwen IMU/audio use five seeds; other paired cells use three. LFM vision P is outside the registered grid.", "tab:tasks"))
        linear = self.summary["linear_baselines"]
        linear_text = "; ".join(MOD_NAMES[m] + " " + estimate(linear[m]["linear_feature_classifier"]["rank1"]["macro"]) for m in MODALITIES)
        self.write("task-results", "Table~\\ref{tab:tasks} uses all held-out items, with seed and cluster intervals. "
                   "The frozen-feature linear classifier reaches macro accuracy " + linear_text + ". "
                   "This reference tests whether the bridge offers value beyond information already exposed by its encoder. "
                   "For the same true-trained checkpoint, the correct-minus-first-permutation macro-accuracy differences are "
                   + ("; ".join(contrasts) if contrasts else "unavailable for the registered seed set") + ". "
                   "All five fixed permutations and other feature interventions are retained, so this first-permutation display is not selected by outcome."
                   + ("\n\nThe new vision probes give " + "; ".join(vision) + ". "
                      "They directly test whether image features affect label ranking, addressing a question that U's "
                      "BPB result cannot answer. This is a ten-class ranking test, not visual generation or open-world reasoning."
                      if vision else ""))
        rows = []
        for host in HOSTS:
            for modality in MODALITIES:
                if host == "lfm230m" and modality == "vision":
                    continue
                values = [self.stat(host, modality, "ranking", "rank1", "macro", cohort="P", intervention=i)
                          for i in ("correct", "constant", "off", "different_class", "shuffle-0")]
                rows.append([HOST_NAMES[host], MOD_NAMES[modality], *[estimate(v, interval=False) for v in values]])
        self.write("interventions", table(["Host", "Task", "Correct", "Constant", "Off", "Other class", "Shuffle 0"], rows,
            "Test interventions on identical true-trained P checkpoints at 6000 steps and zero penalty. "
            "Values are macro accuracy. Different-class substitution guarantees a changed class; a permutation can retain it. "
            "Paired intervals, random features, and all five permutations are in the machine-readable analysis.", "tab:interventions"))

    def collapse(self):
        rows, missing = [], []
        gaps = [r["condition"] for r in self.summary.get("unavailable_trajectory_conditions", [])
                if r["reason"] == "numerical_failure"]
        for host in HOSTS:
            for penalty in (0., .01, .05, .1, .2):
                values = []
                for step in (600, 6000):
                    row = self.find(host, "imu", "ranking", cohort="P", penalty=penalty, step=step, trajectory=True)
                    if row:
                        if row["seeds"] != list(TRAJECTORY_SEEDS):
                            raise ValueError("Trajectory intervals require exactly the three shared seeds")
                        statistic = row["statistics"]["rank1"]["macro"]
                        values.append(estimate(statistic))
                    else:
                        values.append("--")
                rows.append([HOST_NAMES[host], f"{penalty:g}", *values])
            unavailable = sum(c["host"] == host and c["step"] == 6000 and c["measurement"] == "ranking" for c in gaps)
            if unavailable:
                missing.append(HOST_NAMES[host] + " (" + str(unavailable) + " of 5 penalty settings)")
        coverage = ("Recorded numerical stops prevent complete three-seed 6000-step estimates for "
                    + "; ".join(missing) + ". "
                    "These gaps do not supply a later accuracy measurement. Earlier per-seed evidence remains available. ") if missing else ""
        zero_penalty, geometry, long_intervals = [], [], []
        for host in ("lfm230m", "qwen08"):
            endpoints = [self.stat(host, "imu", "ranking", "rank1", "macro", cohort="P", step=step, trajectory=True)
                         for step in (600, 6000)]
            if all(value is not None for value in endpoints):
                zero_penalty.append(HOST_NAMES[host] + " " + " to ".join(estimate(v) for v in endpoints))
                long_intervals.append(endpoints[-1])
            variation = self.find(host, "imu", "variation", cohort="P", penalty=.2, trajectory=True)
            sensitivity = self.find(host, "imu", "functional_sensitivity", cohort="P", penalty=.2, trajectory=True)
            if variation is not None and sensitivity is not None:
                v = variation["statistics"]
                kl = sensitivity["statistics"]["mean_kl_to_sensor_mixture"]["mean"]
                geometry.append(HOST_NAMES[host] + ": factor cosine " + number(v["factor_cosine"]["mean"])
                                + ", effective-update cosine " + number(v["effective_cosine"]["mean"])
                                + ", and output KL " + number(kl) + " nats")
        behavior = "At zero penalty, the 600-to-6000-step comparison is " + "; ".join(zero_penalty) + ". " if zero_penalty else ""
        if long_intervals and all(value["ci95"][0] > 1 / 6 for value in long_intervals):
            behavior += ("The available larger-host long-endpoint intervals lie above the six-class chance rate. "
                         "They do not support a universal collapse-to-chance claim at this budget. ")
        geometry_text = ("At the largest declared penalty, $\\lambda=0.2$, and 6000 P steps, the seed means are "
                         + "; ".join(geometry) + ". Factor-space diversity, effective-update direction, and output "
                         "variation are distinct. Cosine also does not measure update magnitude; centered energies "
                         "and norms remain in the full analysis. ") if geometry else ""
        self.write("collapse-results", "Figure~\\ref{fig:trajectories} connects each IMU P accuracy curve to effective-update "
                   "cosine and fixed-text output sensitivity from the same weights. Table~\\ref{tab:budget-endpoints} "
                   "compares 600 with 6000 steps using only seeds 42, 1042, and 2042 at every penalty.\n\n"
                   + table(["Host", "$\\lambda$", "600 steps [95\\%]", "6000 steps [95\\%]"], rows,
                           "IMU P macro rank-1 accuracy and grouped 95\\% intervals for the three shared seeds. "
                           "The zero-penalty row uses the same seed subset as every other penalty; additional primary "
                           "probe seeds do not enter these curves. Dashes mark unavailable full-seed estimates.",
                           "tab:budget-endpoints")
                   + "\n\n" + coverage + behavior + "\n\n" + geometry_text
                   + "Penalty effects differ across hosts and budgets. The full U trajectories are retained "
                   "in the analysis. No penalty is selected by its test-set peak, and the short budget does not define a "
                   "universal collapse threshold.")

    @staticmethod
    def shared_seed_mean(row, statistic):
        pairs = dict(zip(row["seeds"], statistic["per_seed"]))
        if not set(TRAJECTORY_SEEDS) <= pairs.keys():
            raise ValueError("An IMU trajectory estimate requires all three declared shared seeds")
        return float(np.mean([pairs[s] for s in TRAJECTORY_SEEDS]))

    def generation(self):
        rows, references, raw = [], [], []
        for modality in ("imu", "audio"):
            for family in ("label", "sentence", "rule"):
                values = [self.stat("qwen08", modality, "generation", f"{family}/test/{metric}", "macro", cohort="G")
                          for metric in ("semantic_correct", "format_compliant", "joint_success")]
                rows.append([MOD_NAMES[modality], family.capitalize(), *[estimate(v) for v in values]])
                values = [self.stat("qwen08", modality, "generation", f"{family}/test/joint_success", "macro", cohort="G", kind=kind, mode=mode)
                          for kind, mode in (("none", "true"), ("bridge", "constant"), ("bridge", "shuffled"), ("bridge", "random"), ("static", "true"))]
                values += [self.stat("qwen08", modality, "generation", f"{family}/test/joint_success", "macro", cohort="G", kind="none", intervention=i)
                           for i in ("oracle_label_in_text", "classifier_label_in_text")]
                references.append([MOD_NAMES[modality], family.capitalize(), *[estimate(v, interval=False) for v in values]])
        for modality in MODALITIES:
            values = [self.stat("qwen08", modality, "generation", "label/original-ranking/joint_success", "macro", cohort="P", kind=kind)
                      for kind in ("none", "static", "bridge")]
            raw.append([MOD_NAMES[modality], *[estimate(v) for v in values]])
        self.write("generation", table(["Task", "Family", "Semantic [95\\%]", "Format [95\\%]", "Joint [95\\%]"], rows,
            "Qwen G at 6000 steps: class-macro response scores on held-out instruction templates. Each estimate requires all three training seeds. "
            "Joint success requires the exact normalized target; semantic and format scores identify different failure types.", "tab:generation")
            + table(["Task", "Family", "Frozen", "Const.", "Shuf.", "Rand.", "Static", "Oracle", "Classifier"], references,
            "Generation references: class-macro held-out-template joint success, with the same G evaluation prompts and decoding. "
            "Oracle and classifier columns reveal an explicit label to the frozen host. Other columns receive no label text. "
            "Their full intervals and both budgets are retained in the analysis.", "tab:generation-controls")
            + table(["Task", "Frozen [95\\%]", "P static [95\\%]", "P true bridge [95\\%]"], raw,
            "Raw generation from Qwen P checkpoints at 6000 steps, without additional instruction training. "
            "The frozen reference uses the identical original ranking prompt. Exact category output is joint success, averaged by class. "
            "This diagnostic is separate from native-chat G training.", "tab:raw-generation"))
        statements, mapping_pairs, response_results = [], [], []
        for modality in ("imu", "audio"):
            true = self.find("qwen08", modality, "generation", cohort="G")
            swapped = self.find("qwen08", modality, "generation", cohort="G", intervention="different_class")
            family_results, family_values = [], []
            for family in ("label", "sentence"):
                value = self.stat("qwen08", modality, "generation", f"{family}/test/joint_success", "macro", cohort="G")
                delta = self.contrast(true, swapped, f"{family}/test/joint_success", "macro")
                if value is not None:
                    family_values.append(value)
                    family_results.append(family + " success is " + estimate(value)
                                          + " (correct-minus-different-class " + estimate(delta) + ")")
            if family_results:
                text = "For " + MOD_NAMES[modality] + ", " + "; ".join(family_results) + ". "
                if len(family_values) == 2:
                    labels, sentences = family_values
                    text += ("Per-seed label scores are " + ", ".join(number(v) for v in labels["per_seed"])
                             + "; sentence scores are " + ", ".join(number(v) for v in sentences["per_seed"])
                             + ", in seed order " + ", ".join(map(str, true["seeds"])) + ". ")
                    zero = sum(a == 0 and b == 0 for a, b in zip(labels["per_seed"], sentences["per_seed"]))
                    if zero:
                        text += ("One", "Two", "Three")[zero - 1] + " of the three seeds " + ("has" if zero == 1 else "have") + " zero exact success in both families. "
                response_results.append(text)
            delta = self.contrast(true, swapped, "rule/test/joint_success", "macro")
            if delta:
                statements.append(MOD_NAMES[modality] + " " + estimate(delta))
            paired = self.stat("qwen08", modality, "generation", "rule/test/both_rule_mappings", "macro", cohort="G")
            if paired:
                mapping_pairs.append(MOD_NAMES[modality] + " " + estimate(paired))
        self.write("generation-results", "Tables~\\ref{tab:generation}--\\ref{tab:raw-generation} distinguish raw P decoding, "
                   "separately trained instruction responses, and label-in-text references. All values below are "
                   "class-macro scores on held-out templates at 6000 G steps.\n\n"
                   + "\n\n".join(response_results) + "\n\n"
                   "The correct-minus-different-class differences in class-macro held-out-rule joint success are "
                   + ("; ".join(statements) if statements else "unavailable for the registered seed set") + ". "
                   + "The secondary score requiring both reversed mappings to be correct for the same sensor/template is "
                   + ("; ".join(mapping_pairs) if mapping_pairs else "unavailable for the registered seed set") + ". "
                   "The rule panel includes reversed output mappings with each sensor fixed. It therefore tests a bounded "
                   "dependence on the instruction as well as the sensor. Results on these deterministic templates do not "
                   "establish general instruction following. In particular, sensor-dependent labels or fixed sentences "
                   "do not establish reliable rule application. The label-in-text references use a different input interface "
                   "and are practical comparisons, not matched bridge architectures. Every generated response, including invalid and repetitive output, "
                   "is preserved in the evidence.")
        self.generation_examples()

    def generation_examples(self):
        if self.evidence is None:
            self.write("generation-examples", r"\emph{Incomplete draft: generated examples are not yet available.}")
            return
        index = load(self.evidence / "index.json")
        evidence_by_id = {r["evaluation_id"]: r for r in index["evaluations"]}
        rows = []
        for modality in ("imu", "audio"):
            condition = self.find("qwen08", modality, "generation", cohort="G")
            if condition is None:
                continue
            chosen = condition["evaluation_ids"][condition["seeds"].index(42)]
            items = load(self.evidence / evidence_by_id[chosen]["path"])["items"]
            first_id = items[0]["id"]
            selected = [r for r in items if r["id"] == first_id and r["template_split"] == "test"
                        and r["template"] == 0 and r["family"] in ("sentence", "rule")]
            if len(selected) != 3:
                raise ValueError("Expected one sentence and two reversed rules for the fixed example sensor")
            for item in selected:
                response = item["response"].replace("\n", " ")
                if len(response) > 280:
                    response = response[:280] + " [display truncated]"
                sensor = tex(MOD_NAMES[modality] + "/" + str(first_id)).replace("/", r"/\allowbreak{}")
                displayed_response = tex(response).replace("!", r"!\allowbreak{}")
                rows.append([sensor, tex(item["actual_prompt"]), tex(item["target"]), displayed_response])
        if not rows:
            self.write("generation-examples", r"\emph{No generation example is available for a complete registered seed set at this endpoint.}")
            return
        self.write("generation-examples", table(["Sensor", "Held-out instruction", "Target", "Response"], rows,
            "Fixed examples: first registered sensor of each modality, seed 42, 6000 G steps, test template 0. "
            "The sentence and both rule mappings are shown regardless of success. Responses longer than 280 characters "
            "are visibly truncated here; the evidence retains complete output.", "tab:examples", long=True,
            columns=r"@{}p{.13\linewidth}p{.37\linewidth}p{.13\linewidth}p{.28\linewidth}@{}"))

    def composition(self):
        rows = []
        names = {"factor_average": "Factor mean", "effective_sum": "Update sum", "effective_mean": "Update mean"}
        for host in HOSTS:
            cells = [r for r in self.summary["composition"] if r["host"] == host and r["step"] == 1500 and len(r["subset"]) == 3]
            if len(cells) != 3 and not self.draft:
                raise ValueError("Missing full-subset composition operators")
            for cell in sorted(cells, key=lambda r: r["operator"]):
                rows.append([HOST_NAMES[host], names.get(cell["operator"], tex(cell["operator"])),
                             estimate(cell["bpb"]), estimate(cell["delta_to_off"]), estimate(cell["bpb_nonadditivity"])])
        self.write("composition", table(["Host", "Operator", "BPB [95\\%]", "$\\Delta$ frozen [95\\%]", "Nonadditivity [95\\%]"], rows,
            "All three U modalities composed at 1500 constituent steps. Negative changes from frozen improve BPB. "
            "Positive nonadditivity means combined BPB is worse than the prediction from summing standalone BPB improvements. "
            "That prediction is a descriptive reference, not a required law. All subsets and both budgets are in the analysis.", "tab:composition"))

    def resources(self):
        rows, full, costs = [], [], []
        for item in sorted(self.summary["resources"], key=lambda r: (HOSTS.index(r["host"]), r["modality"], r["kind"])):
            row = [HOST_NAMES[item["host"]], MOD_NAMES[item["modality"]], item["kind"].capitalize(),
                   number(item["adapter_parameters"] / 1e6, 2), number(item["memory"]["peak_allocated_gib"], 2),
                   number(item["raw_sensor_to_first_token"]["median_ms"], 1),
                   number(item["cached_sensor_generate_32"]["median_ms"], 1)]
            full.append(row)
            if item["modality"] == "imu" and item["kind"] in ("none", "bridge", "static"):
                rows.append(row)
        for host in HOSTS:
            jobs = [r for r in self.summary["training"] if r["recipe"]["host"] == host]
            seconds = sum(r["elapsed_s"] for r in jobs)
            peak = max((r["memory"].get("peak_allocated_gib", 0) for r in jobs), default=0)
            costs.append(HOST_NAMES[host] + " " + number(seconds / 3600, 2) + " accumulated optimizer-step hours and "
                         + number(peak, 2) + " GiB peak allocated training memory")
        caption = ("Batch-one resource measurements on the isolated RTX PRO 6000. Parameters are trainable adapter millions; "
                   "VRAM is peak allocated GiB with host, adapter, and external encoder resident. First is raw-sensor-to-first-token "
                   "median milliseconds; 32-token is median generation time with cached sensor factors and a fresh text cache. "
                   "Both use 128 input text tokens. Frozen/static paths omit unused sensor work.")
        headers = ["Host", "Task", "Interface", "Params M", "VRAM GiB", "First ms", "32-token ms"]
        self.write("resources", table(headers, rows, caption + " The appendix contains every modality and prefix case.", "tab:resources"))
        self.resource_appendix = table(headers, full, caption, "tab:resources-full", long=True)
        self.write("resource-results", "Table~\\ref{tab:resources} reports measured costs, including the frozen encoder. "
                   "Across successfully trained trajectories, training records contain " + "; ".join(costs) + ". "
                   "These step times include batch preparation and synchronized training, but exclude model loading, "
                   "checkpoint writes, separate evaluation, and failed attempts. The artifact also records the wall time "
                   "of each successful worker attempt. Times overlap under concurrent training and are not isolated GPU-hours "
                   "or total elapsed study time. The measurements include dense-head cost rather than inferring efficiency "
                   "from the number of adapted host parameters.")

    def plots(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                             "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
                             "axes.labelsize": 10, "axes.titlesize": 11, "legend.fontsize": 10})
        fig, axes = plt.subplots(3, 3, figsize=(10, 8), squeeze=False, layout="constrained")
        for column, host in enumerate(HOSTS):
            for row, modality in enumerate(MODALITIES):
                ax = axes[row, column]
                for control, (kind, mode, name) in enumerate(CONTROLS):
                    values = [self.contrast(self.find(host, modality, "native_bpb", kind=kind, mode=mode, step=step),
                                           self.find(host, modality, "native_bpb", kind="none", rows=rows), "bpb")
                              for step, rows in ((300, 256), (3000, 1024))]
                    valid = [(x, v) for x, v in enumerate(values) if v is not None]
                    if valid:
                        xs = [x + (control - 2) * .025 for x, v in valid]
                        ys = [v["mean"] for x, v in valid]
                        errors = [[max(0, v["mean"] - v["ci95"][0]) for x, v in valid],
                                  [max(0, v["ci95"][1] - v["mean"]) for x, v in valid]]
                        ax.errorbar(xs, ys, yerr=errors, marker="o", markersize=3, linewidth=1.2,
                                    color=COLORS[control], label=name, capsize=2)
                ax.axhline(0, color="black", linestyle="--", label="Frozen")
                ax.set_title(HOST_NAMES[host] + " / " + MOD_NAMES[modality])
                ax.set_xticks([0, 1], ["300", "3000"])
                ax.set_xlim(-.15, 1.15)
                ax.grid(axis="y", alpha=.2)
                if column == 0:
                    ax.set_ylabel("BPB change from frozen")
                if row == 2:
                    ax.set_xlabel("Optimizer steps")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=6, frameon=False)
        fig.savefig(self.figures / "bpb-controls.pdf", metadata={"CreationDate": None})
        plt.close(fig)
        fig, axes = plt.subplots(3, 3, figsize=(10, 9), squeeze=False, layout="constrained")
        for column, host in enumerate(HOSTS):
            for row, modality in enumerate(MODALITIES):
                ax = axes[row, column]
                ax.set_title(HOST_NAMES[host] + " / " + MOD_NAMES[modality])
                if host == "lfm230m" and modality == "vision":
                    ax.text(.5, .5, "P vision not registered", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue
                values = [self.stat(host, modality, "ranking", "rank1", "macro", cohort="P", kind=kind, mode=mode)
                          for kind, mode, name in CONTROLS]
                values += [self.stat(host, modality, "ranking", "rank1", "macro", kind="none"),
                           self.summary["linear_baselines"][modality]["linear_feature_classifier"]["rank1"]["macro"]]
                for y, value in enumerate(values):
                    if value:
                        error = [[max(0, value["mean"] - value["ci95"][0])], [max(0, value["ci95"][1] - value["mean"])]]
                        ax.errorbar(value["mean"], y, xerr=error, fmt="o", color=(*COLORS, "black", "#307237")[y], markersize=4, capsize=2)
                ax.set_yticks(range(7), [r[2] for r in CONTROLS] + ["Frozen", "Linear"])
                ax.invert_yaxis()
                ax.set_xlim(-.03, 1.03)
                ax.set_xticks([0, .5, 1.])
                ax.axvline(1 / {"imu": 6, "audio": 50, "vision": 10}[modality], color="black", linestyle=":", linewidth=.8)
                ax.grid(axis="x", alpha=.2)
                if row == 2:
                    ax.set_xlabel("Macro rank-1 accuracy")
        fig.savefig(self.figures / "task-controls.pdf", metadata={"CreationDate": None})
        plt.close(fig)
        fig, axes = plt.subplots(3, 3, figsize=(10, 8), squeeze=False, layout="constrained")
        metrics = (("ranking", ("rank1", "macro"), "Macro rank-1 accuracy"),
                   ("variation", ("effective_cosine",), "Effective-update cosine"),
                   ("functional_sensitivity", ("mean_kl_to_sensor_mixture",), "Mean output KL (nats)"))
        for column, host in enumerate(HOSTS):
            for row, (measurement, keys, label) in enumerate(metrics):
                ax = axes[row, column]
                for control, penalty in enumerate((0., .01, .05, .1, .2)):
                    xs, ys = [], []
                    for step in (0, 100, 300, 600, 1200, 2400, 6000):
                        found = self.find(host, "imu", measurement, cohort="P", penalty=penalty, step=step, trajectory=True)
                        if found:
                            value = found["statistics"]
                            for key in keys:
                                value = value[key]
                            xs.append(step)
                            ys.append(self.shared_seed_mean(found, value))
                    ax.plot(xs, ys, "o-", color=COLORS[control], label=f"λ={penalty:g}", markersize=3, linewidth=1.2)
                ax.set_xscale("symlog", linthresh=100)
                ax.set_xticks([0, 600, 6000], ["0", "600", "6000"])
                ax.grid(alpha=.2)
                if row == 0:
                    ax.set_title(HOST_NAMES[host])
                    ax.set_ylim(-.03, 1.03)
                if row == 2:
                    ax.set_xlabel("Optimizer steps")
                    ax.set_ylim(bottom=min(0., ax.dataLim.ymin))
                if column == 0:
                    ax.set_ylabel(label)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=5, frameon=False)
        fig.savefig(self.figures / "imu-trajectories.pdf", metadata={"CreationDate": None})
        plt.close(fig)

    def appendix(self):
        fragments = ["The machine-readable summary contains every condition, seed estimate, paired contrast, "
                     "intervention, and intermediate checkpoint. The following tables retain both primary endpoints "
                     "and the declared penalty values. A final release accounts for every declared trial, including numerical failures. "
                     "For an unavailable aggregate, $n=a/b$ gives available/declared seeds; no survivor-only estimate is reported.\n",
                     self.failure_appendix]
        all_conditions = [*self.summary["conditions"], *[{**r, "statistics": None, "seeds": r["available_seeds"]}
                                                       for r in self.summary.get("unavailable_conditions", [])]]
        for cohort, measurement in (("U", "native_bpb"), ("U", "common_bpb"), ("P", "ranking")):
            rows = []
            for record in sorted(all_conditions, key=lambda r: json.dumps(r["condition"], sort_keys=True)):
                c, stats = record["condition"], record["statistics"]
                if c["cohort"] != cohort or c["measurement"] != measurement:
                    continue
                if c["step"] not in ((0, 300, 3000) if cohort == "U" else (600, 6000)):
                    continue
                if measurement == "ranking" and c["settings"] != {"prompt": PROMPTS[c["modality"]][0], "intervention": "correct"}:
                    continue
                interface = "Frozen" if c["kind"] == "none" else (c["feature_mode"] if c["kind"] == "bridge" else c["kind"])
                row = [HOST_NAMES[c["host"]], MOD_NAMES[c["modality"]], tex(interface), f'{c["diversity"]:g}', str(c["step"]), str(len(record["seeds"]))]
                if stats is None:
                    row[5] += "/" + str(len(record["expected_seeds"]))
                    row += ["--", "--"]
                    if measurement == "native_bpb":
                        row[4] += "/" + str(c["settings"]["rows"])
                elif measurement == "ranking":
                    row += [estimate(stats["rank1"]["macro"]), estimate(stats["rank1"]["micro"])]
                else:
                    row += [estimate(stats["bpb"]), number(stats["bpb"]["seed_sd"])]
                    if measurement == "native_bpb":
                        row[4] += "/" + str(c["settings"]["rows"])
                rows.append(row)
            title = {"native_bpb": "Native BPB; step/row count is shown explicitly", "common_bpb": "Common-byte BPB on identical held-out spans", "ranking": "Paired P ranking on the full held-out population"}[measurement]
            headers = ["Host", "Task", "Interface", "$\\lambda$", "Step" + ("/rows" if measurement == "native_bpb" else ""), "$n$"]
            headers += ["Macro [95\\%]", "Micro [95\\%]"] if measurement == "ranking" else ["BPB [95\\%]", "Seed SD"]
            fragments.append(table(headers, rows, title + ". $n$ is the number of training seeds; a frozen host is a fixed reference.", "tab:full-" + measurement, long=True))
        audit_rows = []
        for modality, value in self.summary["linear_baselines"].items():
            for name, baseline in value.items():
                audit_rows.append([MOD_NAMES[modality], "Linear" if name == "linear_feature_classifier" else "Original head",
                                   estimate(baseline["rank1"]["macro"]), estimate(baseline["rank1"]["micro"])])
        fragments.append(table(["Task", "Frozen-feature reference", "Macro [95\\%]", "Micro [95\\%]"], audit_rows,
            "Full-population feature references. Their classifiers are fixed in the bootstrap; intervals reflect held-out clusters, "
            "not independently trained encoder seeds. Audio exposure remains as stated in the protocol.", "tab:linear"))
        sensitivity = []
        for record in self.summary["conditions"]:
            c, stats = record["condition"], record["statistics"]
            if (c["cohort"] == "P" and c["measurement"] == "ranking" and c["modality"] == "audio"
                and c["step"] == 6000 and c["diversity"] == 0 and c["kind"] == "bridge"
                and c["feature_mode"] == "true" and c["settings"] == {"prompt": PROMPTS["audio"][0], "intervention": "correct"}):
                secondary = stats["source_disjoint_sensitivity"]
                sensitivity.append([HOST_NAMES[c["host"]], str(secondary["excluded_items"]),
                                    estimate(stats["rank1"]["macro"]), estimate(secondary["macro"])])
        fragments.append(table(["Host", "Excluded clips", "All test [95\\%]", "Source-disjoint [95\\%]"], sensitivity,
            "Audio sensitivity analysis for known original sources crossing train/test. Excluding these sources does not undo "
            "the encoder's held-out normalization exposure.", "tab:audio-audit"))
        fragments.append(self.resource_appendix)
        self.write("appendix", "\n".join(fragments))
        # A lossless tabular index keeps all cells accessible without hundreds of PDF pages.
        with (self.output / "condition-index.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, lineterminator="\n", fieldnames=["scope", "condition_id", "condition", "status", "seeds", "expected_seeds", "statistics"])
            writer.writeheader()
            trajectories = [*self.summary.get("trajectory_conditions", []),
                            *[{**r, "statistics": None, "seeds": r["available_seeds"]}
                              for r in self.summary.get("unavailable_trajectory_conditions", [])]]
            for scope, record in [("primary", r) for r in all_conditions] + [("fixed_shared_seeds", r) for r in trajectories]:
                value = {**record, "scope": scope, "status": "estimated" if record["statistics"] is not None else "unavailable_" + record["reason"],
                         "expected_seeds": record.get("expected_seeds", record["seeds"])}
                writer.writerow({k: json.dumps(value[k], sort_keys=True) if k not in ("scope", "condition_id", "status") else value[k] for k in writer.fieldnames})

    def abstract(self):
        capacity = []
        for host in HOSTS:
            for modality in MODALITIES:
                delta = self.contrast(self.find(host, modality, "native_bpb"),
                                      self.find(host, modality, "native_bpb", mode="constant"), "bpb")
                if delta:
                    capacity.append(delta)
        capacity_text = ""
        if capacity:
            zero = sum(d["ci95"][0] <= 0 <= d["ci95"][1] for d in capacity)
            capacity_text = ("Under independent sampling, " + str(zero) + " of " + str(len(capacity))
                             + " true-minus-constant BPB intervals include zero. This does not establish equivalence "
                             "or extend to all paired likelihood objectives. ")
        results = []
        for host in ("mini", "qwen08"):
            values = [(MOD_NAMES[modality], self.stat(host, modality, "ranking", "rank1", "macro", cohort="P"))
                      for modality in MODALITIES]
            available = [number(value["mean"]) + " on " + (name if name == "IMU" else name.lower())
                         for name, value in values if value is not None]
            if available:
                results.append(HOST_NAMES[host] + " reaches macro rank-1 accuracy " + ", ".join(available))
        generation = [self.stat("qwen08", "imu", "generation", f"{family}/test/joint_success", "macro", cohort="G")
                      for family in ("label", "sentence")]
        mappings = self.stat("qwen08", "imu", "generation", "rule/test/both_rule_mappings", "macro", cohort="G")
        generated = ("On held-out templates, Qwen's IMU joint label and sentence success is "
                     + " and ".join(number(v["mean"]) for v in generation)
                     + "; sensor replacement tests their input dependence. "
                     + ("Success on both reversed rule mappings is " + number(mappings["mean"]) + ". " if mappings else "")
                     + "Audio generation varies substantially across seeds. ") if all(v is not None for v in generation) else ""
        self.write("abstract", "A hypernetwork can condition a frozen language model by generating low-rank weight updates from "
                   "sensor features. The same bridge also adds trainable capacity, so improved text likelihood alone is an "
                   "ambiguous indicator of sensor use. We separate these effects with independently sampled text, paired "
                   "category probes, and a separately trained instruction-generation cohort. The study retains the original "
                   "33.6M host and adds pinned 230M and 0.8B hosts, with matched constant, shuffled, random, static-adapter, "
                   "and frozen references. " + capacity_text
                   + "Full held-out populations, grouped intervals, and same-checkpoint interventions assess conditioning. "
                   + ("Numerical training failures affected " + str(len(self.summary["coverage"].get("numerical_failures", [])))
                      + " of the " + str(self.summary["coverage"].get("registered_trajectories", 411))
                      + " declared trials; the corresponding endpoint estimates are unavailable. "
                      if self.summary["coverage"].get("numerical_failures") else "")
                   + (("; ".join(results) + ". ") if results else "") + generated
                   + "We report training-budget trajectories, effective-update variation, three composition operators, and "
                   "measured bridge costs. The dense output head grows with adapted dimensions, and workstation results do "
                   "not establish edge efficiency. Audio evidence is limited by the original encoder's use of held-out "
                   "inputs in normalization. Claims about likelihood, input prefixes, and generation are scoped to the "
                   "tested objectives and interfaces; the complete evidence and per-paper reproduction code accompany the revision.")


def template_directory():
    here = Path(__file__).resolve()
    for candidate in (here.parents[2] / "manuscript", here.parent / "manuscript"):
        if (candidate / "template.tex").is_file() and (candidate / "references.bib").is_file():
            return candidate
    raise FileNotFoundError("The installed package is missing its manuscript template or bibliography")


def validate_summary(summary_path, evidence, *, draft=False):
    summary = load(summary_path)
    if summary["schema"] != "paper1-summary-v2":
        raise ValueError("Unknown summary schema")
    if not draft:
        if summary["status"] != "complete" or evidence is None:
            raise ValueError("A final manuscript requires complete, locally verifiable evidence")
        coverage = verify(evidence)
        if coverage != summary["coverage"]:
            raise ValueError("Summary coverage differs from verified evidence")
        if file_hash(Path(evidence) / "manifest.json") != summary["evidence_manifest_sha256"]:
            raise ValueError("Summary does not refer to this evidence manifest")
        for name, sha in summary["analysis_source"].items():
            if file_hash(Path(__file__).with_name(name)) != sha:
                raise ValueError("Statistical analysis source changed; aggregate before rendering")
        if summary["uncertainty"]["replicates"] != 10000:
            raise ValueError("Final analysis requires the registered 10000 bootstrap replicates")
    elif evidence is not None:
        verify(evidence, incomplete=True)
        if file_hash(Path(evidence) / "manifest.json") != summary["evidence_manifest_sha256"]:
            raise ValueError("Draft summary/evidence identity mismatch")
    return summary


def compile_document(directory):
    directory = Path(directory)
    result = subprocess.run(["tectonic", "--keep-logs", "--keep-intermediates", "main.tex"], cwd=directory,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (directory / "compile-output.txt").write_text(result.stdout)
    log = result.stdout + "\n" + (directory / "main.log").read_text(errors="replace") if (directory / "main.log").exists() else result.stdout
    bib = directory / "main.bbl"
    failures = ("There were undefined references", "Citation `", "I couldn't open database", "Missing character:")
    if result.returncode or not bib.exists() or bib.stat().st_size < 100 or any(s in log for s in failures):
        raise RuntimeError("Manuscript compilation or bibliography validation failed; inspect compile-output.txt and main.log")
    if not (directory / "main.pdf").is_file():
        raise RuntimeError("Compiler did not produce main.pdf")
    return {"pdf_sha256": file_hash(directory / "main.pdf"), "pdf_bytes": (directory / "main.pdf").stat().st_size,
            "bibliography_bytes": bib.stat().st_size, "overfull_boxes": log.count("Overfull \\hbox")}


def package_source(directory, archive):
    directory, archive = Path(directory), Path(archive)
    files = [directory / name for name in ("main.tex", "references.bib", "release.json", "condition-index.csv")]
    files += sorted((directory / "generated").glob("*.tex")) + sorted((directory / "figures").glob("*.pdf"))
    manifest = {str(p.relative_to(directory)): file_hash(p) for p in files}
    write_json(directory / "source-manifest.json", manifest)
    files.append(directory / "source-manifest.json")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in files:
            output.write(path, path.relative_to(directory))
    with tempfile.TemporaryDirectory(prefix="paper1-source-verify-") as temporary:
        with zipfile.ZipFile(archive) as source:
            source.extractall(temporary)
        root = Path(temporary)
        if any(file_hash(root / name) != sha for name, sha in load(root / "source-manifest.json").items()):
            raise ValueError("Packaged manuscript source integrity failure")
        compiled = compile_document(root)
    return {"archive_sha256": file_hash(archive), "archive_bytes": archive.stat().st_size,
            "fresh_directory_compile": compiled, "passed": True}


def run(summary, evidence, output, date, *, draft=False, compile=False, package=None):
    calendar_date.fromisoformat(date)
    data = validate_summary(summary, evidence, draft=draft)
    report = Report(data, output, draft=draft, evidence=evidence)
    templates = template_directory()
    report.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(templates / "template.tex", report.output / "main.tex")
    if (templates / "references.bib").resolve() != (report.output / "references.bib").resolve():
        shutil.copyfile(templates / "references.bib", report.output / "references.bib")
    shown_date = calendar_date.fromisoformat(date).strftime("%B %d, %Y")
    report.write("release", "\\newcommand{\\RevisionDate}{" + shown_date + (r"\\DRAFT: incomplete evidence" if draft else " --- revision 2") + "}\n"
                 + (r"\usepackage{fancyhdr}\pagestyle{fancy}\fancyhf{}\fancyhead[C]{\small DRAFT: incomplete evidence}\fancyfoot[C]{\thepage}\setlength{\headheight}{14pt}" if draft else ""))
    report.host_table()
    report.failures()
    report.bpb()
    report.tasks()
    report.collapse()
    report.generation()
    report.composition()
    report.resources()
    report.plots()
    report.appendix()
    report.abstract()
    report.write("revision-notes", "The original root-level \\texttt{arxiv/} and \\texttt{src/} remain the historical release. "
                 "This revision replaces its primary estimates with the corrected, explicitly registered matrix. "
                 "Changes include matched prefix text budgets, separation of training and test corruption, native tokenization "
                 "at label boundaries, full-population evaluation, and fresh larger-host runs. The new results must not be "
                 "pooled with historical scores from a different evaluation protocol. The matrix identifier is "
                 + r"\texttt{" + data["matrix_id"][:12] + "}. The evidence manifest SHA-256 is recorded in "
                 r"\texttt{release.json}, along with the exact summary and rendering sources. "
                 "The source archive includes its bibliography and is compiled in a fresh directory before release. "
                 "The original mini tokenizer is distributed exactly; it is not reconstructed from a new tokenizer-training run.")
    metadata = {"schema": "paper1-manuscript-v2", "status": "draft" if draft else "final", "date": date,
                "matrix_id": data["matrix_id"], "summary_sha256": file_hash(summary),
                "evidence_manifest_sha256": data["evidence_manifest_sha256"],
                "renderer_sha256": file_hash(Path(__file__)), "template_sha256": file_hash(templates / "template.tex"),
                "bibliography_sha256": file_hash(templates / "references.bib")}
    write_json(report.output / "release.json", metadata)
    result = {"status": metadata["status"], "source": str(report.output / "main.tex")}
    if compile:
        result["compile"] = compile_document(report.output)
    if package:
        if draft:
            raise ValueError("A draft cannot be packaged as a release")
        result["source_package"] = package_source(report.output, package)
    write_json(report.output / "render-validation.json", result)
    return result
