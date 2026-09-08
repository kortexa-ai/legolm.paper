"""Recompute manuscript evidence from saved responses; no model is loaded.

Bootstrap intervals resample images or episodes, retaining all tested seeds
and paired interventions for each selected unit. They describe sampling
variation conditional on the tested model/adapter seeds and source population.
They are pointwise, post hoc intervals, not replacements for recorded bands.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE = Path(__file__).resolve().parents[2]
MODES = {"social": "greet", "caution": "stop", "explore": "proceed"}
SEEDS = (42, 1042, 2042)
HOSTS = ("450m", "3b", "27b")
REPLICATES = 10000
BOOTSTRAP_SEED = 20260907
JSON_OBJECT = re.compile(r"\{[^{}]*\}")
ACTION_WORD = re.compile(r"\b(greet|stop|proceed)\b", re.IGNORECASE)


class EvidenceError(ValueError):
    """The saved records do not support the requested comparison."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: Any, expected: Any, where: str, tolerance: float = 2.1e-6) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise EvidenceError(f"{where}: {actual!r} != {expected!r}")
    elif not (math.isfinite(float(actual)) and math.isfinite(float(expected))):
        raise EvidenceError(f"{where}: non-finite value")
    elif abs(float(actual) - float(expected)) > tolerance:
        raise EvidenceError(f"{where}: recomputed {actual} != recorded {expected}")


def row_key(row: dict) -> tuple:
    return (row["image_id"], row["label"])


def parsed_action(text: str) -> tuple[str, str]:
    """Replay the published parser without importing a model runtime."""
    for match in JSON_OBJECT.finditer(text):
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        action = str(value.get("action", "")).strip().lower() if isinstance(value, dict) else ""
        if action in MODES.values():
            return action, "json"
    word = ACTION_WORD.search(text)
    return (word.group(1).lower(), "word") if word else ("other", "fallback")


def indexed(rows: list[dict], label: str | None = None) -> dict[tuple, dict]:
    chosen = [r for r in rows if label is None or r["label"] == label]
    result = {row_key(r): r for r in chosen}
    if len(result) != len(chosen):
        raise EvidenceError("duplicate observation identity in an arm")
    if not result:
        raise EvidenceError("empty analysis population")
    return result


def paired_values(records: list[dict], mode: str, terms: dict[str, float],
                  label: str = "present") -> tuple[np.ndarray, list[tuple]]:
    """Return seed x image paired contrasts, aligning by identity, not row order."""
    keys = None
    values = []
    for record in records:
        arrays = {arm: indexed(record["arms"][mode][arm], label) for arm in terms}
        for rows in arrays.values():
            current = sorted(rows)
            if keys is None:
                keys = current
            if current != keys:
                raise EvidenceError("paired comparison changed the observation population")
        values.append([sum(coefficient * (arrays[arm][key]["action"] == MODES[mode])
                           for arm, coefficient in terms.items()) for key in keys])
    return np.asarray(values, dtype=float), keys or []


def estimate(values: np.ndarray, *, key: str, unit: str = "image",
             replicates: int = REPLICATES) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
        raise EvidenceError("estimate requires finite seed x observation values")
    per_unit = values.mean(axis=0)
    seed = int.from_bytes(hashlib.sha256(f"{BOOTSTRAP_SEED}:{key}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(per_unit), size=(replicates, len(per_unit)))
    boot = per_unit[draws].mean(axis=1)
    interval = np.quantile(boot, [0.025, 0.975])
    return {"mean": float(per_unit.mean()), "interval": interval.tolist(),
            "seed_means": values.mean(axis=1).tolist(), "n_units": len(per_unit),
            "n_seeds": len(values), "unit": unit,
            "degenerate_bootstrap": bool(interval[0] == interval[1])}


def cuts_to_wakes(rows: list[dict]) -> dict:
    """Measure every post-start cut, including right-censored episode endings.

The event is the next gate wake, irrespective of whether the model's action
would change. Episode starts are reported separately. Multiple cuts can await
the same wake. Censoring is administrative at the last observed step.
"""
    episodes = collections.defaultdict(list)
    for row in rows:
        episodes[row["episode"]].append(row)
    cuts = []
    starts = 0
    for episode, items in sorted(episodes.items()):
        items = sorted(items, key=lambda r: r["step"])
        if len({r["step"] for r in items}) != len(items):
            raise EvidenceError("duplicate episode step")
        end = items[-1]["step"]
        wakes = [r["step"] for r in items if r["woke"]]
        for row in items:
            if not row["cut"]:
                continue
            if row["step"] == 0:
                starts += 1
                continue
            next_wake = next((step for step in wakes if step >= row["step"]), None)
            cuts.append({"episode": episode, "cut_step": row["step"],
                         "observed_wake": next_wake is not None,
                         "duration": (next_wake if next_wake is not None else end) - row["step"]})
    observed = [r["duration"] for r in cuts if r["observed_wake"]]
    censored = [r["duration"] for r in cuts if not r["observed_wake"]]
    # Survival is over cuts; these are descriptive counts, not independent
    # survival observations for a population-level significance test.
    survival = 1.0
    km = []
    for duration in sorted({r["duration"] for r in cuts}):
        risk = sum(r["duration"] >= duration for r in cuts)
        events = sum(r["duration"] == duration and r["observed_wake"] for r in cuts)
        censor = sum(r["duration"] == duration and not r["observed_wake"] for r in cuts)
        survival *= 1 - events / risk
        km.append({"duration": duration, "at_risk": risk, "wakes": events,
                   "censored": censor, "survival": survival})
    return {"episode_starts": starts, "post_start_cuts": len(cuts),
            "observed_wakes": len(observed), "right_censored": len(censored),
            "observed_mean": float(np.mean(observed)) if observed else None,
            "observed_median": float(np.median(observed)) if observed else None,
            "observed_max": max(observed) if observed else None,
            "observed_over_one": sum(x > 1 for x in observed),
            "censored_duration": censored, "records": cuts, "kaplan_meier": km}


def detector_gap(cells: dict[str, list[dict]], mode: str, label: str = "present") -> dict:
    oracle = indexed(cells["oracle"], label)
    classifier = indexed(cells["classifier"], label)
    flipped = indexed(cells["flipped"], label)
    if oracle.keys() != classifier.keys() or oracle.keys() != flipped.keys():
        raise EvidenceError("detector comparison changed the image population")
    target = MODES[mode]
    errors = []
    contributions = []
    for key, row in classifier.items():
        wrong = row["verdict"] != row["label"]
        # FLIPPED inverts the detector verdict, not the annotation. On an
        # error it therefore selects the oracle packet; on a correct verdict
        # the ordinary detector arm selects the oracle packet.
        selected = flipped[key] if wrong else row
        if selected["action"] != oracle[key]["action"]:
            raise EvidenceError("the switch did not reproduce its selected packet action")
        delta = int(oracle[key]["action"] == target) - int(row["action"] == target)
        contributions.append(int(oracle[key]["action"] == target) - int(row["action"] == target))
        if wrong:
            errors.append(delta)
    error_rate = len(errors) / len(oracle)
    conditional_cost = float(np.mean(errors)) if errors else None
    gap = float(np.mean(contributions))
    identity = error_rate * conditional_cost if errors else 0.0
    close(gap, identity, "error-conditioned detector-gap identity", 1e-12)
    return {"oracle_minus_detector": gap, "error_rate": error_rate,
            "error_conditional_cost": conditional_cost, "errors": len(errors),
            "n_images": len(oracle), "absolute_gap_upper_bound": error_rate}


def audit_image_record(record: dict, stem: str) -> dict:
    checked = 0
    parses = collections.Counter()
    for mode, cells in record["arms"].items():
        if mode not in MODES:
            raise EvidenceError(f"{stem}: unknown mode {mode}")
        expected_keys = None
        metrics = record["metrics"][mode]
        for arm, rows in cells.items():
            keys = set(indexed(rows))
            if expected_keys is None:
                expected_keys = keys
            if keys != expected_keys:
                raise EvidenceError(f"{stem}: unpaired image arms")
            parses.update(r.get("parse", "unrecorded") for r in rows)
            for row in rows:
                if "decoded" not in row or parsed_action(row["decoded"]) != (row["action"], row["parse"]):
                    raise EvidenceError(f"{stem}:{mode}:{arm}: stored parse differs from decoded text")
            for label in ("present", "absent"):
                subset = [r for r in rows if r["label"] == label]
                if not subset:
                    raise EvidenceError(f"{stem}: missing {label} observations")
                share = sum(r["action"] == MODES[mode] for r in subset) / len(subset)
                close(share, metrics[f"{label}_share"][arm], f"{stem}:{mode}:{label}:{arm}")
                checked += 1
                for action, reported in metrics.get(f"{label}_mix", {}).get(arm, {}).items():
                    close(sum(r["action"] == action for r in subset) / len(subset), reported,
                          f"{stem}:{mode}:{label}:{arm}:{action}")
                    checked += 1
            if "parse_fallback_share" in metrics:
                close(sum(r.get("parse") == "fallback" for r in rows) / len(rows),
                      metrics["parse_fallback_share"][arm], f"{stem}:{mode}:{arm}:fallback")
                checked += 1
        p = metrics["present_share"]
        linear = {"lean": ("packet", "habit"), "text_lean": ("text", "habit"),
                  "swap": ("packet", "shuffled"), "oracle_lean": ("oracle", "habit"),
                  "fidelity": ("classifier", "oracle"), "miss_cost": ("oracle", "flipped"),
                  "gain": ("boxes", "switch"), "silence_cost": ("boxes", "lying"),
                  "switch_miss_cost": ("switch", "habit"),
                  "additivity": ("packet+congruent", "packet")}
        for name, (a, b) in linear.items():
            if name in metrics:
                close(p[a] - p[b], metrics[name], f"{stem}:{mode}:{name}")
                checked += 1
        if "ownership" in metrics:
            ownership = metrics["ownership"]
            numerator = p["packet+hostile"] - p["hostile-text"]
            denominator = p["packet"] - p["habit"]
            close(numerator, ownership["numerator"], f"{stem}:{mode}:ownership numerator")
            close(denominator, ownership["denominator"], f"{stem}:{mode}:ownership denominator")
            guarded = abs(denominator) < ownership["guard"]
            if guarded != ownership["guarded"]:
                raise EvidenceError(f"{stem}:{mode}: ownership guard differs")
            close(None if guarded else numerator / denominator, ownership["value"], f"{stem}:{mode}:ownership")
            checked += 3
    return {"scalar_checks": checked, "recorded_parse_counts_with_repeated_arms": dict(parses)}


def audit_host_probe(record: dict) -> dict:
    """The capsule diagnostic has flat text/habit arms, both in caution mode."""
    arms = {arm: indexed(rows) for arm, rows in record["arms"].items()}
    if arms["habit"].keys() != arms["text"].keys():
        raise EvidenceError("capsule host probe changed its image population")
    shares = {}
    for arm, rows in arms.items():
        for row in rows.values():
            if parsed_action(row["decoded"]) != (row["action"], row["parse"]):
                raise EvidenceError("capsule host probe parse differs from decoded text")
        shares[arm] = {}
        for label in ("present", "absent"):
            subset = [r for r in rows.values() if r["label"] == label]
            shares[arm][label] = sum(r["action"] == "stop" for r in subset) / len(subset)
            close(shares[arm][label], record["stop_shares"][arm][label],
                  f"capsule host probe:{arm}:{label}")
    discrimination = shares["text"]["present"] - shares["text"]["absent"]
    close(discrimination, record["reading"]["discrimination"], "capsule host discrimination")
    return {"scalar_checks": 5, "shares": shares, "discrimination": discrimination,
            "n_images": len(arms["text"])}


def load_evidence(directory: Path, *, new_run: bool = False) -> tuple[dict[str, dict], dict]:
    if new_run:
        artifacts = {}
        entries = []
        for path in sorted(directory.glob("vision-mode-*.json")):
            record = json.loads(path.read_text())
            if record.get("smoke"):
                raise EvidenceError(f"smoke artifact in a full-study rerun: {path.name}")
            artifacts[path.stem] = record
            entries.append({"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
        if not artifacts:
            raise EvidenceError("no new experiment records found")
        heads = sorted({r["git_head"] for r in artifacts.values() if r.get("git_head")})
        return artifacts, {"evidence_kind": "new run; not the published observation set",
                           "source_commit": heads[0] if len(heads) == 1 else None,
                           "recorded_source_commits": heads, "files": len(entries),
                           "bytes": sum(r["bytes"] for r in entries), "records": entries}
    manifest_path = directory.parent / "provenance/evidence-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifacts = {}
    for entry in manifest["records"]:
        path = directory / Path(entry["path"]).name
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise EvidenceError(f"evidence hash mismatch: {path.name}")
        data = json.loads(path.read_text())
        if data.get("smoke"):
            raise EvidenceError(f"smoke artifact in publication evidence: {path.name}")
        artifacts[path.stem] = data
    if {p.stem for p in directory.glob("*.json")} != set(artifacts):
        raise EvidenceError("artifact inventory differs from the evidence manifest")
    return artifacts, {"manifest_sha256": sha256(manifest_path),
                       "source_commit": manifest["source_commit"],
                       "files": len(artifacts), "bytes": sum(r["bytes"] for r in manifest["records"])}


def seeded(artifacts: dict, stem: str, seeds: tuple[int, ...] = SEEDS) -> list[dict]:
    records = []
    for seed in seeds:
        record = artifacts[f"{stem}-s{seed}"]
        if record.get("seed") != seed:
            raise EvidenceError(f"{stem}: seed identity mismatch")
        records.append(record)
    return records


def image_summary(records: list[dict], stem: str, mode: str, *, unit: str = "image") -> dict:
    cells = records[0]["arms"][mode]
    estimates = {}
    for label in ("present", "absent"):
        for arm in cells:
            values, _ = paired_values(records, mode, {arm: 1}, label)
            estimates[f"{label}:{arm}"] = estimate(values, key=f"{stem}:{mode}:{label}:{arm}", unit=unit)
    contrasts = {"packet-habit": {"packet": 1, "habit": -1},
                 "packet-text": {"packet": 1, "text": -1},
                 "packet-shuffled": {"packet": 1, "shuffled": -1},
                 "detector-oracle": {"classifier": 1, "oracle": -1},
                 "boxes-switch": {"boxes": 1, "switch": -1},
                 "hostile-effect": {"packet+hostile": 1, "hostile-text": -1}}
    for name, terms in contrasts.items():
        if set(terms).issubset(cells):
            values, _ = paired_values(records, mode, terms)
            estimates[name] = estimate(values, key=f"{stem}:{mode}:{name}", unit=unit)
    return {"stem": stem, "mode": mode, "seeds": [r["seed"] for r in records],
            "estimates": estimates, "recorded_readings": [r.get("reading") for r in records],
            "packet_present_by_phrase": [{str(position): float(np.mean([row["action"] == MODES[mode] for row in r["arms"][mode].get("packet", []) if row["label"] == "present" and row.get("position") == position])) for position in sorted({row["position"] for row in r["arms"][mode].get("packet", []) if row["label"] == "present" and "position" in row})} for r in records],
            "checkpoint_identities": [r["doors"].get("checkpoints", {k: r["doors"][k] for k in ("mode", "constant") if k in r["doors"]}) for r in records]}


def gate_summary(records: list[dict], stem: str, mode: str, gate: str) -> dict:
    arrays = collections.defaultdict(list)
    delays = []
    if len({r["episodes"]["manifest_sha256"] for r in records}) != 1:
        raise EvidenceError("episode pixels changed between tested seeds")
    for record in records:
        baseline = record["arms"][mode]["always"]
        rows = record["arms"][mode][gate]
        reference = {(r["episode"], r["step"]): r for r in baseline}
        if len(reference) != len(baseline) or len(rows) != len(reference):
            raise EvidenceError(f"{stem}: duplicate or missing gate steps")
        by_episode = collections.defaultdict(list)
        for row in rows:
            other = reference[(row["episode"], row["step"])]
            if (row["base_image_id"], row["label"], row["cut"]) != (other["base_image_id"], other["label"], other["cut"]):
                raise EvidenceError(f"{stem}: changed episode contents")
            oracle = MODES[mode] if row["label"] == "present" else "proceed"
            by_episode[row["episode"]].append({
                "agreement": float(row["action"] == other["action"]),
                "wake_fraction": float(row["woke"]),
                "invocations_avoided": 1.0 - float(row["woke"]),
                "oracle_correctness": float(row["action"] == oracle),
                "correctness_change": float(row["action"] == oracle) - float(other["action"] == oracle)})
        lengths = {len(v) for v in by_episode.values()}
        if len(lengths) != 1:
            raise EvidenceError("unequal episode lengths require weighted estimation")
        for name in next(iter(by_episode.values()))[0]:
            values = [np.mean([r[name] for r in by_episode[e]]) for e in sorted(by_episode)]
            arrays[name].append(values)
            stored_name = {"invocations_avoided": "compute_saved"}.get(name, name)
            if stored_name in record["metrics"][mode][gate]:
                close(np.mean(values), record["metrics"][mode][gate][stored_name], f"{stem}:{gate}:{name}")
        delays.append(cuts_to_wakes(rows))
    return {"stem": stem, "mode": mode, "gate": gate,
            "seeds": [r["seed"] for r in records],
            "estimates": {name: estimate(np.asarray(values), key=f"{stem}:{mode}:{gate}:{name}", unit="episode") for name, values in arrays.items()},
            "delay": delays, "decode_seconds": [r["decode"]["mean_decode_seconds"] for r in records],
            "episode_manifest_sha256": records[0]["episodes"]["manifest_sha256"],
            "base_images_in_stream": len({r["base_image_id"] for r in records[0]["arms"][mode][gate]})}


def audit_populations_and_baselines(artifacts: dict) -> dict:
    manifest = json.loads((PACKAGE / "provenance/coco-inputs.json").read_text())
    populations = {name: {(r["image_id"], r["label"]) for r in rows}
                   for name, rows in manifest["image_sets"].items()}
    seen = set()
    for name, population in populations.items():
        if seen & population or len(population) != len(manifest["image_sets"][name]):
            raise EvidenceError(f"COCO populations overlap: {name}")
        seen |= population
    checked = 0
    for stem, record in artifacts.items():
        if "seed" not in record or "arms" not in record:
            continue
        if re.match(r"vision-mode-r1[b]?-", stem):
            population = populations["eval"]
        elif re.match(r"vision-mode-r2(?:b|c|y)?-", stem):
            population = populations["rung2"]
        else:
            continue
        for cells in record["arms"].values():
            for rows in cells.values():
                if set(indexed(rows)) != population:
                    raise EvidenceError(f"{stem}: photographic evaluation population changed")
                checked += 1
    baseline_checks = 0
    rendered_manifests = {}
    reference_draws = None
    repeated_text_differences = []
    for stem, record in artifacts.items():
        if stem.startswith("vision-mode-r3b-") and "arms" in record:
            manifest = record["manifest"]
            if reference_draws is None:
                reference_draws = manifest["draws"]
            if manifest["draws"] != reference_draws:
                raise EvidenceError(f"{stem}: rendered scene parameters changed")
            rendered_manifests.setdefault(manifest["sha256"], []).append(stem)
    for host in ("450m", "27b"):
        for seed in SEEDS:
            base = artifacts[f"vision-mode-r3b-{host}-s{seed}"]
            mixed = artifacts[f"vision-mode-r3b-mx2door-{host}-s{seed}"]
            fixed = artifacts[f"vision-mode-r3b-mx2ndoor-{host}-s{seed}"]
            if mixed["doors"]["mode"]["sha256"] != fixed["doors"]["mode"]["sha256"]:
                raise EvidenceError("fixed-baseline comparison changed its mode checkpoint")
            if base["doors"]["constant"]["sha256"] != fixed["doors"]["constant"]["sha256"]:
                raise EvidenceError("fixed baseline is not the original constant checkpoint")
            if mixed["manifest"]["sha256"] != fixed["manifest"]["sha256"]:
                raise EvidenceError("fixed-baseline comparison changed its rendered pixels")
            for mode in MODES:
                for arm, reference in (("packet", mixed), ("shuffled", mixed),
                                       ("habit", base), ("text", base)):
                    a = indexed(fixed["arms"][mode][arm])
                    b = indexed(reference["arms"][mode][arm])
                    if a.keys() != b.keys():
                        raise EvidenceError(f"fixed-baseline {host}/{seed}/{mode}/{arm} changed actions")
                    differences = [k for k in a if a[k]["action"] != b[k]["action"]]
                    if differences and arm != "text":
                        raise EvidenceError(f"fixed-baseline {host}/{seed}/{mode}/{arm} changed actions")
                    if differences:
                        repeated_text_differences.append({
                            "host": host, "seed": seed, "mode": mode, "images": differences,
                            "original_pixel_manifest": base["manifest"]["sha256"],
                            "followup_pixel_manifest": fixed["manifest"]["sha256"],
                        })
                    baseline_checks += 1
    return {"disjoint_coco_sets": {k: len(v) for k, v in populations.items()},
            "photographic_arm_population_checks": checked,
            "fixed_baseline_arm_comparisons": baseline_checks,
            "rendered_scene_draws_identical": True,
            "rendered_pixel_manifests": {k: sorted(v) for k, v in sorted(rendered_manifests.items())},
            "repeated_text_differences_across_pixel_versions": repeated_text_differences}


def analyze(directory: Path, *, new_run: bool = False) -> dict:
    artifacts, provenance = load_evidence(directory, new_run=new_run)
    audits = {}
    for stem, record in artifacts.items():
        if stem == "vision-mode-r3-hostprobe-3b":
            audits[stem] = audit_host_probe(record)
        elif "seed" in record and "arms" in record and "r2t-" not in stem:
            audits[stem] = audit_image_record(record, stem)
    result = {"format": "lookout-captain-analysis-v1", "provenance": provenance,
              "population_and_baseline_audit": audit_populations_and_baselines(artifacts),
              "uncertainty": {"method": "paired percentile bootstrap of images/scenes or complete episodes; seed realizations held fixed",
                              "replicates": REPLICATES, "seed": BOOTSTRAP_SEED,
                              "coverage": "pointwise 95%, post hoc, conditional on the tested seeds and source population",
                              "boundary_caveat": "a degenerate percentile interval at an empirical boundary does not establish zero population error"},
              "image_record_audits": audits, "conditioning": [], "switch": [], "conflict": [],
              "boxes": [], "blind": [], "gating": [], "transfer": []}
    for host in HOSTS:
        stem = f"vision-mode-r1-{host}"
        records = seeded(artifacts, stem)
        result["conditioning"].extend(image_summary(records, stem, mode) for mode in MODES)
        for dose in ("x1", "x1p5"):
            stem = f"vision-mode-r1b-{host}-{dose}-oracle"
            records = seeded(artifacts, stem)
            result["conflict"].extend(image_summary(records, stem, mode) for mode in MODES)
        stem = f"vision-mode-r2t-{host}"
        records = seeded(artifacts, stem)
        result["gating"].extend(gate_summary(records, stem, mode, gate) for mode in ("social", "caution") for gate in records[0]["arms"][mode])
    switch_families = [("r2", ("450m", "3b")), ("r2y", HOSTS), ("r2y-26s", ("450m", "3b")), ("r2y-26m", ("450m", "3b"))]
    for family, hosts in switch_families:
        for host in hosts:
            stem = f"vision-mode-{family}-{host}"
            records = seeded(artifacts, stem)
            for mode in ("social", "caution"):
                item = image_summary(records, stem, mode)
                item["error_identity"] = {label: [detector_gap(r["arms"][mode], mode, label) for r in records] for label in ("present", "absent")}
                result["switch"].append(item)
    for section, family in (("boxes", "r2b"), ("blind", "r2c")):
        for host in ("450m", "3b"):
            stem = f"vision-mode-{family}-{host}"
            records = seeded(artifacts, stem)
            result[section].extend(image_summary(records, stem, mode) for mode in ("social", "caution"))
    transfer_stems = sorted({re.sub(r"-s\d+$", "", stem) for stem, r in artifacts.items() if stem.startswith("vision-mode-r3b-") and "seed" in r and "arms" in r})
    for stem in transfer_stems:
        records = sorted([r for name, r in artifacts.items() if re.sub(r"-s\d+$", "", name) == stem and "seed" in r], key=lambda r: r["seed"])
        for mode in MODES:
            result["transfer"].append(image_summary(records, stem, mode, unit="rendered scene"))
    # Include follow-up prompt evaluations without pretending that single- or
    # two-seed mechanism probes have the main study's three-seed design.
    existing = {row["stem"] for row in result["conflict"]}
    extra = sorted({re.sub(r"-s\d+$", "", name) for name, r in artifacts.items() if name.startswith("vision-mode-r1b-") and "seed" in r and "arms" in r} - existing)
    for stem in extra:
        records = sorted([r for name, r in artifacts.items() if re.sub(r"-s\d+$", "", name) == stem and "seed" in r], key=lambda r:r["seed"])
        result["conflict"].extend(image_summary(records, stem, mode) for mode in MODES)
    result["detectors"] = {stem: r for stem, r in artifacts.items() if stem in ("vision-mode-r2y-detectors", "vision-mode-r3-detector", "vision-mode-r3b-detector", "vision-mode-r3-hostprobe-3b")}
    result["inventory"] = [{"stem": stem, "seed": r.get("seed"), "seeds": r.get("seeds"), "recorded_experiment": r.get("experiment"), "has_individual_arms": "arms" in r} for stem, r in sorted(artifacts.items())]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=PACKAGE / "artifacts")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--new-run", action="store_true",
                        help="analyze a complete rerun separately from the hash-checked publication records")
    args = parser.parse_args()
    if args.output is None:
        args.output = PACKAGE / ("runs/summary.json" if args.new_run else "analysis/summary.json")
    result = analyze(args.artifacts, new_run=args.new_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output), "artifacts": result["provenance"]["files"], "audited_image_records": len(result["image_record_audits"]), "gating_comparisons": len(result["gating"])}))


if __name__ == "__main__":
    main()
