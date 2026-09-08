"""Derive all reported estimates from the verified item-level evidence."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from .artifacts import load, verify, milestones, modalities_for, expected_measurements, PROMPTS
from .provenance import digest, file_hash, write_json
from .statistics import ranking_interval, likelihood_interval, generation_interval, ranking_score_diagnostics

TRAJECTORY_SEEDS = (42, 1042, 2042)


def condition(item):
    r = item["recipe"]
    settings = item["settings"]
    name = item["measurement"]
    keys = {"native_bpb": ("rows",), "ranking": ("prompt", "intervention"),
            "generation": ("objective", "intervention")}.get(name, ())
    return {"host": r["host"], "cohort": r["cohort"], "modality": item["evaluation_modality"],
            "kind": r["kind"], "feature_mode": r["feature_mode"], "diversity": r["diversity"],
            "step": item["step"], "measurement": name, "settings": {k: settings[k] for k in keys}}


def paired_seeds(first, second):
    """Pair actual training seeds; a single frozen host is a fixed reference."""
    left = {r["recipe"]["seed"]: r for r in first}
    right = {r["recipe"]["seed"]: r for r in second}
    if len(left) != len(first) or len(right) != len(second):
        raise ValueError("Repeated training seed in an analysis cell")
    if len(second) == 1 and second[0]["recipe"]["kind"] == "none":
        seeds = sorted(left)
        return seeds, [left[s] for s in seeds], second
    seeds = sorted(left.keys() & right.keys())
    if not seeds:
        raise ValueError("No shared training seeds for a paired contrast")
    return seeds, [left[s] for s in seeds], [right[s] for s in seeds]


def seed_summary(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite seed-level measurement")
    return {"mean": float(values.mean()), "per_seed": values.tolist(),
            "seed_sd": float(values.std(ddof=1)) if len(values) > 1 else None}


def registered_conditions(protocol):
    """The intended seed population is defined by recipes, never by survivors."""
    result = {}
    for recipe in protocol["identity"]["recipes"]:
        for step in milestones(recipe):
            for modality in modalities_for(recipe):
                for measurement in expected_measurements(recipe, step, modality).values():
                    c = condition({"recipe": recipe, "step": step, "evaluation_modality": modality, **measurement})
                    key = digest(c)
                    result.setdefault(key, {"condition": c, "recipes_by_seed": {}})["recipes_by_seed"][recipe["seed"]] = digest(recipe)
    return result


def trajectory_conditions(planned):
    """The IMU penalty curves use the same three declared seeds at every penalty."""
    result = {}
    for key, declaration in planned.items():
        c = declaration["condition"]
        if (c["cohort"] != "P" or c["modality"] != "imu" or c["kind"] != "bridge"
                or c["feature_mode"] != "true" or c["measurement"] not in
                ("ranking", "variation", "functional_sensitivity")):
            continue
        if c["measurement"] == "ranking" and c["settings"] != {
                "prompt": PROMPTS["imu"][0], "intervention": "correct"}:
            continue
        seeds = declaration["recipes_by_seed"]
        if not set(TRAJECTORY_SEEDS) <= seeds.keys():
            raise ValueError("An IMU trajectory condition lacks the declared shared seeds")
        result[key] = {**declaration, "recipes_by_seed": {s: seeds[s] for s in TRAJECTORY_SEEDS}}
    return result


def complete_conditions(groups, planned, failures, *, incomplete=False):
    """Suppress a partial seed aggregate; retain all underlying item evidence."""
    complete, unavailable = {}, []
    by_recipe = {digest(f["recipe"]): f for f in failures}
    if groups.keys() - planned.keys():
        raise ValueError("An aggregate condition was not registered")
    for key, declaration in planned.items():
        items = groups.get(key, [])
        actual = {r["recipe"]["seed"] for r in items}
        expected = set(declaration["recipes_by_seed"])
        if actual - expected or len(actual) != len(items):
            raise ValueError("An aggregate has extra or repeated training seeds")
        if actual == expected:
            complete[key] = items
            continue
        missing = expected - actual
        failed = [by_recipe.get(declaration["recipes_by_seed"][s]) for s in sorted(missing)]
        justified = all(f is not None and declaration["condition"]["step"] in f["unavailable_checkpoints"] for f in failed)
        if not incomplete and not justified:
            raise ValueError("Missing seed evidence is not justified by a recorded numerical failure")
        unavailable.append({"condition_id": key, "condition": declaration["condition"],
                            "expected_seeds": sorted(expected), "available_seeds": sorted(actual), "missing_seeds": sorted(missing),
                            "failed_training_ids": [f["training_id"] for f in failed if f is not None],
                            "reason": "numerical_failure" if justified else "incomplete_evidence"})
    return complete, unavailable


def run(evidence, output, *, incomplete=False, replicates=10000):
    evidence, output = Path(evidence), Path(output)
    coverage = verify(evidence, incomplete=incomplete)
    index = load(evidence / "index.json")
    manifest = load(evidence / "manifest.json")
    audit = load(evidence / "audit/encoder-audit.json")
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "analysis-cache"
    cache.mkdir(exist_ok=True)
    numerical_source = {n: file_hash(Path(__file__).with_name(n)) for n in ("aggregate.py", "statistics.py")}
    groups = defaultdict(list)
    conditions = {}
    for item in index["evaluations"]:
        c = condition(item)
        key = digest(c)
        groups[key].append(item)
        conditions[key] = c
    for items in groups.values():
        items.sort(key=lambda r: r["recipe"]["seed"])
        if len({i["recipe"]["seed"] for i in items}) != len(items):
            raise ValueError("An analysis group repeats a training seed")
    planned = registered_conditions(load(evidence / "protocol.json"))
    trajectory_plan = trajectory_conditions(planned)
    trajectory_groups = {k: [i for i in groups.get(k, []) if i["recipe"]["seed"] in TRAJECTORY_SEEDS]
                         for k in trajectory_plan}
    trajectory_groups, unavailable_trajectories = complete_conditions(
        trajectory_groups, trajectory_plan, coverage["numerical_failures"], incomplete=incomplete)
    groups, unavailable = complete_conditions(groups, planned, coverage["numerical_failures"], incomplete=incomplete)

    @lru_cache(maxsize=12)
    def records(relative):
        return load(evidence / relative)

    def rows(items):
        return [records(item["path"])["items"] for item in items]

    def label_likelihood(runs):
        return [[{"id":r["id"], "group":r["group"], "nats":r["target_nats"], "bytes":r["target_bytes"]} for r in run] for run in runs]

    def strict(runs, modality):
        detail = audit["modalities"][modality]
        bad_groups = set(detail["overlapping_groups"])
        bad_ids = set(detail.get("secondary_strict_test_excluded_ids", []))
        return [[r for r in run if r["group"] not in bad_groups and r["id"] not in bad_ids] for run in runs]

    def estimate(c, items, comparison=None):
        name = c["measurement"]
        options = {"replicates": replicates, "seed": 20260904}
        runs = rows(items) if name in ("ranking", "native_bpb", "common_bpb", "generation") else None
        other = rows(comparison) if comparison else None
        if name == "ranking":
            result = {"rank1": ranking_interval(runs, comparison=other, **options),
                      "paired_label_bpb": likelihood_interval(label_likelihood(runs), comparison=label_likelihood(other) if other else None, **options)}
            detail = audit["modalities"][c["modality"]]
            if detail["overlapping_groups"] or detail.get("secondary_strict_test_excluded_ids"):
                filtered = strict(runs, c["modality"])
                result["source_disjoint_sensitivity"] = ranking_interval(filtered, comparison=strict(other,c["modality"]) if other else None, **options)
                result["source_disjoint_sensitivity"]["excluded_items"] = len(runs[0]) - len(filtered[0])
                result["source_disjoint_sensitivity"]["scope"] = "Excludes known shared original sources or exact images; does not remove prior encoder normalization exposure."
            if comparison is None:
                result["legacy64_secondary"] = {key: seed_summary([records(i["path"])["summary"]["legacy_64_size_diagnostic"][key] for i in items])
                                                for key in ("micro_accuracy", "macro_accuracy", "average_rank")}
                result["ranking_diagnostics"] = {key: seed_summary([records(i["path"])["summary"][key] for i in items])
                                                 for key in ("top2", "average_rank")}
                diagnostics=[ranking_score_diagnostics(rows) for rows in runs]
                result["ranking_diagnostics"].update({key:seed_summary([r[key] for r in diagnostics])
                                                      for key in diagnostics[0]})
            return result
        if name in ("native_bpb", "common_bpb"):
            return {"bpb": likelihood_interval(runs, comparison=other, **options)}
        if name == "generation":
            cells = sorted({(r["family"], r["template_split"]) for r in runs[0]})
            result = {}
            for family, split in cells:
                for metric in ("semantic_correct", "format_compliant", "joint_success"):
                    key = f"{family}/{split}/{metric}"
                    result[key] = generation_interval(runs, family, split, metric, comparison=other, **options)
                if family == "rule":
                    result[f"{family}/{split}/both_rule_mappings"] = generation_interval(runs, family, split, "both_rule_mappings", comparison=other, **options)
            return result
        if comparison is not None:
            raise ValueError("No paired estimator declared for this measurement")
        if name == "variation":
            return {k: seed_summary([records(i["path"])[k] for i in items]) for k in
                    ("factor_cosine", "factor_variance", "effective_cosine", "effective_centered_energy", "effective_mean_squared_norm")}
        if name == "functional_sensitivity":
            return {k: seed_summary([np.mean([r[k] for r in records(i["path"])["contexts"]]) for i in items]) for k in
                    ("mean_kl_to_sensor_mixture", "max_kl_to_sensor_mixture", "distinct_argmax_tokens")}
        raise ValueError(f"Unknown measurement: {name}")

    def cached(c, items, comparison=None):
        identity = {"condition": c, "first": [i["evaluation_id"] for i in items],
                    "comparison": [i["evaluation_id"] for i in comparison or []],
                    "source": numerical_source, "replicates": replicates,
                    "audit_sha256": manifest["files"]["audit/encoder-audit.json"]["sha256"]}
        key = digest(identity)
        path = cache / f"{key}.json"
        if path.exists():
            result = load(path)
            if result["identity"] != identity:
                raise ValueError("Analysis cache identity changed")
            return result["result"]
        result = estimate(c, items, comparison)
        write_json(path, {"identity": identity, "result": result})
        return result

    summaries = []
    for key in sorted(groups):
        items = groups[key]
        c = conditions[key]
        summaries.append({"condition_id": key, "condition": c, "seeds": [i["recipe"]["seed"] for i in items],
                          "evaluation_ids": [i["evaluation_id"] for i in items], "statistics": cached(c,items)})
        print(f"Aggregated {len(summaries)}/{len(groups)} conditions", flush=True)

    trajectories = [{"condition_id": key, "condition": trajectory_plan[key]["condition"],
                     "seeds": [i["recipe"]["seed"] for i in items],
                     "evaluation_ids": [i["evaluation_id"] for i in items],
                     "statistics": cached(trajectory_plan[key]["condition"], items)}
                    for key, items in sorted(trajectory_groups.items())]

    # Pair true features with each training control and with interventions on
    # the identical checkpoint. No comparison is selected by its outcome.
    contrasts = []
    def compare(first_key, second_key, reason):
        if second_key not in groups:
            return
        seeds, first, second = paired_seeds(groups[first_key], groups[second_key])
        c = conditions[first_key]
        contrasts.append({"first": first_key, "comparison": second_key, "reason": reason,
                          "paired_training_seeds": seeds, "first_available_seeds": [i["recipe"]["seed"] for i in groups[first_key]],
                          "comparison_available_seeds": [i["recipe"]["seed"] for i in groups[second_key]],
                          "statistics": cached(c,first,second)})
    for key in sorted(groups):
        c = conditions[key]
        if c["measurement"] not in ("ranking", "native_bpb", "common_bpb", "generation") or c["kind"] == "none":
            continue
        if c["settings"].get("intervention", "correct") != "correct":
            continue
        if c["diversity"] == 0 and c["kind"] == "bridge" and c["feature_mode"] == "true":
            for mode in ("constant", "shuffled", "random"):
                compare(key, digest({**c,"feature_mode":mode}), "training control")
            compare(key, digest({**c,"kind":"static"}), "static adapter")
        baseline = {**c, "kind":"none", "feature_mode":"true", "diversity":0., "step":0, "cohort":"G" if c["cohort"]=="G" else "U"}
        # Raw P decoding matches the frozen U checkpoint's P-format diagnostic.
        # Native-chat G decoding has its own separately evaluated frozen baseline.
        compare(key, digest(baseline), "frozen host")
        if c["measurement"] in ("ranking", "generation") and c["kind"] == "bridge" and c["feature_mode"] == "true":
            for intervention in ("constant", "random", "off", "different_class", *[f"shuffle-{i}" for i in range(5)]):
                compare(key, digest({**c,"settings":{**c["settings"],"intervention":intervention}}), "same-checkpoint feature intervention")
        if c["diversity"] > 0:
            compare(key, digest({**c,"diversity":0.}), "diversity penalty versus zero; shared seeds only")

    linear = load(evidence / "baselines/linear-probes.json")
    baseline_results = {}
    for modality, item in linear.items():
        values = [("linear_feature_classifier",item)]
        if "original_encoder_head" in item:
            values.append(("original_encoder_head",item["original_encoder_head"]))
        baseline_results[modality] = {}
        for name, value in values:
            items = records("baselines/" + value["path"])["items"]
            baseline_results[modality][name] = {"rank1": ranking_interval([items],replicates=replicates), "fixed_classifier": True}
            filtered = strict([items],modality)
            if len(filtered[0]) != len(items):
                baseline_results[modality][name]["source_disjoint_sensitivity"] = ranking_interval(filtered,replicates=replicates)
    composition_groups=defaultdict(list)
    for item in index.get("composition",[]):
        key=(item["host"],item["step"],tuple(item["subset"]),item["operator"])
        composition_groups[key].append(item)
    composition=[]
    for key,items in sorted(composition_groups.items()):
        host,step,subset,operator=key
        items.sort(key=lambda i:i["seed"])
        runs=rows(items)
        record={"host":host,"step":step,"subset":list(subset),"operator":operator,"seeds":[i["seed"] for i in items],
                "evaluation_ids":[i["evaluation_id"] for i in items],"bpb":likelihood_interval(runs,replicates=replicates)}
        baseline=composition_groups[(host,step,(),"off")]
        baseline=sorted(baseline,key=lambda i:i["seed"])
        if [i["seed"] for i in items]!=[i["seed"] for i in baseline]:
            raise ValueError("Composition constituent seeds are not paired")
        record["delta_to_off"]=likelihood_interval(runs,comparison=rows(baseline),replicates=replicates)
        if len(subset)>1:
            singles=[sorted(composition_groups[(host,step,(modality,),operator)],key=lambda i:i["seed"]) for modality in subset]
            single_rows=[rows(part) for part in singles]
            baseline_rows=rows(baseline)
            prediction=[]
            for seed_index,run in enumerate(runs):
                estimate=[]
                for position,row in enumerate(run):
                    candidates=[part[seed_index][position] for part in single_rows]
                    base=baseline_rows[seed_index][position]
                    if any((v["row"],v["bytes"])!=(row["row"],row["bytes"]) for v in [*candidates,base]):
                        raise ValueError("Composition additivity diagnostic has unpaired text")
                    estimate.append({**row,"nats":sum(v["nats"] for v in candidates)-(len(subset)-1)*base["nats"]})
                prediction.append(estimate)
            record["bpb_nonadditivity"]=likelihood_interval(runs,comparison=prediction,replicates=replicates)
            record["bpb_nonadditivity"]["interpretation"]="Positive: combined BPB exceeds the prediction from summing standalone BPB improvements. Additivity is a diagnostic, not an assumed law."
        composition.append(record)
    resources=[load(evidence/i["path"]) for i in index.get("resources",[])]
    training=[]
    for item in index["trajectories"]:
        data=load(evidence/item["path"])
        if data.get("status", "complete") != "complete":
            continue
        training.append({k:v for k,v in data.items() if k in ("training_id","recipe","elapsed_s","successful_attempt_wall_s","trainable_parameters","memory","exposure","peak_process_rss_bytes")})
    host_identities={}
    for item in index["trajectories"]:
        host=item["recipe"]["host"]
        identity=load(evidence/item["path"])["identity"]["host"]
        if host in host_identities and host_identities[host]!=identity:
            raise ValueError("One host label resolves to multiple model identities")
        host_identities[host]=identity
    result = {"schema":"paper1-summary-v2", "status":coverage["status"], "matrix_id":manifest["matrix_id"],
              "evidence_manifest_sha256":file_hash(evidence/"manifest.json"), "analysis_source":numerical_source,
              "uncertainty":{"replicates":replicates,"seed":20260904,"unit_by_modality":{"imu":"subject","audio":"original source","vision":"image"},
                             "method":"crossed seed/cluster percentile; class-stratified when clusters each belong to one class; equal seed weight",
                             "multiplicity":"Intervals are descriptive; no multiplicity-adjusted significance claims."},
              "coverage":coverage, "conditions":summaries, "unavailable_conditions":unavailable,
              "trajectory_conditions":trajectories, "unavailable_trajectory_conditions":unavailable_trajectories,
              "contrasts":contrasts, "linear_baselines":baseline_results,
              "composition":composition,"resources":resources,"training":training,"hosts":host_identities,"encoder_audit":audit}
    write_json(output/"summary.json",result)
    return {"status":result["status"],"conditions":len(summaries),"contrasts":len(contrasts),"summary":str(output/"summary.json")}
