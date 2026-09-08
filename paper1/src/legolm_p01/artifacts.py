"""Portable evidence export and fail-closed verification; standard library only.

The release records checkpoint hashes, not checkpoint tensors. It is sufficient
to audit populations, controls, uncertainty and rendering without downloading a
language model. Repeating model execution uses the separate locked environment.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations, product
import gzip
import json
import math
from pathlib import Path
import shutil

from .provenance import digest, file_hash, write_json
from .bootstrap import bundled_assets
from .outcomes import TERMINAL, terminal_record, validate_failure

MODALITIES = ("imu", "audio", "vision")
PROMPTS = {
    "imu": ["The activity is: ", "The person is: ", "Detected human activity: ", "This motion corresponds to: "],
    "audio": ["The sound is: ", "The recorded sound comes from: ", "Detected audio event: ", "The sound category is: "],
    "vision": ["The image shows: ", "The pictured object is: ", "The image category is: ", "This is an image of: "],
}


def load(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def milestones(r):
    if r["kind"] == "none":
        return [0]
    short = 300 if r["cohort"] == "U" else 600
    points = {short, r["steps"]}
    if r["kind"] == "prefix":
        points.add(0)
    if r["cohort"] == "U" and r["kind"] == "bridge" and r["feature_mode"] == "true":
        points.update((150, 1500))
    if r["kind"] == "bridge" and r["modality"] == "imu" and r["feature_mode"] == "true":
        points.update((0, 30, 100, 300, 1000, 3000) if r["cohort"] == "U" else (0, 100, 300, 600, 1200, 2400, 6000))
    return sorted(s for s in points if s <= r["steps"])


def checkpoint_gaps(record):
    """Only milestones after a recorded numerical failure may be unavailable."""
    expected = milestones(record["recipe"])
    unavailable = []
    if record.get("status", "complete") == "numerical_failure":
        first = validate_failure(record)
        unavailable = [s for s in expected if s >= first]
        expected = [s for s in expected if s < first]
    if sorted(p["step"] for p in record["checkpoints"]) != expected:
        raise ValueError("Incomplete checkpoint trajectory")
    return unavailable


def measurement_key(measurement, settings):
    """Scientific cell, excluding implementation and source fingerprints."""
    keys = {"native_bpb": ("rows",), "ranking": ("prompt", "intervention"),
            "generation": ("objective", "intervention")}.get(measurement, ())
    return digest([measurement, {k: settings[k] for k in keys}])


def expected_measurements(r, step, modality):
    endpoints = (300, 3000) if r["cohort"] == "U" else (600, 6000)
    main = step in endpoints or r["kind"] == "none" or (r["kind"] == "prefix" and step == 0)
    trajectory = r["cohort"] in ("U", "P") and r["kind"] == "bridge" and r["modality"] == "imu" and r["feature_mode"] == "true" and step not in (150, 1500)
    if not (main or trajectory):
        return {}
    result = {}
    def add(name, **settings):
        result[measurement_key(name, settings)] = {"measurement": name, "settings": settings}
    add("ranking", prompt=PROMPTS[modality][0], intervention="correct")
    if main:
        add("native_bpb", rows=256 if step <= endpoints[0] else 1024)
        if r["kind"] == "none":
            add("native_bpb", rows=1024)
        add("common_bpb")
        if r["kind"] == "none" or r["cohort"] == "P":
            for prompt in PROMPTS[modality][1:]:
                add("ranking", prompt=prompt, intervention="correct")
        if r["kind"]=="none" and r["cohort"]=="U" and r["host"]=="qwen08":
            add("generation",objective="P",intervention="correct")
        if r["cohort"] == "P" and r["kind"] == "bridge" and r["feature_mode"] == "true":
            for intervention in ("constant", "random", "off", "different_class", *[f"shuffle-{i}" for i in range(5)]):
                add("ranking", prompt=PROMPTS[modality][0], intervention=intervention)
        if r["cohort"] == "G" or (r["cohort"] == "P" and r["host"] == "qwen08" and r["diversity"] == 0):
            interventions = ("correct", "constant", "off", "different_class", "shuffle-0") if r["kind"] == "bridge" and r["feature_mode"] == "true" else ("correct",)
            for intervention in interventions:
                add("generation", objective=r["cohort"], intervention=intervention)
            if r["kind"] == "none" and r["cohort"] == "G":
                for intervention in ("oracle_label_in_text", "classifier_label_in_text"):
                    add("generation", objective="G", intervention=intervention)
    if r["kind"] == "bridge":
        add("variation")
        add("functional_sensitivity")
    return result


def modalities_for(r):
    return MODALITIES if r["cohort"] == "U" and r["kind"] in ("none", "static") else (r["modality"],)


def checked_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or escaping evidence path: {relative}")
    return path


def validate_population(evidence, dataset):
    identity = evidence["identity"]
    name = identity["measurement"]
    items = evidence.get("items", [])
    if name == "ranking":
        test = dataset["test"]
        actual = [(i["id"], i["label"], i["group"]) for i in items]
        expected = list(zip(test["ids"], test["labels"], test["groups"]))
        if actual != expected:
            raise ValueError("Ranking population/order differs from the full registered holdout")
        classes = len(dataset["categories"])
        for row in items:
            if len(row["scores"]) != classes or not all(math.isfinite(s) for s in row["scores"]):
                raise ValueError("Missing or nonfinite class scores")
    elif name in ("native_bpb", "common_bpb"):
        expected_count = identity["rows"] if name == "native_bpb" else 512
        if len(items) != expected_count or any(not math.isfinite(r["nats"]) or r["bytes"] <= 0 for r in items):
            raise ValueError("Incomplete or invalid held-out likelihood records")
    elif name == "generation":
        # Sixteen predeclared prompt cases per sensor; labels and groups must
        # remain those of the same held-out sensor under feature intervention.
        canonical = dict(zip(dataset["test"]["ids"], zip(dataset["test"]["labels"], dataset["test"]["groups"])))
        counts = Counter(r["id"] for r in items)
        expected_count = 400 if dataset["modality"] == "audio" else 300
        cases_per_sensor = 16 if identity["objective"] == "G" else 1
        if len(counts) != expected_count or set(counts.values()) != {cases_per_sensor}:
            raise ValueError("Generation panel is incomplete")
        for row in items:
            if canonical.get(row["id"]) != (row["label"], row["group"]):
                raise ValueError("Generation target identity changed")
            if any(not isinstance(row[k], bool) for k in ("semantic_correct", "format_compliant", "joint_success")):
                raise ValueError("Generation outcomes must be explicit booleans")


def export(matrix, run_root, output, *, incomplete=False):
    """Export only the current registered matrix; no glob-based result pooling."""
    matrix, run_root, output = Path(matrix), Path(run_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    declaration = load(matrix / "matrix.json")
    if declaration.get("excluded_from_results"):
        raise ValueError("Acceptance or excluded attempts cannot become a release")
    files = {}
    def put(relative, *, source=None, value=None):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source is not None:
            if not target.exists() or file_hash(target) != file_hash(source):
                shutil.copyfile(source, target)
        else:
            write_json(target, value)
        files[relative] = {"sha256": file_hash(target), "bytes": target.stat().st_size}
        return relative
    put("protocol.json", value=declaration)
    for m in MODALITIES:
        put(f"data/{m}.json", source=run_root / "features-fp32" / f"{m}.json")
    for h in ("mini", "lfm230m", "qwen08"):
        put(f"data/text-{h}.json", source=run_root / "text" / f"text-{h}.json")
    put("data/common-text.json", source=run_root / "text/common-text.json")
    put("audit/encoder-audit.json", source=run_root / "audit/encoder-audit.json")
    for host in ("mini","lfm230m","qwen08"):
        put(f"acceptance/{host}-fp32.json",source=run_root/"canaries"/f"{host}-fp32.json")
    put("acceptance/training.json",source=run_root/"canaries/training-acceptance.json")
    if declaration["workers"]>1:
        put("acceptance/parallel.json",source=run_root/"grid/parallel-canary.json")
    put("baselines/linear-probes.json", source=run_root / "linear-v2/linear-probes.json")
    for item in load(run_root / "linear-v2/linear-probes.json").values():
        for value in [item, *([item["original_encoder_head"]] if "original_encoder_head" in item else [])]:
            source = checked_path(run_root / "linear-v2", value["path"])
            if file_hash(source) != value["sha256"]:
                raise ValueError("Linear classifier evidence changed")
            put("baselines/" + source.name, source=source)
    trajectories, evaluations = [], []
    missing = []
    if (matrix/"evaluation-version.json").exists():
        put("acceptance/evaluation-version.json",source=matrix/"evaluation-version.json")
    else:
        missing.append({"reason":"Final evaluator reconciliation is incomplete"})
    for recipe in declaration["identity"]["recipes"]:
        job = matrix / f"job-{digest(recipe)}.json"
        if not job.exists() or load(job)["status"] not in TERMINAL:
            missing.append({"recipe": recipe, "reason": "training incomplete"})
            continue
        record = load(job)
        completed = terminal_record(record)
        directory = Path(completed["directory"])
        config = load(directory / "config.json")
        if config["training_id"] != completed["training_id"] or config["identity"]["recipe"] != recipe:
            raise ValueError("Training metadata identity mismatch")
        clean = {k: v for k, v in completed.items() if k != "directory"}
        clean["identity"] = config["identity"]
        clean["status"] = record["status"]
        if record["status"] == "complete":
            clean["successful_attempt_wall_s"] = record["elapsed_wall_s"]
        else:
            log = put(f"audit/training-failures/{completed['training_id']}.txt", source=Path(record["log"]))
            clean["execution_log"] = {"path": log, "sha256": file_hash(output / log)}
            if "elapsed_wall_s" in record:
                clean["failed_attempt_wall_s"] = record["elapsed_wall_s"]
        for point in clean["checkpoints"]:
            checkpoint = directory / point["path"]
            # Full tensor hashes were checked at each evaluation. Checking file
            # sizes here avoids rereading a terabyte on every progress export.
            point["checkpoint_bytes"] = checkpoint.stat().st_size
        relative = put(f"training/{completed['training_id']}.json", value=clean)
        trajectories.append({"training_id": completed["training_id"], "recipe": recipe, "path": relative, "status": record["status"]})
        for point in completed["checkpoints"]:
            for modality in modalities_for(recipe):
                expected = expected_measurements(recipe, point["step"], modality)
                if not expected:
                    continue
                directory = matrix / "evaluations" / (point["sha256"] + "-" + modality)
                catalog_path = directory / "evaluations.json"
                catalog = load(catalog_path) if catalog_path.exists() else {}
                selected = {}
                # Multiple implementations of one measurement require an
                # explicit evaluator revision, never best-result selection.
                for item in catalog.values():
                    key = measurement_key(item["measurement"], item["settings"])
                    if key not in expected:
                        raise ValueError("Unregistered evaluation in the selected checkpoint catalog")
                    if key in selected:
                        raise ValueError(f"Ambiguous evaluator revisions in {catalog_path}; reconcile explicitly")
                    selected[key] = item
                for key in expected.keys() - selected.keys():
                    missing.append({"recipe": recipe, "step": point["step"], "modality": modality, **expected[key]})
                for item in selected.values():
                    source = checked_path(directory, item["path"])
                    if file_hash(source) != item["sha256"]:
                        raise ValueError("Evaluation catalog hash mismatch")
                    relative = put("evaluations/" + source.name, source=source)
                    evaluations.append({**item, "path": relative})
    composition, resources = [], []
    for host, seed, step in product(("mini","lfm230m","qwen08"),(42,1042,2042),(150,1500)):
        name=f"composition-{host}-s{seed}-step{step}"
        directory=run_root/"supplements"/name
        if not (directory/"complete.json").exists():
            missing.append({"supplement":name,"reason":"composition incomplete"})
            continue
        complete=load(directory/"complete.json")
        for relative, checksum in complete["outputs"].items():
            if file_hash(checked_path(directory,relative))!=checksum:
                raise ValueError("Composition completion record changed")
        for item in load(directory/"composition.json"):
            source=checked_path(directory,item["path"])
            if file_hash(source)!=item["sha256"]:
                raise ValueError("Composition evidence changed")
            relative=put("composition/"+source.name,source=source)
            composition.append({**item,"path":relative})
    for host, modality, kind in product(("mini","lfm230m","qwen08"),MODALITIES,("none","static","bridge","prefix")):
        name=f"resources-{host}-{modality}-{kind}"
        directory=run_root/"supplements"/name
        filename=f"{host}-{modality}-{kind}.json"
        if not (directory/"complete.json").exists():
            missing.append({"supplement":name,"reason":"isolated resource measurement incomplete"})
            continue
        expected=load(directory/"complete.json")["outputs"][filename]
        if file_hash(directory/filename)!=expected:
            raise ValueError("Resource measurement changed")
        relative=put("resources/"+filename,source=directory/filename)
        resources.append({"host":host,"modality":modality,"kind":kind,"path":relative})
    put("index.json", value={"trajectories": trajectories, "evaluations": evaluations,"composition":composition,"resources":resources})
    status = "incomplete" if missing else "complete"
    put("coverage.json", value={"status": status, "expected_trajectories": len(declaration["identity"]["recipes"]),
                               "terminal_trajectories": len(trajectories),
                               "completed_trajectories": sum(t["status"] == "complete" for t in trajectories),
                               "numerical_failures": sum(t["status"] == "numerical_failure" for t in trajectories),
                               "evaluations": len(evaluations), "missing": missing})
    manifest = {"schema": "paper1-evidence-v2", "matrix_id": declaration["matrix_id"], "status": status, "files": files}
    write_json(output / "manifest.json", manifest)
    if missing and not incomplete:
        raise ValueError(f"Release refused: {len(missing)} missing obligations; inspect coverage.json")
    verify(output, incomplete=incomplete)
    return {"status": status, "trajectories": len(trajectories), "evaluations": len(evaluations), "missing_obligations": len(missing)}


def verify(root, *, incomplete=False):
    root = Path(root)
    manifest = load(root / "manifest.json")
    if manifest["schema"] != "paper1-evidence-v2":
        raise ValueError("Unsupported evidence schema")
    for name, item in manifest["files"].items():
        path = checked_path(root, name)
        if file_hash(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Evidence integrity failure: {name}")
    protocol, index = load(root / "protocol.json"), load(root / "index.json")
    if digest(protocol["identity"]) != manifest["matrix_id"]:
        raise ValueError("Matrix identity mismatch")
    evaluator = None
    if "acceptance/evaluation-version.json" in manifest["files"]:
        evaluator = load(root/"acceptance/evaluation-version.json")
        if (not evaluator["passed"] or evaluator["matrix_id"] != manifest["matrix_id"]
            or digest(evaluator["source"]) != evaluator["source_id"] or evaluator["backend"] != protocol["identity"]["backend"]):
            raise ValueError("Final evaluator identity mismatch")
        for name, sha in evaluator["source"]["evaluation_files"].items():
            if file_hash(checked_path(Path(__file__).parent,name)) != sha:
                raise ValueError("Released evaluator source differs from the final measurements")
        if file_hash(bundled_assets()/"text-cache-identities.json") != evaluator["source"].get("text_cache_identities_sha256"):
            raise ValueError("Released token-reference manifest differs from the final measurements")
    elif not incomplete:
        raise ValueError("Final evaluator reconciliation is required")
    required_checks={"zero-LoRA","zero-length-prefix","candidate-batch-parity","bridge-gradient","prefix-gradient","frozen-host","matching-prefix-targets"}
    for host in ("mini","lfm230m","qwen08"):
        canary=load(root/f"acceptance/{host}-fp32.json")
        if not required_checks.issubset(canary["checks"]) or canary["host"]["dtype"]!="float32":
            raise ValueError("Numerical acceptance evidence is missing")
    accepted=load(root/"acceptance/training.json")
    if {(r["host"],r["cohort"]) for r in accepted}!={(h,c) for h in ("mini","lfm230m","qwen08") for c in (("U","P","G") if h=="qwen08" else ("U","P"))}:
        raise ValueError("Training acceptance coverage incomplete")
    if any(not r["excluded_from_results"] for r in accepted) or not next(r for r in accepted if r["cohort"]=="G")["generation_batch_scalar_equal"]:
        raise ValueError("Generation acceptance failed or canaries were pooled with results")
    if protocol["workers"]>1:
        parallel=load(root/"acceptance/parallel.json")
        if not parallel["passed"] or parallel["training_source"]!=protocol["identity"]["training_source"] or parallel["backend"]!=protocol["identity"]["backend"]:
            raise ValueError("Parallel resource acceptance differs from the main matrix")
    recipes = {digest(r): r for r in protocol["identity"]["recipes"]}
    if len(recipes) != 411:
        raise ValueError("This release requires all 411 registered trajectories")
    actual = {digest(t["recipe"]): t for t in index["trajectories"]}
    if len(actual) != len(index["trajectories"]) or actual.keys() - recipes.keys():
        raise ValueError("Duplicate or unregistered trajectory")
    if not incomplete and actual.keys() != recipes.keys():
        raise ValueError("Missing registered trajectories")
    datasets = {m: load(root / f"data/{m}.json") for m in MODALITIES}
    for modality, item in load(root / "baselines/linear-probes.json").items():
        for value in [item, *([item["original_encoder_head"]] if "original_encoder_head" in item else [])]:
            path = checked_path(root, "baselines/" + value["path"])
            data = load(path)
            if file_hash(path) != value["sha256"] or digest(data["identity"]) != value["evaluation_id"]:
                raise ValueError("Feature classifier identity mismatch")
            validate_population({**data, "identity": {"measurement": "ranking"}}, datasets[modality])
    evaluations = {}
    for item in index["evaluations"]:
        path = checked_path(root, item["path"])
        if file_hash(path) != item["sha256"]:
            raise ValueError("Index/evidence hash mismatch")
        evidence = load(path)
        if digest(evidence["identity"]) != item["evaluation_id"] or evidence["evaluation_id"] != item["evaluation_id"]:
            raise ValueError("Evaluation identity mismatch")
        identity = evidence["identity"]
        if evaluator is not None and (identity["evaluation_source"] != evaluator["source"] or identity["sensor_batch"] != evaluator["sensor_batch"]):
            raise ValueError("Mixed evaluation implementations in final evidence")
        if identity["backend"] != protocol["identity"]["backend"]:
            raise ValueError("Evaluation backend differs from the registered matrix")
        modality = item["evaluation_modality"]
        if identity["evaluation_features_sha256"] != datasets[modality]["cache_sha256"]:
            raise ValueError("Evaluation feature population identity mismatch")
        validate_population(evidence, datasets[modality])
        key = (identity["checkpoint_sha256"], modality, measurement_key(item["measurement"], item["settings"]))
        if key in evaluations:
            raise ValueError("Duplicate scientific measurement")
        evaluations[key] = item
    if evaluator is not None:
        selected_ids = [key for catalog in evaluator["catalogs"] for key in catalog["evaluation_ids"]]
        if len(selected_ids) != len(set(selected_ids)) or set(selected_ids) != {r["evaluation_id"] for r in evaluations.values()}:
            raise ValueError("Final evaluator selection and exported evidence differ")
    expected_keys = set()
    checkpoints = {}
    numerical_failures = []
    for record in actual.values():
        data = load(root / record["path"])
        r = record["recipe"]
        identity = data["identity"]
        if identity["recipe"] != r or data["recipe"] != r or data["training_id"] != record["training_id"]:
            raise ValueError("Training index/identity mismatch")
        source = identity["source"]
        fields = {k: v for k, v in source.items() if k not in ("git_commit", "files", "source_hash")}
        protected = protocol["identity"]["training_source"]["training_files"]
        fields["training_files"] = {k: source["files"][k] for k in protected}
        if fields != protocol["identity"]["training_source"] or identity["backend"] != protocol["identity"]["backend"]:
            raise ValueError("Training source/backend differs from the registered matrix")
        if digest({**identity, "source": fields}) != record["training_id"]:
            raise ValueError("Training content identity mismatch")
        status = data.get("status", "complete")
        if status != record.get("status", "complete") or status not in TERMINAL:
            raise ValueError("Training outcome status is missing or inconsistent")
        unavailable = checkpoint_gaps(data)
        if status == "numerical_failure":
            receipt = data["execution_log"]
            log = checked_path(root, receipt["path"])
            if file_hash(log) != receipt["sha256"] or data["failure"]["exception"] not in log.read_text():
                raise ValueError("Numerical failure lacks its original execution log")
            if data["failure"].get("original_attempt_log_sha256", receipt["sha256"]) != receipt["sha256"]:
                raise ValueError("Original failed attempt was replaced")
            numerical_failures.append({"training_id": data["training_id"], "recipe": r,
                                       "failure": data["failure"], "unavailable_checkpoints": unavailable})
        for point in data["checkpoints"]:
            checkpoints[point["sha256"]]=(r,point["step"])
            for modality in modalities_for(r):
                keys = [(point["sha256"], modality, key) for key in expected_measurements(r, point["step"], modality)]
                expected_keys.update(keys)
                for key in keys:
                    item = evaluations.get(key)
                    if item is not None and (item["recipe"] != r or item["step"] != point["step"]):
                        raise ValueError("Evaluation recipe/step differs from its checkpoint")
    if evaluations.keys() - expected_keys:
        raise ValueError("Evaluation is not attached to its registered checkpoint")
    missing = expected_keys - evaluations.keys()
    if not incomplete and missing:
        raise ValueError(f"Missing {len(missing)} checkpoint measurements")
    supplement_counts=verify_supplements(root,index,checkpoints,protocol,datasets,incomplete=incomplete)
    if not incomplete and manifest["status"] != "complete":
        raise ValueError("Incomplete exports are not releases")
    return {"verified_files": len(manifest["files"]), "trajectories": len(actual), "evaluations": len(evaluations),
            "registered_trajectories": len(recipes),
            "successful_training": len(actual) - len(numerical_failures), "numerical_failures": numerical_failures,
            "unavailable_checkpoints": sum(len(f["unavailable_checkpoints"]) for f in numerical_failures),
            "missing_checkpoint_measurements": len(missing), "status": manifest["status"],**supplement_counts}


def verify_supplements(root,index,checkpoints,protocol,datasets,*,incomplete=False):
    hosts=("mini","lfm230m","qwen08")
    subsets=[()] + [s for n in (1,2,3) for s in combinations(sorted(MODALITIES),n)]
    expected={(h,s,t,subset,op) for h,s,t in product(hosts,(42,1042,2042),(150,1500))
              for subset in subsets for op in (("off",) if not subset else ("factor_average","effective_sum","effective_mean"))}
    found=set()
    for item in index.get("composition",[]):
        data=load(checked_path(root,item["path"]))
        identity=data["identity"]
        if digest(identity)!=item["evaluation_id"] or file_hash(root/item["path"])!=item["sha256"]:
            raise ValueError("Composition content identity mismatch")
        key=(identity["host"],identity["seed"],identity["step"],tuple(identity["subset"]),identity["operator"])
        if key not in expected or key in found:
            raise ValueError("Unregistered or repeated composition cell")
        found.add(key)
        if identity["backend"]!=protocol["identity"]["backend"]:
            raise ValueError("Composition backend differs from the registered study")
        if sorted(c["modality"] for c in identity["checkpoints"])!=sorted(MODALITIES):
            raise ValueError("Composition omitted a constituent identity")
        for c in identity["checkpoints"]:
            r,step=checkpoints[c["sha256"]]
            if any(r[k]!=v for k,v in {"cohort":"U","host":identity["host"],"seed":identity["seed"],
                                       "modality":c["modality"],"kind":"bridge","feature_mode":"true","diversity":0.}.items()) or step!=identity["step"]:
                raise ValueError("Composition constituent belongs to another trajectory")
        count=256 if identity["step"]==150 else 1024
        if [r["row"] for r in data["items"]]!=list(range(count)) or any(not math.isfinite(r["nats"]) or r["bytes"]<=0 for r in data["items"]):
            raise ValueError("Composition text population incomplete")
    if not incomplete and found!=expected:
        raise ValueError(f"Missing {len(expected-found)} composition cells")
    expected_resources=set(product(hosts,MODALITIES,("none","static","bridge","prefix")))
    measured=set()
    for item in index.get("resources",[]):
        data=load(checked_path(root,item["path"]))
        key=(data["host"],data["modality"],data["kind"])
        if key not in expected_resources or key in measured:
            raise ValueError("Unregistered or repeated resource measurement")
        measured.add(key)
        if data["backend"]!=protocol["identity"]["backend"] or data["feature_sha256"]!=datasets[data["modality"]]["cache_sha256"]:
            raise ValueError("Resource measurement uses a different backend or feature population")
        r,step=checkpoints[data["checkpoint_sha256"]]
        if any(r[k]!=v for k,v in {"host":data["host"],"cohort":"U","kind":data["kind"],"seed":42,"diversity":0.,"feature_mode":"true"}.items()) or step!=(0 if data["kind"]=="none" else 3000):
            raise ValueError("Resource measurement uses a different scientific checkpoint")
        if not data["cached_decode_matches_four_token_reference"] or (data["text_tokens"],data["output_tokens"],data["batch_size"])!=(128,32,1):
            raise ValueError("Resource acceptance or length contract failed")
        for metric in ("preprocessing","encoder_only","sensor_path","fresh_sensor_prefill","cached_sensor_prefill","cached_sensor_generate_32","raw_sensor_to_first_token"):
            if any(not math.isfinite(v) or v<=0 for v in data[metric]["samples_ms"]):
                raise ValueError("Invalid resource timing")
    if not incomplete and measured!=expected_resources:
        raise ValueError(f"Missing {len(expected_resources-measured)} isolated resource measurements")
    return {"composition_cells":len(found),"resource_measurements":len(measured),
            "missing_composition_cells":len(expected-found),"missing_resource_measurements":len(expected_resources-measured)}
