"""A separate evaluation identity for each immutable training checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .hosts import load_host
from .train import Recipe, make_adapter, load_adapter_state, train_indices
from .evaluate import ranking, bpb, common_bpb, functional_sensitivity, variation, stratified_indices, save_evaluation, PARAPHRASES
from .provenance import file_hash, digest, write_json, source_identity
from .backend import identity as backend_identity
from .portable_text import load_verified_text, reference_path


def evaluator_source():
    source=source_identity()
    files={n:h for n,h in source["files"].items() if n in ("hosts.py","text.py","portable_text.py","canary.py","generation.py","generation_cached.py","evaluate.py","checkpoint_eval.py","train.py","provenance.py","backend.py") or n.startswith("legacy/")}
    source={k:v for k,v in source.items() if k not in ("files","source_hash","git_commit")}
    return {**source,"evaluation_files":files,"text_cache_identities_sha256":file_hash(reference_path())}


def run(checkpoint, asset_root, data_cache, features_dir, text_dir, output, device, suites=("primary",), sensor_batch=4, eval_modality=None, linear_dir=None):
    checkpoint=Path(checkpoint)
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    recipe=Recipe(**state["identity"]["recipe"])
    host=load_host(recipe.host,asset_root,data_cache,device)
    if host.identity != state["identity"]["host"]:
        raise ValueError("Checkpoint host/tokenizer/target map does not match the loaded host")
    path=features_dir/f"{recipe.modality}.pt"
    if file_hash(path)!=state["identity"]["features_sha256"]:
        raise ValueError("Sensor cache changed since training")
    modality=eval_modality or recipe.modality
    if modality!=recipe.modality:
        if recipe.kind not in ("static","none") or recipe.cohort!="U":
            raise ValueError("Only frozen hosts and static text adapters are shared across modalities")
        path=features_dir/f"{modality}.pt"
    dataset=torch.load(path,map_location="cpu",weights_only=False)
    text,text_validation=load_verified_text(text_dir,recipe.host,state["identity"]["text_sha256"])
    adaptation,_=make_adapter(host,recipe,dataset["train"]["features"].shape[1])
    load_adapter_state(host,adaptation,recipe.kind,state["adapter"])
    host.model.eval()
    if adaptation is not None:
        adaptation.eval()
    pool=dataset["unpaired"]["features"] if recipe.cohort=="U" else dataset["train"]["features"][train_indices(dataset)]
    constant=pool.mean(dim=0)
    # Scoring identity binds numerical evaluation sources, rather than unrelated
    # documentation or renderer edits. The checkpoint retains its Git provenance.
    source=evaluator_source()
    base={"training_id":state["training_id"],"checkpoint_sha256":file_hash(checkpoint),"step":state["step"],
          "evaluation_source":source,"backend":backend_identity(device),"sensor_batch":sensor_batch,"evaluation_modality":modality,"evaluation_features_sha256":file_hash(path),"text_validation":text_validation}
    del state
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    catalog_path=output/"evaluations.json"
    catalog=json.loads(catalog_path.read_text()) if catalog_path.exists() else {}
    def measure(name, settings, function):
        identity={**base,"measurement":name,**settings}
        evaluation_id=digest(identity)
        if evaluation_id in catalog:
            item=catalog[evaluation_id]
            if file_hash(output/item["path"])!=item["sha256"]:
                raise ValueError("Evaluation cache integrity mismatch")
            return item
        result=function()
        item=save_evaluation(output,identity,result)
        item.update({"measurement":name,"settings":settings,"checkpoint_sha256":base["checkpoint_sha256"],"step":base["step"],"recipe":recipe.__dict__,"evaluation_modality":modality})
        catalog[evaluation_id]=item
        write_json(catalog_path,catalog)
        print(json.dumps({"evaluated":name,"recipe":recipe.__dict__,"step":base["step"],"summary":item["summary"]},allow_nan=False),flush=True)
        return item
    if "primary" in suites:
        count=256 if base["step"]<=(300 if recipe.cohort=="U" else 600) else 1024
        measure("native_bpb",{"rows":count},lambda:bpb(host,adaptation,recipe.kind,text["val"],dataset["unpaired"]["features"],constant,feature_mode=recipe.feature_mode,seed=900042,count=count))
        if recipe.kind=="none":
            measure("native_bpb",{"rows":1024},lambda:bpb(host,adaptation,recipe.kind,text["val"],dataset["unpaired"]["features"],constant,count=1024))
        measure("ranking",{"prompt":dataset["prompt"],"intervention":"correct"},lambda:ranking(host,adaptation,recipe.kind,dataset,constant,seed=900042,sensor_batch=sensor_batch,legacy_seed=recipe.seed))
    if "trajectory" in suites:
        measure("ranking",{"prompt":dataset["prompt"],"intervention":"correct"},lambda:ranking(host,adaptation,recipe.kind,dataset,constant,seed=900042,sensor_batch=sensor_batch,legacy_seed=recipe.seed))
    if "common" in suites:
        path=text_dir/"common-text.json"
        common=json.loads(path.read_text())
        measure("common_bpb",{"manifest_sha256":file_hash(path)},lambda:common_bpb(host,adaptation,recipe.kind,common["examples"],dataset["unpaired"]["features"],constant,feature_mode=recipe.feature_mode,seed=900042))
    if "interventions" in suites:
        for intervention in ("constant","random","off","different_class","shuffle-0","shuffle-1","shuffle-2","shuffle-3","shuffle-4"):
            permutation=int(intervention.rsplit("-",1)[-1]) if intervention.startswith("shuffle-") else 0
            seed=900042+permutation
            measure("ranking",{"prompt":dataset["prompt"],"intervention":intervention,"seed":seed},
                lambda intervention=intervention,seed=seed:ranking(host,adaptation,recipe.kind,dataset,constant,intervention=intervention,seed=seed,sensor_batch=sensor_batch,legacy_seed=recipe.seed))
    if "paraphrases" in suites:
        for prompt in PARAPHRASES[modality][1:]:
            measure("ranking",{"prompt":prompt,"intervention":"correct"},lambda prompt=prompt:ranking(host,adaptation,recipe.kind,dataset,constant,prompt=prompt,seed=900042,sensor_batch=sensor_batch,legacy_seed=recipe.seed))
    if "geometry" in suites and recipe.kind=="bridge":
        indices=stratified_indices(dataset["test"]["labels"],32 if recipe.modality=="imu" else 2,42)
        measure("variation",{"sample_ids":[dataset["test"]["ids"][i] for i in indices]},lambda:variation(host,adaptation,dataset["test"]["features"][indices]))
        indices=stratified_indices(dataset["test"]["labels"],2,42)[:32]
        measure("functional_sensitivity",{"sample_ids":[dataset["test"]["ids"][i] for i in indices],"text_rows":[0,1,2,3]},lambda:functional_sensitivity(host,adaptation,recipe.kind,dataset["test"]["features"][indices],text["val"]))
    if "generation" in suites:
        from .generation_cached import evaluate_generation
        objective="P" if recipe.kind=="none" and recipe.cohort=="U" else recipe.cohort
        interventions=("correct","constant","off","different_class","shuffle-0") if recipe.kind=="bridge" and recipe.feature_mode=="true" else ("correct",)
        for intervention in interventions:
            measure("generation",{"objective":objective,"intervention":intervention},
                lambda intervention=intervention:evaluate_generation(host,adaptation,recipe.kind,dataset,constant,objective=objective,intervention=intervention,batch_size=1 if recipe.host=="mini" else 8))
        if recipe.kind=="none" and recipe.cohort=="G":
            measure("generation",{"objective":"G","intervention":"oracle_label_in_text"},lambda:evaluate_generation(host,None,"none",dataset,constant,objective="G",oracle_labels=dataset["test"]["labels"]))
            if linear_dir is None:
                raise ValueError("Generation baseline requires the separately fitted feature classifier")
            linear_path=Path(linear_dir)/"linear-probes.json"
            labels=json.loads(linear_path.read_text())[modality]["predicted_labels"]
            measure("generation",{"objective":"G","intervention":"classifier_label_in_text","linear_manifest_sha256":file_hash(linear_path)},
                lambda:evaluate_generation(host,None,"none",dataset,constant,objective="G",oracle_labels=labels))
    return catalog
