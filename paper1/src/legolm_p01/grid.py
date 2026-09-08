"""Finite, outcome-independent trajectory matrix for the revised paper."""
from dataclasses import asdict
from pathlib import Path

from .train import Recipe
from .provenance import digest,write_json

SEEDS=(42,1042,2042)
TASK_SEEDS=(42,1042,2042,3042,4042)
HOSTS=("mini","lfm230m","qwen08")
MODALITIES=("imu","audio","vision")
MODES=("true","constant","shuffled","random")
LAMBDAS=(0.,.01,.05,.10,.20)


def recipes():
    result=[]
    def add(**kwargs):
        recipe=Recipe(**kwargs)
        if recipe in result:
            raise ValueError("Duplicate scientific trajectory")
        result.append(recipe)
    for host in HOSTS:
        # One frozen host and one static text adapter per seed are reused for
        # all modality evaluations. Paired static adapters are task-specific.
        add(host=host,cohort="U",modality="imu",kind="none",steps=0)
        for seed in SEEDS:
            add(host=host,cohort="U",modality="imu",kind="static",seed=seed)
            for modality in MODALITIES:
                for mode in MODES:
                    add(host=host,cohort="U",modality=modality,feature_mode=mode,seed=seed)
                add(host=host,cohort="U",modality=modality,kind="prefix",seed=seed)
            for penalty in LAMBDAS[1:]:
                add(host=host,cohort="U",modality="imu",seed=seed,diversity=penalty)
        task_seeds=SEEDS if host=="lfm230m" else TASK_SEEDS
        for modality in ("imu","audio"):
            for seed in task_seeds:
                for mode in MODES:
                    add(host=host,cohort="P",modality=modality,feature_mode=mode,seed=seed,steps=6000,batch_size=4)
                add(host=host,cohort="P",modality=modality,kind="static",seed=seed,steps=6000,batch_size=4)
        for seed in SEEDS:
            for penalty in LAMBDAS[1:]:
                add(host=host,cohort="P",modality="imu",seed=seed,diversity=penalty,steps=6000,batch_size=4)
        if host!="lfm230m":
            for seed in SEEDS:
                for mode in MODES:
                    add(host=host,cohort="P",modality="vision",feature_mode=mode,seed=seed,steps=6000,batch_size=4)
                add(host=host,cohort="P",modality="vision",kind="static",seed=seed,steps=6000,batch_size=4)
    for modality in ("imu","audio"):
        add(host="qwen08",cohort="G",modality=modality,kind="none",steps=0,batch_size=4,text_length=256)
        for seed in SEEDS:
            for mode in MODES:
                add(host="qwen08",cohort="G",modality=modality,seed=seed,feature_mode=mode,steps=6000,batch_size=4,text_length=256)
            add(host="qwen08",cohort="G",modality=modality,seed=seed,kind="static",steps=6000,batch_size=4,text_length=256)
    return result


def manifest(output):
    output=Path(output)
    rows=[]
    for r in recipes():
        config=asdict(r)
        name=f"{r.host}-{r.cohort}-{r.modality}-{r.kind}-{r.feature_mode}-s{r.seed}-d{r.diversity:g}.json"
        write_json(output/name,config)
        rows.append({"config":name,"recipe":config,"recipe_id":digest(config)})
    counts={}
    for r in rows:
        key=r["recipe"]["host"]+"/"+r["recipe"]["cohort"]
        counts[key]=counts.get(key,0)+1
    result={"protocol":"p1-corrected-v2","trajectories":rows,"counts":counts,
            "unique_trajectories":len(rows),"optimizer_steps":sum(r["recipe"]["steps"] for r in rows),
            "required_hosts":list(HOSTS),"optional_2b_included":False,
            "evaluation_seeds":{"base":900042,"generation":700044},
            "composition":{"steps":[150,1500],"operators":["factor_average","effective_sum","effective_mean"]}}
    write_json(output/"matrix.json",result)
    return result
