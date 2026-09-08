"""Seed and cluster uncertainty, retaining pairing across conditions.

Clusters are subjects (IMU), original sources (audio), or images (vision).
If clusters belong to one class, resampling is stratified by class. Subjects
span classes and are sampled jointly. Equal cluster profiles are compressed
only for computation; multinomial frequencies preserve their sampling law.
"""
from __future__ import annotations

import math

import numpy as np


def aligned_items(runs):
    if not runs:
        raise ValueError("No runs supplied")
    keys=[r["id"] for r in runs[0]]
    if len(keys)!=len(set(keys)):
        raise ValueError("Duplicate item IDs; collapse repeated prompts to sensor units first")
    aligned=[]
    for rows in runs:
        by_id={r["id"]:r for r in rows}
        if len(by_id)!=len(rows) or set(by_id)!=set(keys):
            raise ValueError("Seed or paired-condition populations differ")
        ordered=[by_id[k] for k in keys]
        for left,right in zip(runs[0],ordered):
            if left["label"]!=right["label"] or left["group"]!=right["group"]:
                raise ValueError("Labels or cluster identities changed")
        aligned.append(ordered)
    return aligned


def rank_success(rows):
    return np.array([int(np.argmax(r["scores"])==r["label"]) for r in rows],dtype=float)


def ranking_score_diagnostics(rows):
    """Descriptive margins and candidate-score entropy, not calibrated confidence."""
    scores=np.asarray([r["scores"] for r in rows],dtype=float)
    labels=np.asarray([r["label"] for r in rows],dtype=int)
    if scores.ndim!=2 or scores.shape[1]<2 or not np.isfinite(scores).all():
        raise ValueError("Expected finite scores for at least two candidate labels")
    others=scores.copy()
    others[np.arange(len(rows)),labels]=-np.inf
    margin=scores[np.arange(len(rows)),labels]-others.max(axis=1)
    shifted=scores-scores.max(axis=1,keepdims=True)
    logp=shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True))
    entropy=-(np.exp(logp)*logp).sum(axis=1)
    return {"mean_correct_label_margin":float(margin.mean()),
            "mean_candidate_score_entropy_nats":float(entropy.mean())}


def _profiles(success,labels,groups):
    labels=np.asarray(labels,dtype=int)
    names=sorted(set(groups))
    classes=sorted(set(labels.tolist()))
    gmap={name:i for i,name in enumerate(names)}
    cmap={name:i for i,name in enumerate(classes)}
    correct=np.zeros((success.shape[0],len(names),len(classes)))
    counts=np.zeros((len(names),len(classes)))
    for i,(label,group) in enumerate(zip(labels,groups)):
        g,c=gmap[group],cmap[int(label)]
        correct[:,g,c]+=success[:,i]
        counts[g,c]+=1
    stratified=bool(((counts>0).sum(1)==1).all())
    partitions=[np.flatnonzero(counts[:,c]>0) for c in range(len(classes))] if stratified else [np.arange(len(names))]
    profiles=[]
    for indices in partitions:
        values=np.concatenate([correct[:,indices,:].transpose(1,0,2).reshape(len(indices),-1),counts[indices]],axis=1)
        unique,frequencies=np.unique(values,axis=0,return_counts=True)
        nseeds=success.shape[0];nclasses=len(classes)
        profiles.append((unique[:,:nseeds*nclasses].reshape(-1,nseeds,nclasses).transpose(1,0,2),unique[:,nseeds*nclasses:],frequencies))
    return correct,counts,profiles,{"clusters":len(names),"classes":len(classes),"stratified_by_class":stratified}


def accuracy_interval(success,labels,groups,*,replicates=10000,seed=20260904):
    """Crossed seed/cluster percentile bootstrap, for accuracy or paired deltas."""
    success=np.asarray(success,dtype=float)
    if success.ndim!=2 or success.shape[1]!=len(labels) or len(groups)!=len(labels) or not np.isfinite(success).all():
        raise ValueError("Expected finite [training seed, held-out sensor] values")
    correct,counts,profiles,details=_profiles(success,labels,groups)
    per_seed_class=correct.sum(1)/counts.sum(0)[None]
    per_seed_micro=correct.sum((1,2))/counts.sum()
    per_seed_macro=per_seed_class.mean(1)
    rng=np.random.default_rng(seed)
    micros=[];macros=[];invalid=0
    while len(micros)<replicates:
        batch=min(256,replicates-len(micros))
        seed_weights=rng.multinomial(len(success),np.full(len(success),1/len(success)),size=batch)/len(success)
        numerator=np.zeros((batch,counts.shape[1]));denominator=np.zeros_like(numerator)
        for values,sizes,frequencies in profiles:
            weights=rng.multinomial(int(frequencies.sum()),frequencies/frequencies.sum(),size=batch)
            seed_means=np.einsum("bs,sgc->bgc",seed_weights,values)
            numerator+=np.einsum("bg,bgc->bc",weights,seed_means)
            denominator+=weights@sizes
        valid=(denominator>0).all(1)
        invalid+=int((~valid).sum())
        if invalid>replicates:
            raise ValueError("Too few independent clusters to retain every class")
        micros.extend((numerator[valid].sum(1)/denominator[valid].sum(1)).tolist())
        macros.extend((numerator[valid]/denominator[valid]).mean(1).tolist())
    def summary(point,values,per_seed):
        return {"mean":float(point),"ci95":np.quantile(values,[.025,.975]).tolist(),"per_seed":per_seed.tolist(),
                "seed_sd":float(np.std(per_seed,ddof=1)) if len(per_seed)>1 else None}
    return {"micro":summary(per_seed_micro.mean(),micros,per_seed_micro),"macro":summary(per_seed_macro.mean(),macros,per_seed_macro),
            "sensors":len(labels),"training_seeds":len(success),**details,
            "bootstrap":{"replicates":replicates,"seed":seed,"method":"crossed seed/cluster percentile; equal training-seed weight", "rejected_missing_class_draws":invalid}}


def ranking_interval(runs,comparison=None,**kwargs):
    runs=aligned_items(runs)
    values=np.stack([rank_success(r) for r in runs])
    if comparison is not None:
        if len(comparison)==1:
            comparison=comparison*len(runs)
        if len(comparison)!=len(runs):
            raise ValueError("Paired training seeds must match explicitly")
        comparison=aligned_items([runs[0],*comparison])[1:]
        values-=np.stack([rank_success(r) for r in comparison])
    result=accuracy_interval(values,[r["label"] for r in runs[0]],[r["group"] for r in runs[0]],**kwargs)
    result["estimand"]="rank-1 accuracy difference (first minus comparison)" if comparison is not None else "rank-1 accuracy"
    return result


def likelihood_interval(runs,comparison=None,*,replicates=10000,seed=20260904):
    """Byte-weighted likelihood, resampling seeds and shared text units."""
    def canonical(rows):
        return [{**r,"id":str(r.get("id",r.get("row"))),"label":0,"group":str(r.get("group",r.get("id",r.get("row"))))} for r in rows]
    runs=aligned_items([canonical(r) for r in runs])
    nats=np.array([[r["nats"] for r in run] for run in runs])
    byte_counts=np.array([r["bytes"] for r in runs[0]],dtype=float)
    if any([r["bytes"] for r in run]!=[r["bytes"] for r in runs[0]] for run in runs) or (byte_counts<=0).any():
        raise ValueError("Likelihood comparisons require identical positive target-byte counts")
    if comparison is not None:
        if len(comparison)==1:
            comparison=comparison*len(runs)
        if len(comparison)!=len(runs):
            raise ValueError("Paired likelihood training seeds do not match")
        comparison=aligned_items([runs[0],*[canonical(r) for r in comparison]])[1:]
        if any([r["bytes"] for r in run]!=byte_counts.tolist() for run in comparison):
            raise ValueError("Paired likelihood targets differ")
        nats-=np.array([[r["nats"] for r in run] for run in comparison])
    groups=sorted({r["group"] for r in runs[0]})
    mapping={g:i for i,g in enumerate(groups)}
    totals=np.zeros((len(runs),len(groups)));sizes=np.zeros(len(groups))
    for i,row in enumerate(runs[0]):
        group=mapping[row["group"]];totals[:,group]+=nats[:,i];sizes[group]+=byte_counts[i]
    per_seed=totals.sum(1)/(math.log(2)*sizes.sum())
    rng=np.random.default_rng(seed);samples=[]
    for start in range(0,replicates,256):
        b=min(256,replicates-start)
        sw=rng.multinomial(len(runs),np.full(len(runs),1/len(runs)),size=b)/len(runs)
        gw=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=b)
        numerator=np.einsum("bs,sg,bg->b",sw,totals,gw)
        samples.extend((numerator/(math.log(2)*(gw@sizes))).tolist())
    return {"mean":float(per_seed.mean()),"ci95":np.quantile(samples,[.025,.975]).tolist(),"per_seed":per_seed.tolist(),
            "seed_sd":float(per_seed.std(ddof=1)) if len(per_seed)>1 else None,"training_seeds":len(runs),"text_units":len(groups),
            "target_bytes":int(byte_counts.sum()),"bootstrap":{"replicates":replicates,"seed":seed,"method":"crossed seed/text-unit percentile"},
            "estimand":"BPB difference (first minus comparison)" if comparison is not None else "BPB"}


def generation_units(items,family,split,metric="joint_success"):
    grouped={}
    pairs={}
    for row in items:
        if row["family"]!=family or row["template_split"]!=split:
            continue
        item=grouped.setdefault(row["id"],{"id":row["id"],"label":row["label"],"group":row["group"],"values":[]})
        if metric=="both_rule_mappings":
            if family!="rule":
                raise ValueError("Reversed-mapping success is defined only for rule cases")
            key=(row["id"],row["template"],row["query_label"])
            pair=pairs.setdefault(key,{})
            reverse=row["reverse_rule"]
            if reverse in pair:
                raise ValueError("Repeated response for one rule mapping")
            pair[reverse]=row["joint_success"]
        else:
            item["values"].append(float(row[metric]))
    for (sensor,template,query),pair in pairs.items():
        if set(pair)!={False,True}:
            raise ValueError("A rule case is missing its reversed mapping")
        grouped[sensor]["values"].append(float(all(pair.values())))
    return [{**row,"success":float(np.mean(row["values"]))} for row in grouped.values()]


def generation_interval(runs,family,split,metric="joint_success",comparison=None,**kwargs):
    units=aligned_items([generation_units(r,family,split,metric) for r in runs])
    values=np.array([[r["success"] for r in rows] for rows in units])
    if comparison is not None:
        if len(comparison)==1:
            comparison=comparison*len(runs)
        if len(comparison)!=len(runs):
            raise ValueError("Generation comparisons require matching seeds")
        other=aligned_items([units[0],*[generation_units(r,family,split,metric) for r in comparison]])[1:]
        values-=np.array([[r["success"] for r in rows] for rows in other])
    result=accuracy_interval(values,[r["label"] for r in units[0]],[r["group"] for r in units[0]],**kwargs)
    result.update({"family":family,"template_split":split,"metric":metric,"paired_difference":comparison is not None})
    return result
