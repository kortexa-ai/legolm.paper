"""Evaluate saved adapters without optimizer steps or training RNG side effects."""
from __future__ import annotations

import gzip
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .canary import precision
from .text import label_rows
from .provenance import digest, file_hash, write_json

PARAPHRASES = {
    "imu": ["The activity is: ", "The person is: ", "Detected human activity: ", "This motion corresponds to: "],
    "audio": ["The sound is: ", "The recorded sound comes from: ", "Detected audio event: ", "The sound category is: "],
    "vision": ["The image shows: ", "The pictured object is: ", "The image category is: ", "This is an image of: "],
}


def stratified_indices(labels, per_class=32, seed=42):
    rng=random.Random(seed)
    result=[]
    for label in sorted(set(labels)):
        candidates=[i for i,y in enumerate(labels) if y==label]
        rng.shuffle(candidates)
        result.extend(candidates[:per_class])
    return result


def intervene(features, labels, mode, seed, constant):
    generator = torch.Generator().manual_seed(seed)
    if mode in ("correct", "off"):
        return features, list(range(len(features)))
    if mode == "constant":
        return constant[None].expand(len(features), -1), None
    if mode == "random":
        return torch.randn(features.shape, generator=generator), None
    if mode.startswith("shuffle"):
        indices = torch.randperm(len(features), generator=generator)
        return features[indices], indices.tolist()
    if mode == "different_class":
        y = np.array(labels)
        rng = np.random.default_rng(seed)
        indices = [int(rng.choice(np.flatnonzero(y != label))) for label in y]
        return features[indices], indices
    raise ValueError(mode)


def configure(host, adaptation, kind, features, *, intervention="correct", repeat=1):
    if kind == "none" or intervention == "off":
        host.inject(enabled=False)
        return None
    if kind == "prefix":
        host.inject(enabled=False)
        prefix = adaptation(features)
        return prefix.repeat_interleave(repeat,dim=0) if repeat != 1 else prefix
    if kind == "static":
        host.inject()
        return None
    host.inject(adaptation(features), repeat=repeat)
    return None


def accuracy_summary(scores, labels):
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if scores.shape[0] != len(y) or not np.isfinite(scores).all():
        raise ValueError("Incomplete or nonfinite candidate scores")
    # Stable category ordering is the registered tie rule.
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.argmax(order == y[:,None], axis=1) + 1
    correct = ranks == 1
    classes = sorted(set(y.tolist()))
    return {"n": len(y), "micro_accuracy": float(correct.mean()),
            "macro_accuracy": float(np.mean([correct[y == c].mean() for c in classes])),
            "top2": float((ranks <= 2).mean()), "average_rank": float(ranks.mean()),
            "per_class": {str(c): {"n": int((y == c).sum()), "accuracy": float(correct[y == c].mean())} for c in classes}}


def legacy_subset_indices(dataset,count=64,seed=42):
    """Original insertion-order shuffle, then label-name round robin and pop."""
    groups={}
    for i,label in enumerate(dataset["test"]["labels"]):
        groups.setdefault(dataset["categories"][label],[]).append(i)
    rng=random.Random(seed)
    for items in groups.values():
        rng.shuffle(items)
    chosen=[]
    while len(chosen)<min(count,len(dataset["test"]["ids"])):
        for label in sorted(groups):
            if groups[label] and len(chosen)<count:
                chosen.append(groups[label].pop())
    return chosen


@torch.no_grad()
def ranking(host, adaptation, kind, dataset, constant, *, prompt=None, intervention="correct", seed=42, sensor_batch=4, legacy_seed=42):
    test = dataset["test"]
    selected, mapping = intervene(test["features"], test["labels"], intervention, seed, constant.cpu())
    x,y,mask = label_rows(host.tokens, prompt or dataset["prompt"], dataset["categories"])
    x,y,mask = x.to(host.device),y.to(host.device),mask.to(host.device)
    nclasses = len(dataset["categories"])
    all_scores, target_nats, target_bytes = [], [], []
    start_time = time.monotonic()
    fixed_losses=None
    if kind in ("none","static") or intervention in ("off","constant"):
        # These cases have no input-dependent weights. Evaluate their shared
        # prompt/category sequences once, retaining every held-out label below.
        prefix=configure(host,adaptation,kind,selected[:1].to(host.device),intervention=intervention,repeat=nclasses)
        fixed_losses=host.losses(x,y,prefix)[None]
    for start in range(0, len(selected), sensor_batch):
        f = selected[start:start+sensor_batch].to(host.device)
        batch = len(f)
        with precision(str(host.device)):
            if fixed_losses is None:
                prefix = configure(host, adaptation, kind, f, intervention=intervention, repeat=nclasses)
                losses = host.losses(x.repeat(batch,1),y.repeat(batch,1),prefix).reshape(batch,nclasses,-1)
            else:
                losses=fixed_losses.expand(batch,-1,-1)
            nats = (losses * mask[None]).sum(dim=-1)
            scores = -nats / mask.sum(dim=-1)[None]
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Nonfinite ranking score")
        all_scores.extend(scores.float().cpu().tolist())
        widths = host.tokens.widths.to(host.device)[y.clamp_min(0)] * mask
        for row in range(batch):
            label = test["labels"][start+row]
            target_nats.append(float(nats[row,label]))
            target_bytes.append(int(widths[label].sum()))
    summary = accuracy_summary(all_scores,test["labels"])
    legacy=legacy_subset_indices(dataset,64,legacy_seed)
    summary["legacy_64_size_diagnostic"]=accuracy_summary([all_scores[i] for i in legacy],[test["labels"][i] for i in legacy])
    summary["legacy_64_seed"]=legacy_seed
    summary["legacy_64_sample_ids"]=[test["ids"][i] for i in legacy]
    summary["paired_label_bpb"] = sum(target_nats)/(math.log(2)*sum(target_bytes))
    summary["elapsed_s"] = time.monotonic()-start_time
    records = [{"id": ident, "label": label, "group": group, "scores": scores, "target_nats": nats, "target_bytes": nbytes,
                "sensor_id": None if mapping is None else test["ids"][mapping[i]]}
               for i,(ident,label,group,scores,nats,nbytes) in enumerate(zip(test["ids"], test["labels"], test["groups"], all_scores,target_nats,target_bytes))]
    return {"summary": summary,"items": records,"intervention": intervention,"intervention_seed": seed,"prompt": prompt or dataset["prompt"]}


@torch.no_grad()
def bpb(host, adaptation, kind, rows, features, constant, *, feature_mode="true", seed=42, count=1024, batch_size=8):
    rows = rows[:count]
    generator = torch.Generator().manual_seed(seed)
    if feature_mode == "shuffled":
        features = features[torch.randperm(len(features),generator=generator)]
    totals, records = [0.,0], []
    widths = host.tokens.widths.to(host.device)
    start_time=time.monotonic()
    for start in range(0,len(rows),batch_size):
        data=rows[start:start+batch_size].to(host.device)
        f=features[(torch.arange(len(data))+start)%len(features)]
        if feature_mode == "constant":
            f=constant.cpu()[None].expand(len(data),-1)
        elif feature_mode == "random":
            f=torch.randn(f.shape,generator=generator)
        with precision(str(host.device)):
            prefix=configure(host,adaptation,kind,f.to(host.device))
            losses=host.losses(data[:,:-1],data[:,1:],prefix)
        nbytes=widths[data[:,1:]]
        nats=(losses*(nbytes>0)).sum(dim=1).double().cpu().tolist()
        byte_counts=nbytes.sum(dim=1).cpu().tolist()
        for i,(n,b) in enumerate(zip(nats,byte_counts)):
            if not math.isfinite(n):
                raise FloatingPointError("Nonfinite BPB")
            totals[0]+=n;totals[1]+=b
            records.append({"row":start+i,"nats":n,"bytes":b})
    return {"bpb":totals[0]/(math.log(2)*totals[1]),"nats":totals[0],"bytes":totals[1],
            "rows":len(rows),"scored_tokens":int((widths[rows[:,1:].to(host.device)]>0).sum()),
            "items":records,"elapsed_s":time.monotonic()-start_time}


@torch.no_grad()
def variation(host, bridge, features):
    """Factor and effective-update geometry use the same saved bridge."""
    with precision(str(host.device)):
        vectors=bridge(features.to(host.device)).float()
    normed=F.normalize(vectors,dim=1)
    cos=normed@normed.T
    mask=~torch.eye(len(vectors),device=vectors.device,dtype=torch.bool)
    # Frobenius products of AB without materializing full dense host updates:
    # <AiBi,AjBj> = sum((Ai^T Aj) * (Bi Bj^T)).
    gram=torch.zeros((len(vectors),len(vectors)),device=vectors.device)
    module_stats=[]
    for entry in host.target_manifest():
        o=entry["offset"]; ashape=entry["a_shape"];bshape=entry["b_shape"]
        na=math.prod(ashape);nb=math.prod(bshape)
        a=vectors[:,o:o+na].reshape(-1,*ashape)
        b=vectors[:,o+na:o+na+nb].reshape(-1,*bshape)
        aa=torch.einsum("nir,mik->nmrk",a,a)
        bb=torch.einsum("nrj,mkj->nmrk",b,b)
        g=(aa*bb).sum(dim=(-1,-2))
        gram+=g
        module_stats.append({"module":entry["module"],"mean_squared_frobenius":float(g.diag().mean())})
    lengths=gram.diag().clamp_min(0).sqrt()
    effective=gram/(lengths[:,None]*lengths[None,:]).clamp_min(1e-30)
    centered=gram.diag().mean()-gram.mean()
    return {"n":len(vectors),"factor_cosine":float(cos[mask].mean()),"factor_variance":float(vectors.var(dim=0,unbiased=False).mean()),
            "effective_cosine":float(effective[mask].mean()),"effective_centered_energy":float(centered),
            "effective_mean_squared_norm":float(gram.diag().mean()),"modules":module_stats}


def save_evaluation(directory, identity, result):
    evaluation_id=digest(identity)
    path=Path(directory)/f"{evaluation_id}.json.gz"
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path.with_suffix(".tmp"),"wt") as f:
        json.dump({"evaluation_id":evaluation_id,"identity":identity,**result},f,allow_nan=False,separators=(",",":"))
    path.with_suffix(".tmp").replace(path)
    return {"evaluation_id":evaluation_id,"path":path.name,"sha256":file_hash(path),"summary":result.get("summary",{k:v for k,v in result.items() if k!="items"})}


@torch.no_grad()
def common_bpb(host, adaptation, kind, examples, features, constant, *, feature_mode="true", seed=42, batch_size=8):
    records=[]
    generator=torch.Generator().manual_seed(seed)
    for start in range(0,len(examples),batch_size):
        chosen=examples[start:start+batch_size]
        views=[r["views"][host.name] for r in chosen]
        width=max(len(v["x"]) for v in views)
        x=torch.zeros((len(chosen),width),dtype=torch.long,device=host.device)
        y=torch.full_like(x,-1)
        mask=torch.zeros_like(x,dtype=torch.bool)
        for i,v in enumerate(views):
            n=len(v["x"])
            x[i,:n]=torch.tensor(v["x"],device=host.device)
            y[i,:n]=torch.tensor(v["y"],device=host.device)
            mask[i,:n]=torch.tensor(v["mask"],device=host.device)
        f=features[(torch.arange(len(chosen))+start)%len(features)]
        if feature_mode == "constant":
            f=constant.cpu()[None].expand(len(chosen),-1)
        elif feature_mode == "random":
            f=torch.randn(f.shape,generator=generator)
        elif feature_mode == "shuffled":
            permutation=torch.randperm(len(features),generator=torch.Generator().manual_seed(seed+17))
            f=features[permutation[(torch.arange(len(chosen))+start)%len(features)]]
        with precision(str(host.device)):
            prefix=configure(host,adaptation,kind,f.to(host.device))
            losses=host.losses(x,y,prefix)
        for i,r in enumerate(chosen):
            records.append({"id":r["id"],"group":r["document_id"],"nats":float((losses[i]*mask[i]).sum()),"bytes":len(bytes.fromhex(r["target_hex"]))})
    return {"bpb":sum(r["nats"] for r in records)/(math.log(2)*sum(r["bytes"] for r in records)),"items":records,"rows":len(records)}


@torch.no_grad()
def functional_sensitivity(host, adaptation, kind, features, text_rows):
    results=[]
    for i,row in enumerate(text_rows[:4]):
        x=row[:32][None].to(host.device).expand(len(features),-1)
        with precision(str(host.device)):
            prefix=configure(host,adaptation,kind,features.to(host.device))
            logp=F.log_softmax(host.logits(x,prefix)[:,-1,:],dim=-1)
        p=logp.exp()
        mean=p.mean(dim=0).clamp_min(1e-30)
        kl=(p*(logp-mean.log())).sum(dim=-1)
        results.append({"text_row":i,"n_sensors":len(features),"mean_kl_to_sensor_mixture":float(kl.mean()),
                        "max_kl_to_sensor_mixture":float(kl.max()),"distinct_argmax_tokens":len(set(logp.argmax(dim=-1).tolist()))})
    return {"contexts":results}
