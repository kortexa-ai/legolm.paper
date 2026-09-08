"""Fixed-step training, immutable endpoints, and resumable optimizer state."""
from __future__ import annotations

import gc
import json
import math
import random
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .hosts import Bridge, Prefix, load_host
from .canary import precision, memory
from .provenance import digest, file_hash, write_json, source_identity
from .text import label_rows
from .backend import identity as backend_identity


@dataclass(frozen=True)
class Recipe:
    host: str
    cohort: str
    modality: str
    kind: str = "bridge"
    feature_mode: str = "true"
    seed: int = 42
    steps: int = 3000
    diversity: float = 0.
    lr: float = .001
    rank: int = 4
    text_length: int = 128
    prefix_length: int = 8
    batch_size: int = 8
    protocol: str = "p1-corrected-v2"

    def __post_init__(self):
        if self.cohort not in ("U","P","G") or self.kind not in ("bridge","prefix","static","none"):
            raise ValueError("Unknown training interface or objective")
        if self.feature_mode not in ("true","shuffled","random","constant"):
            raise ValueError("Unknown feature control")
        if self.rank != 4 or self.text_length != (256 if self.cohort=="G" else 128) or self.prefix_length != 8:
            raise ValueError("This release preserves the registered rank/context/prefix recipe")
        if self.cohort=="G" and self.host!="qwen08":
            raise ValueError("The registered generation training host is Qwen 0.8B")
        if self.diversity and self.kind!="bridge":
            raise ValueError("The diversity penalty is defined only for generated LoRA factors")


def rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(value,temp)
    temp.replace(path)
    return file_hash(path)


def adapter_state(host, adaptation, kind):
    if kind == "static":
        return {name:{"a":m.a.detach().cpu(),"b":m.b.detach().cpu()} for name,m in host.targets}
    return {} if adaptation is None else {k:v.detach().cpu() for k,v in adaptation.state_dict().items()}


def load_adapter_state(host, adaptation, kind, state):
    if kind == "static":
        if set(state) != {n for n,_ in host.targets}:
            raise ValueError("Static checkpoint target map mismatch")
        with torch.no_grad():
            for name,m in host.targets:
                m.a.copy_(state[name]["a"]);m.b.copy_(state[name]["b"])
    elif adaptation is not None:
        adaptation.load_state_dict(state,strict=True)


def make_adapter(host, recipe, feature_dim):
    torch.manual_seed(recipe.seed)
    if recipe.kind == "bridge":
        module=Bridge(feature_dim,host.dimension).to(host.device)
        return module,list(module.parameters())
    if recipe.kind == "prefix":
        module=Prefix(feature_dim,host.hidden,recipe.prefix_length).to(host.device)
        return module,list(module.parameters())
    if recipe.kind == "static":
        # Explicit seed initialization removes dependence on HF loader internals.
        with torch.no_grad():
            for _,m in host.targets:
                m.a.normal_(0,.01);m.b.zero_()
        return None,[p for _,m in host.targets for p in (m.a,m.b)]
    return None,[]


def training_source(source):
    files={n:h for n,h in source["files"].items() if n in
        ("train.py","hosts.py","text.py","canary.py","generation.py","provenance.py","backend.py") or n.startswith("legacy/")}
    result={k:v for k,v in source.items() if k not in ("git_commit","files","source_hash")}
    return {**result,"training_files":files}


def train_indices(dataset):
    if "_train_indices" not in dataset:
        excluded=set(dataset.get("dev_indices",[]))
        dataset["_train_indices"]=[i for i in range(len(dataset["train"]["ids"])) if i not in excluded]
    return dataset["_train_indices"]


def make_batch(recipe, step, dataset, text_rows, host):
    # Independent, stateless sampling streams allow exact restart at any step.
    generator=torch.Generator().manual_seed(recipe.seed*1000003+step)
    if recipe.cohort == "U":
        pool=dataset["unpaired"]["features"]
        indices=torch.randperm(len(pool),generator=generator)[:recipe.batch_size]
        offset=step*recipe.batch_size
        rows=text_rows[offset:offset+recipe.batch_size]
        if len(rows) != recipe.batch_size:
            raise ValueError("Materialized text stream is shorter than the registered budget")
        x,y=rows[:,:-1],rows[:,1:]
        labels=None
    else:
        pool=dataset["train"]["features"]
        available=train_indices(dataset)
        selector=random.Random(recipe.seed*1000003+step)
        indices=torch.tensor(selector.sample(available,recipe.batch_size))
        labels=[dataset["train"]["labels"][i] for i in indices.tolist()]
        if recipe.cohort == "G":
            from .generation import training_rows
            x,y=training_rows(host.tokens,dataset,labels,step)
        else:
            key="_label_rows_"+host.name
            if key not in dataset:
                dataset[key]=label_rows(host.tokens,dataset["prompt"],dataset["categories"])
            all_x,all_y,_=dataset[key]
            x,y=all_x[labels],all_y[labels]
    f=pool[indices]
    key="_constant_"+recipe.cohort
    if key not in dataset:
        dataset[key]=pool.mean(dim=0) if recipe.cohort == "U" else pool[train_indices(dataset)].mean(dim=0)
    constant=dataset[key]
    if recipe.feature_mode == "constant":
        f=constant[None].expand(len(indices),-1)
    elif recipe.feature_mode == "random":
        f=torch.randn(f.shape,generator=generator)
    elif recipe.feature_mode == "shuffled":
        if recipe.cohort == "U":
            permutation=torch.randperm(len(pool),generator=torch.Generator().manual_seed(recipe.seed+17))
            f=pool[permutation[indices]]
        else:
            f=torch.roll(f,-1,dims=0)
    return x.to(host.device),y.to(host.device),f.to(host.device),constant


def optimize_step(host, adaptation, params, optimizer, recipe, batch):
    x,y,features,_=batch
    optimizer.zero_grad(set_to_none=True)
    with precision(str(host.device)):
        if recipe.kind == "bridge":
            vectors=adaptation(features)
            host.inject(vectors)
            prefix=None
        elif recipe.kind == "prefix":
            host.inject(enabled=False)
            prefix=adaptation(features)
        else:
            host.inject();prefix=None
        losses=host.losses(x,y,prefix)
        # Equal weighting of examples preserves the original mean task loss.
        loss=(losses.sum(dim=1)/(y>=0).sum(dim=1)).mean()
        diversity=loss.new_tensor(0.)
        if recipe.diversity:
            z=F.normalize(vectors.float(),dim=1)
            mask=~torch.eye(len(z),device=z.device,dtype=torch.bool)
            diversity=(z@z.T)[mask].mean()
            loss=loss+recipe.diversity*diversity
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    optimizer.step()
    return {"loss":float(loss.detach()),"diversity":float(diversity.detach()),"gradient_norm":float(norm)}


def run(recipe, asset_root, data_cache, features_dir, text_dir, output, device, *, stop_after=None):
    from .evaluate import variation, stratified_indices
    dataset_path=features_dir/f"{recipe.modality}.pt"
    dataset=torch.load(dataset_path,map_location="cpu",weights_only=False)
    text_manifest=json.loads((text_dir/f"text-{recipe.host}.json").read_text())
    text_path=text_dir/text_manifest["file"]
    if file_hash(text_path) != text_manifest["sha256"]:
        raise ValueError("Text cache integrity mismatch")
    text=torch.load(text_path,map_location="cpu",weights_only=False)
    torch.manual_seed(recipe.seed);random.seed(recipe.seed);np.random.seed(recipe.seed)
    host=load_host(recipe.host,asset_root,data_cache,device)
    adaptation,params=make_adapter(host,recipe,dataset["train"]["features"].shape[1])
    source=source_identity()
    identity={"recipe":asdict(recipe),"host":host.identity,"backend":backend_identity(device),"features_sha256":file_hash(dataset_path),
              "text_sha256":text_manifest["sha256"],"source":source,
              "optimizer":{"type":"AdamW","lr":recipe.lr,"betas":[.9,.999],"eps":1e-8,"weight_decay":.01,"clip_norm":1.},
              "precision":"FP32 parameters, optimizer and execution; TF32 and autocast disabled"}
    scientific_source=training_source(source)
    # Retain the repository commit as provenance, but a documentation-only
    # commit must not invalidate numerically identical source and dependencies.
    training_id=digest({**identity,"source":scientific_source})
    label=f"{recipe.host}-{recipe.cohort}-{recipe.modality}-{recipe.kind}-{recipe.feature_mode}-s{recipe.seed}-d{recipe.diversity:g}"
    directory=output/f"{label}-{training_id[:12]}"
    directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/"config.json",{"training_id":training_id,"identity":identity})
    done=directory/"complete.json"
    if done.exists():
        completed=json.loads(done.read_text())
        if completed["training_id"] != training_id:
            raise ValueError("Completed run identity mismatch")
        for item in completed["checkpoints"]:
            if file_hash(directory/item["path"]) != item["sha256"]:
                raise ValueError("Completed checkpoint integrity mismatch")
        return completed
    optimizer=torch.optim.AdamW(params,lr=recipe.lr,betas=(.9,.999),eps=1e-8,weight_decay=.01) if params else None
    resume=directory/"resume.pt"
    current,checkpoints,elapsed=0,[],0.
    exposure={"examples":0,"supervised_tokens":0,"supervised_bytes":0}
    if resume.exists():
        state=torch.load(resume,map_location="cpu",weights_only=False)
        if state["training_id"] != training_id:
            raise ValueError("Resume rejected: training identity mismatch")
        load_adapter_state(host,adaptation,recipe.kind,state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        current,checkpoints,elapsed=state["step"],state["checkpoints"],state["elapsed_s"]
        exposure=state["exposure"]
        del state
    frozen_versions={n:p._version for n,p in host.model.named_parameters() if not p.requires_grad}
    short=300 if recipe.cohort == "U" else 600
    milestones={short,recipe.steps}
    if recipe.kind=="prefix":
        milestones.add(0)
    if recipe.cohort == "U" and recipe.kind == "bridge" and recipe.feature_mode == "true":
        milestones.update([150,1500])
    trajectory=recipe.kind == "bridge" and recipe.modality == "imu" and recipe.feature_mode == "true"
    if trajectory:
        milestones.update([0,30,100,300,1000,3000] if recipe.cohort == "U" else [0,100,300,600,1200,2400,6000])
    milestones={s for s in milestones if s<=recipe.steps}
    if recipe.kind == "none":
        milestones={0}
    host.model.train()
    def endpoint(step):
        state={"training_id":training_id,"identity":identity,"step":step,"adapter":adapter_state(host,adaptation,recipe.kind),
               "rng":rng_state(),"elapsed_s":elapsed,"exposure":dict(exposure)}
        path=directory/f"step-{step:06d}.pt"
        checksum=save_torch(path,state)
        item={"step":step,"path":path.name,"sha256":checksum,"elapsed_s":elapsed,"exposure":dict(exposure)}
        if trajectory:
            # Geometry uses held-out inputs but no held-out labels or selection.
            saved=rng_state()
            indices=stratified_indices(dataset["test"]["labels"],32,42)
            item["variation"]=variation(host,adaptation,dataset["test"]["features"][indices])
            item["variation"]["sample_ids"]=[dataset["test"]["ids"][i] for i in indices]
            restore_rng(saved)
        checkpoints.append(item)
        write_json(directory/"checkpoints.json",checkpoints)
        print(json.dumps({"checkpoint":label,"step":step,"sha256":checksum,"elapsed_s":elapsed}),flush=True)
    if current==0 and 0 in milestones and not checkpoints:
        endpoint(0)
    for step in range(current,recipe.steps if optimizer else 0):
        start=time.monotonic()
        batch=make_batch(recipe,step,dataset,text["train"],host)
        metrics=optimize_step(host,adaptation,params,optimizer,recipe,batch)
        targets=batch[1].detach().cpu()
        scored=targets>=0
        exposure["examples"]+=len(targets)
        exposure["supervised_tokens"]+=int(scored.sum())
        exposure["supervised_bytes"]+=int((host.tokens.widths[targets.clamp_min(0)]*scored).sum())
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed+=time.monotonic()-start
        if step==current:
            snapshot=memory(device)
            if snapshot and snapshot["free_gib"]<10:
                raise MemoryError("Training crosses the 10 GiB free-memory floor")
        if (step+1)%50==0 or step==0:
            record={"step":step+1,"elapsed_s":elapsed,**metrics,**memory(device)}
            with (directory/"training.jsonl").open("a") as f:
                f.write(json.dumps(record,allow_nan=False)+"\n")
            print(json.dumps({"training":label,**record}),flush=True)
        if step+1 in milestones:
            endpoint(step+1)
        if (step+1)%300==0 or (stop_after is not None and step+1==stop_after):
            save_torch(resume,{"training_id":training_id,"step":step+1,"adapter":adapter_state(host,adaptation,recipe.kind),
                "optimizer":optimizer.state_dict(),"rng":rng_state(),"checkpoints":checkpoints,"elapsed_s":elapsed,"exposure":dict(exposure)})
        if stop_after is not None and step+1==stop_after:
            return {"training_id":training_id,"interrupted_at":step+1,"directory":str(directory)}
    for n,p in host.model.named_parameters():
        if n in frozen_versions and p._version!=frozen_versions[n]:
            raise AssertionError(f"Frozen parameter mutated: {n}")
    completed={"training_id":training_id,"recipe":asdict(recipe),"directory":str(directory),"checkpoints":checkpoints,
               "elapsed_s":elapsed,"trainable_parameters":sum(p.numel() for p in params),"memory":memory(device),"exposure":exposure,
               "peak_process_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)}
    write_json(done,completed)
    if resume.exists():
        # This process created this exact transient optimizer checkpoint. All
        # scientific endpoints remain; completed runs need no active resume file.
        resume.unlink()
    return completed
