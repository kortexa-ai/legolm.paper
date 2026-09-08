"""Same-constituent comparison of factor averaging and effective-update merges."""
from itertools import combinations
import json
import math
from pathlib import Path

import torch

from .hosts import load_host,Bridge
from .evaluate import save_evaluation
from .provenance import file_hash,source_identity,write_json
from .backend import identity as backend_identity


@torch.no_grad()
def run(checkpoints,asset_root,data_cache,features,text,output,device):
    states=[torch.load(p,map_location="cpu",weights_only=False) for p in checkpoints]
    identities=[s["identity"] for s in states]
    first=identities[0]
    names=[i["recipe"]["modality"] for i in identities]
    if sorted(names)!=["audio","imu","vision"]:
        raise ValueError("Composition needs exactly the three registered constituents")
    step=states[0]["step"]
    if step not in (150,1500) or any(s["step"]!=step for s in states):
        raise ValueError("Constituents must share a registered composition milestone")
    for identity in identities:
        r=identity["recipe"]
        if (r["cohort"],r["kind"],r["feature_mode"],r["diversity"])!=("U","bridge","true",0.) or r["seed"]!=first["recipe"]["seed"]:
            raise ValueError("Unmatched composition recipe")
        if identity["host"]!=first["host"] or identity["text_sha256"]!=first["text_sha256"]:
            raise ValueError("Constituents disagree on host or text")
    host=load_host(first["recipe"]["host"],asset_root,data_cache,device)
    if host.identity!=first["host"]:
        raise ValueError("Loaded host differs from composition checkpoints")
    host.model.eval()
    manifest=json.loads((text/f"text-{host.name}.json").read_text())
    if manifest["sha256"]!=first["text_sha256"] or file_hash(text/manifest["file"])!=manifest["sha256"]:
        raise ValueError("Composition text changed")
    rows=torch.load(text/manifest["file"],map_location="cpu",weights_only=False)["val"][:256 if step==150 else 1024]
    pools,vectors={},{}
    for identity,state in zip(identities,states):
        modality=identity["recipe"]["modality"]
        path=features/f"{modality}.pt"
        if file_hash(path)!=identity["features_sha256"]:
            raise ValueError("Composition feature cache changed")
        pool=torch.load(path,map_location="cpu",weights_only=False)["unpaired"]
        bridge=Bridge(pool["features"].shape[1],host.dimension).to(device).eval()
        bridge.load_state_dict(state["adapter"],strict=True)
        vectors[modality]=bridge(pool["features"].to(device))
        pools[modality]=pool
        del bridge
    del states
    results=[]
    subsets=[()] + [c for n in (1,2,3) for c in combinations(sorted(names),n)]
    widths=host.tokens.widths.to(device)
    for subset in subsets:
        for operator in (("off",) if not subset else ("factor_average","effective_sum","effective_mean")):
            items=[]
            for start in range(0,len(rows),8):
                data=rows[start:start+8].to(device)
                indices={m:(torch.arange(len(data),device=device)+start)%len(vectors[m]) for m in subset}
                parts=[vectors[m][indices[m]] for m in subset]
                if not parts:
                    host.inject(enabled=False)
                elif operator=="factor_average":
                    host.inject(torch.stack(parts).mean(0))
                else:
                    host.inject(components=parts,mean=operator=="effective_mean")
                losses=host.losses(data[:,:-1],data[:,1:])
                byte_widths=widths[data[:,1:]]
                for i in range(len(data)):
                    items.append({"row":start+i,"nats":float((losses[i]*(byte_widths[i]>0)).sum()),"bytes":int(byte_widths[i].sum()),
                        "sensor_ids":{m:pools[m]["ids"][int(indices[m][i])] for m in subset}})
            identity={"measurement":"composition","host":host.name,"seed":first["recipe"]["seed"],"step":step,
                      "subset":list(subset),"operator":operator,"checkpoints":[{"modality":m,"sha256":file_hash(p)} for m,p in zip(names,checkpoints)],
                      "source":source_identity(),"backend":backend_identity(device),"text_sha256":manifest["sha256"],"rows":len(rows),"sensor_selection":"original deterministic cycles, one independent source pool per modality"}
            value=sum(i["nats"] for i in items)/(math.log(2)*sum(i["bytes"] for i in items))
            record=save_evaluation(output,identity,{"summary":{"bpb":value,"rows":len(rows)},"items":items})
            record.update({k:identity[k] for k in ("host","seed","step","subset","operator")})
            results.append(record)
            print(json.dumps({"composition":{k:record[k] for k in ("host","seed","step","subset","operator","summary")}}),flush=True)
    write_json(Path(output)/"composition.json",results)
    return results
