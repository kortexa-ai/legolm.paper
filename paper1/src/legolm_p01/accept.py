"""Small real-host acceptance runs, explicitly excluded from paper results."""
from pathlib import Path
import gc
import json

import torch

from .train import Recipe,run,make_adapter,load_adapter_state
from .hosts import load_host
from .evaluate import ranking,bpb
from .generation import training_rows,decode,evaluation_cases
from .provenance import write_json


def acceptance(asset_root,data_cache,features,text,output,device):
    dataset=torch.load(features/"audio.pt",map_location="cpu",weights_only=False)
    subset={**dataset,"test":{k:v[:8] for k,v in dataset["test"].items()}}
    reports=[]
    for name in ("mini","lfm230m","qwen08"):
        for cohort in (("U","P","G") if name=="qwen08" else ("U","P")):
            recipe=Recipe(host=name,cohort=cohort,modality="audio",steps=4,batch_size=8 if cohort=="U" else 4,text_length=256 if cohort=="G" else 128)
            completed=run(recipe,asset_root,data_cache,features,text,output/"acceptance-only",device)
            path=Path(completed["directory"])/completed["checkpoints"][-1]["path"]
            state=torch.load(path,map_location="cpu",weights_only=False)
            host=load_host(name,asset_root,data_cache,device)
            module,_=make_adapter(host,recipe,dataset["train"]["features"].shape[1])
            load_adapter_state(host,module,"bridge",state["adapter"])
            del state
            host.model.eval();module.eval()
            constant=dataset["train"]["features"].mean(0)
            manifest=json.loads((text/f"text-{name}.json").read_text())
            rows=torch.load(text/manifest["file"],weights_only=False)["val"]
            report={"host":name,"cohort":cohort,"steps":4,"excluded_from_results":True,
                    "training_elapsed_s":completed["elapsed_s"],"memory":completed["memory"],
                    "ranking_8_items":ranking(host,module,"bridge",subset,constant,sensor_batch=1 if name=="mini" else 2)["summary"],
                    "bpb_8_rows":bpb(host,module,"bridge",rows,dataset["unpaired"]["features"],constant,count=8)["bpb"]}
            if cohort=="G":
                items=evaluation_cases(dataset,[0,1])
                prompts=[items[0]["prompt"],items[-1]["prompt"]]
                f=dataset["test"]["features"][:2]
                batch=decode(host,module,"bridge",f,prompts,native_chat=True,max_new_tokens=8)
                singles=[decode(host,module,"bridge",f[i:i+1],[p],native_chat=True,max_new_tokens=8)[0] for i,p in enumerate(prompts)]
                if batch!=singles:
                    raise AssertionError({"generation_batch":batch,"generation_scalar":singles})
                report["generation_batch_scalar_equal"]=True
                report["generation_fixture_outputs"]=batch
            reports.append(report)
            write_json(output/"training-acceptance.json",reports)
            print(json.dumps({"acceptance":report}),flush=True)
            del host,module
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    return reports
