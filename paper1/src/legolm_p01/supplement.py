"""Serial composition and isolated resource measurements after the main grid."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import signal

from .artifacts import load
from .provenance import file_hash, write_json
from .outcomes import TERMINAL, terminal_record


def run(matrix,asset_root,data_cache,features,text,output,device):
    matrix,output=Path(matrix),Path(output)
    done=load(matrix/"matrix-complete.json")
    if not done["passed"] or done["completed"]!=411 or not done["evaluated"] or done["excluded_from_results"]:
        raise ValueError("Supplemental measurement requires the complete evaluated matrix")
    jobs=[load(p) for p in matrix.glob("job-*.json")]
    if len(jobs)!=411 or any(j["status"] not in TERMINAL for j in jobs):
        raise ValueError("Main matrix jobs are incomplete")
    output.mkdir(parents=True,exist_ok=True)
    def checkpoint(host,modality,kind,seed,step):
        selected=[j for j in jobs if all(j["recipe"][k]==v for k,v in {
            "host":host,"cohort":"U","modality":"imu" if kind in ("none","static") else modality,
            "kind":kind,"feature_mode":"true","seed":seed,"diversity":0.}.items())]
        if len(selected)!=1:
            raise ValueError("Missing or ambiguous supplemental constituent")
        completed=terminal_record(selected[0])
        saved=next((p for p in completed["checkpoints"] if p["step"]==step),None)
        if saved is None:
            raise ValueError(f"Supplement requires an unavailable {host}/{modality}/{kind}/seed{seed}/step{step} checkpoint")
        path=Path(completed["directory"])/saved["path"]
        if file_hash(path)!=saved["sha256"]:
            raise ValueError("Supplemental checkpoint hash mismatch")
        return path
    common=["--asset-root",str(asset_root),"--data-cache",str(data_cache),"--features",str(features),"--text",str(text),"--device",device]
    environment=os.environ.copy()
    environment.update(OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false")
    completed=[]
    process_state={"stopped":False,"child":None}
    def stop_owned(signum,frame):
        process_state["stopped"]=True
        child=process_state["child"]
        if child is not None and child.poll() is None:
            child.terminate()
    handlers={s:signal.signal(s,stop_owned) for s in (signal.SIGTERM,signal.SIGINT)}
    def execute(name,args,expected,target=None):
        if process_state["stopped"]:
            raise InterruptedError("Supplemental scheduling stopped")
        directory=Path(target) if target is not None else output/name
        directory.mkdir(parents=True,exist_ok=True)
        obligation={"command":args,"checkpoints":{str(p):file_hash(p) for p in expected}}
        record=directory/"complete.json"
        if record.exists():
            saved=load(record)
            if saved["obligation"]!=obligation or any(file_hash(directory/p)!=sha for p,sha in saved["outputs"].items()):
                raise ValueError("Supplemental cache does not match its declared inputs")
            completed.append(name)
            return
        attempt=1
        while (directory/f"attempt-{attempt}.log").exists():
            attempt+=1
        cmd=[sys.executable,"-m","legolm_p01.cli",*args,*common,"--output",str(directory)]
        with (directory/f"attempt-{attempt}.log").open("w") as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=environment)
            process_state["child"]=child
            if process_state["stopped"]:
                child.terminate()
            code=child.wait()
            process_state["child"]=None
        if code or process_state["stopped"]:
            raise RuntimeError(f"Supplement {name} failed; inspect its attempt log")
        outputs={p.name:file_hash(p) for p in directory.iterdir() if p.suffix in (".json",".gz") and p.name!="complete.json"}
        if not outputs:
            raise ValueError("Supplement exited without evidence")
        write_json(record,{"obligation":obligation,"outputs":outputs})
        completed.append(name)
        write_json(output/"progress.json",{"completed":completed,"expected":57})
        print(json.dumps({"supplement_complete":name,"completed":len(completed),"expected":57}),flush=True)
    try:
        for modality in ("imu","audio","vision"):
            path=checkpoint("qwen08",modality,"none",42,0)
            target=matrix/"evaluations"/(file_hash(path)+"-"+modality)
            execute(f"raw-generation-qwen08-{modality}",["evaluate","--checkpoint",str(path),"--suites","generation","--eval-modality",modality],[path],target)
        for host in ("mini","lfm230m","qwen08"):
            for seed in (42,1042,2042):
                for step in (150,1500):
                    paths=[checkpoint(host,m,"bridge",seed,step) for m in ("imu","audio","vision")]
                    execute(f"composition-{host}-s{seed}-step{step}",["compose","--checkpoints",*[str(p) for p in paths]],paths)
            for modality in ("imu","audio","vision"):
                for kind in ("none","static","bridge","prefix"):
                    path=checkpoint(host,modality,kind,42,0 if kind=="none" else 3000)
                    execute(f"resources-{host}-{modality}-{kind}",["resources","--checkpoint",str(path),"--eval-modality",modality],[path])
    finally:
        for signum,handler in handlers.items():
            signal.signal(signum,handler)
    result={"matrix_id":done["matrix_id"],"completed":completed,"expected":57,"passed":len(completed)==57}
    write_json(output/"supplement-complete.json",result)
    return result
