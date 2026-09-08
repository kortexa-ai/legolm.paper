"""A bounded local process pool. CUDA service ownership stays with its caller."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor,as_completed,CancelledError
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal
import threading

from .grid import recipes,manifest
from .train import Recipe,training_source
from .provenance import digest,file_hash,source_identity,write_json
from .backend import identity as backend_identity
from .job_evaluation import obligations
from .checkpoint_eval import evaluator_source
from .artifacts import expected_measurements
from .reconcile import select_catalog
from .outcomes import TERMINAL, terminal_record


def run(asset_root,data_cache,features,text,output,device,*,workers=1,parallel_canary=False,evaluate=False,linear_dir=None):
    if workers not in (1,2,3):
        raise ValueError("At most three coordinated workers were budgeted")
    output=Path(output)
    registered=recipes()
    if parallel_canary:
        registered=[Recipe(host="qwen08",cohort="U",modality="audio",steps=10,seed=s) for s in (42,1042,2042)]
    elif workers>1:
        path=output/"parallel-canary.json"
        if not path.exists():
            raise ValueError("Run the three-worker resource canary before a parallel matrix")
        canary=json.loads(path.read_text())
        if canary["training_source"]!=training_source(source_identity()) or canary.get("backend")!=backend_identity(device) or not canary["passed"]:
            raise ValueError("Parallel canary source changed or did not pass")
    identity={"recipes":[asdict(r) for r in registered],"training_source":training_source(source_identity()),"backend":backend_identity(device),
              "features":{m:file_hash(features/f"{m}.pt") for m in ("imu","audio","vision")},
              "text":{h:file_hash(text/f"text-{h}.json") for h in ("mini","lfm230m","qwen08")}}
    matrix_id=digest(identity)
    directory=output/f"{'parallel-acceptance' if parallel_canary else 'matrix'}-{matrix_id[:12]}"
    directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/"matrix.json",{"matrix_id":matrix_id,"identity":identity,"workers":workers,"evaluate":evaluate,"excluded_from_results":parallel_canary})
    configs=directory/"configs";configs.mkdir(exist_ok=True)
    logs=directory/"logs";logs.mkdir(exist_ok=True)
    jobs=[]
    for recipe in registered:
        key=digest(asdict(recipe))
        config=configs/f"{key}.json"
        write_json(config,asdict(recipe))
        jobs.append((recipe,key,config))
    # Breadth is chosen before results: one seed across hosts first, then other
    # seeds; unregularized comparisons precede the registered penalty sweeps.
    jobs.sort(key=lambda x:(x[0].seed,x[0].diversity,{"U":0,"P":1,"G":2}[x[0].cohort],x[0].kind!="none",x[0].feature_mode!="true",x[0].modality,x[0].host,x[0].feature_mode))
    stopped=threading.Event()
    active=set()
    active_lock=threading.Lock()
    def stop_owned_processes(signum,frame):
        stopped.set()
        with active_lock:
            for process in list(active):
                if process.poll() is None:
                    process.terminate()
    old_handlers={sig:signal.signal(sig,stop_owned_processes) for sig in (signal.SIGTERM,signal.SIGINT)}
    def launch(cmd,log,started):
        with active_lock:
            if stopped.is_set():
                raise CancelledError("Scheduling stopped")
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=environment())
            active.add(process)
        try:
            started(process.pid)
            return process.wait()
        finally:
            with active_lock:
                active.discard(process)
    def environment():
        env=os.environ.copy()
        if device.startswith("cuda") and workers>1:
            # Three hard 25% allocator caps plus contexts leave over 20 GiB
            # physical headroom. Each measured process fits below this cap.
            env.update(SMARTY_GPU_ALLOCATOR_FRACTION="0.25",SMARTY_GPU_PRELOAD_MIN_FREE_GIB="40")
        env.update(OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false",HF_HUB_DISABLE_PROGRESS_BARS="1")
        return env
    def score(record):
        if not evaluate or parallel_canary:
            return
        if linear_dir is None:
            raise ValueError("The complete evaluation requires --linear-dir")
        completed=terminal_record(record)
        planned=obligations(completed)
        source=evaluator_source()
        for item in planned:
            if file_hash(item["checkpoint"])!=item["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed before scheduled evaluation")
            name=item["checkpoint_sha256"]+"-"+item["modality"]
            target=directory/"evaluations"/name
            target.mkdir(parents=True,exist_ok=True)
            expected=expected_measurements(completed["recipe"],item["step"],item["modality"])
            _,retained,_,missing=select_catalog(target,expected,source,identity["backend"])
            if not missing:
                # Reuse only after checking every item hash, source, backend,
                # batch and registered measurement. Old versions are retained
                # for final reconciliation; scores never select the version.
                print(json.dumps({"evaluation_reused":name,"measurements":len(retained),
                                  "source_id":digest(source)}),flush=True)
                continue
            attempt=1
            while (target/f"attempt-{attempt}.log").exists():
                attempt+=1
            cmd=[sys.executable,"-m","legolm_p01.cli","evaluate","--checkpoint",item["checkpoint"],"--asset-root",str(asset_root),
                 "--data-cache",str(data_cache),"--features",str(features),"--text",str(text),"--output",str(target),"--device",device,
                 "--eval-modality",item["modality"],"--linear-dir",str(linear_dir),"--suites",*item["suites"]]
            with (target/f"attempt-{attempt}.log").open("w") as log:
                code=launch(cmd,log,lambda pid:write_json(target/"process.json",{"pid":pid,"checkpoint":item["checkpoint"],"attempt":attempt}))
            if code:
                raise RuntimeError(f"Checkpoint evaluation failed ({code}); inspect {target}")
        write_json(Path(completed["directory"])/"evaluation-obligations.json",planned)
    def execute_body(job):
        recipe,key,config=job
        if training_source(source_identity())!=identity["training_source"]:
            raise ValueError("Training sources changed while the matrix was running")
        record_path=directory/f"job-{key}.json"
        if record_path.exists():
            record=json.loads(record_path.read_text())
            if record["status"] in TERMINAL:
                terminal_record(record)
                score(record)
                return record
        attempt=1
        while (logs/f"{key}-attempt-{attempt}.log").exists():
            attempt+=1
        log=logs/f"{key}-attempt-{attempt}.log"
        code="from legolm_p01.cli import main; main()"
        cmd=[sys.executable,"-c",code,"train","--config",str(config),"--asset-root",str(asset_root),"--data-cache",str(data_cache),
             "--features",str(features),"--text",str(text),"--output",str(directory/"training"),"--device",device]
        started=time.monotonic()
        with log.open("w") as f:
            returncode=launch(cmd,f,lambda pid:write_json(record_path,{"status":"running","pid":pid,"recipe":asdict(recipe),"log":str(log),"attempt":attempt}))
        if returncode:
            record={"status":"failed","returncode":returncode,"recipe":asdict(recipe),"log":str(log),"attempt":attempt}
            write_json(record_path,record)
            raise RuntimeError(f"Recipe failed ({returncode}); inspect {log}")
        lines=log.read_text().splitlines()
        completed=json.loads(lines[-1])
        if "checkpoints" not in completed:
            raise ValueError("Worker exited without a completed trajectory")
        if completed.get("status")=="numerical_failure":
            record={"status":"numerical_failure","recipe":asdict(recipe),"outcome":completed,"elapsed_wall_s":time.monotonic()-started,
                    "outcome_sha256":file_hash(Path(completed["directory"])/"numerical-failure.json"),"log":str(log),"attempt":attempt}
        else:
            record={"status":"complete","recipe":asdict(recipe),"completed":completed,"elapsed_wall_s":time.monotonic()-started,
                    "complete_sha256":file_hash(Path(completed["directory"])/"complete.json"),"log":str(log),"attempt":attempt}
        write_json(record_path,record)
        print(json.dumps({"trajectory_terminal":asdict(recipe),"status":record["status"],"elapsed_s":completed.get("elapsed_s"),"matrix_id":matrix_id}),flush=True)
        score(record)
        return record
    def execute(job):
        if stopped.is_set():
            raise CancelledError("Scheduling stopped")
        try:
            return execute_body(job)
        except BaseException:
            # Fence the next worker before publishing the failed future.
            stopped.set()
            raise
    finished=[]
    failures=[]
    # A worker failure cancels queued work; the already-running bounded workers
    # finish before the caller's production-restore trap can run.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(execute,j):j for j in jobs}
        for future in as_completed(pending):
            if future.cancelled():
                continue
            try:
                finished.append(future.result())
            except CancelledError:
                pass
            except Exception as exc:
                failures.append(str(exc))
                for queued in pending:
                    queued.cancel()
            write_json(directory/"progress.json",{"completed":len(finished),"expected":len(jobs),"failures":failures,
                                                  "numerical_failures":sum(r["status"]=="numerical_failure" for r in finished)})
    for sig,handler in old_handlers.items():
        signal.signal(sig,handler)
    if stopped.is_set() and not failures:
        failures.append("Run interrupted before all obligations completed")
    if failures:
        raise RuntimeError("Matrix incomplete: "+"; ".join(failures))
    result={"matrix_id":matrix_id,"directory":str(directory),"completed":len(finished),"expected":len(jobs),"passed":True,
            "successful_training":sum(r["status"]=="complete" for r in finished),
            "numerical_failures":sum(r["status"]=="numerical_failure" for r in finished),
            "completion_meaning":"Every registered recipe has a terminal outcome and every retained checkpoint was evaluated as requested.",
            "training_source":identity["training_source"],"backend":identity["backend"],"evaluated":evaluate,"excluded_from_results":parallel_canary}
    write_json(directory/"matrix-complete.json",result)
    write_json(output/("parallel-canary.json" if parallel_canary else "active-matrix.json"),result)
    return result
