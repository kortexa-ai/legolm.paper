from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Paper 1 reproduction; isolated from other papers")
    sub = parser.add_subparsers(dest="command", required=True)
    p=sub.add_parser("bootstrap")
    p.add_argument("--asset-root",type=Path,required=True)
    p.add_argument("--data-cache",type=Path,required=True)
    p=sub.add_parser("publish-checkpoints",help="Materialize an immutable checkpoint archive on the same filesystem")
    for name in ("evidence","matrix","features","objects","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--base-url",required=True)
    p=sub.add_parser("fetch-features",help="Download the exact three frozen feature caches")
    for name in ("index","evidence","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p=sub.add_parser("fetch-checkpoint",help="Download one declared checkpoint with SHA-256 verification")
    p.add_argument("--index",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--host",choices=["mini","lfm230m","qwen08"],required=True)
    p.add_argument("--cohort",choices=["U","P","G"],required=True)
    p.add_argument("--modality",choices=["imu","audio","vision"],required=True)
    p.add_argument("--kind",choices=["none","static","bridge","prefix"],required=True)
    p.add_argument("--feature-mode",choices=["true","constant","shuffled","random"],default="true")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--step",type=int,required=True)
    p.add_argument("--diversity",type=float,default=0.)
    p=sub.add_parser("export-artifacts")
    p.add_argument("--matrix",type=Path,required=True)
    p.add_argument("--run-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--incomplete",action="store_true",help="Progress audit only; cannot render a final release")
    p=sub.add_parser("verify-artifacts")
    p.add_argument("directory",type=Path)
    p=sub.add_parser("diagnose-generation",help="Inspect one fixed G generation batch without changing task scores")
    for name in ("checkpoint","reference","asset-root","data-cache","features","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",default="cpu")
    p=sub.add_parser("aggregate")
    p.add_argument("--evidence",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--incomplete",action="store_true")
    p=sub.add_parser("render")
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--evidence",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--date",required=True,help="ISO release date, preserved in the source archive")
    p.add_argument("--draft",action="store_true",help="Explicitly mark incomplete evidence; never package it as a release")
    p.add_argument("--compile",action="store_true")
    p.add_argument("--package",type=Path)
    p=sub.add_parser("matrix")
    p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("prepare-common-text")
    p.add_argument("--data-cache",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    for name in ("prepare", "prepare-text", "canary"):
        p = sub.add_parser(name)
        p.add_argument("--asset-root", type=Path, required=True, help="Public repo asset root with checkpoints/ and data/")
        p.add_argument("--data-cache", type=Path, required=True, help="Explicit original dataset/tokenizer cache")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--device", default="cpu")
        p.add_argument("--host", choices=["mini", "lfm230m", "qwen08"], default="mini")
        if name == "prepare":
            p.add_argument("--modalities", nargs="+", default=["imu", "audio", "vision"])
    for name in ("train","evaluate","accept","schedule","linear-probes","compose","encoder-audit","resources","supplement","reconcile"):
        p=sub.add_parser(name)
        p.add_argument("--asset-root",type=Path,required=True)
        p.add_argument("--data-cache",type=Path,required=True)
        p.add_argument("--features",type=Path,required=True)
        p.add_argument("--text",type=Path,required=True)
        p.add_argument("--output",type=Path,required=True)
        p.add_argument("--device",default="cpu")
        if name=="train":
            p.add_argument("--config",type=Path,required=True)
            p.add_argument("--stop-after",type=int)
        elif name=="evaluate":
            p.add_argument("--checkpoint",type=Path,required=True)
            p.add_argument("--suites",nargs="+",default=["primary","common","geometry"])
            p.add_argument("--sensor-batch",type=int,default=4)
            p.add_argument("--eval-modality",choices=["imu","audio","vision"])
            p.add_argument("--linear-dir",type=Path)
        elif name=="schedule":
            p.add_argument("--workers",type=int,default=1)
            p.add_argument("--parallel-canary",action="store_true")
            p.add_argument("--evaluate",action="store_true")
            p.add_argument("--linear-dir",type=Path)
        elif name=="compose":
            p.add_argument("--checkpoints",nargs=3,type=Path,required=True)
        elif name=="resources":
            p.add_argument("--checkpoint",type=Path,required=True)
            p.add_argument("--eval-modality",choices=["imu","audio","vision"])
        elif name=="supplement":
            p.add_argument("--matrix",type=Path,required=True)
        elif name=="reconcile":
            p.add_argument("--matrix",type=Path,required=True)
            p.add_argument("--linear-dir",type=Path,required=True)
    args = parser.parse_args()
    if hasattr(args,"device") and args.command not in ("schedule","supplement","reconcile"):
        from .gpu_limits import configure
        configure(args.device)
    if args.command=="bootstrap":
        from .bootstrap import run
        print(json.dumps(run(args.asset_root,args.data_cache)))
    elif args.command=="export-artifacts":
        from .artifacts import export
        print(json.dumps(export(args.matrix,args.run_root,args.output,incomplete=args.incomplete)))
    elif args.command=="publish-checkpoints":
        from .checkpoints import publish
        print(json.dumps(publish(args.evidence,args.matrix,args.features,args.objects,args.base_url,args.output)))
    elif args.command=="diagnose-generation":
        from .generation_diagnostic import run
        print(json.dumps(run(args.checkpoint,args.reference,args.asset_root,args.data_cache,args.features,args.output,args.device)))
    elif args.command=="fetch-features":
        from .checkpoints import fetch_features
        print(json.dumps(fetch_features(args.index,args.evidence,args.output)))
    elif args.command=="fetch-checkpoint":
        from .checkpoints import fetch
        print(json.dumps(fetch(args.index,args.output,host=args.host,cohort=args.cohort,modality=args.modality,
                               kind=args.kind,feature_mode=args.feature_mode,seed=args.seed,step=args.step,diversity=args.diversity)))
    elif args.command=="verify-artifacts":
        from .artifacts import verify
        print(json.dumps(verify(args.directory)))
    elif args.command=="aggregate":
        from .aggregate import run
        print(json.dumps(run(args.evidence,args.output,incomplete=args.incomplete)))
    elif args.command=="render":
        from .render import run
        print(json.dumps(run(args.summary,args.evidence,args.output,args.date,draft=args.draft,compile=args.compile,package=args.package)))
    elif args.command=="matrix":
        from .grid import manifest
        result=manifest(args.output)
        print(json.dumps({k:v for k,v in result.items() if k!="trajectories"}))
    elif args.command=="accept":
        from .accept import acceptance
        acceptance(args.asset_root,args.data_cache,args.features,args.text,args.output,args.device)
    elif args.command=="schedule":
        from .scheduler import run
        print(json.dumps(run(args.asset_root,args.data_cache,args.features,args.text,args.output,args.device,workers=args.workers,parallel_canary=args.parallel_canary,evaluate=args.evaluate,linear_dir=args.linear_dir)))
    elif args.command=="linear-probes":
        from .linear_probe import run
        run(args.features,args.asset_root,args.output,args.device,args.data_cache)
    elif args.command=="compose":
        from .composition import run
        run(args.checkpoints,args.asset_root,args.data_cache,args.features,args.text,args.output,args.device)
    elif args.command=="encoder-audit":
        from .encoder_audit import run
        run(args.asset_root,args.data_cache,args.features,args.output)
    elif args.command=="resources":
        from .resources import run
        print(json.dumps(run(args.checkpoint,args.asset_root,args.data_cache,args.features,args.text,args.output,args.device,args.eval_modality)))
    elif args.command=="supplement":
        from .supplement import run
        print(json.dumps(run(args.matrix,args.asset_root,args.data_cache,args.features,args.text,args.output,args.device)))
    elif args.command=="reconcile":
        from .reconcile import run
        print(json.dumps(run(args.matrix,args.asset_root,args.data_cache,args.features,args.text,args.output,args.device,args.linear_dir)))
    elif args.command=="prepare-common-text":
        from .text import prepare_common_text
        result=prepare_common_text(args.data_cache,args.output)
        print(json.dumps({"examples":len(result["examples"])}))
    elif args.command=="train":
        from .train import Recipe
        from .execution import run
        result=run(Recipe(**json.loads(args.config.read_text())),args.asset_root,args.data_cache,args.features,args.text,args.output,args.device,stop_after=args.stop_after)
        print(json.dumps(result))
    elif args.command=="evaluate":
        from .checkpoint_eval import run
        run(args.checkpoint,args.asset_root,args.data_cache,args.features,args.text,args.output,args.device,args.suites,args.sensor_batch,args.eval_modality,args.linear_dir)
    elif args.command == "prepare":
        from .assets import prepare
        prepare(args.asset_root, args.data_cache, args.output, args.device, args.modalities)
    elif args.command == "prepare-text":
        from .text import prepare_text
        print(prepare_text(args.host, args.data_cache, args.output))
    elif args.command == "canary":
        from .canary import run
        result = run(args.host, args.asset_root, args.data_cache, args.output, args.device)
        print(json.dumps({"host": args.host, "checks": result["checks"], "memory": result["final_memory"]}))


if __name__ == "__main__":
    main()
