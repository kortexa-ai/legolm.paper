"""Batch-one component timings at fixed lengths, separate from task accuracy."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from .assets import Vision, _checkpoint
from .backend import identity as backend_identity
from .canary import memory
from .evaluate import configure
from .hosts import load_host
from .legacy import audio_snapshot as audio, imu_snapshot as imu
from .provenance import file_hash, source_identity, write_json
from .train import Recipe, make_adapter, load_adapter_state


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elif str(device).startswith("mps"):
        torch.mps.synchronize()


def timed(function, device, *, repeats=30, warmup=5):
    for _ in range(warmup):
        function()
    synchronize(device)
    values=[]
    for _ in range(repeats):
        start=time.perf_counter()
        function()
        synchronize(device)
        values.append((time.perf_counter()-start)*1000)
    return {"median_ms":float(np.median(values)),"p10_ms":float(np.quantile(values,.1)),
            "p90_ms":float(np.quantile(values,.9)),"samples_ms":values,"warmup":warmup,"repeats":repeats}


def external_encoder(modality,asset_root,data_cache,device,dataset):
    """Raw inputs are already in RAM; preprocessing and transfer remain timed."""
    if modality=="imu":
        ckpt=_checkpoint(asset_root/"checkpoints/encoders/imu.pt")
        encoder=imu.IMUConv1DEncoder(ckpt["n_channels"],ckpt["n_classes"],ckpt["feature_dim"])
        encoder.load_state_dict(ckpt["model"],strict=True)
        imu.CACHE_DIR=data_cache/"sensor-fusion"
        raw=imu.load_signals(imu.download_uci_har(),"test")[:1]
        def prepare():
            return ((raw-ckpt["normalization"]["mean"])/(ckpt["normalization"]["std"]+1e-8)).to(device)
    elif modality=="audio":
        ckpt=_checkpoint(asset_root/"checkpoints/experiments/esc50-512d.pt")
        encoder=audio.AudioConv1DEncoder(ckpt["n_mels"],ckpt["n_classes"],ckpt["feature_dim"])
        encoder.load_state_dict(ckpt["model"],strict=True)
        audio.CACHE_DIR=data_cache/"audio-encoder"
        _,directory=audio.load_esc50()
        raw=audio.load_audio_wav(directory/dataset["test"]["ids"][0])
        if raw is None:
            raise ValueError("Missing registered raw audio input")
        def prepare():
            mel=audio.compute_mel_spectrogram(raw,n_mels=ckpt["n_mels"])
            return torch.tensor((mel-ckpt["normalization"]["mean"])/ckpt["normalization"]["std"],dtype=torch.float32,device=device)[None]
    else:
        import cv2
        from torchvision.datasets import CIFAR10
        encoder=Vision(_checkpoint(asset_root/"checkpoints/experiments/vision-perceiver.pt"),device)
        raw=CIFAR10(data_cache/"cifar10",train=False,download=False).data[0][...,::-1]
        mean=torch.tensor([.485,.456,.406]).view(3,1,1)
        std=torch.tensor([.229,.224,.225]).view(3,1,1)
        def prepare():
            resized=cv2.resize(raw,(224,224))
            return ((torch.from_numpy(resized.copy()).permute(2,0,1).float()/255-mean)/std)[None].to(device)
    encoder.to(device).eval().requires_grad_(False)
    return encoder,prepare


@torch.no_grad()
def fixed_decode(host, ids, prefix, output_tokens=32, measure=False):
    """Exactly 32 greedy tokens, with EOS stopping disabled for timing."""
    if measure:
        synchronize(host.device)
        start=time.perf_counter()
    if host.name=="mini":
        row=ids
        for position in range(output_tokens):
            token=host.logits(row,prefix)[:,-1].argmax(-1,keepdim=True)
            row=torch.cat((row,token),dim=1)
            if measure and position==0:
                synchronize(host.device);first=time.perf_counter()
        if measure:
            synchronize(host.device);end=time.perf_counter()
            return {"first_token_ms":(first-start)*1000,"remaining_tokens_ms":(end-first)*1000,"total_ms":(end-start)*1000}
        return row[:,ids.shape[1]:]
    initial={"input_ids":ids} if prefix is None else {"inputs_embeds":torch.cat((prefix,host.model.get_input_embeddings()(ids)),dim=1)}
    width=ids.shape[1]+(0 if prefix is None else prefix.shape[1])
    mask=torch.ones((1,width),dtype=torch.long,device=host.device)
    out=host.model(**initial,attention_mask=mask,use_cache=True)
    token=out.logits[:,-1].argmax(-1,keepdim=True)
    result=[token]
    cache=out.past_key_values
    if measure:
        synchronize(host.device);first=time.perf_counter()
    for _ in range(output_tokens-1):
        mask=torch.cat((mask,torch.ones_like(token)),dim=1)
        out=host.model(input_ids=token,attention_mask=mask,past_key_values=cache,use_cache=True)
        cache=out.past_key_values
        token=out.logits[:,-1].argmax(-1,keepdim=True)
        result.append(token)
    if measure:
        synchronize(host.device);end=time.perf_counter()
        return {"first_token_ms":(first-start)*1000,"remaining_tokens_ms":(end-first)*1000,"total_ms":(end-start)*1000}
    return torch.cat(result,dim=1)


def ensure_isolated(device):
    if not str(device).startswith("cuda"):
        return
    uuid=os.environ.get("CUDA_VISIBLE_DEVICES","")
    if not uuid.startswith("GPU-") or "," in uuid:
        raise ValueError("CUDA latency measurement requires one UUID-pinned device")
    result=subprocess.run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True)
    others=[line for line in result.stdout.splitlines() if line.split(",")[0].strip()==uuid and int(line.split(",")[1].strip())!=os.getpid()]
    if others:
        raise RuntimeError("Latency measurement refused: another CUDA process uses this GPU")


def operating_state(device):
    """Observe operating conditions outside the timed sections; change nothing."""
    if not str(device).startswith("cuda"):
        return None
    uuid=os.environ["CUDA_VISIBLE_DEVICES"]
    fields="uuid,power.limit,power.default_limit,temperature.gpu,clocks.current.sm,clocks.current.memory,fan.speed,pstate"
    result=subprocess.run(["nvidia-smi","--id="+uuid,"--query-gpu="+fields,
                           "--format=csv,noheader,nounits"],capture_output=True,text=True,check=True)
    row=[value.strip() for value in result.stdout.strip().split(",")]
    if len(row)!=8 or row[0]!=uuid:
        raise ValueError("GPU operating-state query did not identify the allocated device")
    return {"gpu_uuid":row[0],"power_limit_w":float(row[1]),"default_power_limit_w":float(row[2]),
            "temperature_c":float(row[3]),"sm_clock_mhz":float(row[4]),"memory_clock_mhz":float(row[5]),
            "fan_percent":row[6],"performance_state":row[7]}


@torch.no_grad()
def run(checkpoint,asset_root,data_cache,features,text,output,device,modality=None):
    ensure_isolated(device)
    operating_before=operating_state(device)
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    recipe=Recipe(**state["identity"]["recipe"])
    modality=modality or recipe.modality
    if modality!=recipe.modality and recipe.kind not in ("none","static"):
        raise ValueError("Only input-independent adapters can share modality resource measurements")
    if recipe.cohort!="U" or recipe.seed!=42 or recipe.feature_mode!="true" or recipe.diversity!=0:
        raise ValueError("Resource protocol uses seed-42 unpaired reference interfaces")
    if state["step"]!=(0 if recipe.kind=="none" else 3000):
        raise ValueError("Resource measurement requires the registered final checkpoint")
    dataset=torch.load(features/f"{modality}.pt",map_location="cpu",weights_only=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    encoder,prepare=external_encoder(modality,asset_root,data_cache,device,dataset)
    observed=encoder.features(prepare()).float().cpu()
    torch.testing.assert_close(observed,dataset["test"]["features"][:1],atol=2e-5,rtol=2e-5)
    preprocessing=timed(prepare,device)
    preprocessed=prepare()
    encoding=timed(lambda:encoder.features(preprocessed),device)
    sensor_path=timed(lambda:encoder.features(prepare()),device)
    encoder_parameters=sum(p.numel() for p in encoder.parameters())
    del preprocessed
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    host=load_host(recipe.host,asset_root,data_cache,device)
    adapter,parameters=make_adapter(host,recipe,dataset["test"]["features"].shape[1])
    load_adapter_state(host,adapter,recipe.kind,state["adapter"])
    if host.identity!=state["identity"]["host"]:
        raise ValueError("Resource checkpoint host mismatch")
    host.model.eval()
    if adapter is not None:
        adapter.eval()
    synchronize(device)
    loading_s=time.perf_counter()-started
    manifest=json.loads((text/f"text-{recipe.host}.json").read_text())
    rows=torch.load(text/manifest["file"],map_location="cpu",weights_only=False)["val"]
    ids=rows[:1,:128].to(device)
    feature=dataset["test"]["features"][:1].to(device)
    adapter_parameters=sum(p.numel() for p in parameters)
    del parameters,state
    if adapter is not None:
        adapter_time=timed(lambda:adapter(feature),device)
    else:
        adapter_time=None
    def fresh():
        prefix=configure(host,adapter,recipe.kind,feature)
        return host.logits(ids,prefix)[:,-1].argmax(-1)
    fresh_prefill=timed(fresh,device)
    def end_to_end():
        # Sensor-free references do not pay for an unused encoder invocation.
        value=encoder.features(prepare()) if recipe.kind in ("bridge","prefix") else feature
        prefix=configure(host,adapter,recipe.kind,value)
        return host.logits(ids,prefix)[:,-1].argmax(-1)
    full_sensor_prefill=timed(end_to_end,device)
    prefix=configure(host,adapter,recipe.kind,feature)
    cached_tokens=fixed_decode(host,ids[:,:16],prefix,4)
    reference=ids[:,:16]
    for _ in range(4):
        token=host.logits(reference,prefix)[:,-1].argmax(-1,keepdim=True)
        reference=torch.cat((reference,token),dim=1)
    torch.testing.assert_close(cached_tokens,reference[:,16:],atol=0,rtol=0)
    cached_prefill=timed(lambda:host.logits(ids,prefix)[:,-1].argmax(-1),device)
    full_decode=timed(lambda:fixed_decode(host,ids,prefix),device,repeats=10,warmup=2)
    decode_parts=[fixed_decode(host,ids,prefix,measure=True) for _ in range(10)]
    ensure_isolated(device)
    operating_after=operating_state(device)
    if operating_before is not None and operating_before["power_limit_w"]!=operating_after["power_limit_w"]:
        raise RuntimeError("GPU power limit changed during the resource measurement")
    result={"measurement":"resources","host":recipe.host,"modality":modality,"kind":recipe.kind,
            "checkpoint_sha256":file_hash(checkpoint),"checkpoint_bytes":Path(checkpoint).stat().st_size,
            "backend":backend_identity(device),"source":source_identity(),"host_identity":host.identity,
            "operating_conditions":{"before":operating_before,"after":operating_after},
            "feature_sha256":file_hash(features/f"{modality}.pt"),"sample_id":dataset["test"]["ids"][0],
            "text_sha256":manifest["sha256"],"text_row":0,"text_tokens":128,"output_tokens":32,
            "eos_stopping":False,"batch_size":1,"prefix_tokens":8 if recipe.kind=="prefix" else 0,
            "encoder_parameters":encoder_parameters,"adapter_parameters":adapter_parameters,
            "generated_factor_dimension":host.dimension if recipe.kind=="bridge" else 0,
            "adapter_tensor_bytes_fp32":adapter_parameters*4,"checkpoint_loading_s":loading_s,
            "preprocessing":preprocessing,"encoder_only":encoding,"sensor_path":sensor_path,"adapter_generation":adapter_time,
            "raw_sensor_to_first_token":full_sensor_prefill,"cached_decode_matches_four_token_reference":True,
            "fresh_sensor_prefill":fresh_prefill,"cached_sensor_prefill":cached_prefill,"cached_sensor_generate_32":full_decode,
            "generate_32_tokens_per_second":32/(full_decode["median_ms"]/1000),"memory":memory(device),
            "decode_breakdown_ms":decode_parts,
            "decode_after_first_token_per_second":31/(float(np.median([r["remaining_tokens_ms"] for r in decode_parts]))/1000),
            "scope":{"raw_input":"Already decoded in RAM; excludes file/network acquisition.",
                     "fresh_sensor_prefill":"Feature already on GPU; regenerates adapter/prefix before 128-token prefill.",
                     "raw_sensor_to_first_token":"Preprocessing, transfer, encoder, adapter and prefill; sensor-free references omit unused sensor computation.",
                     "memory":"Language host, adapter and external encoder simultaneously resident; no optimizer state.",
                     "cached_sensor":"Reuses generated factors/prefix; each request starts a new text cache.",
                     "decoding":"Native model cache for LFM/Qwen; mini recomputes its prefix because the released mini has no KV-cache interface.",
                     "timing_spread":"Repeat dispersion on one isolated workstation; not uncertainty across hardware or inputs.",
                     "cold_weights":"Checkpoint-loading time is observed with existing local files; no claim of an empty OS file cache."}}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    write_json(output/f"{recipe.host}-{modality}-{recipe.kind}.json",result)
    return {k:result[k] for k in ("host","modality","kind","adapter_parameters","generate_32_tokens_per_second")}
