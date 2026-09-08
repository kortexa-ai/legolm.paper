"""Inspect the original assets without fitting or changing an encoder."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .legacy import audio_snapshot as audio,imu_snapshot as imu
from .provenance import file_hash,write_json,source_identity


def run(asset_root,data_cache,features,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    datasets={m:torch.load(features/f"{m}.pt",map_location="cpu",weights_only=False) for m in ("imu","audio","vision")}
    result={"source":source_identity(),"features":{m:file_hash(features/f"{m}.pt") for m in datasets},"modalities":{}}
    for modality,data in datasets.items():
        overlap=sorted(set(data["train"]["groups"]) & set(data["test"]["groups"]))
        result["modalities"][modality]={"train_items":len(data["train"]["ids"]),"test_items":len(data["test"]["ids"]),
            "train_groups":len(set(data["train"]["groups"])),"test_groups":len(set(data["test"]["groups"])),
            "overlapping_groups":overlap,"encoder_parameters":data["encoder_parameters"],"encoder_sha256":data["encoder_sha256"]}
    path=asset_root/"checkpoints/experiments/esc50-512d.pt"
    ckpt=torch.load(path,map_location="cpu",weights_only=False)
    audio.CACHE_DIR=data_cache/"audio-encoder"
    samples,audio_dir=audio.load_esc50()
    raw=[]
    for sample in samples:
        waveform=audio.load_audio_wav(audio_dir/sample["filename"])
        if waveform is None:
            raise ValueError("The encoder audit cannot silently exclude clips")
        raw.append(torch.tensor(audio.compute_mel_spectrogram(waveform,n_mels=ckpt["n_mels"]),dtype=torch.float32))
    raw=torch.stack(raw)
    train=torch.tensor([s["fold"]!=5 for s in samples])
    def stats(tensor):
        return {"mean":float(tensor.mean()),"std":float(tensor.std()+1e-8)}
    all_stats,train_stats=stats(raw),stats(raw[train])
    saved={k:float(v) for k,v in ckpt["normalization"].items()}
    result["modalities"]["audio"]["normalizer"]={"checkpoint":saved,"all_2000_clips":all_stats,"training_1600_clips":train_stats,
        "absolute_error_all":{k:abs(saved[k]-all_stats[k]) for k in saved},"absolute_error_train":{k:abs(saved[k]-train_stats[k]) for k in saved},
        "scope_note":"A matching all-clip normalizer includes held-out inputs; this is separate from bridge fitting and encoder gradient updates."}
    result["modalities"]["audio"]["checkpoint_reported_best_accuracy"]=ckpt.get("best_acc")
    result["modalities"]["audio"]["checkpoint_epoch"]=ckpt.get("epoch")
    result["modalities"]["audio"]["run_provenance_complete"]=False
    del raw,ckpt
    path=asset_root/"checkpoints/encoders/imu.pt"
    ckpt=torch.load(path,map_location="cpu",weights_only=False)
    imu.CACHE_DIR=data_cache/"sensor-fusion"
    raw=imu.load_signals(imu.download_uci_har(),"train")
    mean=raw.mean(dim=(0,2),keepdim=True)
    std=raw.std(dim=(0,2),keepdim=True)+1e-8
    result["modalities"]["imu"]["normalizer"]={"max_abs_error_from_training_mean":float((mean-ckpt["normalization"]["mean"]).abs().max()),
        "max_abs_error_from_training_std":float((std-ckpt["normalization"]["std"]).abs().max())}
    result["modalities"]["imu"]["run_provenance_complete"]=False
    del raw,ckpt
    from torchvision.datasets import CIFAR10
    training=CIFAR10(data_cache/"cifar10",train=True,download=False)
    testing=CIFAR10(data_cache/"cifar10",train=False,download=False)
    excluded=set(datasets["vision"]["dev_indices"])
    hashes={}
    for i,image in enumerate(training.data):
        hashes.setdefault(hashlib.sha256(image.tobytes()).hexdigest(),[]).append(i)
    overlaps=[]
    for i,image in enumerate(testing.data):
        matched=hashes.get(hashlib.sha256(image.tobytes()).hexdigest(),[])
        fitted=[j for j in matched if j not in excluded]
        if matched:
            overlaps.append({"test_id":datasets["vision"]["test"]["ids"][i],"training_indices":matched,"bridge_fitting_indices":fitted})
    result["modalities"]["vision"].update({"exact_train_test_image_matches":overlaps,
        "secondary_strict_test_excluded_ids":[r["test_id"] for r in overlaps if r["bridge_fitting_indices"]],
        "bridge_train_items":len(training)-len(excluded),"bridge_dev_items":len(excluded),
        "overlap_scope":"Exact RGB array equality only; near-duplicates and upstream ImageNet/host pretraining overlap are not excluded.",
        "known_encoder_sources":["ImageNet MobileNetV3-Small","Big Buck Bunny perceiver fitting"],"upstream_exposure_exhaustive":False})
    write_json(output/"encoder-audit.json",result)
    print({m:{k:v for k,v in r.items() if k in ("overlapping_groups","normalizer","secondary_strict_test_excluded_ids")} for m,r in result["modalities"].items()},flush=True)
    return result
