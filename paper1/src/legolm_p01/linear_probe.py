"""Train-only multinomial linear feature baseline; no language model required."""
from pathlib import Path
import json
import time

import torch
from torch import nn

from .train import train_indices,save_torch
from .evaluate import accuracy_summary,save_evaluation
from .provenance import file_hash,write_json,source_identity


def run(features_dir,asset_root,output,device="cpu",data_cache=None):
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    results={}
    for modality in ("imu","audio","vision"):
        path=features_dir/f"{modality}.pt"
        data=torch.load(path,map_location="cpu",weights_only=False)
        selected=train_indices(data)
        x=data["train"]["features"][selected].to(device)
        y=torch.tensor([data["train"]["labels"][i] for i in selected],device=device)
        mean=x.mean(0);std=x.std(0,unbiased=False).clamp_min(1e-6)
        x=(x-mean)/std
        classifier=nn.Linear(x.shape[1],len(data["categories"])).to(device)
        nn.init.zeros_(classifier.weight);nn.init.zeros_(classifier.bias)
        optimizer=torch.optim.LBFGS(classifier.parameters(),lr=1.,max_iter=100,history_size=10,line_search_fn="strong_wolfe")
        evaluations=0
        def closure():
            nonlocal evaluations
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(classifier(x),y)+.0001*classifier.weight.square().sum()/2
            loss.backward();evaluations+=1
            return loss
        started=time.monotonic()
        optimizer.step(closure)
        with torch.no_grad():
            train_loss=float(nn.functional.cross_entropy(classifier(x),y))
            scores=classifier((data["test"]["features"].to(device)-mean)/std).cpu().tolist()
        state={"weights":{k:v.detach().cpu() for k,v in classifier.state_dict().items()},"mean":mean.cpu(),"std":std.cpu(),
               "categories":data["categories"],"feature_sha256":file_hash(path)}
        checkpoint=output/f"linear-{modality}.pt"
        checksum=save_torch(checkpoint,state)
        identity={"measurement":"linear_feature_classifier","modality":modality,"features_sha256":file_hash(path),
                  "checkpoint_sha256":checksum,"source":source_identity(),"optimizer":"LBFGS strong Wolfe, max_iter=100, history=10",
                  "l2_penalty":.0001,"training_ids":[data["train"]["ids"][i] for i in selected],
                  "standardization":"training mean and population std; std floor 1e-6","initialization":"zero"}
        items=[{"id":ident,"label":label,"group":group,"scores":s} for ident,label,group,s in zip(data["test"]["ids"],data["test"]["labels"],data["test"]["groups"],scores)]
        summary={**accuracy_summary(scores,data["test"]["labels"]),"train_cross_entropy":train_loss,
                 "training_items":len(selected),"closure_evaluations":evaluations,"elapsed_s":time.monotonic()-started}
        row=save_evaluation(output,identity,{"summary":summary,"items":items})
        row["predicted_labels"]=torch.tensor(scores).argmax(-1).tolist()
        row["checkpoint"]={"path":checkpoint.name,"sha256":checksum}
        # The published encoder classifier is an additional, separately named
        # reference. Its class ordering is asserted against checkpoint metadata.
        if modality in ("imu","audio"):
            p=asset_root/("checkpoints/encoders/imu.pt" if modality=="imu" else "checkpoints/experiments/esc50-512d.pt")
            ckpt=torch.load(p,map_location="cpu",weights_only=False)
            if modality=="audio":
                from .legacy import audio_snapshot as audio
                if data_cache is None:
                    raise ValueError("Audio head scoring requires the explicit ESC-50 metadata cache")
                audio.CACHE_DIR=Path(data_cache)/"audio-encoder"
                samples,audio_dir=audio.load_esc50()
                taxonomy={}
                for sample in samples:
                    previous=taxonomy.setdefault(sample["category"],sample["target"])
                    if previous!=sample["target"]:
                        raise ValueError("ESC-50 category/target mapping is inconsistent")
                order=[taxonomy[c] for c in data["categories"]]
                taxonomy_source={"metadata_sha256":file_hash(audio_dir.parent/"meta/esc50.csv"),"numeric_targets_in_scoring_order":order}
            else:
                from .legacy.imu_snapshot import ACTIVITIES
                if [c.lower() for c in ACTIVITIES]!=data["categories"]:
                    raise ValueError("IMU encoder target order mismatch")
                order=list(range(len(ACTIVITIES)))
                taxonomy_source={"activities":ACTIVITIES,"numeric_targets_in_scoring_order":order}
            model=ckpt["model"]
            old_scores=nn.functional.linear(data["test"]["features"],model["classifier.weight"],model["classifier.bias"])[:,order].tolist()
            old_items=[{**item,"scores":s} for item,s in zip(items,old_scores)]
            row["original_encoder_head"]=save_evaluation(output,{"measurement":"original_encoder_head","modality":modality,"encoder_sha256":file_hash(p),"features_sha256":file_hash(path),"taxonomy":taxonomy_source},
                {"summary":accuracy_summary(old_scores,data["test"]["labels"]),"items":old_items})
        results[modality]=row
        write_json(output/"linear-probes.json",results)
        print(json.dumps({"linear_probe":modality,"summary":summary}),flush=True)
    return results
