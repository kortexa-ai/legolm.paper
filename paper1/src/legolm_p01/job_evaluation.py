"""Evaluation obligations for each saved training trajectory."""
from pathlib import Path

from .train import Recipe


def obligations(completed):
    r=Recipe(**completed["recipe"])
    endpoints=(300,3000) if r.cohort=="U" else (600,6000)
    result=[]
    for saved in completed["checkpoints"]:
        step=saved["step"]
        main=step in endpoints or r.kind=="none" or (r.kind=="prefix" and step==0)
        trajectory=r.cohort in ("U","P") and r.kind=="bridge" and r.modality=="imu" and r.feature_mode=="true" and step not in (150,1500)
        if not main and not trajectory:
            continue
        modalities=("imu","audio","vision") if r.cohort=="U" and r.kind in ("none","static") else (r.modality,)
        for modality in modalities:
            suites=["primary","common","geometry"] if main else ["trajectory","geometry"]
            if main and r.kind=="none":
                suites.append("paraphrases")
                if r.cohort=="U" and r.host=="qwen08":
                    suites.append("generation")
            if main and r.cohort=="P":
                suites.append("paraphrases")
                if r.kind=="bridge" and r.feature_mode=="true":
                    suites.append("interventions")
                if r.host=="qwen08" and r.diversity==0:
                    suites.append("generation")
            if main and r.cohort=="G":
                suites.append("generation")
            result.append({"checkpoint":str(Path(completed["directory"])/saved["path"]),"checkpoint_sha256":saved["sha256"],
                           "step":step,"modality":modality,"suites":suites})
    return result
