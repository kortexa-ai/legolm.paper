"""Read-only inspection of the first fixed native-chat generation batch.

These observations are a separate diagnostic, never replacement task scores.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import inspect

import torch

from .artifacts import load
from .backend import identity as backend_identity
from .generation import evaluation_cases, panel_indices
from .generation_cached import chat_stop_ids, decode
from .hosts import load_host
from .provenance import digest, file_hash, write_json
from .train import Recipe, load_adapter_state, make_adapter


def tensor_stats(value):
    value = value.detach()
    finite = torch.isfinite(value)
    clean = value[finite]
    return {"shape": list(value.shape), "elements": value.numel(),
            "nonfinite": int((~finite).sum()), "zeros": int((value == 0).sum()),
            "max_abs": float(clean.abs().max()) if len(clean) else None}


def scalar(value):
    return float(value) if torch.isfinite(value) else str(float(value))


def observe(host, adapter, features, prompts):
    """Observe unchanged decoding; remove every installed hook on failure too."""
    result = {"steps": [], "normalizations": []}

    def check_logits(module, args, output):
        values = []
        for value in output.logits[:, -1].detach():
            finite = torch.isfinite(value)
            clean = value[finite]
            token = int(value.argmax())
            values.append({"vocab": value.numel(), "nonfinite": int((~finite).sum()),
                "zeros": int((value == 0).sum()),
                "finite_min": float(clean.min()) if len(clean) else None,
                "finite_max": float(clean.max()) if len(clean) else None,
                "greedy_token_id": token, "greedy_token": host.tokens.decode([token])})
        result["steps"].append({"step": len(result["steps"]), "rows": values})

    def check_norm(name, module, args, output):
        if result["steps"]:
            return
        value = args[0].detach()
        variance = value.float().pow(2).mean(-1)
        detail = {"name": name, "class": type(module).__name__,
                  "input": tensor_stats(value), "output": tensor_stats(output),
                  "fp32_variance_nonfinite": int((~torch.isfinite(variance)).sum()),
                  "variance_count": variance.numel(),
                  "max_fp64_variance": scalar(value.double().pow(2).mean(-1).max())}
        if name == "model.norm":
            detail.update(last_token_input=tensor_stats(value[:, -1]),
                          last_token_output=tensor_stats(output[:, -1]),
                          last_token_fp32_variance=[scalar(v) for v in variance[:, -1]])
        result["normalizations"].append(detail)

    hooks = []
    try:
        for name, module in host.model.named_modules():
            if "RMSNorm" in type(module).__name__:
                hooks.append(module.register_forward_hook(
                    lambda module, args, output, name=name: check_norm(name, module, args, output)))
        hooks.append(host.model.register_forward_hook(check_logits))
        result["responses"] = decode(host, adapter, "bridge", features, prompts,
                                     native_chat=True, intervention="correct")
    finally:
        for handle in hooks:
            handle.remove()
    return result


def run(checkpoint, reference, asset_root, data_cache, features_dir, output, device):
    """Inspect one downloaded G checkpoint against its published measurement."""
    checkpoint, reference, output = Path(checkpoint), Path(reference), Path(output)
    if output.exists():
        raise FileExistsError("Use a new diagnostic output path")
    saved = load(reference)
    if digest(saved["identity"]) != saved["evaluation_id"]:
        raise ValueError("Reference evaluation identity is invalid")
    checksum = file_hash(checkpoint)
    if saved["identity"]["checkpoint_sha256"] != checksum:
        raise ValueError("Reference evaluation belongs to another checkpoint")
    from .checkpoint_eval import evaluator_source
    if saved["identity"]["evaluation_source"]["evaluation_files"] != evaluator_source()["evaluation_files"]:
        raise ValueError("The diagnostic and reference use different numerical evaluator files")
    torch.set_num_threads(saved["identity"]["backend"]["intraop_threads"])
    torch.set_num_interop_threads(saved["identity"]["backend"]["interop_threads"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    recipe = Recipe(**state["identity"]["recipe"])
    if (recipe.host != "qwen08" or recipe.cohort != "G" or recipe.kind != "bridge"
            or recipe.feature_mode != "true" or recipe.modality not in ("imu", "audio")):
        raise ValueError("This diagnostic requires a true-feature Qwen G checkpoint")
    feature_path = Path(features_dir) / (recipe.modality + ".pt")
    if (file_hash(feature_path) != state["identity"]["features_sha256"]
            or saved["identity"]["evaluation_features_sha256"] != state["identity"]["features_sha256"]):
        raise ValueError("Features differ from the checkpoint's frozen inputs")
    dataset = torch.load(feature_path, map_location="cpu", weights_only=False)
    cases = evaluation_cases(dataset, panel_indices(dataset))[:8]
    if len(cases) != 8 or len({c["sensor_index"] for c in cases}) != 1:
        raise ValueError("The fixed first batch must contain eight cases for one sensor")
    if (not saved["decoding"]["native_chat"] or not saved["decoding"]["greedy"]
            or saved["decoding"]["max_new_tokens"] != 64
            or saved["execution"]["batch_size"] != 8
            or saved["identity"]["measurement"] != "generation"
            or saved["identity"]["objective"] != "G"
            or saved["identity"]["intervention"] != "correct"):
        raise ValueError("Reference is not the registered correct-feature G decoder")
    for case, original in zip(cases, saved["items"][:8]):
        if (original["id"] != dataset["test"]["ids"][case["sensor_index"]]
                or original["actual_prompt"] != case["prompt"]):
            raise ValueError("Reference cases differ from the fixed first batch")
    if len(saved["items"]) < 8:
        raise ValueError("Reference first batch is incomplete")
    host = load_host(recipe.host, Path(asset_root), Path(data_cache), device)
    host.model.eval()
    if host.identity != state["identity"]["host"] or chat_stop_ids(host) != saved["decoding"]["eos_token_ids"]:
        raise ValueError("Host or native stopping tokens differ from the checkpoint/reference")
    adapter, _ = make_adapter(host, recipe, dataset["train"]["features"].shape[1])
    load_adapter_state(host, adapter, recipe.kind, state["adapter"])
    adapter.eval()
    del state
    observations = observe(host, adapter,
        dataset["test"]["features"][[c["sensor_index"] for c in cases]],
        [c["prompt"] for c in cases])
    original_responses = [r["response"] for r in saved["items"][:8]]
    report = {"schema": "paper1-generation-diagnostic-v2", "excluded_from_results": True,
              "scope": "First eight registered G cases, one sensor, unchanged 64-token greedy decoding. "
                       "Normalization observations cover the first forward pass only; no full-panel incidence or timing claim.",
              "observed_at_utc": datetime.now(timezone.utc).isoformat(),
              "checkpoint_sha256": checksum, "reference_sha256": file_hash(reference),
              "reference_evaluation_id": saved["evaluation_id"],
              "feature_sha256": file_hash(feature_path), "backend": backend_identity(device),
              "reference_backend": saved["identity"]["backend"],
              "backend_matches_reference": backend_identity(device) == saved["identity"]["backend"],
              "batch": [{"id": dataset["test"]["ids"][c["sensor_index"]], **c} for c in cases],
              "source_sha256": file_hash(Path(__file__)), "exact_source": Path(__file__).read_text(),
              "normalization_source": {name: inspect.getsource(type(module))
                  for name, module in host.model.named_modules()
                  if name in ("model.norm", "model.layers.0.input_layernorm", "model.layers.0.linear_attn.norm")},
              "saved_responses": original_responses,
              "matches_saved_responses": observations["responses"] == original_responses, **observations}
    write_json(output, report)
    return {"output": str(output), "sha256": file_hash(output),
            "backend_matches_reference": report["backend_matches_reference"],
            "matches_saved_responses": report["matches_saved_responses"], "excluded_from_results": True}
