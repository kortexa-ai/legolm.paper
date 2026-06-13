from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    DEFAULT_MINI_CHECKPOINT,
    EVAL_TOKENS,
    MAX_SEQ_LEN,
    Tokenizer,
    apply_hypernet_weights,
    apply_lora,
    autocast_for,
    cycle_batch,
    evaluate_bpb,
    freeze_non_lora,
    get_device,
    get_token_bytes,
    get_lora_params,
    load_lm,
    make_dataloader,
    seed_everything,
    sensor_limit_for,
    total_lora_dim,
)
from .modalities import ModalityBundle, TaskDataset, load_modality_bundle, load_task_dataset
from .step_log import StepLogger


class PaperBridgeHyper(nn.Module):
    """Paper-facing bridge hypernetwork with explicit per-sample outputs."""

    def __init__(self, input_dim: int, output_dim: int, context_dim: int = 64):
        super().__init__()
        self.context = nn.Parameter(torch.randn(context_dim) * 0.02)
        self.proj = nn.Sequential(nn.Linear(input_dim, context_dim), nn.GELU())
        self.mlp = nn.Sequential(
            nn.Linear(context_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, output_dim),
        )
        nn.init.normal_(self.mlp[-1].weight, std=0.001)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features: torch.Tensor, pool_batch: bool = False) -> torch.Tensor:
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if features.dim() > 2:
            features = features.mean(dim=tuple(range(1, features.dim() - 1)))
        if pool_batch:
            features = features.mean(dim=0, keepdim=True)
        feat = self.proj(features)
        context = self.context.unsqueeze(0).expand(feat.size(0), -1)
        output = self.mlp(torch.cat([feat, context], dim=1))
        if pool_batch:
            return output[0]
        return output


class PrefixProjection(nn.Module):
    """Project sensor features into continuous prefix tokens."""

    def __init__(self, feature_dim: int, d_model: int, n_prefix: int):
        super().__init__()
        self.n_prefix = n_prefix
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, d_model * n_prefix),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if features.dim() > 2:
            features = features.mean(dim=tuple(range(1, features.dim() - 1)))
        projected = self.proj(features)
        return projected.view(features.size(0), self.n_prefix, self.d_model)


@dataclass
class BridgeArtifacts:
    bridge: PaperBridgeHyper
    bundle: ModalityBundle


def _throughput_metrics(total_time: float, steps: int, batch_size: int, seq_len: int = MAX_SEQ_LEN) -> dict:
    examples_seen = steps * batch_size
    tokens_seen = examples_seen * seq_len
    return {
        "train_elapsed_s": total_time,
        "steps_per_second": (steps / total_time) if total_time > 0 else 0.0,
        "examples_seen": examples_seen,
        "tokens_seen": tokens_seen,
    }


def _display_activity(label: str) -> str:
    return label.replace("_", " ")


def _suffix_log_path(log_path: str | None, suffix: str) -> str | None:
    if not log_path:
        return None
    path = Path(log_path)
    return str(path.with_name(f"{path.stem}-{suffix}{path.suffix}"))


def _make_global_shuffle_map(size: int, seed: int, device: torch.device) -> torch.Tensor | None:
    if size < 2:
        return None
    base = torch.arange(size, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(size, generator=generator)
    while bool(torch.any(perm == base)):
        perm = torch.randperm(size, generator=generator)
    return perm.to(device)


def _cycle_indices(size: int, batch_size: int, offset: int, device: torch.device) -> tuple[torch.Tensor, int]:
    if size <= 0:
        raise ValueError("Cannot cycle an empty tensor")
    indices = (torch.arange(batch_size, device=device) + offset) % size
    return indices, (offset + batch_size) % size


def _offdiag_cosine(vectors: torch.Tensor) -> torch.Tensor:
    if vectors.dim() == 1 or vectors.size(0) < 2:
        return vectors.new_tensor(0.0)
    normed = F.normalize(vectors, dim=1)
    sim = normed @ normed.T
    mask = ~torch.eye(vectors.size(0), dtype=torch.bool, device=vectors.device)
    return sim[mask].mean()


def _mean_pairwise_cosine(vectors: list[torch.Tensor]) -> float:
    if len(vectors) < 2:
        return 0.0
    stacked = torch.stack(vectors)
    return float(_offdiag_cosine(stacked).item())


def _forward_with_prefix(raw_model, prefix_tokens: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
    batch_size, text_len = text_ids.size()
    n_prefix = prefix_tokens.size(1)
    cos_sin = raw_model.cos[:, : text_len + n_prefix], raw_model.sin[:, : text_len + n_prefix]
    text_embeds = raw_model.transformer.wte(text_ids)
    combined = torch.cat([prefix_tokens, text_embeds], dim=1)
    hidden = combined
    for block in raw_model.transformer.h:
        hidden = block(hidden, cos_sin)
    hidden = raw_model.transformer.ln_f(hidden)
    logits = raw_model.lm_head(hidden).float()
    softcap = 30
    return softcap * torch.tanh(logits / softcap)


def _bridge_features_for_indices(
    bundle: ModalityBundle,
    indices: torch.Tensor,
    feature_mode: str,
    shuffle_map: torch.Tensor | None = None,
    constant_feature: torch.Tensor | None = None,
) -> torch.Tensor:
    if feature_mode == "true":
        return bundle.get_features(bundle.data[indices])
    if feature_mode == "shuffled":
        if shuffle_map is None:
            return bundle.get_features(bundle.data[indices])
        return bundle.get_features(bundle.data[shuffle_map[indices]])
    if feature_mode == "random":
        features = bundle.get_features(bundle.data[indices])
        return torch.randn_like(features)
    if feature_mode == "constant":
        if constant_feature is None:
            raise ValueError("constant feature mode requires a precomputed feature vector")
        return constant_feature.unsqueeze(0).expand(len(indices), -1)
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def _backprop_conditioned_batch_loss(
    model: torch.nn.Module,
    tx: torch.Tensor,
    ty: torch.Tensor,
    bridge: PaperBridgeHyper,
    features: torch.Tensor,
    rank: int,
    target: str,
    autocast_ctx,
) -> torch.Tensor:
    total_loss = torch.zeros((), device=tx.device)
    batch_size = tx.size(0)
    for idx, feature in enumerate(features):
        weight_vector = bridge(feature.unsqueeze(0))[0]
        apply_hypernet_weights(model, weight_vector, rank, target)
        with autocast_ctx:
            sample_loss = model(tx[idx : idx + 1], ty[idx : idx + 1]) / batch_size
        total_loss = total_loss + sample_loss.detach()
        sample_loss.backward()
    return total_loss


def _evaluate_bridge_bpb(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    bridge: PaperBridgeHyper,
    bundle: ModalityBundle,
    rank: int,
    target: str,
    feature_mode: str,
    eval_tokens: int | None,
    autocast_ctx,
    batch_size: int = 8,
    shuffle_map: torch.Tensor | None = None,
    constant_feature: torch.Tensor | None = None,
) -> float:
    was_training = bridge.training
    bridge.eval()
    device = next(model.parameters()).device
    token_bytes = get_token_bytes(device=device)
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = max(1, (eval_tokens or EVAL_TOKENS) // (batch_size * MAX_SEQ_LEN))
    total_nats = 0.0
    total_bytes = 0
    sensor_offset = 0

    try:
        with torch.no_grad():
            for _ in range(steps):
                x, y, _ = next(val_loader)
                x = x.to(device)
                y = y.to(device)
                indices, sensor_offset = _cycle_indices(len(bundle.data), x.size(0), sensor_offset, device)
                features = _bridge_features_for_indices(bundle, indices, feature_mode, shuffle_map, constant_feature)
                weight_vectors = bridge(features)
                for idx, weight_vector in enumerate(weight_vectors):
                    apply_hypernet_weights(model, weight_vector, rank, target)
                    with autocast_ctx:
                        loss_flat = model(x[idx : idx + 1], y[idx : idx + 1], reduction="none").view(-1)
                    y_flat = y[idx : idx + 1].view(-1)
                    nbytes = token_bytes[y_flat]
                    mask = nbytes > 0
                    total_nats += (loss_flat * mask).sum().item()
                    total_bytes += nbytes.sum().item()
    finally:
        if was_training:
            bridge.train()
    return total_nats / (math.log(2) * total_bytes)


def probe_imu_diversity(
    bridge: PaperBridgeHyper,
    *,
    max_items_per_activity: int = 32,
    selection_seed: int = 42,
) -> dict:
    device = next(bridge.parameters()).device
    was_training = bridge.training
    bridge.eval()
    task_data = load_task_dataset("imu", device)
    grouped: dict[str, list[torch.Tensor]] = {}
    for sensor_tensor, label in task_data.test_pairs:
        grouped.setdefault(label, []).append(sensor_tensor)

    selector = random.Random(selection_seed)
    selected_pairs: list[tuple[torch.Tensor, str]] = []
    for label in sorted(grouped):
        items = list(grouped[label])
        selector.shuffle(items)
        limit = min(max_items_per_activity, len(items))
        selected_pairs.extend((item, label) for item in items[:limit])

    labels: list[str] = []
    weight_vectors: list[torch.Tensor] = []
    try:
        with torch.no_grad():
            for sensor_tensor, label in selected_pairs:
                feature = task_data.get_feature(sensor_tensor)
                weight_vector = bridge(feature)[0].detach().float().cpu()
                labels.append(label)
                weight_vectors.append(weight_vector)
    finally:
        if was_training:
            bridge.train()

    if not weight_vectors:
        raise ValueError("IMU diversity probe selected no held-out examples")

    all_weights = torch.stack(weight_vectors)
    cross_input_cosine = float(_offdiag_cosine(all_weights).item())

    by_label: dict[str, list[torch.Tensor]] = {}
    for label, weight_vector in zip(labels, weight_vectors):
        by_label.setdefault(label, []).append(weight_vector)

    centroids: dict[str, torch.Tensor] = {
        label: torch.stack(vectors).mean(dim=0)
        for label, vectors in sorted(by_label.items())
    }
    centroid_labels = list(centroids.keys())
    centroid_stack = torch.stack([centroids[label] for label in centroid_labels])
    centroid_normed = F.normalize(centroid_stack, dim=1)
    centroid_sim = centroid_normed @ centroid_normed.T

    centroid_pair_cosines: dict[str, float] = {}
    for i, left in enumerate(centroid_labels):
        for j in range(i + 1, len(centroid_labels)):
            right = centroid_labels[j]
            pair_name = f"{_display_activity(left)} <-> {_display_activity(right)}"
            centroid_pair_cosines[pair_name] = float(centroid_sim[i, j].item())

    motion_labels = [label for label in centroid_labels if label.startswith("walking")]
    still_labels = [label for label in centroid_labels if label in {"sitting", "standing", "laying"}]
    motion_centroids = [centroids[label] for label in motion_labels]
    still_centroids = [centroids[label] for label in still_labels]
    motion_vs_still = [
        float(F.cosine_similarity(centroids[left].unsqueeze(0), centroids[right].unsqueeze(0)).item())
        for left in motion_labels
        for right in still_labels
    ]

    paper_pairs = {}
    for left, right in [
        ("walking", "walking_upstairs"),
        ("walking", "sitting"),
        ("walking", "standing"),
        ("sitting", "standing"),
        ("walking_downstairs", "laying"),
    ]:
        if left in centroids and right in centroids:
            paper_pairs[f"{_display_activity(left)} <-> {_display_activity(right)}"] = float(
                F.cosine_similarity(centroids[left].unsqueeze(0), centroids[right].unsqueeze(0)).item()
            )

    return {
        "probe_dataset": "uci-har-test",
        "selection_seed": selection_seed,
        "max_items_per_activity": max_items_per_activity,
        "selected_examples": len(weight_vectors),
        "activity_counts": { _display_activity(label): len(vectors) for label, vectors in sorted(by_label.items()) },
        "cross_input_cosine_mean": cross_input_cosine,
        "within_activity_cosine_mean": {
            _display_activity(label): _mean_pairwise_cosine(vectors)
            for label, vectors in sorted(by_label.items())
        },
        "motion_cluster_centroid_cosine_mean": _mean_pairwise_cosine(motion_centroids),
        "still_cluster_centroid_cosine_mean": _mean_pairwise_cosine(still_centroids),
        "motion_vs_still_centroid_cosine_mean": sum(motion_vs_still) / len(motion_vs_still) if motion_vs_still else 0.0,
        "activity_centroid_pair_cosines": centroid_pair_cosines,
        "paper_pairs": paper_pairs,
    }


def run_diversity_experiment(
    checkpoint: str | Path = DEFAULT_MINI_CHECKPOINT,
    train_steps: int = 300,
    rank: int = 4,
    target: str = "all",
    lr: float = 1e-3,
    diversity_weight: float = 0.1,
    log_csv: str | None = None,
    eval_tokens: int | None = None,
    sensor_limit: int | None = None,
    seed: int = 42,
    probe_max_items_per_activity: int = 32,
    probe_seed: int | None = None,
):
    result = run_bridge_experiment(
        modality="imu",
        feature_mode="true",
        checkpoint=checkpoint,
        train_steps=train_steps,
        rank=rank,
        target=target,
        lr=lr,
        diversity_weight=diversity_weight,
        log_csv=log_csv,
        eval_tokens=eval_tokens,
        sensor_limit=sensor_limit,
        seed=seed,
        return_artifacts=True,
    )
    artifacts = result.pop("artifacts")
    result["experiment"] = f"diversity-imu-l{diversity_weight:.2f}"
    result["heldout_probe"] = probe_imu_diversity(
        artifacts.bridge,
        max_items_per_activity=probe_max_items_per_activity,
        selection_seed=seed if probe_seed is None else probe_seed,
    )
    result["paper_summary"] = {
        "lambda": diversity_weight,
        "bpb_improvement": result["improvement"],
        "cross_input_cosine_mean": result["heldout_probe"]["cross_input_cosine_mean"],
    }
    return result


def _load_bridge_setup(
    modality: str,
    checkpoint: str | Path,
    rank: int,
    target: str,
    sensor_limit: int | None,
    eval_tokens: int | None,
    seed: int,
):
    seed_everything(seed)
    device, device_type = get_device()
    model, _, _ = load_lm(checkpoint)
    model = model.to(device)
    tokenizer = Tokenizer.from_directory()
    autocast_ctx = autocast_for(device_type)
    model.eval()
    with torch.no_grad(), autocast_ctx:
        baseline = evaluate_bpb(model, tokenizer, 8, eval_tokens=eval_tokens)

    apply_lora(model, rank=rank, target=target)
    model = model.to(device)
    freeze_non_lora(model)
    bundle = load_modality_bundle(modality, device, limit=sensor_limit_for(modality, sensor_limit))
    bridge = PaperBridgeHyper(bundle.feature_dim, total_lora_dim(model)).to(device)
    return device, device_type, model, tokenizer, autocast_ctx, baseline, bundle, bridge


def run_static_lora(
    modality: str,
    checkpoint: str | Path = DEFAULT_MINI_CHECKPOINT,
    train_steps: int = 300,
    rank: int = 4,
    target: str = "all",
    lr: float = 1e-3,
    log_csv: str | None = None,
    eval_tokens: int | None = None,
    seed: int = 42,
):
    seed_everything(seed)
    device, device_type = get_device()
    model, _, _ = load_lm(checkpoint)
    model = model.to(device)
    tokenizer = Tokenizer.from_directory()
    autocast_ctx = autocast_for(device_type)

    model.eval()
    with torch.no_grad(), autocast_ctx:
        baseline = evaluate_bpb(model, tokenizer, 8, eval_tokens=eval_tokens)

    apply_lora(model, rank=rank, target=target)
    model = model.to(device)
    freeze_non_lora(model)

    # Same optimizer recipe as the bridge experiments (METHOD_CONTRACT §1.2):
    # only the source of the LoRA weights may differ between conditions.
    lora_params = get_lora_params(model)
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)
    text_loader = make_dataloader(tokenizer, 8, MAX_SEQ_LEN, "train")
    step_log = StepLogger(log_csv) if log_csv else None

    total_time = 0.0
    steps = 0
    model.train()
    while steps < train_steps:
        t0 = time.time()
        tx, ty, _ = next(text_loader)
        tx = tx.to(device)
        ty = ty.to(device)
        with autocast_ctx:
            loss = model(tx, ty)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.time() - t0
        if step_log and steps % 10 == 0:
            step_log.log(step=steps, elapsed_s=round(total_time, 1), loss=round(loss.item(), 6))
        steps += 1
    if step_log:
        step_log.close()

    model.eval()
    eval_t0 = time.time()
    with torch.no_grad(), autocast_ctx:
        final = evaluate_bpb(model, tokenizer, 8, eval_tokens=eval_tokens)
    eval_elapsed = time.time() - eval_t0
    result = {
        "experiment": f"static-lora-{modality}",
        "baseline": baseline,
        "final": final,
        "improvement": baseline - final,
        "steps": steps,
        "lr": lr,
        "trainable_params": sum(p.numel() for p in lora_params),
        "eval_elapsed_s": eval_elapsed,
    }
    result.update(_throughput_metrics(total_time, steps, batch_size=8))
    return result


def run_bridge_experiment(
    modality: str,
    feature_mode: str = "true",
    checkpoint: str | Path = DEFAULT_MINI_CHECKPOINT,
    train_steps: int = 300,
    rank: int = 4,
    target: str = "all",
    lr: float = 1e-3,
    diversity_weight: float = 0.0,
    log_csv: str | None = None,
    eval_tokens: int | None = None,
    sensor_limit: int | None = None,
    seed: int = 42,
    return_artifacts: bool = False,
):
    device, _, model, tokenizer, autocast_ctx, baseline, bundle, bridge = _load_bridge_setup(
        modality=modality,
        checkpoint=checkpoint,
        rank=rank,
        target=target,
        sensor_limit=sensor_limit,
        eval_tokens=eval_tokens,
        seed=seed,
    )

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=0.01)
    text_loader = make_dataloader(tokenizer, 8, MAX_SEQ_LEN, "train")
    step_log = StepLogger(log_csv) if log_csv else None
    shuffle_map = _make_global_shuffle_map(len(bundle.data), seed + 17, device) if feature_mode == "shuffled" else None
    # Capacity-matched control: the bridge sees one fixed feature vector (the
    # bundle mean) on every step, isolating "extra trainable parameters" from
    # any per-input conditioning signal.
    constant_feature = bundle.get_features(bundle.data).mean(dim=0) if feature_mode == "constant" else None

    total_time = 0.0
    steps = 0
    last_diversity = 0.0
    model.train()
    while steps < train_steps:
        t0 = time.time()
        indices = torch.randperm(len(bundle.data), device=device)[:8]
        features = _bridge_features_for_indices(bundle, indices, feature_mode, shuffle_map, constant_feature)

        tx, ty, _ = next(text_loader)
        tx = tx.to(device)
        ty = ty.to(device)
        loss = _backprop_conditioned_batch_loss(model, tx, ty, bridge, features, rank, target, autocast_ctx)

        if diversity_weight > 0:
            per_sample_weights = bridge(features)
            diversity = _offdiag_cosine(per_sample_weights)
            (diversity_weight * diversity).backward()
            loss = loss + diversity_weight * diversity.detach()
            last_diversity = diversity.item()

        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.time() - t0
        if step_log and steps % 10 == 0:
            log = {"step": steps, "elapsed_s": round(total_time, 1), "loss": round(loss.item(), 6)}
            if diversity_weight > 0:
                log["diversity"] = round(last_diversity, 6)
            step_log.log(**log)
        steps += 1
    if step_log:
        step_log.close()

    model.eval()
    eval_t0 = time.time()
    final = _evaluate_bridge_bpb(
        model,
        tokenizer,
        bridge,
        bundle,
        rank,
        target,
        feature_mode,
        eval_tokens,
        autocast_ctx,
        shuffle_map=shuffle_map,
        constant_feature=constant_feature,
    )
    eval_elapsed = time.time() - eval_t0

    result = {
        "experiment": f"{feature_mode}-{modality}" if feature_mode != "true" else f"bridge-{modality}",
        "baseline": baseline,
        "final": final,
        "improvement": baseline - final,
        "steps": steps,
        "lr": lr,
        "diversity_weight": diversity_weight,
        "last_diversity": last_diversity,
        "trainable_params": sum(p.numel() for p in bridge.parameters()),
        "lora_dim": total_lora_dim(model),
        "eval_elapsed_s": eval_elapsed,
    }
    result.update(_throughput_metrics(total_time, steps, batch_size=8))
    if return_artifacts:
        result["artifacts"] = BridgeArtifacts(bridge=bridge, bundle=bundle)
    return result


def run_composition(
    bricks: list[str],
    checkpoint: str | Path = DEFAULT_MINI_CHECKPOINT,
    steps_per_brick: int = 150,
    rank: int = 4,
    target: str = "all",
    lr: float = 1e-3,
    eval_tokens: int | None = None,
    sensor_limit: int | None = None,
    seed: int = 42,
    log_csv: str | None = None,
    eval_mode: str = "conditioned",
):
    trained = []
    train_elapsed_s = 0.0
    total_steps = 0
    total_examples_seen = 0
    total_tokens_seen = 0
    for brick in bricks:
        result = run_bridge_experiment(
            modality=brick,
            feature_mode="true",
            checkpoint=checkpoint,
            train_steps=steps_per_brick,
            rank=rank,
            target=target,
            lr=lr,
            eval_tokens=eval_tokens,
            sensor_limit=sensor_limit,
            seed=seed,
            log_csv=_suffix_log_path(log_csv, brick),
            return_artifacts=True,
        )
        trained.append(result["artifacts"])
        train_elapsed_s += result.get("train_elapsed_s", 0.0)
        total_steps += result.get("steps", 0)
        total_examples_seen += result.get("examples_seen", 0)
        total_tokens_seen += result.get("tokens_seen", 0)

    seed_everything(seed)
    device, device_type = get_device()
    model, _, _ = load_lm(checkpoint)
    model = model.to(device)
    tokenizer = Tokenizer.from_directory()
    autocast_ctx = autocast_for(device_type)

    model.eval()
    with torch.no_grad(), autocast_ctx:
        baseline = evaluate_bpb(model, tokenizer, 8, eval_tokens=eval_tokens)

    apply_lora(model, rank=rank, target=target)
    model = model.to(device)

    model.eval()
    for artifact in trained:
        artifact.bridge.eval()
    eval_t0 = time.time()
    if eval_mode == "fixed":
        with torch.no_grad():
            vectors = []
            for artifact in trained:
                indices, _ = _cycle_indices(len(artifact.bundle.data), 8, 0, device)
                features = artifact.bundle.get_features(artifact.bundle.data[indices])
                vectors.append(artifact.bridge(features).mean(dim=0))
            combined = torch.stack(vectors).mean(dim=0)
        apply_hypernet_weights(model, combined, rank, target)
        with torch.no_grad(), autocast_ctx:
            final = evaluate_bpb(model, tokenizer, 8, eval_tokens=eval_tokens)
    elif eval_mode == "conditioned":
        device = next(model.parameters()).device
        token_bytes = get_token_bytes(device=device)
        val_loader = make_dataloader(tokenizer, 8, MAX_SEQ_LEN, "val")
        steps = max(1, (eval_tokens or EVAL_TOKENS) // (8 * MAX_SEQ_LEN))
        total_nats = 0.0
        total_bytes = 0
        offsets = [0 for _ in trained]
        with torch.no_grad():
            for _ in range(steps):
                x, y, _ = next(val_loader)
                x = x.to(device)
                y = y.to(device)
                weight_vectors = []
                for idx, artifact in enumerate(trained):
                    sensor_indices, offsets[idx] = _cycle_indices(len(artifact.bundle.data), x.size(0), offsets[idx], device)
                    features = artifact.bundle.get_features(artifact.bundle.data[sensor_indices])
                    weight_vectors.append(artifact.bridge(features))
                merged = torch.stack(weight_vectors).mean(dim=0)
                for row, weight_vector in enumerate(merged):
                    apply_hypernet_weights(model, weight_vector, rank, target)
                    with autocast_ctx:
                        loss_flat = model(x[row : row + 1], y[row : row + 1], reduction="none").view(-1)
                    y_flat = y[row : row + 1].view(-1)
                    nbytes = token_bytes[y_flat]
                    mask = nbytes > 0
                    total_nats += (loss_flat * mask).sum().item()
                    total_bytes += nbytes.sum().item()
        final = total_nats / (math.log(2) * total_bytes)
    else:
        raise ValueError(f"Unsupported composition eval mode: {eval_mode}")
    eval_elapsed = time.time() - eval_t0
    return {
        "experiment": f"compose-{''.join(brick[0].upper() for brick in bricks)}",
        "baseline": baseline,
        "final": final,
        "improvement": baseline - final,
        "bricks": bricks,
        "steps_per_brick": steps_per_brick,
        "lr": lr,
        "eval_mode": eval_mode,
        "steps": total_steps,
        "train_elapsed_s": train_elapsed_s,
        "eval_elapsed_s": eval_elapsed,
        "steps_per_second": (total_steps / train_elapsed_s) if train_elapsed_s > 0 else 0.0,
        "examples_seen": total_examples_seen,
        "tokens_seen": total_tokens_seen,
    }


def run_prefix_experiment(
    modality: str,
    checkpoint: str | Path = DEFAULT_MINI_CHECKPOINT,
    train_steps: int = 300,
    n_prefix: int = 8,
    lr: float = 1e-3,
    log_csv: str | None = None,
    eval_tokens: int | None = None,
    sensor_limit: int | None = None,
    seed: int = 42,
):
    seed_everything(seed)
    device, device_type = get_device()
    model, config, _ = load_lm(checkpoint)
    model = model.to(device)
    tokenizer = Tokenizer.from_directory()
    autocast_ctx = autocast_for(device_type)

    model.eval()
    with torch.no_grad(), autocast_ctx:
        baseline = evaluate_bpb(model, tokenizer, 8, eval_tokens=eval_tokens)

    for param in model.parameters():
        param.requires_grad_(False)
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model

    bundle = load_modality_bundle(modality, device, limit=sensor_limit_for(modality, sensor_limit))
    prefix_proj = PrefixProjection(bundle.feature_dim, config.n_embd, n_prefix).to(device)

    optimizer = torch.optim.AdamW(prefix_proj.parameters(), lr=lr, weight_decay=0.01)
    text_loader = make_dataloader(tokenizer, 8, MAX_SEQ_LEN - n_prefix, "train")
    val_loader = make_dataloader(tokenizer, 8, MAX_SEQ_LEN - n_prefix, "val")
    step_log = StepLogger(log_csv) if log_csv else None

    total_time = 0.0
    steps = 0
    model.train()
    while steps < train_steps:
        t0 = time.time()
        indices = torch.randperm(len(bundle.data), device=device)[:8]
        sensor_batch = bundle.data[indices]
        features = bundle.get_features(sensor_batch)
        prefix_tokens = prefix_proj(features)

        tx, ty, _ = next(text_loader)
        tx = tx.to(device)
        ty = ty.to(device)
        with autocast_ctx:
            logits = _forward_with_prefix(raw, prefix_tokens, tx)
        # Dataloader targets are already shifted (y = row[1:]), so the logit at
        # combined position n_prefix+k predicts ty[k]: drop only the prefix part.
        text_logits = logits[:, n_prefix:]
        loss = F.cross_entropy(text_logits.reshape(-1, text_logits.size(-1)), ty.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(prefix_proj.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.time() - t0
        if step_log and steps % 10 == 0:
            step_log.log(step=steps, elapsed_s=round(total_time, 1), loss=round(loss.item(), 6))
        steps += 1
    if step_log:
        step_log.close()

    model.eval()
    # Byte-weighted BPB, matching evaluate_bpb/_evaluate_bridge_bpb exactly, so
    # the prefix final is in the same units as the baseline it is compared to.
    token_bytes = get_token_bytes(device=device)
    total_nats = 0.0
    total_bytes = 0
    sensor_offset = 0
    eval_steps = max(1, (eval_tokens or EVAL_TOKENS) // (8 * (MAX_SEQ_LEN - n_prefix)))
    eval_t0 = time.time()
    with torch.no_grad():
        for _ in range(eval_steps):
            vx, vy, _ = next(val_loader)
            vx = vx.to(device)
            vy = vy.to(device)
            sensor_batch, sensor_offset = cycle_batch(bundle.data, vx.size(0), sensor_offset)
            prefix_tokens = prefix_proj(bundle.get_features(sensor_batch))
            # Same precision context as evaluate_bpb/_evaluate_bridge_bpb so the
            # baseline-vs-final comparison stays like-for-like on CUDA (bf16).
            with autocast_ctx:
                logits = _forward_with_prefix(raw, prefix_tokens, vx)
            text_logits = logits[:, n_prefix:]
            loss_flat = F.cross_entropy(
                text_logits.reshape(-1, text_logits.size(-1)),
                vy.reshape(-1),
                reduction="none",
            )
            vy_flat = vy.reshape(-1)
            nbytes = token_bytes[vy_flat]
            mask = nbytes > 0
            total_nats += (loss_flat * mask).sum().item()
            total_bytes += nbytes.sum().item()
    final = total_nats / (math.log(2) * total_bytes)
    eval_elapsed = time.time() - eval_t0
    result = {
        "experiment": f"prefix-{modality}-{n_prefix}tok",
        "baseline": baseline,
        "final": final,
        "improvement": baseline - final,
        "steps": steps,
        "n_prefix": n_prefix,
        "lr": lr,
        "params": sum(param.numel() for param in prefix_proj.parameters()),
        "eval_elapsed_s": eval_elapsed,
    }
    result.update(_throughput_metrics(total_time, steps, batch_size=8, seq_len=MAX_SEQ_LEN - n_prefix))
    return result


def _task_feature_for_condition(
    cond_mode: str,
    own_feature: torch.Tensor,
    alt_feature: torch.Tensor | None = None,
) -> torch.Tensor:
    if cond_mode == "true":
        return own_feature
    if cond_mode == "shuffled":
        if alt_feature is None:
            return own_feature
        return alt_feature
    if cond_mode == "random":
        return torch.randn_like(own_feature)
    raise ValueError(f"Unsupported task condition: {cond_mode}")


def _split_label_ids(prompt_ids: list[int], full_ids: list[int]) -> tuple[list[int], list[int]]:
    """Split the natural tokenization of prompt+label into shared context and label tokens.

    The template's trailing space can merge into the first label token, so the
    shared context can be shorter than the standalone prompt tokenization. Scoring
    everything after the longest common prefix guarantees no label token is lost.
    """
    common = 0
    for prompt_token, full_token in zip(prompt_ids, full_ids):
        if prompt_token != full_token:
            break
        common += 1
    return full_ids[:common], full_ids[common:]


def _stratified_subset(
    pairs: list[tuple[torch.Tensor, str]],
    max_items: int,
    seed: int,
) -> list[tuple[torch.Tensor, str]]:
    """Label-stratified, seed-shuffled eval subset (round-robin across labels)."""
    grouped: dict[str, list[tuple[torch.Tensor, str]]] = {}
    for pair in pairs:
        grouped.setdefault(pair[1], []).append(pair)
    selector = random.Random(seed)
    for items in grouped.values():
        selector.shuffle(items)
    labels = sorted(grouped)
    target = min(max_items, len(pairs))
    subset: list[tuple[torch.Tensor, str]] = []
    while len(subset) < target:
        progressed = False
        for label in labels:
            if grouped[label] and len(subset) < target:
                subset.append(grouped[label].pop())
                progressed = True
        if not progressed:
            break
    return subset


def _score_label(model, tokenizer: Tokenizer, context_ids: list[int], label_ids: list[int]) -> float:
    if not label_ids or not context_ids:
        return float("-inf")
    full = context_ids + label_ids
    x = torch.tensor([full[:-1]], dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        logits = model(x, targets=None)[0]
    start = len(context_ids) - 1
    positions = logits[start : start + len(label_ids)]
    log_probs = F.log_softmax(positions, dim=-1)
    targets = torch.tensor(label_ids, dtype=torch.long, device=logits.device)
    return log_probs[torch.arange(len(label_ids), device=logits.device), targets].mean().item()


def run_task_eval(
    modality: str,
    checkpoint: str | Path = DEFAULT_MINI_CHECKPOINT,
    train_steps: int = 600,
    rank: int = 4,
    target: str = "all",
    lr: float = 1e-3,
    max_eval_items: int = 200,
    seed: int = 42,
):
    device, _, = get_device()
    seed_everything(seed)
    task_data = load_task_dataset(modality, device)
    tokenizer = Tokenizer.from_directory()
    prompt_ids = tokenizer.encode(task_data.prompt_template)

    results = {}
    conditions = {
        "true": "true",
        "shuffled": "shuffled",
        "random": "random",
        "no_bridge": "none",
    }

    for cond_name, cond_mode in conditions.items():
        seed_everything(seed)
        model, _, ckpt = load_lm(checkpoint)
        model = model.to(device)
        apply_lora(model, rank=rank, target=target)
        model = model.to(device)
        freeze_non_lora(model)
        lora_dim = total_lora_dim(model)

        if cond_mode != "none":
            bridge = PaperBridgeHyper(task_data.feature_dim, lora_dim).to(device)
            trainable = list(bridge.parameters())
        else:
            bridge = None
            trainable = []

        # Same optimizer recipe as run_bridge_experiment (METHOD_CONTRACT §1.2):
        # mean loss over the batch, weight decay 0.01, grad clip 1.0.
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01) if trainable else None
        total_time = 0.0
        steps = 0
        if cond_mode != "none":
            model.train()
            while steps < train_steps:
                t0 = time.time()
                batch = random.sample(task_data.train_pairs, min(4, len(task_data.train_pairs)))
                alt_batch = batch[1:] + batch[:1]
                loss_total = 0.0
                for (sensor_data, label), (alt_sensor, _) in zip(batch, alt_batch):
                    own_feature = task_data.get_feature(sensor_data)
                    alt_feature = task_data.get_feature(alt_sensor)
                    feature = _task_feature_for_condition(cond_mode, own_feature, alt_feature)
                    weight_vector = bridge(feature)[0]
                    apply_hypernet_weights(model, weight_vector, rank, target)
                    ids = tokenizer.encode(task_data.prompt_template + label)
                    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
                    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
                    loss_total = loss_total + model(x, y)
                (loss_total / len(batch)).backward()
                torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if device.type == "mps":
                    torch.mps.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize()
                total_time += time.time() - t0
                steps += 1

        model.eval()
        test_subset = _stratified_subset(task_data.test_pairs, max_eval_items, seed)
        cached_features = [task_data.get_feature(sensor_data) for sensor_data, _ in test_subset]
        rank1 = 0
        top2 = 0
        ranks = []
        for idx, ((_, true_label), own_feature) in enumerate(zip(test_subset, cached_features)):
            if cond_mode == "none":
                zeros = torch.zeros(lora_dim, device=device)
                apply_hypernet_weights(model, zeros, rank, target)
            else:
                alt_feature = cached_features[(idx + 1) % len(cached_features)]
                feature = _task_feature_for_condition(cond_mode, own_feature, alt_feature)
                weight_vector = bridge(feature)[0]
                apply_hypernet_weights(model, weight_vector, rank, target)

            scored = []
            for category in task_data.categories:
                full_ids = tokenizer.encode(task_data.prompt_template + category)
                context_ids, label_ids = _split_label_ids(prompt_ids, full_ids)
                scored.append((category, _score_label(model, tokenizer, context_ids, label_ids)))
            scored.sort(key=lambda item: item[1], reverse=True)
            ordered = [label for label, _ in scored]
            label_rank = ordered.index(true_label) + 1
            ranks.append(label_rank)
            if label_rank == 1:
                rank1 += 1
            if label_rank <= 2:
                top2 += 1

        n_eval = len(ranks)
        results[cond_name] = {
            "rank1": rank1 / n_eval if n_eval else 0.0,
            "top2": top2 / n_eval if n_eval else 0.0,
            "avg_rank": sum(ranks) / n_eval if n_eval else 0.0,
            "count": n_eval,
            "steps": steps,
            "lr": lr,
            "train_elapsed_s": total_time,
            "examples_seen": steps * min(4, len(task_data.train_pairs)),
        }

    return {
        "experiment": f"task-{modality}",
        "modality": modality,
        "results": results,
    }


def results_to_jsonable(result):
    if isinstance(result, dict):
        return {key: results_to_jsonable(value) for key, value in result.items() if key != "artifacts"}
    if isinstance(result, list):
        return [results_to_jsonable(value) for value in result]
    if isinstance(result, Path):
        return str(result)
    return result


def dump_result(result) -> str:
    return json.dumps(results_to_jsonable(result), indent=2, sort_keys=True)
