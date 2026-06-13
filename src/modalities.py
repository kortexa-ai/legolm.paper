from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from .audio_snapshot import AudioConv1DEncoder, compute_mel_spectrogram, load_audio_wav, load_esc50
from .common import (
    DEFAULT_AUDIO_ENCODER,
    DEFAULT_IMU_ENCODER,
    DEFAULT_VISION_PERCEIVER,
    REPO_ROOT,
    assert_materialized_asset,
)
from .imu_snapshot import ACTIVITIES, IMUConv1DEncoder, download_uci_har, load_labels, load_signals
from .vision_snapshot import FrameEncoder, PerceiverResampler


@dataclass
class ModalityBundle:
    modality: str
    feature_dim: int
    data: torch.Tensor
    get_features: Callable[[torch.Tensor], torch.Tensor]


@dataclass
class TaskDataset:
    modality: str
    feature_dim: int
    train_pairs: list[tuple[torch.Tensor, str]]
    test_pairs: list[tuple[torch.Tensor, str]]
    categories: list[str]
    prompt_template: str
    get_feature: Callable[[torch.Tensor], torch.Tensor]


def _resolve_frame_path(frame_path: str) -> Path:
    rel = Path(frame_path)
    candidate = REPO_ROOT / rel
    if candidate.exists():
        return candidate

    parts = rel.parts
    if parts and parts[0] == "vision_data":
        mapped = REPO_ROOT / "data" / "vision" / Path(*parts[1:])
        if mapped.exists():
            return mapped

    data_candidate = REPO_ROOT / "data" / "vision" / rel.name
    if data_candidate.exists():
        return data_candidate

    return candidate


def load_audio_bundle(device: torch.device, limit: int = 200) -> ModalityBundle:
    ckpt = torch.load(assert_materialized_asset(DEFAULT_AUDIO_ENCODER), map_location="cpu", weights_only=False)
    enc = AudioConv1DEncoder(
        n_mels=ckpt["n_mels"],
        n_classes=ckpt["n_classes"],
        feature_dim=ckpt["feature_dim"],
    ).to(device)
    enc.load_state_dict(ckpt["model"])
    enc.eval()
    norm = ckpt["normalization"]

    samples, audio_dir = load_esc50()
    mels = []
    for sample in samples:
        wav = load_audio_wav(audio_dir / sample["filename"])
        if wav is None:
            continue
        mel = compute_mel_spectrogram(wav, n_mels=ckpt["n_mels"])
        mel = (mel - norm["mean"]) / norm["std"]
        mels.append(torch.tensor(mel, dtype=torch.float32))
        if len(mels) >= limit:
            break
    data = torch.stack(mels).to(device)

    def get_features(batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return enc.features(batch)

    return ModalityBundle("audio", ckpt["feature_dim"], data, get_features)


def load_imu_bundle(device: torch.device, limit: int = 200) -> ModalityBundle:
    ckpt = torch.load(assert_materialized_asset(DEFAULT_IMU_ENCODER), map_location="cpu", weights_only=False)
    enc = IMUConv1DEncoder(
        n_channels=ckpt["n_channels"],
        n_classes=ckpt["n_classes"],
        feature_dim=ckpt["feature_dim"],
    ).to(device)
    enc.load_state_dict(ckpt["model"])
    enc.eval()
    norm = ckpt["normalization"]

    data_dir = download_uci_har()
    data = load_signals(data_dir, "train")
    data = (data - norm["mean"]) / (norm["std"] + 1e-8)
    data = data[:limit].to(device)

    def get_features(batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return enc.features(batch)

    return ModalityBundle("imu", ckpt["feature_dim"], data, get_features)


def load_vision_bundle(device: torch.device, limit: int = 50) -> ModalityBundle:
    import cv2

    enc = FrameEncoder().to(device)
    enc.eval()

    per_ckpt = torch.load(assert_materialized_asset(DEFAULT_VISION_PERCEIVER), map_location="cpu", weights_only=False)
    per_cfg = per_ckpt.get("config", {})
    perceiver = PerceiverResampler(
        n_queries=per_cfg.get("n_vision_tokens", 32),
        feat_dim=per_cfg.get("feat_dim", 576),
        d_model=per_cfg.get("d_model", 512),
        n_heads=per_cfg.get("n_heads", 8),
        n_layers=per_cfg.get("n_perceiver_layers", 2),
    ).to(device)
    perceiver.load_state_dict(per_ckpt["perceiver"], strict=True)
    perceiver.eval()

    dataset_path = REPO_ROOT / "data" / "vision" / "dataset.json"
    with open(dataset_path) as f:
        dataset = json.load(f)

    frames = []
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    for sample in dataset:
        img = cv2.imread(str(_resolve_frame_path(sample["frame_path"])))
        if img is None:
            continue
        img = cv2.resize(img, (224, 224))
        frame = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        frame = (frame - mean) / std
        frames.append(frame)
        if len(frames) >= limit:
            break
    data = torch.stack(frames).to(device)

    def get_features(batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            spatial = enc(batch)
            tokens = perceiver(spatial)
            return tokens.mean(dim=1)

    return ModalityBundle("vision", per_cfg.get("d_model", 512), data, get_features)


def load_modality_bundle(modality: str, device: torch.device, limit: int | None = None) -> ModalityBundle:
    if modality == "audio":
        return load_audio_bundle(device, limit or 200)
    if modality == "imu":
        return load_imu_bundle(device, limit or 200)
    if modality == "vision":
        return load_vision_bundle(device, limit or 50)
    raise ValueError(f"Unsupported modality: {modality}")


def load_audio_task_dataset(device: torch.device) -> TaskDataset:
    ckpt = torch.load(assert_materialized_asset(DEFAULT_AUDIO_ENCODER), map_location="cpu", weights_only=False)
    enc = AudioConv1DEncoder(
        n_mels=ckpt["n_mels"],
        n_classes=ckpt["n_classes"],
        feature_dim=ckpt["feature_dim"],
    ).to(device)
    enc.load_state_dict(ckpt["model"])
    enc.eval()
    norm = ckpt["normalization"]

    samples, audio_dir = load_esc50()
    train_pairs = []
    test_pairs = []
    for sample in samples:
        wav = load_audio_wav(audio_dir / sample["filename"])
        if wav is None:
            continue
        mel = compute_mel_spectrogram(wav, n_mels=ckpt["n_mels"])
        mel = (mel - norm["mean"]) / norm["std"]
        pair = (torch.tensor(mel, dtype=torch.float32), sample["category"])
        if sample["fold"] == 5:
            test_pairs.append(pair)
        else:
            train_pairs.append(pair)
    categories = sorted({sample["category"] for sample in samples})

    def get_feature(sensor_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return enc.features(sensor_tensor.unsqueeze(0).to(device)).squeeze(0)

    return TaskDataset(
        modality="audio",
        feature_dim=ckpt["feature_dim"],
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        categories=categories,
        prompt_template="The sound is: ",
        get_feature=get_feature,
    )


def load_imu_task_dataset(device: torch.device) -> TaskDataset:
    ckpt = torch.load(assert_materialized_asset(DEFAULT_IMU_ENCODER), map_location="cpu", weights_only=False)
    enc = IMUConv1DEncoder(
        n_channels=ckpt["n_channels"],
        n_classes=ckpt["n_classes"],
        feature_dim=ckpt["feature_dim"],
    ).to(device)
    enc.load_state_dict(ckpt["model"])
    enc.eval()
    norm = ckpt["normalization"]

    data_dir = download_uci_har()
    x_train = load_signals(data_dir, "train")
    y_train = load_labels(data_dir, "train")
    x_test = load_signals(data_dir, "test")
    y_test = load_labels(data_dir, "test")
    x_train = (x_train - norm["mean"]) / (norm["std"] + 1e-8)
    x_test = (x_test - norm["mean"]) / (norm["std"] + 1e-8)

    categories = [activity.lower() for activity in ACTIVITIES]
    train_pairs = [(x_train[idx], categories[y_train[idx].item()]) for idx in range(len(x_train))]
    test_pairs = [(x_test[idx], categories[y_test[idx].item()]) for idx in range(len(x_test))]

    def get_feature(sensor_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return enc.features(sensor_tensor.unsqueeze(0).to(device)).squeeze(0)

    return TaskDataset(
        modality="imu",
        feature_dim=ckpt["feature_dim"],
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        categories=categories,
        prompt_template="The activity is: ",
        get_feature=get_feature,
    )


def load_task_dataset(modality: str, device: torch.device) -> TaskDataset:
    if modality == "audio":
        return load_audio_task_dataset(device)
    if modality == "imu":
        return load_imu_task_dataset(device)
    raise ValueError(f"Task evaluation is only implemented for audio/imu, got: {modality}")
