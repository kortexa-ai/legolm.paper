"""Frozen sensor features with sample, split, group and encoder provenance."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .legacy import audio_snapshot as audio
from .legacy import imu_snapshot as imu
from .legacy.vision_snapshot import FrameEncoder, PerceiverResampler
from .provenance import file_hash, write_json, digest


def _encode(encoder, data, batch=64):
    device = next(encoder.parameters()).device
    with torch.inference_mode():
        features = torch.cat([encoder.features(x.to(device)).float().cpu() for x in data.split(batch)])
        # Batch feature caching must preserve the per-item evaluation path.
        for i in range(min(3, len(data))):
            single = encoder.features(data[i:i+1].to(device)).float().cpu()[0]
            torch.testing.assert_close(single, features[i], atol=2e-5, rtol=2e-5)
    return features


def _checkpoint(path):
    with path.open("rb") as f:
        if f.read(42).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Materialize Git LFS asset: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def prepare_imu(asset_root, cache, device):
    path = asset_root / "checkpoints/encoders/imu.pt"
    ckpt = _checkpoint(path)
    encoder = imu.IMUConv1DEncoder(ckpt["n_channels"], ckpt["n_classes"], ckpt["feature_dim"]).to(device).eval()
    encoder.load_state_dict(ckpt["model"], strict=True)
    encoder.requires_grad_(False)
    imu.CACHE_DIR = cache / "sensor-fusion"
    data_dir = imu.download_uci_har()
    categories = [a.lower() for a in imu.ACTIVITIES]
    result = {"categories": categories, "prompt": "The activity is: ", "modality": "imu",
              "encoder_sha256": file_hash(path), "encoder_parameters": sum(p.numel() for p in encoder.parameters()),
              "encoder_split": "UCI-HAR official training split; verify against encoder checkpoint provenance",
              "raw_files": {str(p.relative_to(data_dir)): file_hash(p) for p in sorted(data_dir.rglob("*.txt"))}}
    for split in ("train", "test"):
        raw = imu.load_signals(data_dir, split)
        x = (raw - ckpt["normalization"]["mean"]) / (ckpt["normalization"]["std"] + 1e-8)
        labels = imu.load_labels(data_dir, split).tolist()
        groups = np.loadtxt(data_dir / split / f"subject_{split}.txt", dtype=int).tolist()
        result[split] = {"features": _encode(encoder, x), "labels": labels,
                         "ids": [f"uci-har/{split}/{i:05d}" for i in range(len(x))],
                         "groups": [f"subject-{g}" for g in groups]}
    if len(result["test"]["labels"]) != 2947 or len(result["train"]["labels"]) != 7352:
        raise ValueError("UCI-HAR official split count mismatch")
    result["unpaired"] = {"features": result["train"]["features"][:64], "ids": result["train"]["ids"][:64]}
    return result


def prepare_audio(asset_root, cache, device):
    path = asset_root / "checkpoints/experiments/esc50-512d.pt"
    ckpt = _checkpoint(path)
    encoder = audio.AudioConv1DEncoder(ckpt["n_mels"], ckpt["n_classes"], ckpt["feature_dim"]).to(device).eval()
    encoder.load_state_dict(ckpt["model"], strict=True)
    encoder.requires_grad_(False)
    audio.CACHE_DIR = cache / "audio-encoder"
    samples, audio_dir = audio.load_esc50()
    with (audio_dir.parent / "meta/esc50.csv").open() as f:
        metadata = {r["filename"]: r for r in csv.DictReader(f)}
    categories = sorted({s["category"] for s in samples})
    raw, hashes = [], {}
    for s in samples:
        p = audio_dir / s["filename"]
        wav = audio.load_audio_wav(p)
        if wav is None:
            raise ValueError(f"Unusable ESC-50 clip; no silent exclusions: {p}")
        mel = audio.compute_mel_spectrogram(wav, n_mels=ckpt["n_mels"])
        raw.append(torch.tensor((mel - ckpt["normalization"]["mean"]) / ckpt["normalization"]["std"], dtype=torch.float32))
        hashes[s["filename"]] = file_hash(p)
    features = _encode(encoder, torch.stack(raw))
    result = {"categories": categories, "prompt": "The sound is: ", "modality": "audio",
              "encoder_sha256": file_hash(path), "encoder_parameters": sum(p.numel() for p in encoder.parameters()),
              "encoder_split": "ESC-50 folds 1-4; fold 5 excluded", "raw_files": hashes}
    for split in ("train", "test"):
        indices = [i for i, s in enumerate(samples) if (s["fold"] == 5) == (split == "test")]
        result[split] = {"features": features[indices], "labels": [categories.index(samples[i]["category"]) for i in indices],
                         "ids": [samples[i]["filename"] for i in indices],
                         "groups": [str(metadata[samples[i]["filename"]]["src_file"]) for i in indices]}
    if len(result["test"]["labels"]) != 400 or len(result["train"]["labels"]) != 1600:
        raise ValueError("ESC-50 split count mismatch")
    result["unpaired"] = {"features": features[:64], "ids": [s["filename"] for s in samples[:64]]}
    return result


class Vision(torch.nn.Module):
    def __init__(self, checkpoint, device):
        super().__init__()
        cfg = checkpoint.get("config", {})
        self.encoder = FrameEncoder()
        self.perceiver = PerceiverResampler(n_queries=cfg.get("n_vision_tokens", 32), feat_dim=cfg.get("feat_dim", 576),
            d_model=cfg.get("d_model", 512), n_heads=cfg.get("n_heads", 8), n_layers=cfg.get("n_perceiver_layers", 2))
        self.perceiver.load_state_dict(checkpoint["perceiver"], strict=True)
        self.to(device).eval().requires_grad_(False)

    def features(self, images):
        return self.perceiver(self.encoder(images)).mean(dim=1)


def prepare_vision(asset_root, cache, device):
    import cv2
    from torchvision.datasets import CIFAR10
    path = asset_root / "checkpoints/experiments/vision-perceiver.pt"
    encoder = Vision(_checkpoint(path), device)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    def transform(bgr):
        arr = cv2.resize(bgr, (224, 224))
        return (torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255 - mean) / std
    index = json.loads((asset_root / "data/vision/dataset.json").read_text())
    frames, frame_ids, raw_hashes = [], [], {}
    for sample in index[:64]:
        relative = Path(sample["frame_path"])
        candidates = [asset_root / relative, asset_root / "data/vision" / Path(*relative.parts[1:]), asset_root / "data/vision" / relative.name]
        p = next((p for p in candidates if p.exists()), None)
        if p is None:
            raise FileNotFoundError(relative)
        frames.append(transform(cv2.imread(str(p))))
        frame_ids.append(str(relative))
        raw_hashes[str(relative)] = file_hash(p)
    result = {"categories": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
              "prompt": "The image shows: ", "modality": "vision", "encoder_sha256": file_hash(path),
              "encoder_parameters": sum(p.numel() for p in encoder.parameters()),
              "preprocessing": "224x224 OpenCV linear resize; BGR channels and original ImageNet normalization; original pipeline preserved",
              "encoder_split": "ImageNet pretrained MobileNetV3-Small plus BBB perceiver; CIFAR excluded from known fitting, upstream exposure not exhaustively known",
              "raw_files": raw_hashes,
              "unpaired": {"features": _encode(encoder, torch.stack(frames), batch=16), "ids": frame_ids}}
    for train in (True, False):
        dataset = CIFAR10(cache / "cifar10", train=train, download=True)
        features = []
        # Bound image residency; never materialize 60k full-resolution images.
        for start in range(0, len(dataset), 32):
            inputs = torch.stack([transform(dataset.data[i][..., ::-1]) for i in range(start, min(start+32, len(dataset)))])
            with torch.inference_mode():
                features.append(encoder.features(inputs.to(device)).float().cpu())
        split = "train" if train else "test"
        result[split] = {"features": torch.cat(features), "labels": list(map(int, dataset.targets)),
                         "ids": [f"cifar10/{split}/{i:05d}" for i in range(len(dataset))],
                         "groups": [f"cifar10/{split}/{i:05d}" for i in range(len(dataset))]}
    rng = np.random.default_rng(42)
    dev = []
    y = np.array(result["train"]["labels"])
    for label in range(10):
        dev.extend(rng.permutation(np.flatnonzero(y == label))[:500].tolist())
    result["dev_indices"] = sorted(dev)
    for p in sorted((cache / "cifar10/cifar-10-batches-py").glob("*")):
        if p.is_file():
            result["raw_files"][str(p.relative_to(cache))] = file_hash(p)
    backbone = Path(torch.hub.get_dir()) / "checkpoints/mobilenet_v3_small-047dcff4.pth"
    if not backbone.exists():
        raise FileNotFoundError("Pin the exact MobileNet backbone checkpoint")
    result["backbone_sha256"] = file_hash(backbone)
    return result


def prepare(asset_root: Path, cache: Path, output: Path, device: str, modalities=("imu", "audio", "vision")):
    import importlib.metadata
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    output.mkdir(parents=True, exist_ok=True)
    index = {}
    preparation={"source_sha256":file_hash(__file__),"torch":torch.__version__,
                 "torchvision":importlib.metadata.version("torchvision"),"precision":"FP32; TF32 disabled"}
    for modality in modalities:
        path = output / f"{modality}.pt"
        if not path.exists():
            result = {"imu": prepare_imu, "audio": prepare_audio, "vision": prepare_vision}[modality](asset_root, cache, device)
            result["preparation"]=preparation
            temporary = path.with_suffix(".tmp")
            torch.save(result, temporary)
            temporary.replace(path)
        result = torch.load(path, map_location="cpu", weights_only=False)
        if result.get("preparation")!=preparation:
            raise ValueError(f"Feature preparation provenance changed; use a new output directory: {path}")
        summary = {k: v for k, v in result.items() if k not in ("train", "test", "unpaired")}
        for split in ("train", "test", "unpaired"):
            summary[split] = {k: v for k, v in result[split].items() if k != "features"}
            summary[split]["feature_shape"] = list(result[split]["features"].shape)
        summary["cache_sha256"] = file_hash(path)
        write_json(output / f"{modality}.json", summary)
        index[modality] = summary["cache_sha256"]
        print(json.dumps({"prepared": modality, "sha256": index[modality], "train": len(result["train"]["ids"]), "test": len(result["test"]["ids"])}), flush=True)
    write_json(output / "index.json", index)
    return index
