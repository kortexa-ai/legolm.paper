from __future__ import annotations

import urllib.request
import wave
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CACHE_DIR = Path.home() / ".cache" / "autoresearch" / "audio-encoder"
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
SAMPLE_RATE = 22050
N_MELS = 64
HOP_LENGTH = 512
N_FFT = 1024
AUDIO_LEN = 22050 * 5


def compute_mel_spectrogram(waveform, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    if len(waveform) < n_fft:
        waveform = np.pad(waveform, (0, n_fft - len(waveform)))
    if len(waveform) < AUDIO_LEN:
        waveform = np.pad(waveform, (0, AUDIO_LEN - len(waveform)))
    else:
        waveform = waveform[:AUDIO_LEN]

    window = np.hanning(n_fft)
    n_frames = 1 + (len(waveform) - n_fft) // hop_length
    frames = np.stack([waveform[i * hop_length : i * hop_length + n_fft] * window for i in range(n_frames)])
    spectrum = np.abs(np.fft.rfft(frames, n=n_fft))
    power = spectrum ** 2

    mel_points = np.linspace(0, 2595 * np.log10(1 + sr / 2 / 700), n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    fft_bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for idx in range(n_mels):
        start, center, end = fft_bins[idx], fft_bins[idx + 1], fft_bins[idx + 2]
        if center > start:
            filterbank[idx, start:center] = np.linspace(0, 1, center - start)
        if end > center:
            filterbank[idx, center:end] = np.linspace(1, 0, end - center)

    mel_spec = np.dot(power, filterbank.T)
    mel_spec = np.log(mel_spec + 1e-9)
    return mel_spec.T


def load_esc50():
    data_dir = CACHE_DIR / "ESC-50-master"
    if not data_dir.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = CACHE_DIR / "esc50.zip"
        if not zip_path.exists():
            urllib.request.urlretrieve(ESC50_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(CACHE_DIR)

    audio_dir = data_dir / "audio"
    meta_path = data_dir / "meta" / "esc50.csv"

    import csv

    samples = []
    with open(meta_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(
                {
                    "filename": row["filename"],
                    "fold": int(row["fold"]),
                    "target": int(row["target"]),
                    "category": row["category"],
                }
            )
    return samples, audio_dir


def load_audio_wav(path):
    try:
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if sampwidth == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

        if n_channels > 1:
            data = data.reshape(-1, n_channels).mean(axis=1)

        if framerate != SAMPLE_RATE:
            indices = np.linspace(0, len(data) - 1, int(len(data) * SAMPLE_RATE / framerate))
            data = np.interp(indices, np.arange(len(data)), data)
        return data
    except Exception:
        return None


class AudioConv1DEncoder(nn.Module):
    def __init__(self, n_mels=64, n_classes=50, feature_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(feature_dim, n_classes)
        self.feature_dim = feature_dim

    def forward(self, x):
        return self.classifier(self.features(x))

    def features(self, x):
        return self.conv(x).squeeze(-1)
