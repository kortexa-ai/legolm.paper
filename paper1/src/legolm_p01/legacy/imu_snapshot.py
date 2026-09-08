from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CACHE_DIR = Path.home() / ".cache" / "autoresearch" / "sensor-fusion"
UCI_HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"

SIGNAL_FILES = [
    "body_acc_x_{}.txt",
    "body_acc_y_{}.txt",
    "body_acc_z_{}.txt",
    "body_gyro_x_{}.txt",
    "body_gyro_y_{}.txt",
    "body_gyro_z_{}.txt",
    "total_acc_x_{}.txt",
    "total_acc_y_{}.txt",
    "total_acc_z_{}.txt",
]

ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]


def download_uci_har():
    data_dir = CACHE_DIR / "UCI HAR Dataset"
    if data_dir.exists():
        return data_dir

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / "uci-har.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(UCI_HAR_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(CACHE_DIR)

    inner_zip = CACHE_DIR / "UCI HAR Dataset.zip"
    if inner_zip.exists() and not data_dir.exists():
        with zipfile.ZipFile(inner_zip) as zf:
            zf.extractall(CACHE_DIR)

    assert data_dir.exists(), f"Expected {data_dir} after extraction"
    return data_dir


def load_signals(data_dir, split):
    signals = []
    for sig_file in SIGNAL_FILES:
        path = data_dir / split / "Inertial Signals" / sig_file.format(split)
        data = np.loadtxt(path)
        signals.append(data)
    return torch.tensor(np.stack(signals, axis=1), dtype=torch.float32)


def load_labels(data_dir, split):
    path = data_dir / split / f"y_{split}.txt"
    y = np.loadtxt(path, dtype=int) - 1
    return torch.tensor(y, dtype=torch.long)


class IMUConv1DEncoder(nn.Module):
    def __init__(self, n_channels=9, n_classes=6, feature_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
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
