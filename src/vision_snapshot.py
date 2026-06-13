from __future__ import annotations

import torch
import torch.nn as nn


class FrameEncoder(nn.Module):
    """MobileNetV3-Small feature extractor used by the paper vision brick."""

    def __init__(self):
        super().__init__()
        from torchvision import models

        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = backbone.features
        self.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False
        self.feat_dim = 576

    def forward(self, x):
        return self.features(x)


class PerceiverLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, queries, kv):
        q = self.norm1(queries)
        kv_normed = self.norm_kv(kv)
        queries = queries + self.cross_attn(q, kv_normed, kv_normed, need_weights=False)[0]
        queries = queries + self.ffn(self.norm2(queries))
        return queries


class PerceiverResampler(nn.Module):
    def __init__(self, n_queries=32, feat_dim=576, d_model=512, n_heads=8, n_layers=2):
        super().__init__()
        self.n_queries = n_queries
        self.d_model = d_model
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.layers = nn.ModuleList([PerceiverLayer(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, spatial_features):
        batch = spatial_features.size(0)
        x = spatial_features.flatten(2).transpose(1, 2)
        x = self.input_proj(x)
        queries = self.queries.expand(batch, -1, -1)
        for layer in self.layers:
            queries = layer(queries, x)
        return self.norm(queries)
