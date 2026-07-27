"""Continuous prefix parameterizations and deterministic training schedules."""

from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .data import STYLE_AXIS_BY_NAME, StylePair


def repeat_to_length(ids: torch.Tensor, length: int) -> torch.Tensor:
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("seed token IDs must be a non-empty vector")
    if length <= 0:
        raise ValueError("prefix token count must be positive")
    repeats = (length + ids.numel() - 1) // ids.numel()
    return ids.repeat(repeats)[:length]


def seed_embedding_matrix(
    model: nn.Module,
    tokenizer: Any,
    *,
    text: str,
    prefix_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    ids = repeat_to_length(encoded["input_ids"][0].long().cpu(), prefix_tokens)
    ids = ids.to(device)
    with torch.no_grad():
        values = model.get_input_embeddings()(ids)
    return values.detach().float()


def _axis_geometry(delta: torch.Tensor) -> tuple[float, float]:
    matrix = F.normalize(delta.detach().float().flatten(1), dim=1)
    if matrix.shape[0] < 2:
        return 0.0, 0.0
    cosine = matrix @ matrix.T
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    values = cosine[mask].abs()
    return float(values.mean().cpu()), float(values.max().cpu())


def _diversity_loss(delta: torch.Tensor) -> torch.Tensor:
    matrix = F.normalize(delta.flatten(1), dim=1)
    if matrix.shape[0] < 2:
        return matrix.sum() * 0.0
    cosine = matrix @ matrix.T
    mask = ~torch.eye(
        matrix.shape[0],
        dtype=torch.bool,
        device=matrix.device,
    )
    return cosine[mask].square().mean()


class SoftPrefixBank(nn.Module):
    """One axis-specific center plus a signed direction per axis."""

    format_name = "soft-prefix-expression-repro-v1"

    def __init__(
        self,
        axis_names: Sequence[str],
        positive: torch.Tensor,
        negative: torch.Tensor,
        *,
        norm_cap_multiplier: float = 1.5,
    ) -> None:
        super().__init__()
        names = tuple(str(axis) for axis in axis_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("axis names must be non-empty and unique")
        if positive.shape != negative.shape or positive.ndim != 3:
            raise ValueError(
                "positive and negative must share [axis, token, hidden]"
            )
        if positive.shape[0] != len(names):
            raise ValueError("prefix rows must match axis names")
        if norm_cap_multiplier < 1.0:
            raise ValueError("norm cap multiplier must be at least one")
        positive = positive.detach().float()
        negative = negative.detach().float()
        if not bool(torch.isfinite(positive).all()) or not bool(
            torch.isfinite(negative).all()
        ):
            raise ValueError("prefix initialization must be finite")
        center = (positive + negative) / 2.0
        delta = (positive - negative) / 2.0
        self.axis_names = names
        self._axis_slots = {axis: index for index, axis in enumerate(names)}
        self.center = nn.Parameter(center.clone())
        self.delta = nn.Parameter(delta.clone())
        self.register_buffer("initial_center", center.clone())
        self.register_buffer("initial_delta", delta.clone())
        initial_norm = torch.linalg.vector_norm(
            torch.cat([positive, negative], dim=1),
            dim=-1,
        ).max()
        self.max_token_norm = float(initial_norm) * norm_cap_multiplier

    @classmethod
    def initialize(
        cls,
        model: nn.Module,
        tokenizer: Any,
        *,
        axes: Sequence[str],
        prefix_tokens: int,
        device: torch.device,
        norm_cap_multiplier: float = 1.5,
    ) -> "SoftPrefixBank":
        positive = torch.stack(
            [
                seed_embedding_matrix(
                    model,
                    tokenizer,
                    text=STYLE_AXIS_BY_NAME[axis].positive_seed,
                    prefix_tokens=prefix_tokens,
                    device=device,
                )
                for axis in axes
            ]
        )
        negative = torch.stack(
            [
                seed_embedding_matrix(
                    model,
                    tokenizer,
                    text=STYLE_AXIS_BY_NAME[axis].negative_seed,
                    prefix_tokens=prefix_tokens,
                    device=device,
                )
                for axis in axes
            ]
        )
        return cls(
            axes,
            positive,
            negative,
            norm_cap_multiplier=norm_cap_multiplier,
        ).to(device=device, dtype=torch.float32)

    @property
    def prefix_tokens(self) -> int:
        return int(self.center.shape[1])

    @property
    def hidden_size(self) -> int:
        return int(self.center.shape[2])

    def prefix(
        self,
        axis: str,
        sign: int,
        *,
        strength: float = 1.0,
    ) -> torch.Tensor:
        if sign not in {-1, 1} or strength < 0.0:
            raise ValueError("sign must be -1 or 1 and strength nonnegative")
        try:
            slot = self._axis_slots[axis]
        except KeyError as error:
            raise ValueError(f"unknown axis {axis}") from error
        return self.center[slot] + sign * float(strength) * self.delta[slot]

    def interpolated_prefix(
        self,
        axis: str,
        state: float,
        *,
        include_center: bool = True,
    ) -> torch.Tensor:
        if not math.isfinite(state):
            raise ValueError("state must be finite")
        try:
            slot = self._axis_slots[axis]
        except KeyError as error:
            raise ValueError(f"unknown axis {axis}") from error
        center: torch.Tensor | float = self.center[slot] if include_center else 0.0
        return center + float(state) * self.delta[slot]

    def diversity_loss(self) -> torch.Tensor:
        return _diversity_loss(self.delta)

    def anchor_loss(self) -> torch.Tensor:
        numerator = (
            (self.center - self.initial_center).square().mean()
            + (self.delta - self.initial_delta).square().mean()
        )
        denominator = (
            self.initial_center.square().mean()
            + self.initial_delta.square().mean()
        ).clamp_min(1e-8)
        return numerator / denominator

    @torch.no_grad()
    def project_norms_(self) -> None:
        for parameter in (self.center, self.delta):
            norms = torch.linalg.vector_norm(
                parameter,
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
            parameter.mul_(torch.clamp(self.max_token_norm / norms, max=1.0))

    def geometry(self) -> dict[str, float]:
        mean_cosine, max_cosine = _axis_geometry(self.delta)
        pole_norms = torch.stack(
            [
                torch.linalg.vector_norm(
                    self.center.detach().float() + self.delta.detach().float(),
                    dim=-1,
                ),
                torch.linalg.vector_norm(
                    self.center.detach().float() - self.delta.detach().float(),
                    dim=-1,
                ),
            ]
        )
        return {
            "mean_abs_axis_cosine": mean_cosine,
            "max_abs_axis_cosine": max_cosine,
            "mean_pole_token_norm": float(pole_norms.mean().cpu()),
            "max_pole_token_norm": float(pole_norms.max().cpu()),
        }

    def payload(
        self,
        *,
        model_id: str,
        model_revision: str,
        seed: int,
        history: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "format": self.format_name,
            "model_id": model_id,
            "model_revision": model_revision,
            "seed": seed,
            "axes": list(self.axis_names),
            "prefix_tokens": self.prefix_tokens,
            "hidden_size": self.hidden_size,
            "center": self.center.detach().float().cpu(),
            "delta": self.delta.detach().float().cpu(),
            "initial_center": self.initial_center.detach().float().cpu(),
            "initial_delta": self.initial_delta.detach().float().cpu(),
            "max_token_norm": self.max_token_norm,
            "geometry": self.geometry(),
            "training_history": list(history),
        }


class NeutralAnchoredPrefixBank(nn.Module):
    """One always-present center plus additive signed axis directions."""

    format_name = "neutral-anchored-prefix-expression-repro-v1"

    def __init__(
        self,
        axis_names: Sequence[str],
        neutral: torch.Tensor,
        delta: torch.Tensor,
        *,
        max_token_norm: float,
    ) -> None:
        super().__init__()
        names = tuple(str(axis) for axis in axis_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("axis names must be non-empty and unique")
        if neutral.ndim != 2 or delta.ndim != 3:
            raise ValueError("neutral and delta have invalid ranks")
        if delta.shape[0] != len(names) or delta.shape[1:] != neutral.shape:
            raise ValueError("neutral and delta shapes do not match")
        if max_token_norm <= 0.0:
            raise ValueError("maximum token norm must be positive")
        neutral = neutral.detach().float()
        delta = delta.detach().float()
        if not bool(torch.isfinite(neutral).all()) or not bool(
            torch.isfinite(delta).all()
        ):
            raise ValueError("prefix initialization must be finite")
        self.axis_names = names
        self._axis_slots = {axis: index for index, axis in enumerate(names)}
        self.neutral = nn.Parameter(neutral.clone())
        self.delta = nn.Parameter(delta.clone())
        self.register_buffer("initial_neutral", neutral.clone())
        self.register_buffer("initial_delta", delta.clone())
        self.max_token_norm = float(max_token_norm)

    @classmethod
    def from_stage_h(cls, stage_h: SoftPrefixBank) -> "NeutralAnchoredPrefixBank":
        return cls(
            stage_h.axis_names,
            stage_h.center.detach().float().mean(dim=0),
            stage_h.delta.detach().float(),
            max_token_norm=stage_h.max_token_norm,
        ).to(device=stage_h.center.device, dtype=torch.float32)

    @property
    def prefix_tokens(self) -> int:
        return int(self.neutral.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.neutral.shape[1])

    def neutral_prefix(self) -> torch.Tensor:
        return self.neutral

    def prefix(
        self,
        axis: str,
        sign: int,
        *,
        strength: float = 1.0,
    ) -> torch.Tensor:
        if sign not in {-1, 1} or strength < 0.0:
            raise ValueError("sign must be -1 or 1 and strength nonnegative")
        try:
            slot = self._axis_slots[axis]
        except KeyError as error:
            raise ValueError(f"unknown axis {axis}") from error
        return self.neutral + sign * float(strength) * self.delta[slot]

    def diversity_loss(self) -> torch.Tensor:
        return _diversity_loss(self.delta)

    def anchor_loss(self) -> torch.Tensor:
        numerator = (
            (self.neutral - self.initial_neutral).square().mean()
            + (self.delta - self.initial_delta).square().mean()
        )
        denominator = (
            self.initial_neutral.square().mean()
            + self.initial_delta.square().mean()
        ).clamp_min(1e-8)
        return numerator / denominator

    @torch.no_grad()
    def project_norms_(self) -> None:
        for parameter in (self.neutral, self.delta):
            norms = torch.linalg.vector_norm(
                parameter,
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
            parameter.mul_(torch.clamp(self.max_token_norm / norms, max=1.0))

    def geometry(self) -> dict[str, float]:
        mean_cosine, max_cosine = _axis_geometry(self.delta)
        neutral_norms = torch.linalg.vector_norm(
            self.neutral.detach().float(),
            dim=-1,
        )
        pole_norms = torch.stack(
            [
                torch.linalg.vector_norm(
                    self.neutral.detach().float() + self.delta.detach().float(),
                    dim=-1,
                ),
                torch.linalg.vector_norm(
                    self.neutral.detach().float() - self.delta.detach().float(),
                    dim=-1,
                ),
            ]
        )
        return {
            "mean_abs_axis_cosine": mean_cosine,
            "max_abs_axis_cosine": max_cosine,
            "mean_neutral_token_norm": float(neutral_norms.mean().cpu()),
            "max_neutral_token_norm": float(neutral_norms.max().cpu()),
            "mean_pole_token_norm": float(pole_norms.mean().cpu()),
            "max_pole_token_norm": float(pole_norms.max().cpu()),
        }

    def payload(
        self,
        *,
        model_id: str,
        model_revision: str,
        seed: int,
        history: Sequence[Mapping[str, Any]],
        source_stage_h: str,
    ) -> dict[str, Any]:
        return {
            "format": self.format_name,
            "model_id": model_id,
            "model_revision": model_revision,
            "seed": seed,
            "source_stage_h": source_stage_h,
            "axes": list(self.axis_names),
            "prefix_tokens": self.prefix_tokens,
            "hidden_size": self.hidden_size,
            "neutral": self.neutral.detach().float().cpu(),
            "delta": self.delta.detach().float().cpu(),
            "initial_neutral": self.initial_neutral.detach().float().cpu(),
            "initial_delta": self.initial_delta.detach().float().cpu(),
            "max_token_norm": self.max_token_norm,
            "geometry": self.geometry(),
            "training_history": list(history),
        }


def insert_prefix_embeddings(
    token_embeddings: torch.Tensor,
    prefix: torch.Tensor,
    *,
    insertion: int,
) -> torch.Tensor:
    if token_embeddings.ndim != 3 or prefix.ndim != 2:
        raise ValueError("token embeddings and prefix must have ranks 3 and 2")
    if token_embeddings.shape[0] != 1:
        raise ValueError("the reproduction uses batch size one")
    if prefix.shape[-1] != token_embeddings.shape[-1]:
        raise ValueError("prefix hidden width does not match token embeddings")
    if not 0 <= insertion <= token_embeddings.shape[1]:
        raise ValueError("prefix insertion is outside the sequence")
    typed = prefix.to(
        device=token_embeddings.device,
        dtype=token_embeddings.dtype,
    ).unsqueeze(0)
    return torch.cat(
        (
            token_embeddings[:, :insertion],
            typed,
            token_embeddings[:, insertion:],
        ),
        dim=1,
    )


def pole_training_schedule(
    pairs: Sequence[StylePair],
    *,
    steps: int,
    seed: int,
) -> list[tuple[StylePair, int]]:
    if steps <= 0:
        raise ValueError("training steps must be positive")
    base = [(pair, sign) for pair in pairs for sign in (1, -1)]
    if not base:
        raise ValueError("training pairs are empty")
    randomizer = random.Random(seed)
    schedule: list[tuple[StylePair, int]] = []
    while len(schedule) < steps:
        epoch = list(base)
        randomizer.shuffle(epoch)
        schedule.extend(epoch)
    return schedule[:steps]


def neutral_training_schedule(
    pairs: Sequence[StylePair],
    *,
    pole_steps: int,
    neutral_steps: int,
    seed: int,
) -> list[tuple[str, StylePair, int]]:
    if pole_steps < 0 or neutral_steps < 0 or pole_steps + neutral_steps <= 0:
        raise ValueError("invalid Stage I step counts")
    events = [
        ("pole", pair, sign)
        for pair, sign in pole_training_schedule(
            pairs,
            steps=pole_steps,
            seed=seed,
        )
    ]
    randomizer = random.Random(seed + 1)
    neutral_pairs: list[StylePair] = []
    while len(neutral_pairs) < neutral_steps:
        epoch = list(pairs)
        randomizer.shuffle(epoch)
        neutral_pairs.extend(epoch)
    events.extend(
        ("neutral", pair, 0)
        for pair in neutral_pairs[:neutral_steps]
    )
    randomizer.shuffle(events)
    return events
