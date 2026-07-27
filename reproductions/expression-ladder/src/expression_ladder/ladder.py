"""Residual-space writers: the rungs below the soft prefix.

The expression ladder asks which mechanisms can change a frozen model's
interpersonal style, and — the point of the study — which ones only appear to.
This module holds the mechanisms that act on residual and attention
activations. The soft-prefix rungs live in ``prefix.py``.

Stages, in the order they were run and in increasing order of how convincing
their teacher-forced numbers are:

* **B** ``fit_static_directions`` — one direction per axis, from
  positive-minus-negative activation differences at the prompt frontier.
* **C** ``screen_doses`` — sweep residual fraction and token window, scored by
  complete-response likelihood.
* **F** ``contextual_direction`` plus multi-site injection under a fixed norm
  budget, distributed as ``total / sqrt(sites)``.
* **G** ``TrainedResidualWriter`` — one antisymmetric direction per axis
  trained directly against paired full-response likelihood.

Every stage here is scored two ways, and the gap between them is the result the
paper reports. Teacher-forced likelihood is measured in the space the writer was
optimized in. Free generation is not.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .data import BASE_SYSTEM_PROMPT, StylePair, style_system_prompt
from .runtime import bounded_context, render_prompt


def decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    """The decoder block list, wherever this architecture keeps it."""
    for path in (("model", "layers"), ("layers",), ("transformer", "h")):
        node: Any = model
        for attribute in path:
            node = getattr(node, attribute, None)
            if node is None:
                break
        if node is not None:
            return node
    raise AttributeError("cannot locate decoder layers on this model")


def hidden_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"cannot find hidden tensor in {type(output)!r}")


def replace_hidden(output: Any, tensor: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    if isinstance(output, list):
        return [tensor, *output[1:]]
    raise TypeError(f"cannot replace hidden tensor in {type(output)!r}")


def component_module(layers: Sequence[nn.Module], *, layer: int, component: str) -> nn.Module:
    decoder = layers[layer]
    if component == "residual":
        return decoder
    if component == "attention":
        attention = getattr(decoder, "self_attn", None)
        if attention is None:
            raise ValueError(f"layer {layer} has no self_attn module")
        return attention
    raise ValueError(f"unknown component {component}")


class ActivationCapture(contextlib.AbstractContextManager):
    """Capture the final prompt-token activation from several modules at once."""

    def __init__(
        self,
        layers: Sequence[nn.Module],
        *,
        layer_indices: Sequence[int],
        components: Sequence[str],
    ) -> None:
        self.layers = layers
        self.layer_indices = list(layer_indices)
        self.components = list(components)
        self.records: dict[tuple[str, int], torch.Tensor] = {}
        self.handles: list[Any] = []

    def _hook(self, key: tuple[str, int]):
        def hook(module: nn.Module, args: Any, output: Any) -> None:
            self.records[key] = hidden_tensor(output)[:, -1, :].detach().float().cpu()
            return None

        return hook

    def clear(self) -> None:
        self.records = {}

    def snapshot(self) -> dict[tuple[str, int], torch.Tensor]:
        expected = {(c, l) for c in self.components for l in self.layer_indices}
        missing = sorted(expected - self.records.keys())
        if missing:
            raise RuntimeError(f"activation capture is missing {missing}")
        return {key: value.clone() for key, value in self.records.items()}

    def __enter__(self):
        for component in self.components:
            for layer in self.layer_indices:
                module = component_module(self.layers, layer=layer, component=component)
                self.handles.append(module.register_forward_hook(self._hook((component, layer))))
        return self

    def __exit__(self, *exc: Any):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return None


@dataclass(frozen=True)
class SteeringSpec:
    """One injection site: where, how hard, which way, for how many tokens."""

    axis: str
    layer: int
    component: str
    fraction: float
    sign: int
    window: int  # 0 means every response position

    @property
    def name(self) -> str:
        window = "all" if self.window == 0 else str(self.window)
        sign = "positive" if self.sign > 0 else "negative"
        return f"{self.axis}-{self.component}-l{self.layer}-f{self.fraction:g}-w{window}-{sign}"


class DirectionSteering(contextlib.AbstractContextManager):
    """Add a normalized direction at response-predicting positions.

    The delta is scaled by the norm of the activation it is added to, so
    ``fraction`` means the same thing at every layer and every position.
    """

    def __init__(
        self,
        module: nn.Module,
        direction: torch.Tensor,
        *,
        fraction: float,
        sign: int,
        window: int,
        scoring_start: int | None = None,
        generation_mode: bool = False,
    ) -> None:
        if fraction < 0.0:
            raise ValueError("fraction must be nonnegative")
        if sign not in {-1, 1}:
            raise ValueError("sign must be -1 or 1")
        if window < 0:
            raise ValueError("window must be nonnegative")
        if scoring_start is None and not generation_mode:
            raise ValueError("a scoring start or generation mode is required")
        norm = torch.linalg.vector_norm(direction.float())
        if not bool(torch.isfinite(norm)) or float(norm) <= 1e-12:
            raise ValueError("steering direction must be finite and nonzero")
        self.module = module
        self.direction = direction.detach().float().cpu() / norm
        self.fraction = float(fraction)
        self.sign = int(sign)
        self.window = int(window)
        self.scoring_start = scoring_start
        self.generation_mode = generation_mode
        self.calls = 0
        self.applied_positions = 0
        self.handle: Any | None = None

    def _replace(self, output: Any, tensor: torch.Tensor, positions: slice | list[int]) -> Any:
        changed = tensor.clone()
        selected = changed[:, positions, :]
        direction = self.direction.to(device=selected.device, dtype=selected.dtype)
        scale = torch.linalg.vector_norm(selected.float(), dim=-1, keepdim=True).to(selected.dtype)
        delta = scale * self.fraction * float(self.sign) * direction.reshape(1, 1, -1)
        changed[:, positions, :] = selected + delta
        self.applied_positions += int(selected.shape[1])
        return replace_hidden(output, changed)

    def _hook(self, module: nn.Module, args: Any, output: Any) -> Any:
        tensor = hidden_tensor(output)
        if self.generation_mode:
            token_index = self.calls
            self.calls += 1
            if self.window and token_index >= self.window:
                return output
            return self._replace(output, tensor, [-1])

        start = int(self.scoring_start or 0)
        if start >= tensor.shape[1]:
            return output
        stop = tensor.shape[1]
        if self.window:
            stop = min(stop, start + self.window)
        if stop <= start:
            return output
        self.calls += 1
        return self._replace(output, tensor, slice(start, stop))

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc: Any):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        return None


def direction_tensor(
    directions: Mapping[str, Any],
    *,
    axis: str,
    component: str,
    layer: int,
) -> torch.Tensor:
    return directions[axis][component][layer]


def steering_context(
    model: nn.Module,
    directions: Mapping[str, Any],
    spec: SteeringSpec | Sequence[SteeringSpec],
    *,
    scoring_start: int | None = None,
    generation_mode: bool = False,
) -> contextlib.ExitStack:
    """Install one or more injection sites for the duration of a block.

    Multi-site injection (Stage F) is just several specs sharing a norm budget;
    see ``distribute_budget``.
    """
    specs = [spec] if isinstance(spec, SteeringSpec) else list(spec)
    if not specs:
        raise ValueError("at least one steering specification is required")
    layers = decoder_layers(model)
    stack = contextlib.ExitStack()
    try:
        for item in specs:
            module = component_module(layers, layer=item.layer, component=item.component)
            stack.enter_context(
                DirectionSteering(
                    module,
                    direction_tensor(
                        directions,
                        axis=item.axis,
                        component=item.component,
                        layer=item.layer,
                    ),
                    fraction=item.fraction,
                    sign=item.sign,
                    window=item.window,
                    scoring_start=scoring_start,
                    generation_mode=generation_mode,
                )
            )
    except BaseException:
        stack.close()
        raise
    return stack


def distribute_budget(total: float, sites: int) -> float:
    """Split a total residual fraction across sites.

    Deltas at different sites are close to orthogonal, so their norms add in
    quadrature rather than linearly; dividing by sqrt(sites) keeps the combined
    perturbation near the single-site budget.
    """
    if sites < 1:
        raise ValueError("at least one site is required")
    return float(total) / (float(sites) ** 0.5)


def collect_prompt_activations(
    model: nn.Module,
    tokenizer: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    capture: ActivationCapture,
    max_seq_len: int,
    device: torch.device,
) -> dict[tuple[str, int], torch.Tensor]:
    prompt = bounded_context(
        render_prompt(tokenizer, system_prompt=system_prompt, user_prompt=user_prompt),
        max_seq_len,
    )
    capture.clear()
    with torch.no_grad():
        model(input_ids=prompt.to(device), use_cache=False)
    records = capture.snapshot()
    del prompt
    return records


def fit_static_directions(
    model: nn.Module,
    tokenizer: Any,
    pairs: Sequence[StylePair],
    *,
    axes: Sequence[str],
    layers: Sequence[int],
    components: Sequence[str],
    max_seq_len: int,
    device: torch.device,
) -> dict[str, Any]:
    """Stage B. One direction per (axis, component, layer).

    For each fit prompt the same user turn is run under a positive style
    instruction, a negative one, and the base prompt. The direction is the mean
    positive-minus-negative activation difference at the prompt frontier; the
    neutral pass supplies the scale that ``DirectionSteering`` will work in.
    """
    by_axis = {axis: [p for p in pairs if p.axis == axis] for axis in axes}
    deltas: dict[str, dict[str, dict[int, list[torch.Tensor]]]] = {
        axis: {c: {l: [] for l in layers} for c in components} for axis in axes
    }
    neutral_norms: dict[str, dict[str, dict[int, list[float]]]] = {
        axis: {c: {l: [] for l in layers} for c in components} for axis in axes
    }

    all_layers = decoder_layers(model)
    with ActivationCapture(all_layers, layer_indices=layers, components=components) as capture:
        for axis in axes:
            for pair in by_axis[axis]:
                shared = dict(
                    capture=capture,
                    max_seq_len=max_seq_len,
                    device=device,
                    user_prompt=pair.prompt,
                )
                positive = collect_prompt_activations(
                    model, tokenizer, system_prompt=style_system_prompt(axis, 1), **shared
                )
                negative = collect_prompt_activations(
                    model, tokenizer, system_prompt=style_system_prompt(axis, -1), **shared
                )
                neutral = collect_prompt_activations(
                    model, tokenizer, system_prompt=BASE_SYSTEM_PROMPT, **shared
                )
                for component in components:
                    for layer in layers:
                        key = (component, layer)
                        deltas[axis][component][layer].append(positive[key] - negative[key])
                        neutral_norms[axis][component][layer].append(
                            float(torch.linalg.vector_norm(neutral[key].float()))
                        )

    directions: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    scales: dict[str, dict[str, dict[int, float]]] = {}
    for axis in axes:
        directions[axis] = {}
        scales[axis] = {}
        for component in components:
            directions[axis][component] = {}
            scales[axis][component] = {}
            for layer in layers:
                stacked = torch.stack(deltas[axis][component][layer]).mean(dim=0).squeeze(0)
                directions[axis][component][layer] = stacked
                norms = neutral_norms[axis][component][layer]
                scales[axis][component][layer] = sum(norms) / len(norms)
    return {"directions": directions, "neutral_scale": scales}


def contextual_direction(
    model: nn.Module,
    tokenizer: Any,
    *,
    axis: str,
    user_prompt: str,
    layers: Sequence[int],
    components: Sequence[str],
    max_seq_len: int,
    device: torch.device,
) -> dict[str, dict[int, torch.Tensor]]:
    """Stage F. The positive-minus-negative difference for one specific prompt.

    Stage B averages this over the fit set. Stage F asks whether that averaging
    is what destroys the effect, by deriving the direction from the prompt
    actually being answered.
    """
    all_layers = decoder_layers(model)
    with ActivationCapture(all_layers, layer_indices=layers, components=components) as capture:
        shared = dict(
            capture=capture, max_seq_len=max_seq_len, device=device, user_prompt=user_prompt
        )
        positive = collect_prompt_activations(
            model, tokenizer, system_prompt=style_system_prompt(axis, 1), **shared
        )
        negative = collect_prompt_activations(
            model, tokenizer, system_prompt=style_system_prompt(axis, -1), **shared
        )
    return {
        component: {
            layer: (positive[(component, layer)] - negative[(component, layer)]).squeeze(0)
            for layer in layers
        }
        for component in components
    }


def direction_geometry(
    directions: Mapping[str, Mapping[str, Mapping[int, torch.Tensor]]],
    *,
    axes: Sequence[str],
    components: Sequence[str],
    layers: Sequence[int],
) -> dict[str, Any]:
    """Cross-axis cosine per site.

    Axes fitted this way are strongly correlated, which is why every stage
    reports a wrong-axis control: a large target effect means little if the
    wrong axis moves the response just as far.
    """
    report: dict[str, Any] = {}
    for component in components:
        for layer in layers:
            matrix = torch.stack([directions[a][component][layer].float() for a in axes])
            normalized = F.normalize(matrix, dim=1)
            cosine = normalized @ normalized.T
            off_diagonal = cosine[~torch.eye(len(axes), dtype=torch.bool)]
            report[f"{component}-layer-{layer}"] = {
                "mean_abs_cross_cosine": float(off_diagonal.abs().mean()),
                "max_abs_cross_cosine": float(off_diagonal.abs().max()),
            }
    return report


class TrainedResidualWriter(nn.Module):
    """Stage G. One antisymmetric direction per axis, trained on paired responses.

    Initialized from the Stage B fit, applied as a bounded signed residual
    fraction at response positions, and optimized so the positive state prefers
    the positive reference response and disprefers its content-matched negative.
    A cross-axis cosine penalty keeps the three axes from collapsing into the
    single style factor the fitted geometry already shows.

    This rung is the one to watch. Its teacher-forced numbers are the best in
    the ladder and its generated text is indistinguishable from no intervention.
    """

    def __init__(
        self,
        init: Mapping[str, torch.Tensor],
        *,
        hidden_size: int,
        fraction: float,
    ) -> None:
        super().__init__()
        self.axes = list(init)
        self.fraction = float(fraction)
        stacked = torch.stack([init[axis].float().reshape(hidden_size) for axis in self.axes])
        self.directions = nn.Parameter(F.normalize(stacked, dim=1))

    def direction_for(self, axis: str) -> torch.Tensor:
        return self.directions[self.axes.index(axis)]

    def cross_axis_penalty(self) -> torch.Tensor:
        """Mean squared off-diagonal cosine across the learned directions."""
        if len(self.axes) < 2:
            return self.directions.new_zeros(())
        normalized = F.normalize(self.directions, dim=1)
        cosine = normalized @ normalized.T
        mask = ~torch.eye(len(self.axes), dtype=torch.bool, device=cosine.device)
        return (cosine[mask] ** 2).mean()

    def geometry(self) -> dict[str, float]:
        with torch.no_grad():
            normalized = F.normalize(self.directions.detach(), dim=1)
            cosine = normalized @ normalized.T
            mask = ~torch.eye(len(self.axes), dtype=torch.bool, device=cosine.device)
            off = cosine[mask].abs()
            return {
                "mean_abs_cross_cosine": float(off.mean()),
                "max_abs_cross_cosine": float(off.max()),
            }


class TrainedResidualSteering(contextlib.AbstractContextManager):
    """Install a ``TrainedResidualWriter`` direction, keeping the graph intact.

    ``DirectionSteering`` detaches its direction because it never needs a
    gradient. This one must not, so it is a separate class rather than a flag.
    """

    def __init__(
        self,
        module: nn.Module,
        direction: torch.Tensor,
        *,
        fraction: float,
        sign: int,
        scoring_start: int,
    ) -> None:
        self.module = module
        self.direction = direction
        self.fraction = float(fraction)
        self.sign = int(sign)
        self.scoring_start = int(scoring_start)
        self.handle: Any | None = None

    def _hook(self, module: nn.Module, args: Any, output: Any) -> Any:
        tensor = hidden_tensor(output)
        start = self.scoring_start
        if start >= tensor.shape[1]:
            return output
        direction = F.normalize(self.direction, dim=0).to(tensor.dtype)
        selected = tensor[:, start:, :]
        scale = torch.linalg.vector_norm(selected.float(), dim=-1, keepdim=True).to(tensor.dtype)
        delta = scale * self.fraction * float(self.sign) * direction.reshape(1, 1, -1)
        changed = torch.cat([tensor[:, :start, :], selected + delta], dim=1)
        return replace_hidden(output, changed)

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc: Any):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        return None
