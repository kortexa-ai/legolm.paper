"""Stage orchestration for the expression ladder.

Each stage produces the same two measurements, and the paper's result is the
distance between them:

* a **teacher-forced span** — how much the intervention shifts the likelihood
  of curated positive-versus-negative reference responses, measured in the
  space the writer was optimized in;
* a **generated span** — the same contrast recomputed on text the model
  actually produced under greedy decoding, plus a wrong-axis control.

Stages that pass the first and fail the second are the point of the study.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .data import BASE_SYSTEM_PROMPT, StylePair, style_system_prompt
from .ladder import (
    SteeringSpec,
    TrainedResidualSteering,
    TrainedResidualWriter,
    contextual_direction,
    decoder_layers,
    direction_geometry,
    distribute_budget,
    fit_static_directions,
    steering_context,
)
from .runtime import (
    assistant_header_start,
    emit,
    generate_response,
    render_prompt,
    response_attribution,
    response_nll,
)


@dataclass(frozen=True)
class StageConfig:
    axes: tuple[str, ...]
    layers: tuple[int, ...]
    components: tuple[str, ...]
    fractions: tuple[float, ...]
    windows: tuple[int, ...]
    max_seq_len: int
    max_new_tokens: int
    max_context_tokens: int


def _paired_margin(
    model: nn.Module,
    tokenizer: Any,
    pair: StylePair,
    *,
    max_seq_len: int,
) -> float:
    """Positive-minus-negative reference NLL under whatever hooks are installed.

    Lower NLL on the positive reference and higher on its content-matched
    negative both push this up, so it is the teacher-forced quantity every
    stage below is scored on.
    """
    positive = response_nll(
        model,
        tokenizer,
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=pair.prompt,
        response=pair.positive_response,
        max_seq_len=max_seq_len,
    )
    negative = response_nll(
        model,
        tokenizer,
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=pair.prompt,
        response=pair.negative_response,
        max_seq_len=max_seq_len,
    )
    return float(negative) - float(positive)


def stage_a_prompt_upper_bound(
    model: nn.Module,
    tokenizer: Any,
    pairs: Sequence[StylePair],
    config: StageConfig,
) -> dict[str, Any]:
    """Stage A. What visible style instructions achieve, as the denominator.

    Every later stage reports its span as a fraction of this. Without it a raw
    likelihood shift is uninterpretable.
    """
    by_axis: dict[str, list[float]] = {axis: [] for axis in config.axes}
    for pair in pairs:
        if pair.axis not in by_axis:
            continue
        positive = response_nll(
            model,
            tokenizer,
            system_prompt=style_system_prompt(pair.axis, 1),
            user_prompt=pair.prompt,
            response=pair.positive_response,
            max_seq_len=config.max_seq_len,
        )
        negative = response_nll(
            model,
            tokenizer,
            system_prompt=style_system_prompt(pair.axis, -1),
            user_prompt=pair.prompt,
            response=pair.positive_response,
            max_seq_len=config.max_seq_len,
        )
        by_axis[pair.axis].append(float(negative) - float(positive))
    spans = {axis: sum(v) / len(v) for axis, v in by_axis.items() if v}
    emit("stage_a_complete", spans=spans)
    return {"stage": "A", "explicit_span": spans, "n_pairs": len(pairs)}


def stage_b_fit(
    model: nn.Module,
    tokenizer: Any,
    fit_pairs: Sequence[StylePair],
    config: StageConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Stage B. Static directions, plus the geometry that motivates the controls."""
    fitted = fit_static_directions(
        model,
        tokenizer,
        fit_pairs,
        axes=config.axes,
        layers=config.layers,
        components=config.components,
        max_seq_len=config.max_seq_len,
        device=device,
    )
    geometry = direction_geometry(
        fitted["directions"],
        axes=config.axes,
        components=config.components,
        layers=config.layers,
    )
    emit("stage_b_complete", sites=len(geometry))
    return {"stage": "B", "geometry": geometry, "fitted": fitted}


def stage_c_dose_screen(
    model: nn.Module,
    tokenizer: Any,
    directions: Mapping[str, Any],
    heldout_pairs: Sequence[StylePair],
    config: StageConfig,
) -> dict[str, Any]:
    """Stage C. Sweep dose and window, scored teacher-forced.

    This is where a writer first looks like it is working. The rows with the
    largest spans here are the ones Stage D then fails to reproduce in text.
    """
    rows: list[dict[str, Any]] = []
    for axis in config.axes:
        axis_pairs = [p for p in heldout_pairs if p.axis == axis]
        for component in config.components:
            for layer in config.layers:
                for fraction in config.fractions:
                    for window in config.windows:
                        deltas = []
                        for pair in axis_pairs:
                            start = assistant_header_start(
                                tokenizer,
                                system_prompt=BASE_SYSTEM_PROMPT,
                                user_prompt=pair.prompt,
                            )
                            specs = [
                                SteeringSpec(axis, layer, component, fraction, sign, window)
                                for sign in (1, -1)
                            ]
                            scored = {}
                            for spec in specs:
                                with steering_context(
                                    model, directions, spec, scoring_start=start
                                ):
                                    scored[spec.sign] = _paired_margin(
                                        model, tokenizer, pair, max_seq_len=config.max_seq_len
                                    )
                            deltas.append(scored[1] - scored[-1])
                        if not deltas:
                            continue
                        rows.append(
                            {
                                "axis": axis,
                                "component": component,
                                "layer": layer,
                                "fraction": fraction,
                                "window": window,
                                "span": sum(deltas) / len(deltas),
                                "n": len(deltas),
                                "directional_success": sum(1 for d in deltas if d > 0) / len(deltas),
                            }
                        )
    best = {}
    for axis in config.axes:
        axis_rows = [r for r in rows if r["axis"] == axis]
        if axis_rows:
            best[axis] = max(axis_rows, key=lambda r: r["span"])
    emit("stage_c_complete", rows=len(rows))
    return {"stage": "C", "rows": rows, "best": best}


def _audit_one(
    model: nn.Module,
    tokenizer: Any,
    pair: StylePair,
    axis: str,
    config: StageConfig,
    *,
    make_context,
) -> dict[str, Any]:
    """Generate under each sign and score the produced text, not the reference."""
    out: dict[str, Any] = {}
    for label, sign in (("positive", 1), ("negative", -1)):
        with make_context(sign):
            text = generate_response(
                model,
                tokenizer,
                system_prompt=BASE_SYSTEM_PROMPT,
                user_prompt=pair.prompt,
                prefix=None,
                max_new_tokens=config.max_new_tokens,
                max_context_tokens=config.max_context_tokens,
            )
        attribution = response_attribution(
            model,
            tokenizer,
            axis=axis,
            prompt=pair.prompt,
            response=text,
            max_seq_len=config.max_seq_len,
        )
        out[label] = {"text": text, "attribution": float(attribution["signed_attribution"])}
    out["span"] = out["positive"]["attribution"] - out["negative"]["attribution"]
    return out


def stage_d_generation_audit(
    model: nn.Module,
    tokenizer: Any,
    directions: Mapping[str, Any],
    best: Mapping[str, Mapping[str, Any]],
    heldout_pairs: Sequence[StylePair],
    config: StageConfig,
    *,
    stage: str = "D",
) -> dict[str, Any]:
    """Stage D. The decisive measurement: free generation with a wrong-axis control.

    A stage passes only if its target span beats the shift caused by steering
    along a different axis. The fitted axes are strongly correlated, so a large
    target span on its own proves nothing.
    """
    records: list[dict[str, Any]] = []
    axes = list(config.axes)
    for axis in axes:
        if axis not in best:
            continue
        row = best[axis]
        wrong_axis = axes[(axes.index(axis) + 1) % len(axes)]
        for pair in (p for p in heldout_pairs if p.axis == axis):

            def make_context(sign: int, _row=row, _axis=axis):
                return steering_context(
                    model,
                    directions,
                    SteeringSpec(
                        _axis, _row["layer"], _row["component"], _row["fraction"], sign, _row["window"]
                    ),
                    generation_mode=True,
                )

            def make_wrong(sign: int, _row=row, _axis=wrong_axis):
                return steering_context(
                    model,
                    directions,
                    SteeringSpec(
                        _axis, _row["layer"], _row["component"], _row["fraction"], sign, _row["window"]
                    ),
                    generation_mode=True,
                )

            target = _audit_one(model, tokenizer, pair, axis, config, make_context=make_context)
            control = _audit_one(model, tokenizer, pair, axis, config, make_context=make_wrong)
            records.append(
                {
                    "axis": axis,
                    "prompt": pair.prompt,
                    "target_span": target["span"],
                    "wrong_axis_span": control["span"],
                    "specific": abs(target["span"]) > abs(control["span"]),
                    "signed": target["span"] > 0,
                    "positive_text": target["positive"]["text"],
                    "negative_text": target["negative"]["text"],
                }
            )
    summary = {}
    for axis in axes:
        rows = [r for r in records if r["axis"] == axis]
        if rows:
            summary[axis] = {
                "generated_span": sum(r["target_span"] for r in rows) / len(rows),
                "signed": sum(1 for r in rows if r["signed"]),
                "specific": sum(1 for r in rows if r["specific"]),
                "n": len(rows),
            }
    emit(f"stage_{stage.lower()}_complete", axes=len(summary))
    return {"stage": stage, "records": records, "summary": summary}


def stage_f_contextual(
    model: nn.Module,
    tokenizer: Any,
    heldout_pairs: Sequence[StylePair],
    config: StageConfig,
    device: torch.device,
    *,
    total_fraction: float,
) -> dict[str, Any]:
    """Stage F. Per-prompt directions injected coherently across several sites.

    Tests whether Stage B's failure comes from averaging directions over the fit
    set, or from injecting at only one site. The norm budget is held fixed and
    split across sites so the comparison is not just a larger perturbation.
    """
    sites = [(c, l) for c in config.components for l in config.layers]
    per_site = distribute_budget(total_fraction, len(sites))
    records = []
    for axis in config.axes:
        for pair in (p for p in heldout_pairs if p.axis == axis):
            local = contextual_direction(
                model,
                tokenizer,
                axis=axis,
                user_prompt=pair.prompt,
                layers=config.layers,
                components=config.components,
                max_seq_len=config.max_seq_len,
                device=device,
            )
            directions = {axis: local}
            specs_for = lambda sign: [
                SteeringSpec(axis, layer, component, per_site, sign, 0)
                for component, layer in sites
            ]
            start = assistant_header_start(
                tokenizer, system_prompt=BASE_SYSTEM_PROMPT, user_prompt=pair.prompt
            )
            teacher = {}
            for sign in (1, -1):
                with steering_context(model, directions, specs_for(sign), scoring_start=start):
                    teacher[sign] = _paired_margin(
                        model, tokenizer, pair, max_seq_len=config.max_seq_len
                    )

            def make_context(sign: int, _specs=specs_for, _d=directions):
                return steering_context(model, _d, _specs(sign), generation_mode=True)

            generated = _audit_one(model, tokenizer, pair, axis, config, make_context=make_context)
            records.append(
                {
                    "axis": axis,
                    "prompt": pair.prompt,
                    "teacher_span": teacher[1] - teacher[-1],
                    "generated_span": generated["span"],
                    "positive_text": generated["positive"]["text"],
                    "negative_text": generated["negative"]["text"],
                }
            )
    emit("stage_f_complete", records=len(records))
    return {"stage": "F", "records": records, "per_site_fraction": per_site, "sites": len(sites)}


def stage_g_train_residual(
    model: nn.Module,
    tokenizer: Any,
    fit_pairs: Sequence[StylePair],
    init_directions: Mapping[str, torch.Tensor],
    config: StageConfig,
    *,
    layer: int,
    fraction: float,
    steps: int,
    lr: float,
    cross_axis_weight: float,
) -> dict[str, Any]:
    """Stage G. Train the residual direction directly on paired responses.

    The optimizer sees exactly the teacher-forced objective the stage is scored
    on, and it does extremely well at it. Whether that transfers to generated
    text is the question Stage D answers, and the answer is the paper.
    """
    hidden = int(next(model.parameters()).shape[-1])
    writer = TrainedResidualWriter(init_directions, hidden_size=hidden, fraction=fraction)
    writer.to(next(model.parameters()).device)
    optimizer = torch.optim.AdamW(writer.parameters(), lr=lr)
    layers = decoder_layers(model)
    module = layers[layer]

    history = []
    pairs = [p for p in fit_pairs if p.axis in config.axes]
    for step in range(steps):
        pair = pairs[step % len(pairs)]
        start = assistant_header_start(
            tokenizer, system_prompt=BASE_SYSTEM_PROMPT, user_prompt=pair.prompt
        )
        loss = torch.zeros((), device=writer.directions.device)
        for sign, target_sign in ((1, 1), (-1, -1)):
            with TrainedResidualSteering(
                module,
                writer.direction_for(pair.axis),
                fraction=fraction,
                sign=sign,
                scoring_start=start,
            ):
                wanted = response_nll(
                    model,
                    tokenizer,
                    system_prompt=BASE_SYSTEM_PROMPT,
                    user_prompt=pair.prompt,
                    response=(pair.positive_response if target_sign > 0 else pair.negative_response),
                    max_seq_len=config.max_seq_len,
                )
                opposite = response_nll(
                    model,
                    tokenizer,
                    system_prompt=BASE_SYSTEM_PROMPT,
                    user_prompt=pair.prompt,
                    response=(pair.negative_response if target_sign > 0 else pair.positive_response),
                    max_seq_len=config.max_seq_len,
                )
            loss = loss + wanted - opposite
        loss = loss + cross_axis_weight * writer.cross_axis_penalty()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(writer.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.detach()))
    emit("stage_g_trained", steps=steps, final_loss=history[-1] if history else None)
    return {"stage": "G", "writer": writer, "loss_history": history, "geometry": writer.geometry()}
