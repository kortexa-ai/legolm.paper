# Where is “meh” in J-space?

This directory is a standalone reproduction of a twelve-axis activation
spectrum in Qwen 3.6. It fits targeted residual-stream read directions from
pole-token logit gradients, calibrates them on neutral messages, and measures
held-out social utterances that contain none of the pole words.

The confirmatory model and revision are fixed:

```text
Qwen/Qwen3.6-35B-A3B
995ad96eacd98c81ed38be0c5b274b04031597b0
```

An optional descriptive extension uses the dense
`Qwen/Qwen3.6-27B` revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.

## Install and check

```bash
uv sync
uv run pytest -q
uv run jspace-spectrum-paper check
```

The check verifies the frozen data hash, 12 axes, 144 pole words, 96 held-out
landmarks, 126 atlas utterances, and zero pole-word collisions.

## Small-model plumbing test

```bash
uv run jspace-spectrum-paper reproduce \
  --suite smoke \
  --output-dir results/smoke
```

The smoke suite uses `Qwen/Qwen3.5-2B`. Its measurements only test the runner,
chat-template span detection, gradients, metrics, figures, and HTML replay.
They are not paper evidence.

## Paper run

Run the 35B target on one CUDA GPU with at least 85 GiB free:

```bash
uv run jspace-spectrum-paper reproduce \
  --suite full \
  --model qwen36-35b \
  --output-dir results/qwen36-35b-confirmatory
```

If that completes and the memory gate still passes, run the dense extension:

```bash
uv run jspace-spectrum-paper reproduce \
  --suite full \
  --model qwen36-27b \
  --output-dir results/qwen36-27b-extension
```

Both commands are resumable. The event log saves each completed forward pass,
and the fit-state file saves each completed lens prompt. A resumed command
rejects changes to the model revision, corpus hash, layer choices, or case
inventory.

Compare completed models with:

```bash
uv run jspace-spectrum-paper compare \
  results/qwen36-35b-confirmatory/summary.json \
  results/qwen36-27b-extension/summary.json \
  --output results/model-comparison.json
```

## Artifacts

Each completed model directory contains:

- `lens.pt`: fitted directions and neutral calibration;
- `events.jsonl`: append-only, resumable case records;
- `measurements.json`: every exact user token at every traced layer;
- `summary.json`: metrics, intervals, decisions, and environment;
- `figures/`: four PNG figures made only from `summary.json`;
- `jspace-spectrum.html`: a self-contained token and depth replay.

Regenerate the figures and replay without loading a model:

```bash
uv run jspace-spectrum-paper artifacts \
  results/qwen36-35b-confirmatory/summary.json
```

The model remains frozen and no text is generated. This experiment describes
activation geometry; it does not show persistent memory, response steering,
emotion recognition, or subjective experience.

The preregistered design and decision rules are in [METHOD.md](METHOD.md).
