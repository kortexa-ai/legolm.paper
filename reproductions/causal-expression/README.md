# Causal expression in Qwen 3.6

This directory reproduces the oracle-state expression experiment from LegoLM
without importing LegoLM. It trains eight continuous prefix tokens while
keeping `Qwen/Qwen3.6-35B-A3B` frozen, then tests whether signed social
coordinates change complete generated responses.

The target model and revision are fixed:

```text
Qwen/Qwen3.6-35B-A3B
995ad96eacd98c81ed38be0c5b274b04031597b0
```

The experiment covers warmth versus cold hostility, patience versus annoyance,
and goodwill versus resentment. The corpus has six fit cases, two development
cases, and eight held-out cases. Held-out prompts include repeated demands, a
direct insult, praise, and gratitude.

## Commands

Install and check the CPU-only parts:

```bash
uv sync
uv run pytest -q
uv run causal-expression check
```

Run a small-model plumbing test:

```bash
uv run causal-expression reproduce \
  --suite smoke \
  --output-dir results/smoke
```

Run the paper experiment on one CUDA GPU with at least 85 GiB free:

```bash
uv run causal-expression reproduce \
  --suite full \
  --seeds 20260724,20260725,20260726 \
  --output-dir results/qwen36-35b-confirmatory
```

The full command trains fresh Stage H and Stage I prefix banks for every seed.
It writes checkpoints, teacher-forced screens, all generated responses, warmth
strength sweeps, a consolidated `summary.json`, and figures. It never reads an
old writer checkpoint.

The smoke suite uses `Qwen/Qwen3.5-2B` only to catch broken tensor shapes,
chat-template changes, and gradient problems. Smoke results are not evidence
for the paper.

## Hardware

The reference run uses bfloat16 weights on an RTX Pro 6000 with 96 GB VRAM.
The model streams directly to CUDA. The runner refuses the 35B load when less
than 85 GiB is free and stops training if CUDA headroom falls below 20 GiB.

## Scope

This reproduces a causal expression writer with an oracle social state. It does
not include a recurrent session reader, temporal decay, LegoLM services, MISO,
or a claim about subjective experience.

See [METHOD.md](METHOD.md) for the frozen design and decision rules.

## Reference result

The three-seed reference run is bundled under
`results/qwen36-35b-confirmatory-20260725`. Stage H warmth passed its frozen
rule with a 71.4% median relative generated span and 24/24 successful sign and
specificity comparisons. Stage I patience stopped at 19.5%, just below its 20%
rule, and the shared neutral center drifted too far on every axis.

See [FINDINGS.md](FINDINGS.md) for the response-level findings and artifact
map.

Paper: [Eight Soft Tokens Can Change a Frozen Model's Tone](../../causal-expression-paper/main.pdf).
Public code and data:
[kortexa-ai/legolm.paper](https://github.com/kortexa-ai/legolm.paper/tree/main/reproductions/causal-expression).
