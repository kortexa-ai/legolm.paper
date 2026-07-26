# Conditional LoRA Bridges

Reproducibility artifact for **"Conditional LoRA Bridges for Modular Sensor
Adaptation of Frozen Small Language Models"** (Franci Penov, kortexa.ai).

Can a frozen language model be conditioned on continuous sensor streams by
*generating* LoRA weights at runtime? This repo contains the complete,
controlled study: a 33.6M-parameter Qwen-style LM, three sensor modalities
(vision, audio, IMU), and a harness where every number in the manuscript is
rendered from a bundled result artifact.

**The headline result is a clean dissociation.** Measured by validation
bits-per-byte, conditioning contributes nothing — a capacity-matched bridge fed
a constant feature vector matches the true-feature bridge to within the
per-seed spread. Measured by
task-aligned probes, conditioning is decisive — true sensor features lift
rank-1 label accuracy to 0.80 on six-way IMU activities (chance 0.17) and 0.33
on fifty-way audio events (chance 0.02), while shuffled, random, and
no-bridge controls all sit at chance. Aggregate language-modeling metrics are
the wrong instrument for detecting sensor conditioning. Diversity
regularization governs the failure mode: unregularized hypernetworks collapse
to input-independent weights, and at long training budgets that collapse
can erase task-level conditioning.

## Qwen 3.6 J-space spectrum study

The repository contains the clean reproduction for **"The Latent Geometry of
Meh."** It fits twelve signed residual-stream read directions from pole-token
logit gradients, then measures a lexically disjoint social atlas under three
system prompts.

Both Qwen 3.6 models place `meh` far from neutral and give it a strong
low-engagement coordinate. The confirmatory 35B MoE run orients all twelve
held-out axes but fails its frozen boredom-adjacency rule: discouragement is
nearest at layer 35. The dense 27B extension places `meh` nearest to boredom.
The two twelve-coordinate centroids have cosine 0.951, while the nearest label
changes across models and residual depth.

- Paper: [`jspace-spectrum-paper/main.pdf`](jspace-spectrum-paper/main.pdf)
- Method, code, tests, raw token measurements, figures, and findings:
  [`reproductions/jspace-spectrum/`](reproductions/jspace-spectrum/)
- Interactive 35B replay:
  [`jspace-spectrum.html`](reproductions/jspace-spectrum/results/qwen36-35b-confirmatory-20260725/jspace-spectrum.html)
- Interactive 27B replay:
  [`jspace-spectrum.html`](reproductions/jspace-spectrum/results/qwen36-27b-extension-20260725/jspace-spectrum.html)

## Qwen 3.6 causal-expression study

This repository also contains the clean reproduction for **"Eight Soft Tokens
Can Change a Frozen Model's Tone."** It freezes `Qwen/Qwen3.6-35B-A3B`, trains
eight continuous input embeddings, and audits complete responses under
positive, negative, regular-model, explicit-instruction, and wrong-axis
conditions.

Across three fresh seeds, the warmth writer reaches 71.4% of the response span
caused by visible style instructions. All 24 held-out comparisons have the
intended sign and beat the wrong-axis shift. The shared-center extension does
not pass: patience stops at 19.5% against a 20% rule, and its nominal neutral
center drifts on every axis.

- Paper: [`causal-expression-paper/main.pdf`](causal-expression-paper/main.pdf)
- Method, code, tests, raw responses, and figures:
  [`reproductions/causal-expression/`](reproductions/causal-expression/)

## Reproduce

```bash
./setup.sh                                   # venv + tokenizer + data caches
uv run paper-reproduce --suite smoke --quick # ~5 min sanity pass
uv run paper-reproduce --suite all --include-audio-task \
  --eval-tokens 32768 --sensor-limit 64 --max-eval-items 64 \
  --output-dir results/myrun                 # full suite, ~25 min on one GPU
uv run paper-figures results/myrun/summary.json --output-dir figures
uv run paper-render-arxiv results/myrun/summary.json --output-dir arxiv
```

Runs on CUDA, Apple Silicon (MPS), or CPU. Training loops use fixed step
counts, so a repeated run with the same seed reproduces each number exactly on
the same hardware class. The exact commands and artifacts behind the
manuscript are recorded in its Reproducibility Statement; the bundled
artifacts live at `arxiv/summary.json` (standard budget) and
`arxiv/summary-scaling.json` (10× budget).

## Repository map

- `arxiv/` — the manuscript (`main.tex`, **generated** by `src/render_arxiv.py`
  — edit the renderer, not the tex), figures, bibliography, result artifacts
- `src/` — experiment runner, renderer, and `METHOD_CONTRACT.md` (what each
  experiment must measure)
- `checkpoints/` — frozen base LM and sensor encoders (Git LFS)
- `data/vision/` — Big Buck Bunny keyframes + captions (see provenance below)
- `figures/` — paper figures, regenerated from the artifact
- `ASSET_CHECKLIST.md` — every asset the runner needs and where it lives

## Data provenance

- **Vision**: 172 keyframes from *Big Buck Bunny*, © 2008 Blender Foundation /
  www.bigbuckbunny.org, CC-BY 3.0 (`data/vision/README.md`); captions
  generated with a locally hosted gemma-4-12b.
- **Audio**: ESC-50 (Piczak, 2015), downloaded by `setup.sh`, not
  redistributed here.
- **IMU**: UCI-HAR (Anguita et al., 2013), downloaded by `setup.sh`, not
  redistributed here.
- **Text**: ClimbMix shards (fixed subset, pinned validation shard),
  downloaded by `setup.sh`.

## Citation

arXiv submission pending; citation entry will be added with the arXiv ID. Code: https://github.com/kortexa-ai/legolm.paper

## License

MIT (see `LICENSE`). Big Buck Bunny frames remain CC-BY 3.0 Blender Foundation.
