# LegoLM research reproductions

Each study has its own reproduction commands and environment. All studies stay
on `main`; paper-specific directories separate their code and artifacts.

| Study | Reproduction | Paper |
| --- | --- | --- |
| The Lookout and the Captain | [`reproductions/lookout-captain/`](reproductions/lookout-captain/) | [`PDF`](lookout-captain-paper/main.pdf) |
| Paper 1: conditional LoRA bridges | [`paper1/`](paper1/) | [`Paper 1 manuscript`](paper1/manuscript/main.pdf) |
| Historical 230M companion to Paper 1 | [`Historical command and interpretive erratum`](paper1/REVISION_NOTES.md#impact-on-the-historical-230m-companion-note) | [`PDF`](note/main.pdf) |
| The Latent Geometry of Meh | [`reproductions/jspace-spectrum/`](reproductions/jspace-spectrum/) | [`PDF`](jspace-spectrum-paper/main.pdf) |
| Eight Soft Tokens Can Change a Frozen Model's Tone | [`reproductions/causal-expression/`](reproductions/causal-expression/) | [`PDF`](causal-expression-paper/main.pdf) |
| Five Ways to Fool Yourself About Steering a Frozen Model | [`reproductions/expression-ladder/`](reproductions/expression-ladder/) | [`PDF`](expression-ladder-paper/main.pdf) |

Paper 1 studies sensor-conditioned LoRA weights on frozen language models. Its
revised protocol separates unpaired text adaptation, paired label ranking, and
sensor-plus-instruction generation. It uses the original mini host, LFM2.5-230M,
and Qwen3.5-0.8B. Start with the [Paper 1 guide](paper1/README.md) and
[method contract](paper1/METHOD_CONTRACT.md).
The [revision notes](paper1/REVISION_NOTES.md) identify corrected comparisons,
encoder-provenance limits, and the effect on the historical 230M companion.

The historical root harness and `arxiv/` remain available for the original
33.6M release. An improvement in independently sampled text likelihood alone
does not establish sensor use. The revision measures conditioning with matched
checkpoint evaluations and explicit sensor interventions.

## The Lookout and the Captain

This study separates mode conditioning from event-triggered model invocation
on frozen vision-language hosts of 450M, 3B, and 27B parameters. A mode packet
changes selection among greet, stop, and proceed. A person detector selects
the mode and controls when the host runs. At 27B, verdict gating avoids 83.54%
of calls in the recorded replay while matching every-frame actions on 98.06%
of social and 99.03% of caution steps. The paper also reports action errors,
unresolved invocation delays, and failures across phrases and visual domains.

The [reproduction guide](reproductions/lookout-captain/README.md) provides a
CPU rebuild from 232 saved result records and the complete model-experiment
suite. The [revision notes](reproductions/lookout-captain/REVISION_NOTES.md)
explain corrected comparisons, delay accounting, and provenance limits.

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

## Qwen 3.6 expression ladder study

The repository also contains the clean reproduction for **"Five Ways to Fool
Yourself About Steering a Frozen Model."** It builds seven writers for the same
three social style axes on the same frozen model and scores each one both on
reference likelihood and on text the model actually generates. The two rankings
disagree, and the disagreement is the result.

The trained residual writer drives its own objective from +0.366 to −1.188 and
then moves generated text by 5.1% of the explicit-prompt span on warmth and
−3.7% on patience, where the sign reverses. Stage J, the one rung whose decision
rules were frozen before it ran, distills an always-present soft prefix's zero
state toward the frozen no-prefix model: the center goes nearly silent (drift
1.1% and 0.4% against a 10% rule) and the signed channel collapses with it
(warmth 17.2% against a 40% rule). Its preregistered verdict is a valid FAIL.
Across the three soft-prefix rungs the two quantities trade off monotonically,
which bounds the family.

- Paper: [`expression-ladder-paper/main.pdf`](expression-ladder-paper/main.pdf)
- Preregistration, code, tests, and result artifacts:
  [`reproductions/expression-ladder/`](reproductions/expression-ladder/)

## Historical Paper 1 release

These are the original runner's entry points. Use `paper1/` for the revised
three-host protocol and new evaluations. Its input bootstrap supplies the exact
frozen mini tokenizer; the legacy bootstrap can train a new tokenizer when its
cache is absent, which is not a substitute for the frozen checkpoint's tokenizer.

```bash
./setup.sh                                   # historical setup; see tokenizer note above
uv run paper-reproduce --suite smoke --quick # historical sanity pass
uv run paper-reproduce --suite all --include-audio-task \
  --eval-tokens 32768 --sensor-limit 64 --max-eval-items 64 \
  --output-dir results/myrun                 # historical full suite
uv run paper-figures results/myrun/summary.json --output-dir figures
uv run paper-render-arxiv results/myrun/summary.json --output-dir results/myrun/manuscript
```

Runs on CUDA, Apple Silicon (MPS), or CPU. Training loops use fixed step
counts. Numerical repeatability also depends on software, data and backend;
the revised package records those identities explicitly. The commands and artifacts behind the
manuscript are recorded in its Reproducibility Statement; the bundled
artifacts live at `arxiv/summary.json` (standard budget) and
`arxiv/summary-scaling.json` (10× budget).

For an artifact-only rebuild of the historical paper, supply both preserved
result files and copy its bibliography into the new output directory first:

```bash
mkdir -p paper1/runs/historical-paper1
cp arxiv/references.bib paper1/runs/historical-paper1/references.bib
uv run --project paper1 --frozen python -m src.render_arxiv arxiv/summary.json \
  --scaling-summary arxiv/summary-scaling.json \
  --output-dir paper1/runs/historical-paper1 --figures-dir arxiv/figures --compile \
  --package paper1/runs/historical-paper1-source.zip --verify-package
```

The old renderer does not copy the bibliography automatically. The revised
renderer includes it and validates the source archive in a fresh directory.

## Historical harness and shared assets

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
