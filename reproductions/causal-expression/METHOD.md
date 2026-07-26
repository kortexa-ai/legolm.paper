# Method contract

**Frozen before the confirmatory run:** 2026-07-25

## Question

Can a small continuous prefix, selected by an oracle signed social coordinate,
change the interpersonal style of complete responses from a frozen
`Qwen/Qwen3.6-35B-A3B` model?

The experiment tests expression. It does not test how a session state is read,
updated, or decayed.

## Model

- Model: `Qwen/Qwen3.6-35B-A3B`
- Revision: `995ad96eacd98c81ed38be0c5b274b04031597b0`
- Dtype: bfloat16 on CUDA
- Attention: PyTorch SDPA
- Thinking mode: disabled in the chat template
- Decoding: greedy, at most 96 new tokens
- Model parameters: frozen

The same model scores every completed response under content-matched positive
and negative style instructions. The score is:

`negative-prompt NLL - positive-prompt NLL`

Positive values favor the positive pole.

## Data

The synthetic corpus contains 16 technical requests and three signed axes:

- warmth versus cold hostility;
- patience versus annoyance;
- goodwill versus resentment.

For each request and axis, positive and negative references contain the exact
same technical core. Six requests form the fit split, two the development
split, and eight the held-out split.

## Stage H

Stage H learns one axis-specific prefix:

`prefix(axis, sign) = center[axis] + sign × delta[axis]`

Each prefix has eight continuous tokens. Positive and negative endpoints start
from embeddings of short seed phrases. Training uses all 18 fit pairs and both
signs once, for 36 shuffled steps. At each step the selected prefix lowers NLL
for the desired response and raises NLL for its content-matched opposite.

Fixed settings:

- learning rate `0.001`;
- AdamW with zero weight decay;
- gradient norm cap `1.0`;
- direction-diversity weight `0.01`;
- initialization-anchor weight `0.001`;
- token-norm cap `1.5 ×` the largest initial endpoint norm.

The prefix is inserted immediately before the assistant role header. Placing it
after the header changes the meaning of the seed embeddings and is not this
experiment.

## Stage I

Stage I starts from the Stage H bank produced by the same seed:

`prefix(state) = shared_neutral + Σ state[axis] × delta[axis]`

The shared center starts as the mean of the three Stage H centers. The deltas
come from Stage H. Signed pole events update only axis deltas. Neutral events
update only the shared center.

Fixed settings:

- 36 signed pole steps;
- 18 neutral-center steps;
- learning rate `0.001`;
- neutral loss weight `1.0`;
- diversity weight `0.01`;
- anchor weight `0.001`;
- gradient norm cap `1.0`.

## Generated-response controls

Every held-out prompt is decoded under:

1. regular model, no prefix;
2. explicit positive style instruction;
3. explicit negative style instruction;
4. learned positive prefix;
5. learned negative prefix;
6. a positive prefix from the next axis.

Stage I also includes its shared neutral center. All text and per-response
scores are retained.

The Stage H warmth sweep evaluates states:

`-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1`

It also tests the learned direction without its center.

## Primary measurements

- generated signed span: mean positive-prefix attribution minus mean
  negative-prefix attribution;
- explicit span: mean explicit-positive attribution minus mean
  explicit-negative attribution;
- relative span: generated span divided by explicit span;
- sign success: positive-prefix attribution exceeds negative-prefix
  attribution for a held-out prompt;
- specificity: the target positive-to-negative span exceeds the absolute
  wrong-axis shift from the regular model;
- center drift: shared-center attribution minus regular-model attribution.

Teacher-forced reference likelihood is reported as a diagnostic. Generated
responses decide whether an expression channel exists.

## Confirmatory decision rules

Stage H warmth supports the primary claim when, across three fresh seeds:

- median relative generated span is at least `0.60`;
- at least 21 of 24 held-out comparisons have the intended sign;
- at least 21 of 24 target spans exceed wrong-axis shifts.

Stage I supports the patience result when its median relative patience span is
at least `0.20` and exceeds Stage H patience. The neutral center passes only if
its median absolute drift stays below `0.10` of the explicit span on every
axis. Goodwill remains exploratory.

The run reports the measurements even when a rule fails. Paper claims and
figures must use the clean consolidated artifact rather than the exploratory
LegoLM outputs.

## Seeds

The confirmatory seeds are:

```text
20260724
20260725
20260726
```

The first repeats the exploratory schedule. The other two test schedule
sensitivity. Each seed creates fresh Stage H and Stage I parameters and fresh
optimizers. The frozen language model may remain loaded between seeds.
