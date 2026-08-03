# Preregistration — Stage J: on-policy neutral distillation

**Registered:** 2026-08-01, before any Stage J training step has ever been
executed, in the development tree or anywhere else.
**Status:** frozen. After this document is committed, neither it nor the
threshold dictionary `STAGE_J_PREREG` in `src/expression_ladder/stage_j.py`
may change until the confirmatory run's verdict is committed. The two copies
must stay identical; `tests/test_stage_j.py` pins the code copy.

Stages A–I of this reproduction are confirmatory re-runs of known internal
results and claim no preregistration (see `README.md`). Stage J was designed
in the development tree (`tracks/jspace-memory/EXPRESSION_TRACK.md`, section
J, registered 2026-07-24) but **never run**, so it is the one rung of the
ladder whose decision rules can honestly be frozen before the data exist.
That is what this document does.

## Pins

| Item | Value |
|---|---|
| Model | `Qwen/Qwen3.6-35B-A3B` |
| Revision | `995ad96eacd98c81ed38be0c5b274b04031597b0` (identical to papers 4 and 5) |
| Corpus | frozen synthetic corpus in `src/expression_ladder/data.py`, sha256 `85f6e3599474ad33c37092b583b21ccfc3175fd85b168c65b735c2dd75332346` |
| Seeds | 20260801, 20260802, 20260803 |
| Suite | `full` (3 seeds × 8 held-out prompts × 3 axes) |
| Hardware | smarty, RTX PRO 6000 96 GB, BF16, batch 1 |

A 2B plumbing smoke (`Qwen/Qwen3.5-2B`) runs before the 35B block, per track
convention. Smoke measurements have no paper status and any smoke verdict is
INVALID by construction.

## Question

Stage I made the zero-state tensor path continuous but trained its
always-present shared center with cross-entropy against neutral reference
responses. That objective selected *a different valid neutral continuation*
instead of preserving the base decoder policy: the 35B center had **zero**
exact greedy matches to memory-off, mean center↔off word-Jaccard of only
0.215–0.330 across seeds, and shifted response attribution by 17.0% / 12.5% /
33.4% of the explicit span (warmth / patience / goodwill, 3-seed medians from
`reproductions/causal-expression/results/qwen36-35b-confirmatory-20260725`).

Stage J replaces that objective with distillation against the frozen
no-prefix model's own token distribution, measured on actual decoding
trajectories. The question: can the shared center become behaviorally
indistinguishable from memory-off while the signed axis deltas keep their
causal effect?

## Primary and secondary axes — decided before the run

**Primary (pass/fail): warmth and patience.**
**Secondary (exploratory, reported but unable to affect the verdict):
goodwill.**

Justification from the Stage H/I record, all of it measured before this
registration:

| Evidence | warmth | patience | goodwill |
|---|---|---|---|
| Stage H generated span, dev tree (% of explicit) | 90.9% | 9.9% | 15.6% |
| Stage I generated span, dev tree (% of explicit) | 59.9% | 35.8% | **9.0%** |
| Stage I signed orderings, dev tree (of 8) | 8 | 7 | **4** |
| Stage I beat wrong axis, dev tree (of 8) | 8 | 6 | **3** |
| Stage I relative span, repro 3-seed median | 0.427 | 0.195 | 0.126 |
| Stage I signed, repro (of 24) | 23 | 22 | 18 |
| Stage I center drift fraction, repro median | 0.170 | 0.125 | **0.334** |

Goodwill was already the weakest generated channel in both the development
tree and the confirmatory reproduction — at or below chance on signed
orderings in the dev-tree Stage I audit — and its center drift is the worst
of the three. Stage J's training changes (on-policy center distillation,
more negative-pole coverage) were designed to help patience and goodwill,
but only warmth and patience have demonstrated enough signal that a Stage J
failure on them is informative about the mechanism rather than about axis
strength. Goodwill therefore enters the analysis as a preregistered
secondary axis: every goodwill number will be reported, and none of them can
flip the verdict.

## Training procedure (frozen)

Architecture is the Stage I bank unchanged:
`prefix(state) = shared_neutral + Σ state[axis] × axis_delta`, eight
continuous tokens inserted immediately before the assistant header, base
model frozen. Zero state is `strength = 0` on the same tensor path, never
prefix removal.

Per seed:

1. **Initialization lineage.** Train a Stage H `SoftPrefixBank` for 36 steps
   under the exact frozen H recipe (contrastive paired NLL, lr 0.001,
   diversity 0.01, anchor 0.001, grad clip 1.0), then convert with
   `NeutralAnchoredPrefixBank.from_stage_h` — the same lineage Stage I used.
   This bank is not audited.
2. **Stage J schedule** (`distill_training_schedule`, seeded, deterministic):
   - 48 pole steps: the Stage I signed inventory over the 6 fit cases × 3
     axes × 2 signs, with the **negative patience and negative goodwill
     events doubled** (one exact epoch of the 48-event inventory). Warmth
     events are untouched — the registration's "increase negative patience
     and goodwill coverage without weakening the successful Stage H warmth
     result".
   - **Prefix dropout 0.25**: each pole event is independently converted,
     with seeded probability 0.25, into a zero-state distillation event on
     the same pair.
   - 18 explicit zero-state distillation steps appended, then the whole
     event list is shuffled. Expected composition: ≈36 pole steps, ≈30
     distillation steps, interleaved.
3. **Losses.**
   - Pole steps: Stage I's contrastive paired NLL on `prefix(axis, sign)`,
     gradients applied to the deltas only.
   - Distillation steps: sample an on-policy continuation of up to 24 tokens
     from the center path (temperature 1.0, seeded), then minimize the mean
     token-level **forward KL, `KL(no-prefix ‖ shared-center)`**, over the
     first decoding frontier plus every continuation position. The teacher
     pass has no prefix and no gradient. Gradients applied to the center
     only. The center's loss never sees a curated reference response.
   - Both step kinds keep the Stage I regularizers: diversity 0.01, anchor
     0.001, grad clip 1.0, AdamW lr 0.001, norm projection after each step.
4. **Checkpoint selection** (frozen rule, following the track's precedent of
   selecting on a dev margin rather than the final step): snapshot the bank
   at 1/3, 2/3, and the end of the schedule; score each snapshot on the dev
   split as `mean primary-axis causal margin − mean frontier KL of the
   center` (both in nats); keep the best-scoring snapshot. The KL term
   prevents buying margin with center drift. The test split is never touched
   before the final audit.

Distillation prompts come from the fit split only; dev is used only for
snapshot selection; the eight test cases (repeated demands, crash loop,
deadline, code review, do-it repetition, insult, praise, gratitude) are
reserved for the gates.

## Evaluation protocol (frozen)

All measurements use the audit machinery already validated in the H/I
confirmatory run: response attribution is the NLL difference of the response
under explicit positive vs negative style system prompts; the explicit span
(explicit-positive minus explicit-negative mean attribution, computed per
decoding mode) is the denominator for every relative number.

Conditions per axis × test prompt: `off`, `neutral_center`,
`explicit_positive`, `explicit_negative`, `prefix_positive`,
`prefix_negative`, `wrong_axis` (next axis's positive prefix).

- **Greedy audit:** argmax decoding, 96 new tokens max.
- **Sampled audit:** identical conditions under pure temperature sampling,
  **T = 0.8**, top-p disabled, one sample per condition × prompt, generator
  seeded by CRC32 of `(run seed, condition, pair id)` — fully reproducible,
  and a different stream per condition so identical distributions do not
  trivially produce identical text. Text-identity metrics are therefore not
  gated in sampled mode; attribution metrics are.
- **Trajectory audit (greedy):** for every axis, decode the same prompt at
  states −1, −0.5, −0.25, 0, +0.25, +0.5, +1, each state on the same
  `neutral + state × delta` tensor path. This is the decay trajectory a
  temporal memory would traverse between a signed event and rest;
  intermediate states are measured, not assumed linear from endpoint
  likelihood.

## Decision rules (frozen)

Aggregation: per-seed metrics over 8 test prompts; medians and counts pooled
over the 3 seeds (24 prompts per axis per mode). Count thresholds are stated
of-24 and enforced by exact integer cross-multiplication.

### Gate 1 — center fidelity (both decoding modes)

For each primary axis, in greedy **and** sampled mode:

- median |mean attribution(center) − mean attribution(off)| / explicit span
  **< 0.10**. This is the identical threshold the Stage I confirmatory rules
  froze for the neutral-center decision; Stage I measured 0.170 (warmth) and
  0.125 (patience) and failed it.

And, greedy mode only:

- median across seeds of mean word-Jaccard(center response, off response)
  **≥ 0.60**. Stage I measured 0.215–0.330 while failing; 0.60 is roughly
  double the best failing value and is the point where responses share most
  content words. Exact-match rate is reported but not gated: BF16 KL
  distillation is not expected to reproduce argmax bit-exactly on all
  prompts, and gating on it would test numerics, not behavior.

### Gate 2 — signed axes (both decoding modes)

For each primary axis, in greedy **and** sampled mode:

| Criterion | warmth | patience | Justification |
|---|---|---|---|
| relative span, median | ≥ 0.40 | ≥ 0.20 | warmth: Stage I achieved 0.427 — the center fix must not cost the channel; patience: the already-frozen Stage I rule, which Stage I missed at 0.195 |
| signed prompts (positive > negative), of 24 | ≥ 21 | ≥ 21 | the frozen Stage H warmth rule; Stage I measured 23 and 22 |
| beats wrong-axis shift per prompt, of 24 | ≥ 21 | ≥ 16 | warmth: Stage I measured 24/24, no regression allowed; patience: Stage I measured 12/24 — 16 (two thirds) is the midpoint toward the warmth level and requires a real specificity gain |
| beats center drift per prompt (\|span\| > \|center − off\|), of 24 | ≥ 21 | ≥ 16 | same comparison class as wrong-axis specificity; this is the registration's "beat center drift" clause made per-prompt |

### Gate 3 — trajectories (greedy)

For each primary axis, on the 3-seed pooled mean attribution per state:

- sign correctness at moderate and full states: mean(−1) and mean(−0.5)
  below mean(0); mean(+0.5) and mean(+1) above mean(0). The Stage H
  nine-point sweep found signs reliable at |state| ≥ 0.5 (8/8 prompts) while
  individual branches were non-monotonic.
- at most **1** adjacent inversion across the seven-state mean curve,
  covering the single low-dose wobble that same sweep observed. The ±0.25
  states are measured and count toward the inversion tally but carry no
  separate sign requirement.

### Verdict

- **PASS** — all of Gates 1–3 pass for both primary axes.
- **FAIL** — the run is valid and any primary gate fails.
- **INVALID** — the run is not the preregistered configuration or a
  denominator broke: corpus hash ≠ pin, model ≠ pin, suite ≠ full, seeds ≠
  the frozen set, any non-finite loss/gradient (the run aborts), a CUDA
  memory-gate violation (the run aborts), or an explicit span ≤ 0 in any
  gated mode/axis (the attribution denominator is meaningless).

The verdict is computed by `stage_j_verdict` in `stage_j.py` from these
numbers alone and written into `stage-j-summary.json`. Whatever it says is
what the paper reports.

## Analysis plan

- The paper reports all three axes, both decoding modes, the trajectory
  curves, center content metrics, the distillation-KL training curves, and
  the snapshot-selection record, regardless of outcome.
- Secondary (goodwill) results are labeled exploratory in every table.
- No threshold, axis assignment, or aggregation rule changes after this
  commit. If the run is INVALID, it may be re-run after fixing the invalid
  condition, under the same rules, and the paper discloses every attempt.
- A FAIL is a publishable result: it would say on-policy distillation cannot
  make an always-present prefix behaviorally neutral without destroying its
  signed channels, which bounds the neutral-anchored writer family.

## Operational plan (smarty)

Per `SMARTY_6000_GUIDE.md` and the H/I precedent: allocator fraction capped
at 0.88 before load; ≥ 100 GiB system and ≥ 85 GiB CUDA free required before
load; checkpoint streamed to CUDA; ≥ 24 GiB CUDA free required after load;
batch 1 throughout; ≥ 20 GiB free enforced at every training step and audit
row (stricter than the 10 GiB hard abort floor); stop on OOM, reservation
above 90 GiB, or any non-finite value. Services stopped once from the guide's
safe-list order, restored once and health-checked after artifacts are
committed. The H/I confirmatory block (2 stages × 3 seeds) took 4,735 s at
66.3 GiB peak; Stage J adds ~66 training steps and ~450 generations per seed
and is budgeted one overnight block.
