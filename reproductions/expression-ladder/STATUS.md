# Expression ladder (paper 6a) — build status

**Complete (2026-08-02).** Every rung has run, Stage J included, and the
manuscript ships with the Stage J result folded in. The public snapshot is
`legolm.paper/expression-ladder-paper/` plus
`legolm.paper/reproductions/expression-ladder/`.

## In place (2,530 lines, all imports resolve, all modules parse)

| Module | Lines | Source |
|---|---|---|
| `data.py` | 455 | copied from `causal-expression`; frozen corpus, pinned hash |
| `metrics.py` | 280 | copied; span/attribution/audit machinery |
| `runtime.py` | 757 | extracted from `causal-expression/experiment.py` — model load, CUDA gates, memory snapshots, environment report, prompt rendering, response NLL, generation. Added `bounded_context`, which raises rather than truncating, matching the existing discipline |
| `prefix.py` | 492 | copied; Stage H and I banks |
| `ladder.py` | 543 | **new** — the port |

`ladder.py` covers the rungs below the soft prefix:

- `ActivationCapture` — frontier activations at several sites at once
- `SteeringSpec` / `DirectionSteering` / `steering_context` — norm-scaled
  injection at response positions, scoring and generation modes
- `distribute_budget` — Stage F's `total / sqrt(sites)` split
- `fit_static_directions` — Stage B
- `contextual_direction` — Stage F's per-prompt variant
- `direction_geometry` — cross-axis cosine, which is why every stage needs a
  wrong-axis control
- `TrainedResidualWriter` + `TrainedResidualSteering` — Stage G, kept separate
  from `DirectionSteering` because the trained path must not detach

## In place since (2026-08-01)

- `stages.py` and `cli.py` — stages A–G orchestration and the `reproduce`
  entry point; the 35B ladder block ran and its artifact is
  `artifacts/ladder-35b-summary.json`.
- `stage_j.py` — on-policy neutral distillation: no-prefix KL teacher on the
  frontier plus sampled on-policy continuations, prefix dropout and explicit
  zero-state batches, doubled negative patience/goodwill coverage, dev-split
  best-margin snapshot selection, greedy + sampled + trajectory audits, and
  a machine-readable PASS/FAIL/INVALID verdict. Runs standalone via
  `expression-ladder stage-j`.
- `PREREGISTRATION-stage-j.md` — frozen thresholds, primary axes warmth and
  patience, goodwill preregistered secondary. Frozen before the run and
  unchanged since; the verdict is committed, so it is now history.
- `tests/test_stage_j.py` — 11 CPU-only tests: schedule/dropout batching,
  KL shape and alignment, parameter-targeting, seeded sampling, gate and
  trajectory arithmetic, verdict validity.

## The Stage J run (2026-08-02)

`artifacts/stage-j-35b-{summary,environment}.json`. All three registered
seeds (20260801–03) completed on smarty at the pinned revision and corpus
hash. **Verdict FAIL, `invalid_reasons` empty** — a valid failure, not a
void run. 6,402 s, 65.674 GiB peak reserved, no OOM.

- **Gate 1 (center fidelity) missed by one clause.** Greedy attribution drift
  at zero state fell to 0.011 (warmth) and 0.004 (patience) of the explicit
  span, sampled to 0.048 and 0.019, all four inside the `< 0.10` rule that
  Stage I failed at 0.170 and 0.125. Median center↔off word-Jaccard 0.565
  against `>= 0.60` is the only failing clause.
- **Gate 2 (signed axes) collapsed.** Greedy warmth relative span 0.172
  against `>= 0.40`; patience 0.178 against `>= 0.20` with 18/24 signed
  against 21 required. Sampled is worse on every count.
- **Gate 3 (trajectories) failed both primary axes** — wrong sign at the
  moderate negative state, 2 and 3 adjacent inversions against `<= 1`.

The result is the trade-off curve H → I → J, measured on one model, one
corpus, one audit: warmth span 0.714 → 0.427 → 0.172 as center drift goes
absent → 0.170 → 0.011. A soft prefix with an always-present shared center
cannot be both silent at zero state and causally effective at nonzero state.
That also rules the soft prefix out as 6b's decaying-state substrate, by
measurement rather than by argument.

## Do not forget

6a is a **confirmatory re-run of a known internal result** for stages A–G. It
cannot claim rules were frozen before the run, because the outcomes were
already known from the development tree. The paper says so. Stage J is the
only rung that carries honest preregistration, which was the argument for
folding it in here; its verdict is committed, so
`PREREGISTRATION-stage-j.md` and `STAGE_J_PREREG` are now a historical
record and must not be edited to match anything.
