# Expression ladder (paper 6a) — build status

Work in progress. Lives here, not in `legolm.paper`, until it runs end to end;
the public repo takes only final code.

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

## Remaining

1. **`stages.py`** — orchestration: Stage A prompt upper bound, Stage C
   dose/window sweep, the 144-generation audit loop per stage, Stage J's
   on-policy KL distillation to the no-prefix path. Est. 500–700 lines.
2. **`cli.py`** — `expression-ladder reproduce --stage {a..j}` plus `figures`.
   Model this on `causal-expression/src/causal_expression/cli.py` (105 lines).
3. **`tests/`** — the sibling reproductions carry nine tests each; match that.
4. **2B smoke**, then the confirmatory 35B block. Needs the seven GPU services
   stopped; restore set is in `PROGRAM.md`. Paper 5's comparable run was
   4,735 s at 66.3 GiB peak, so budget one overnight block.
5. **Manuscript** at `legolm.paper/expression-ladder-paper/`, hand-authored
   `.tex` like papers 4 and 5.

## Do not forget

6a is a **confirmatory re-run of a known internal result**. It cannot claim
rules were frozen before the run, because the outcomes are already known from
the development tree. Say so in the paper. Stage J is the only rung that can
carry honest preregistration, which is the argument for folding it in here.
