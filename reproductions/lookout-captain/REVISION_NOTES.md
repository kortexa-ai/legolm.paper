# September 7, 2026 revision

This revision uses the completed August 22–27 vision-mode study. It adds
analysis and publication work; it does not add a new GPU experiment.
The raw observations are preserved in `artifacts/`.

The main findings remain mode conditioning through 27B and a large
reduction in model invocations in the short replay. The revision gives
them a directly measured, bounded interpretation.

- Incorporated 27B event gating and the August 27 fixed-constant-baseline,
  crossed-order, and smaller-host training follow-ups.
- Recomputed action shares and paired contrasts. Added pointwise
  image/scene and episode bootstrap intervals conditional on tested
  seeds and phrase assignment, alongside per-seed results.
- Replaced the population-wide detector miss-cost claim with the exact
  identity using action cost on actual detector errors.
- Corrected delay accounting: 167 post-start cuts, 130 observed wakes,
  37 censored endings, an observed mean of 1.44 steps, and a maximum
  of 12. The previous 0.984 average included starts and omitted
  unresolved cuts.
- Distinguished model agreement from task correctness, and avoided
  calls from measured energy.
- Described the actual emitter and residual receivers, including the
  30-layer 3B host and the constant adapter in the Text control.
- Corrected photographic selection to annotation area; the detector
  rule uses bounding-box area.
- Reported held-out phrase sensitivity, including complete success
  for one phrase and failure for the other in several rendered runs.
- Retained the 450M explore failure and other weak or adverse outcomes.
  Mixed training changes both the mode adapter and its constant baseline.
- Identified two rendered pixel versions with identical scene draws.
  Mixed/fixed-baseline comparisons use identical pixels and mode
  checkpoints. Cross-version changes cannot be attributed solely to
  training. Eight repeated 450M Text actions differ across versions.
- Removed claims of universal prompt fragility, safety, behavioral
  equivalence, a one-frame delay guarantee, and a general transfer recipe.

The release includes all 232 source records, the complete local
dependency set for the study scripts, a pinned new-run environment,
exact run groups, and a CPU manuscript build. All 72 referenced
checkpoint files matched their historical hashes. COCO input hashes
were recovered from the study machine.

Historical limits remain explicit. LFM revisions were recovered from
cache. The original runs lack a complete environment lock and a hash
of uncommitted source. Temporary rendered PNGs were not recovered;
scene draws and both evaluation hash sets remain available. These
limits permit saved-observation reanalysis but prevent an unsupported
promise of bitwise retraining or rendering.
