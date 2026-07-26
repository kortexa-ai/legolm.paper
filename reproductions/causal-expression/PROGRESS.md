# Reproduction progress

## 2026-07-25

- Froze the model revision, synthetic corpus, training schedules, controls,
  metrics, three seeds, and confirmatory decision rules in `METHOD.md`.
- Implemented a standalone package with no imports from LegoLM or `jlens`.
- Added Stage H and Stage I training, teacher-forced diagnostics, greedy
  response audits, wrong-axis controls, the warmth strength sweep, multi-seed
  consolidation, raw-response Markdown, and artifact-driven figures.
- Passed nine local tests covering the data hash, schedules, prefix geometry,
  assistant-header insertion, gradient flow, response metrics, decision rules,
  and figure regeneration.
- Completed an end-to-end Qwen 3.5 2B MPS smoke in 277 seconds. It trained both
  prefix banks, generated and scored every smoke condition, saved checkpoints
  and reports, and rendered all four figures. Its measurements have no paper
  status.
- Committed and pushed the frozen runner at `788db54`.
- Pulled the commit on smarty, passed all nine tests there, recorded the live
  service set, and stopped the seven GPU consumers once.
- Completed all three Qwen 3.6 35B-A3B seeds in 4,735 seconds without an OOM.
  Peak CUDA reservation was about 66.3 GiB.
- The primary Stage H warmth test passed: 71.4% median relative span, 24/24
  signed prompt comparisons, and 24/24 specificity comparisons.
- Stage I improved patience from 15.6% to 19.5% but missed its 20% gate. Its
  neutral center failed the drift rule on every axis.
- Audited the text itself. Two seeds produced openly hostile replies under the
  negative warmth prefix; the third tended toward refusal. Regular responses
  remained neutral.
- Committed and pushed the raw JSON results and four PNGs at `a9a5aab`.

Next: write and render the short paper, publish the exact reproduction subtree
to `legolm.paper`, update `research.kortexa.ai`, then restore the recorded smarty
services.
