# Reproduction progress

## 2026-07-25

- Froze the model revision, synthetic corpus, training schedules, controls,
  metrics, three seeds, and confirmatory decision rules in `METHOD.md`.
- Implemented a self-contained package: no external lens implementation and no
  dependency outside the pinned list in `pyproject.toml`.
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
- Completed all three Qwen 3.6 35B-A3B seeds in 4,735 seconds on one RTX PRO
  6000 without an OOM. Peak CUDA reservation was about 66.3 GiB.
- The primary Stage H warmth test passed: 71.4% median relative span, 24/24
  signed prompt comparisons, and 24/24 specificity comparisons.
- Stage I improved patience from 15.6% to 19.5% but missed its 20% gate. Its
  neutral center failed the drift rule on every axis.
- Audited the text itself. Two seeds produced openly hostile replies under the
  negative warmth prefix; the third tended toward refusal. Regular responses
  remained neutral.

Status: complete.
