# Reproduction progress

## 2026-07-26

- Froze the target models, lens fit, disjoint evaluation inventory, three
  system prompts, calibration, depth trace, bootstrap procedure, metrics,
  confirmatory rules, and memory gates in `METHOD.md`.
- Pinned `Qwen/Qwen3.6-35B-A3B` at
  `995ad96eacd98c81ed38be0c5b274b04031597b0`.
- Pinned the optional dense extension `Qwen/Qwen3.6-27B` at
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Implemented a standalone package with no imports from LegoLM. It fits and
  resumes the targeted lens, calibrates each system prompt and traced layer,
  records exact user tokens, computes the frozen decisions, and saves all
  environment and corpus identifiers.
- Added artifact-only figure regeneration and a self-contained HTML replay
  with family, utterance, system, layer, token, and axis controls.
- Added nine local tests for the corpus hash and disjointness, smoke inventory,
  exact inserted-token spans, profile layers, clustered bootstrap, decision
  rules, model comparison, PNG regeneration, and replay generation.
- Passed a clean `uv sync`, all nine tests, and the inventory command.
- Completed an end-to-end Qwen 3.5 2B MPS smoke in 24.5 seconds after correcting
  its non-paper profile to the checkpoint's 24-layer configuration. The smoke
  saved a lens, 110 measured passes, raw token and depth records, four PNGs,
  and the HTML replay. Its measurements have no paper status.

## 2026-07-25/26: paper runs

- Committed and pushed the standalone reproduction before running either paper
  model. Both artifacts identify source commit
  `0349d5c1ed28e9599dd8e4d0f7037e1b72547474`.
- Stopped the seven GPU services on `smarty` once, after recording the initial
  service state. They were kept stopped until the publication work was
  complete.
- Completed the confirmatory `Qwen/Qwen3.6-35B-A3B` run on the RTX PRO 6000:
  666/666 measurements in 114.2 seconds with the checkpoint already cached,
  with no OOM.
- The 35B run passed held-out orientation, non-null `meh`, and template
  stability. It failed the frozen boredom-adjacency rule: discouragement was
  nearest at layer 35 under all three system prompts.
- Completed the descriptive dense `Qwen/Qwen3.6-27B` extension:
  666/666 measurements in 670.6 seconds including the initial checkpoint
  download, with a 50.5 GiB peak CUDA reservation and no OOM.
- The 27B extension passed all four rules. Boredom was nearest at layer 55
  overall and under all three system prompts.
- Compared the two model artifacts. The median cosine between matching atlas
  family centroids was 0.9549; the `meh` centroids had cosine 0.9512 and agreed
  in sign on all twelve coordinates.
- Regenerated and visually checked the PNG figures and self-contained HTML
  replays for both models.
- Committed and pushed selected raw measurements, summaries, figures, replays,
  and the cross-model comparison to the private repository. Resumable fit
  states, lens checkpoints, and duplicate event streams remain local to
  `smarty`.

## 2026-07-26: paper and publication

- Wrote and built the six-page paper, *The Latent Geometry of "Meh"*, from the
  frozen method and fresh artifacts. The paper reports the failed 35B
  boredom-adjacency rule rather than smoothing it away.
- Committed and pushed the paper and complete track record to the private
  repository, through commit `5c0a27b`.
- Published the clean reproduction, selected artifacts, HTML replays, figures,
  source, and PDF to `kortexa-ai/legolm.paper`, through commit `6214d9f`.
- Published the paper card at `https://research.kortexa.ai/` from research-site
  commit `4c3eb28`, and verified the live card, PDF, and code links.
- Restored all seven services stopped for the experiment exactly once. The
  three model servers, ASR, TTS, vision, and LegoLM each returned HTTP 200
  after restoration.

Status: reproduction, paper, public artifacts, and research-site publication
complete; `smarty` is restored to its pre-experiment service set.
