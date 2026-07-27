# Reproduction progress

## 2026-07-26

- Froze the target models, lens fit, disjoint evaluation inventory, three
  system prompts, calibration, depth trace, bootstrap procedure, metrics,
  confirmatory rules, and memory gates in `METHOD.md`.
- Pinned `Qwen/Qwen3.6-35B-A3B` at
  `995ad96eacd98c81ed38be0c5b274b04031597b0`.
- Pinned the optional dense extension `Qwen/Qwen3.6-27B` at
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Implemented a self-contained package. It fits and resumes the targeted lens,
  calibrates each system prompt and traced layer, records exact user tokens,
  computes the frozen decisions, and saves all environment and corpus
  identifiers.
- Added artifact-only figure regeneration and a self-contained HTML replay
  with family, utterance, system, layer, token, and axis controls.
- Added nine local tests for the corpus hash and disjointness, smoke inventory,
  exact inserted-token spans, profile layers, clustered bootstrap, decision
  rules, model comparison, PNG regeneration, and replay generation.
- Froze the runner before running either paper model. Both artifacts identify
  their source commit.
- Completed the confirmatory `Qwen/Qwen3.6-35B-A3B` run on one RTX PRO 6000:
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

## 2026-07-26: paper

- Wrote and built the six-page paper, *The Latent Geometry of "Meh"*, from the
  frozen method and fresh artifacts. The paper reports the failed 35B
  boredom-adjacency rule rather than smoothing it away.
- Published the clean reproduction, selected artifacts, HTML replays, figures,
  source, and PDF. Resumable fit states, lens checkpoints, and duplicate event
  streams are excluded from the publication bundle.

Status: reproduction, paper, and public artifacts complete.
