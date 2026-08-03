# The expression ladder

A self-contained reproduction of seven mechanisms for changing the
interpersonal style of a frozen language model, ranked by how convincingly they
appear to work.

The result is not that one of them works. It is that five of them produce
persuasive numbers while changing nothing a reader would notice, and that the
measurement which separates the two is not the one most naturally reached for.

## The measurement that matters

Every stage is scored twice:

- **Teacher-forced span** — how far the intervention shifts the likelihood of
  curated positive-versus-negative reference responses. This is measured in the
  same space the writer was optimized in.
- **Generated span** — the same contrast recomputed on text the model actually
  produced under greedy decoding, against a wrong-axis control.

Stage G is the reason the distinction is worth a paper. Trained directly on
paired response likelihood it reaches a teacher-forced span that looks
decisive, and its generated text is indistinguishable from no intervention at
all. A study that reported only the first number would have reported a result
that does not exist.

## Stages

| Stage | Mechanism |
|---|---|
| A | Visible style instructions, as the denominator every later span is quoted against |
| B | Static residual directions from positive-minus-negative frontier activations |
| C | Dose and token-window sweep, scored teacher-forced |
| D | Free-generation audit of the best Stage C writer, with a wrong-axis control |
| F | Per-prompt contextual directions injected across several sites under a fixed norm budget |
| G | A residual direction trained directly against paired full-response likelihood |

Stages H and I, the soft-prefix rungs that do change generated text, are
reported in a companion study; `prefix.py` here carries the same banks so the
ladder can be extended to them without a second corpus.

Stage J — on-policy neutral distillation of the always-present shared center
toward the frozen no-prefix policy — is the one rung that was never run
internally, so it carries genuinely frozen rules. See
`PREREGISTRATION-stage-j.md`.

The preregistered 35B block ran on 2026-08-02 across all three registered
seeds and returned a valid **FAIL** with no invalidating conditions. The
center went nearly silent — greedy attribution drift 1.1% (warmth) and 0.4%
(patience) of the explicit span against a `< 10%` rule that Stage I missed at
17.0% and 12.5% — and the signed channel went with it, warmth relative span
0.172 against a `>= 0.40` rule. Across Stage H, I, and J the two quantities
trade off monotonically on warmth. Every number is in
`artifacts/stage-j-35b-summary.json`.

It runs standalone:

```bash
uv run expression-ladder stage-j --suite smoke --device cpu \
  --output-dir results/stage-j-smoke
uv run expression-ladder stage-j --suite full --device cuda \
  --output-dir results/stage-j-35b
```

## Running it

```bash
uv sync
uv run expression-ladder check                     # validate the frozen corpus
uv run expression-ladder reproduce --suite smoke --device cpu
uv run expression-ladder reproduce --suite full --device cuda \
  --output-dir results/ladder-full
```

The smoke suite runs `Qwen/Qwen3.5-2B` at reduced step counts and shortened
generations. It verifies plumbing, tensor shapes, and hook placement. Its
measurements have no paper status.

The full suite targets `Qwen/Qwen3.6-35B-A3B` at revision
`995ad96eacd98c81ed38be0c5b274b04031597b0`. It needs an 80 GB-class GPU: the
frozen weights alone occupy about 64.6 GiB in BF16 before the backward graph
that Stage G requires. Free-memory gates are advisory and parameterized rather
than hard preflight aborts, so the run is portable to a different card.

## Scope

This measures whether a writer changes complete generated responses on a
synthetic, English, technical-request corpus. It does not include a recurrent
session reader, temporal decay, or any claim about subjective experience.
Social-style names are operational labels for matched response sets.

## Status

This reproduction is a confirmatory re-run: the ladder's outcomes were
established in earlier internal work, so its thresholds were chosen with
knowledge of those outcomes and it makes no preregistration claim. The single
exception is Stage J, which was never run internally, whose decision rules
were frozen in `PREREGISTRATION-stage-j.md` before its first training step,
and which failed them.
