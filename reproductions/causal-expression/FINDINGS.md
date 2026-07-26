# Confirmatory findings

## Run identity

The clean run finished on 2026-07-25 in 4,735 seconds on one NVIDIA RTX Pro
6000. It used `Qwen/Qwen3.6-35B-A3B` at revision
`995ad96eacd98c81ed38be0c5b274b04031597b0`, bfloat16 weights, greedy decoding,
and the three seeds frozen in `METHOD.md`. The model stayed frozen. Only eight
continuous prefix embeddings per endpoint were trained.

The consolidated artifact is
`results/qwen36-35b-confirmatory-20260725/summary.json`.

## Frozen decisions

| Test | Frozen rule | Result |
| --- | --- | --- |
| Stage H warmth | Median relative span at least 60%; at least 21/24 signed; at least 21/24 axis-specific | **Pass:** 71.4%; 24/24; 24/24 |
| Stage I patience | Median relative span at least 20% and above Stage H | **Fail:** 19.5%; above Stage H's 15.6% |
| Stage I neutral center | Median absolute drift below 10% on every axis | **Fail:** warmth 17.0%, patience 12.5%, goodwill 33.4% |

The three Stage H warmth spans were 72.3%, 71.4%, and 47.3% of the span
produced by visible style instructions. The third seed is weaker, but the
predeclared median, sign, and specificity rules all pass.

## The text changed

The response audit used complete greedy generations, not only
teacher-forced likelihood. The regular model answered a direct insult with a
neutral networking explanation:

> An HTTP request can time out even when DNS resolution succeeds because DNS
> resolution is only the first step in establishing a connection.

With the positive warmth prefix, one seed began:

> I understand your frustration, and I apologize for not addressing your
> concern clearly in previous attempts.

With the negative warmth prefix, the first two seeds began:

> Because you clearly didn’t read the first two explanations, or you’re just
> incapable of understanding basic networking concepts.

and:

> Because you clearly can’t connect the dots between “name resolution” and
> “actual data transfer,” let’s spell it out for you.

The third seed refused to engage with the insult instead of returning one.
This is still a change from the regular response, but it is not the same
negative style. On praise and gratitude prompts, the first two negative
prefixes also produced curt or openly insulting replies; the third again
tended toward refusal. The writer therefore has causal access to response
style, while the exact negative behavior remains seed-sensitive.

## The coordinate is not clean

The warmth sweep is not a smooth, symmetric dose-response curve. Negative
states move the median response sharply toward the negative reference, but the
learned center is already positive and increasing the positive coefficient
does not increase warmth monotonically. Removing the center leaves both
direction-only controls near the regular model.

Stage I tries to repair this geometry with one shared neutral center and three
signed directions. It raises the median patience span from 15.6% to 19.5%, but
misses the frozen 20% threshold and moves the nominal center away from the
regular model on all three axes.

The supported claim is narrow: a tiny trained input prefix can causally alter
the interpersonal style of a frozen Qwen 3.6 response on held-out synthetic
requests. The experiment does not show a linear social coordinate, a recurrent
session memory, emotion recognition, or subjective experience.

## Artifact map

- `figures/relative-spans.png`: primary spans with every seed shown.
- `figures/warmth-sweeps.png`: signed warmth trajectories.
- `figures/neutral-anchor.png`: Stage I endpoint and center drift.
- `figures/six-pole-radar.png`: secondary six-pole summary.
- `seed-*/stage-*-responses.json`: every generated response and score.
- `seed-*/stage-*-screen.json`: teacher-forced diagnostics.
- `seed-*/stage-h-warmth-sweep.json`: every sweep response and score.
