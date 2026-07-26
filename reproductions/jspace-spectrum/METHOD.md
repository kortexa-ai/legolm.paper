# Method contract

**Frozen before the confirmatory run:** 2026-07-26

## Question

Can a targeted residual-stream lens place short social utterances in a stable,
measurable affect spectrum without using those utterances to define the lens?

The experiment measures activation geometry. It does not test persistent
memory, response steering, behavior, emotion recognition, or subjective
experience.

## Models

The confirmatory target is:

- `Qwen/Qwen3.6-35B-A3B`
- revision `995ad96eacd98c81ed38be0c5b274b04031597b0`
- bfloat16 on CUDA
- source residual layer 35 of 40
- target layer 39

The optional dense-model extension is:

- `Qwen/Qwen3.6-27B`
- revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
- bfloat16 on CUDA
- source residual layer 55 of 64
- target layer 63

The 27B run is allowed only after the 35B result is complete and only if the
preload memory gate passes. The confirmatory claims do not depend on it.

## The twelve signed axes

The lens has twelve positive-minus-negative coordinates:

1. warmth versus hostility;
2. trust versus wariness;
3. affiliation versus distance;
4. playfulness versus formality;
5. ease versus tension;
6. care versus indifference;
7. engagement versus boredom;
8. patience versus annoyance;
9. efficacy versus frustration;
10. social safety versus hurt or defensiveness;
11. goodwill versus resentment;
12. hope versus discouragement.

Each pole is defined by six words. The tokenizer resolves single-token surface
forms of those words. Evaluation text is rejected if it contains any pole word
as a case-insensitive whole word.

## Lens fit

The base model stays frozen. For each axis, the runner computes the gradient of
the positive-minus-negative pole-token logit contrast with respect to the
source residual stream. It averages the gradient over valid positions and
eight neutral fit prompts. This produces one read direction per axis.

The fit uses:

- maximum sequence length 96;
- the first eight token positions excluded;
- the final transformer layer as the logit target;
- PyTorch SDPA;
- model cache disabled;
- a resumable fit-state artifact.

The fit prompts, calibration messages, landmark cases, and atlas utterances are
separate fixed corpora.

## Calibration and depth traces

The experiment crosses every evaluation utterance with three system prompts:

1. helpful and direct;
2. concise and technical;
3. conversational and natural.

For each system prompt, 24 neutral messages define the mean and population
standard deviation of every axis. The same layer-35 read matrix is also applied
to residual outputs from every full-attention layer:

```text
3, 7, 11, 15, 19, 23, 27, 31, 35, 39
```

Each traced layer receives its own system-prompt-specific neutral center and
scale. These are fixed-direction residual-depth traces, not independently
fitted Jacobian directions at every layer.

The 27B extension uses its full-attention layers:

```text
3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
```

## Evaluation inventory

The held-out landmark set has four positive and four negative paraphrases per
axis: 96 utterances. None contains a lens pole word.

The social atlas has 21 families with six variants each: 126 utterances. It
includes `meh`, greetings, gratitude, apology, praise, insult, dismissal,
complaint, repetition, encouragement, agreement, disagreement, uncertainty,
surprise, amusement, farewell, requests, refusal, relief, confusion, and
neutral statements.

The `meh` family contains three bare forms and three contextual forms. None
uses the words *bored*, *boring*, *indifferent*, or any other pole token.

Every case is measured under all three system prompts, for 666 evaluation
passes per model. Only the exact user-content tokens are read. Role markers,
assistant headers, end markers, and template newlines are excluded by comparing
the populated user turn with an empty user turn.

The artifact retains:

- every user token and its coordinate vector;
- utterance means and final-user-token coordinates;
- every traced layer;
- model, revision, environment, command, corpus hash, and runtime;
- the fitted lens and neutral calibration statistics.

## Metrics

The runner reports:

- held-out positive-minus-negative separation on every target axis;
- the full off-axis separation vector for every landmark pair;
- family centroids, ranges, and 95% bootstrap intervals;
- Euclidean and cosine neighbors among atlas and landmark centroids;
- pairwise cosine stability of family centroids across system prompts;
- bare and contextual `meh` centroids;
- `meh` minus neutral coordinates;
- fixed-direction residual-depth trajectories;
- axis correlation and the fraction of centroid variance in the first
  principal component.

Bootstrap intervals use 2,000 resamples and seed `20260726`. Resampling is
clustered by utterance variant; a system prompt is sampled within each selected
variant. The intervals describe sensitivity to this fixed prompt inventory,
not population or model-sampling uncertainty.

## Confirmatory decisions

1. **Held-out orientation passes** when all twelve target-axis mean
   separations are positive and at least ten have a positive 95% lower bound.
2. **`meh` is non-null** when the upper 95% interval for engagement is below
   zero and the Euclidean distance between `meh` and neutral centroids is at
   least 3 calibrated units.
3. **`meh` is boredom-adjacent** when the held-out boredom centroid is its
   nearest landmark overall and under at least two of the three system prompts.
4. **Template stability passes** when the median pairwise cosine between
   same-family centroids across the three system prompts is at least 0.80.

All measurements are reported when a rule fails. Depth and 27B comparisons are
descriptive extensions.

## Memory gates

The full runner requires at least 85 GiB free CUDA memory and 100 GiB available
system memory before loading either paper model. It aborts if CUDA headroom
falls below 20 GiB after model load or if total CUDA use exceeds 90% during
measurement. Models run sequentially and are released between runs.
