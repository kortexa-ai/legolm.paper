# Findings

The two Qwen 3.6 runs agree on the shape of `meh` and disagree on its nearest
single label. That is the main result.

## Frozen decisions

| Decision | Qwen3.6-35B-A3B | Qwen3.6-27B |
| --- | ---: | ---: |
| Held-out orientation | pass | pass |
| Non-null `meh` | pass | pass |
| Boredom adjacency | **fail** | pass |
| Template stability | pass | pass |

The 35B MoE model is the confirmatory target. The dense 27B model is a
descriptive extension; its result cannot rescue a failed confirmatory rule.

## The twelve directions generalize

The lens pole words do not occur in the 96 held-out landmark utterances. On
35B, all 12 positive-minus-negative landmark means point in the intended
direction, and 10 of 12 clustered-bootstrap lower bounds are above zero. On
27B, the counts are 12 of 12 and 11 of 12. The median target coordinate
accounts for 51.9% of the full landmark separation vector on 35B and 57.3% on
27B. The named axes carry their intended contrast, but they also move together.

## `Meh` is low-engagement and non-null

At the frozen source layer, the 35B `meh` centroid has engagement
-4.941 calibrated units with a 95% interval of [-5.770, -3.727]. Its distance
from the neutral centroid is 5.976. On 27B, engagement is -3.457
[-4.341, -2.043] and the distance from neutral is 6.194. Both models therefore
pass the non-null rule by wide margins.

The cross-model cosine between the two twelve-coordinate `meh` centroids is
0.9512, and all twelve coordinate signs agree. Across all 21 atlas families,
the median matching-family cosine is 0.9549 and mean sign agreement is 94.0%.

## The nearest label is conditional

At the confirmatory 35B source layer 35, discouragement is nearest to `meh`
(Euclidean distance 3.216), followed by frustration (3.297) and boredom
(3.720). Discouragement remains first under each of the three system prompts,
so the frozen boredom-adjacency rule fails.

At the dense 27B source layer 55, boredom is nearest overall (2.654) and under
all three system prompts. The rule passes there.

The label also rotates through residual depth when one fixed read matrix is
applied at every traced layer with layer-specific neutral calibration. On 35B,
the sequence is:

```text
L3 playfulness
L7 playfulness
L11 frustration
L15 frustration
L19 boredom
L23 boredom
L27 boredom
L31 frustration
L35 discouragement
L39 boredom
```

On 27B, late layers alternate mainly between boredom and frustration; boredom
is nearest at layers 27, 35, 39, 43, 55, 59, and 63. These traces reuse the
source-layer directions. They are not independent J-lens fits at each layer.

## The radar axes are correlated

The mean absolute off-diagonal axis correlation is 0.343 on 35B and 0.360 on
27B. The first principal component explains 48.8% and 54.1% of atlas-family
centroid variance. The strongest 35B pair is goodwill/hope at 0.715; the
strongest 27B pair is warmth/trust at 0.652. Euclidean nearest-neighbor labels
therefore summarize a correlated coordinate system, not twelve orthogonal
emotion bins.

## System prompts do little to the family geometry

The median cosine between matching family centroids under the three system
prompts is 0.9958 on 35B and 0.9957 on 27B, above the frozen threshold of 0.80.
This stability covers the three templates in the experiment. It does not claim
invariance to arbitrary prompting.

## Scope

This experiment reads a targeted twelve-direction slice of residual space. It
does not compute the full averaged Jacobian or the sparse nonnegative
decomposition used to define the full J-space. It also does not test response
steering, persistent memory, temporal decay, emotion recognition, behavior, or
subjective experience.

## Archived artifacts

For each paper model, the repository contains:

- `summary.json`, including decisions, intervals, depth traces, and environment;
- `measurements.json`, with every retained token and coordinate;
- four PNG figures;
- a self-contained `jspace-spectrum.html` replay.

`results/model-comparison-20260725.json` contains the cross-model centroid
comparison. Lens checkpoints, resumable fit states, and duplicate JSONL event
streams are excluded from the publication bundle.
