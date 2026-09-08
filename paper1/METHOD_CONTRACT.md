# Corrected Paper 1 protocol

The host, data, optimizer, seed, and budget choices in this contract and
`legolm_p01.grid` were fixed before confirmatory outcomes. The numerical-failure
handling clarification below was added after the first such failure. The
historical release is retained separately; corrected replication preserves its
scientific comparisons, not its old numerical values.

## Hosts and precision

The required hosts are the published mini checkpoint, pinned LFM2.5-230M, and
the pinned Qwen3.5-0.8B language tower. Exact revisions, tokenizer identities,
all adapted module names/shapes, and unused native vision keys are in each run.
The optional 2B extension is excluded before observing revision outcomes.

Parameters, optimizer state and computation use FP32. TF32 and autocast are
disabled. A BF16 acceptance run on LFM failed strict batch/scalar equivalence
(maximum absolute discrepancy 0.365 nats); FP32 passed. The same FP32 policy
applies to all hosts and frozen sensor feature preparation. Results from other
precisions/backends require their own acceptance evidence.

The dense bridge retains rank 4, alpha 4, a 64-dimensional projected sensor
concatenated with a trainable 64-dimensional vector shared across inputs, two
width-256 hidden layers, and a dense output head. Static adapters use the same
target map. Input prefixes contain eight generated embeddings plus 128 text
tokens. The historical prefix comparison used a different text budget and is
not the corrected primary comparison.

## Data and objectives

U trains on independently sampled ClimbMix text and frozen sensor features.
It cannot establish sensor grounding by text likelihood alone. The materialized
native-tokenizer streams have 128 predicted positions, batch 8, and checkpoints
at 300/3000 steps. Three initialization seeds are 42, 1042, and 2042.
Common-byte evaluation separately scores 512 fixed held-out raw-text spans.
Native-tokenizer packing and common-byte scoring are distinct products.

P trains on paired sensor/category examples with the original whole
prompt-plus-label mean loss, batch 4, checkpoints at 600/6000 steps. Ranking
scores only label tokens, including natural tokens crossing the prompt/label
boundary, and uses a single next-token shift. IMU uses 7352 train/2947 test
windows grouped by subject. Audio uses ESC-50 folds 1–4 (1600 clips) and fold 5
(400 clips), grouped by source. Mini and Qwen IMU/audio comparisons have five
seeds (the three above plus 3042 and 4042); LFM has three.

Vision uses the original frozen ImageNet MobileNet and BBB-trained perceiver.
The original BGR preprocessing is retained explicitly. New paired evidence
uses CIFAR-10 with 45000 bridge-training, 5000 development, and all 10000 test
images. Mini and Qwen have three seeds. No development-driven model or budget
selection is part of this fixed-recipe revision. Known encoder training and
unknown upstream pretraining exposure are disclosed separately.

True, constant, shuffled, random, static, and frozen controls are distinct.
Constants use training features only. P shuffled training rotates each batch;
random features use an independent deterministic step stream. All trained
controls receive the same correct-feature primary ranking evaluation. Separate
inference interventions use constant, random, off, different-class sensors, and
five saved permutations. This separation replaces the old practice of combining
training and test corruption in one control score.

The IMU diversity grid is lambda 0/.01/.05/.10/.20 for U and P, three seeds on
all hosts. Saved checkpoints link task accuracy, likelihood, factor geometry,
effective-update geometry and output sensitivity from the same weights. A factor
cosine is not, by itself, a test of useful conditioning.

Intermediate IMU checkpoints use the same full test population for ranking and
fixed test-feature subsets for geometry. These curves are descriptive; only
the declared short/long endpoints are primary comparisons. Unlike the earlier
preparation proposal, this fixed-recipe implementation does not use a separate
development diagnostic panel or select a penalty/checkpoint from a curve.
All penalties and saved points are retained. Candidate-score margins and entropy
are derived from the saved ranking scores; entropy uses a softmax of mean label
log-likelihoods and is not a calibrated class-probability estimate.

## Generation

N5a decodes ranking-trained bridges without further fitting. N5b is separate
Qwen generation training on IMU/audio, batch 4, 600/6000 steps, three seeds,
and up to 256 tokens. Only the native non-thinking chat template is used for
N5b, with assistant-response-only loss and native EOS. Overlong examples fail;
there is no undocumented truncation.

`generation.py` fixes three training and three held-out templates for category,
short-sentence, and binary-rule tasks. Every rule is also tested with its output
mapping reversed, while the sensor stays fixed. Query matches are balanced,
including audio's fifty categories. Ordinary input prompts never reveal the
true category; explicitly marked oracle/classifier baselines do.

The panel contains every audio test clip and 300 IMU windows sampled by
class/subject before outcomes. Multiple instructions do not multiply the number
of independent sensor units. Generation is greedy, at most 64 tokens, with a
new model cache after any sensor change. Publish all responses and failures.
Native-chat decoding stops at either the model EOS (`248044`) or the tokenizer's
assistant-end token (`248046`, `<|im_end|>`). These IDs differ in the pinned Qwen
checkpoint. Raw ranking-prompt generation retains the model's stopping policy.
Primary generation scores remain semantic label match, format compliance, and
exact joint success. A secondary rule diagnostic requires both reversed
mappings for the same sensor/template to succeed, then averages templates within
the sensor. It does not turn repeated prompts into independent sensor samples.
Deterministic category/format/exact-rule scores support bounded controlled
generation claims. They do not establish unrestricted instruction following or
cross-modal reasoning. No model judge substitutes for these primary scores.

## Composition, provenance and uncertainty

The same U constituents at steps 150/1500 supply all seven modality subsets.
Report historical factor averaging, effective-update sum, and effective-update
mean separately. These operators differ mathematically; no rank recompression
is performed. Streams remain independently sampled.

The grid lists 411 unique trials: 406 training trajectories and five frozen
references with zero optimizer steps. Static U adapters and frozen hosts are
reused across modality evaluations; short endpoints and valid composition
milestones share their longer trajectories. Training identity binds all numerical
training sources, dependencies, host/tokenizer, features, text, seed and recipe.
Evaluation identity additionally binds the checkpoint hash and scoring sources.
Documentation and plotting changes do not change trained weights. Incomplete
grids and numerical failures remain visible; never average an undeclared subset.

The execution observer treats nonfinite loss or gradient norms before an optimizer
update as terminal fixed-recipe outcomes. It preserves the unchanged training
operations and records the failed step, error, trace, and saved checkpoints.
Seeds are not retried until they succeed. Earlier checkpoints are evaluated but
cannot replace unavailable later milestones. Each primary aggregate requires
its complete declared seed set; available per-seed evidence remains published
when the aggregate is unavailable. This clarification implements the original
preparation plan's failure-reporting rule after the first observed nonfinite
gradient norm. A gradient failure means the registered norm check rejected a
nonfinite value. The receipt alone does not distinguish nonfinite gradient
elements from overflow during norm computation. CUDA, memory, I/O, and unknown
errors still stop execution for diagnosis.
The IMU penalty curves retain their declared shared seeds 42, 1042, and 2042 at
every penalty. Their coverage is checked separately from the five-seed primary
endpoints. A failure outside those three seeds cannot change a curve's population;
a failure within them makes that curve's later points unavailable. The summary
and CSV identify both scopes explicitly.

Report micro and macro accuracy, class counts, seed variation, and grouped
resampling intervals. Resample IMU subjects and audio sources, not nominally
independent overlapping windows. Statistical nonsignificance is not equivalence.
Resource measurements report the dense bridge's actual size and memory costs;
freezing a host does not establish edge suitability.
