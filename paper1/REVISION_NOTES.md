# Paper 1 revision 2: changes and historical interpretation

Revision 2 is **Conditional LoRA Bridges: Separating Adaptation Capacity from
Sensor Conditioning**. The release date, summary hash, evidence-manifest hash,
and rendering-source hashes are in [manuscript/release.json](manuscript/release.json).

The original Paper 1 manuscript and result artifacts remain in the repository's
root [`arxiv/`](../arxiv/). The original 230M companion note remains in
[`note/`](../note/). Their numbers describe their historical protocols. The
corrected three-host study has separate code and evidence under `paper1/` on
the same `main` branch; its results are not pooled with the historical scores.

## Changes prompted by review

| Review concern | Revision |
| --- | --- |
| Dense bridge size and growth | Report instantiated host/bridge parameter counts, complete target maps, generated-vector dimensions, training memory, and isolated component timings. The dense output head still has 257 parameters per generated factor value. Compression is a separate architectural question. |
| Evidence from one 33.6M host | Re-run the same declared comparisons on the original mini, pinned LFM2.5-230M, and the Qwen3.5-0.8B language tower. Architecture, tokenizer, and model training history remain confounds; this is not a parameter-count scaling law. |
| Small task evaluation panels | Use all 2947 IMU windows and all 400 audio test clips, with multiple training seeds, macro/micro scores, paired contrasts, and grouped intervals. The original 64-item procedure remains a secondary diagnostic. Nine test subjects still limit IMU population coverage. |
| Missing vision behavior | Add paired CIFAR-10 category probes on mini and Qwen, with all 10000 test images and a frozen-feature linear reference. The native Qwen vision tower does not supply these features. |
| Ranking is not generation | Decode ranking-trained Qwen checkpoints with their original prompts and matched frozen references. Separately train/evaluate native-chat generation on category, sentence, and rule responses, including unseen templates, reversed rules, corrupted sensors, and label-in-text references. Publish every response. These tasks remain bounded. |
| BPB claim too broad | State that an improvement in deliberately unpaired text likelihood alone does not establish sensor use. Paired text objectives can measure conditioning. Do not treat a nonsignificant true/constant difference as equivalence. |
| Prefix claim too broad | Identify the tested interface as eight generated input embeddings. Match the 128 text positions, retain initialization and both budgets, and scope conclusions to this parameterization and recipe. |
| Collapse and training-budget claims | Connect saved-checkpoint behavior with factor geometry, effective-update geometry, and output sensitivity. Report the complete penalty grid and both budgets without selecting a test-set peak. |
| Composition | Compare historical factor averaging, effective-update sums, and effective-update means on identical constituents. Unpaired composition measures adapter interference, not semantic multi-sensor reasoning. |
| Reproduction clarity | Provide one Paper 1 namespace, dependency lock, exact input bootstrap, explicit recipes, a standard-library evidence verifier, derived statistics, and a manuscript/source-archive builder. Other paper paths retain their own commands. |

## Additional audit findings

The pinned Qwen model config names `<|endoftext|>` as EOS, while its native chat
template ends assistant responses with `<|im_end|>`. An initial decoder inherited
only the model EOS and continued past valid assistant endings. A saved-checkpoint
replay reproduced repeated category answers after those endings. Native-chat
evaluation now stops at either token and records the stop IDs. Earlier evaluator
versions are retained as audit history and excluded from final scores by the
source-version reconciliation gate. Training rows, optimizer steps, checkpoint
weights, and raw ranking-prompt decoding are unchanged by this correction.

The fixed recipe can terminate with nonfinite loss or gradient norms. A separate
execution observer records these outcomes without changing the registered
training operations. Earlier saved checkpoints remain available, but do not
replace unavailable long-budget endpoints. A primary aggregate requires every
declared seed; partial seed sets remain visible in the item-level evidence and
coverage index without a survivor-only mean. This handling was added after the
first observed numerical failure, in accordance with the preparation plan's
requirement to retain fixed-recipe failures. Hardware and infrastructure errors
remain separate recoverable interruptions.

The original audio encoder's stored normalization matches all 2000 ESC-50 clips,
including bridge-test inputs. Two original source identifiers also cross its
supplied fold boundary. The revision preserves that encoder to retain the
registered replication, discloses the exposure, and reports a source-disjoint
sensitivity analysis. Removing those two sources does not undo the broader
normalization exposure. Complete original encoder training provenance is not
available. Audio test results therefore support the stated bridge-fitting
holdout, with this explicit limit on end-to-end generalization.

Training corruption and inference intervention are now distinct. In paired
training, constant, shuffled, and random inputs are training-signal controls.
All trained controls receive the same correct-feature primary ranking test.
This can put a constant- or random-trained bridge outside its training input
distribution. Static adapters and interventions on the identical true-trained
checkpoint provide the separate comparisons needed to interpret that effect.

The executed fixed-recipe protocol uses the full test population at intermediate
IMU checkpoints, rather than the separate development diagnostic panel proposed
during preparation. Those curves are descriptive. All penalties and checkpoints
are retained; short/long endpoints remain primary, and no test-set peak selects
the reported model. The protocol contract records this choice explicitly.

The original audio classifier head uses ESC-50's numeric category order, while
task ranking uses an alphabetical order. The reference maps those orders
explicitly. A classifier-head score with the columns interpreted in the wrong
order is not a valid encoder baseline.

The frozen mini checkpoint requires its exact tokenizer and byte table. The
new input bootstrap distributes and verifies those files instead of training a
new tokenizer. A historical manuscript rebuild into a fresh directory also
requires an explicit copy of `arxiv/references.bib`; the old renderer does not
copy it. The revision's renderer includes the bibliography and verifies a
fresh-directory compile of its source archive.

A fresh macOS reproduction generated exactly the same training and evaluation
token arrays as Linux for all three hosts. PyTorch's container serialization
identifier differed. Checkpoint evaluation now retains the local file-hash
check and verifies the complete registered token arrays and input identity
when container hashes differ. This permits cross-platform input reconstruction
without accepting changed tokens or distributing the corpus again.

## Impact on the historical 230M companion note

This section is an interpretive erratum for **Conditional LoRA Bridges at 230M:
What Replicates, What Softens, and What Inverts**. It preserves the companion's
published artifact values and narrows claims that those values cannot establish.

1. Read claims that BPB is "blind" to conditioning as claims about the note's
   independently sampled sensor/text objective. They do not apply to all
   language-model likelihood evaluations.
2. Read historical true/constant similarity as a measured comparison under its
   specified seeds and evaluation panel. It is not an equivalence test or proof
   that the conditional contribution is exactly zero.
3. Read the historical prefix result as a result for its input-embedding
   implementation and text budget. It does not establish a universal ordering
   of weight-space and token-space interfaces. Revision 2 supplies a corrected
   matched-text-budget comparison on three hosts.
4. Read the long-budget task/collapse observations as fixed-protocol evidence,
   limited by the original small task panel and seed coverage. Parameter count,
   architecture, tokenizer, pretraining, and post-training change together.
   Neither favorable scaling nor a universal collapse threshold is identified.
5. Apply the audio normalization and source-overlap disclosure above to the
   companion's use of the same original encoder. Its fold-5 task panel is held
   out from bridge fitting, with the stated encoder exposure.
6. The historical standard artifact has **35 jobs**; the long-budget artifact
   has **33**, missing `repro-imu` and `repro-vision`. The new study's three-seed
   long-budget LFM cells are separate corrected evidence. They do not
   retroactively complete the historical 33-job artifact.

Both historical manuscript builders were checked against their preserved input
artifacts and reproduced byte-identical LaTeX. A separate CPU FP32 migration
audit checked mini and LFM frozen logits, identical generated LoRA factors, and
pre-shifted token losses. That check validates the extracted forward interface;
it is not a claim of bitwise historical experiment reproduction across software
versions, devices, or the corrected evaluation protocol.

The companion's historical command path remains available from the repository
root, with all four bundled inputs supplied explicitly:

```bash
uv run --project paper1 --frozen python -m src.render_note \
  --mini arxiv/summary.json --mini-scaling arxiv/summary-scaling.json \
  --lfm note/summary-lfm.json --lfm-scaling note/summary-lfm-scaling.json \
  --output-dir paper1/runs/historical-companion --compile
```

The companion renderer's default LFM paths point to the original run directory;
use the bundled `note/` paths above in a clean checkout. Its preserved layout
has a small overfull line in the introduction. Numerical revision work belongs
to `paper1/`; the historical renderer and artifacts are retained intact.
