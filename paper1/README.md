# Paper 1: conditional LoRA bridges

This directory reproduces **Conditional LoRA Bridges: Separating Adaptation
Capacity from Sensor Conditioning**, revision 2. Its package, dependency lock,
recipes, evidence, and manuscript belong to Paper 1. All papers stay on `main`.
The root `src/` and `arxiv/` are the historical Paper 1 release; other studies
use the paths in the repository's main index.

Read the [method contract](METHOD_CONTRACT.md) before comparing scores. U is
unpaired text adaptation, P is paired category training, and G is separate
sensor-plus-instruction training. A result on one cohort is not evidence about
the weights trained in another. The registered revision has 411 trials: 406
training trajectories and five frozen references with zero optimizer steps.
The planned training budget is 1,896,000 optimizer steps, plus explicit measurement
obligations. Numerical terminations can shorten training trajectories; the released
outcome records identify each termination and its recorded step range.

## Inspect the released paper and evidence

- [Final manuscript and LaTeX source](manuscript/)
- [Verified evidence](artifacts/revision-2/evidence/): input identities, training
  manifests, checkpoint hashes, full per-item evaluations, and resource records
- [Derived summary](artifacts/revision-2/summary.json): grouped intervals, paired
  comparisons, per-seed estimates, and all registered intermediate measurements
- [Source extraction record](extraction.json): original mini and encoder files
- [Checkpoint download index](artifacts/revision-2/checkpoints.json): exact FP32
  adapter tensors and frozen feature caches, with sizes, hashes, and immutable
  HTTPS URLs; adapters also retain their recipe and step

The evidence verifier needs only Python's standard library. It does not load a
model or download a dataset. Materialize the evidence's Git LFS objects first:

```bash
git lfs pull --include 'paper1/artifacts/revision-2/evidence/**'
cd paper1
python3 verify_artifacts.py artifacts/revision-2/evidence
```

Verification checks content hashes, all registered trials and checkpoints,
training/scoring identities, full evaluation populations, numeric finiteness,
acceptance evidence, all 396 composition cells, and all 36 isolated resource
cases. A missing or partial matrix is not a final release. Failed acceptance
attempts and undeclared experiments are excluded explicitly.

## Recompute statistics and build the paper

Install this directory's environment, not the root harness environment:

```bash
uv sync --frozen
uv run --frozen paper1 aggregate \
  --evidence artifacts/revision-2/evidence --output runs/reanalysis
uv run --frozen paper1 render \
  --summary runs/reanalysis/summary.json \
  --evidence artifacts/revision-2/evidence \
  --output runs/manuscript --date YYYY-MM-DD --compile \
  --package runs/paper1-source.zip
```

Replace `YYYY-MM-DD` with the ISO date in `manuscript/release.json`. Compilation
requires [Tectonic](https://tectonic-typesetting.github.io/). The renderer copies
the bibliography, derives numerical text/tables/figures from the summary, and
compiles the source archive in a new temporary directory. It rejects incomplete
evidence, a mismatched summary, and changed statistical source. `--draft` permits
an explicitly marked incomplete preview and cannot produce a release archive.

PDF byte identity can vary with the TeX environment and creation metadata.
Empirical values and source identities must agree. Scientific plots are vector
PDFs; `manuscript/condition-index.csv` indexes every statistical cell without
requiring a model environment to read it.

## Re-run the full registered study

The tested execution environment uses Python 3.14.3, the committed `uv.lock`,
FP32 throughout, and one RTX PRO 6000 with 95.01 GiB reported device memory.
TF32 and autocast are disabled. Three coordinated workers were accepted on that
device. The code also exposes CPU/MPS execution for bounded checks, but those
backends do not inherit the CUDA numerical validation or resource results.
No quantization, different rank, narrowed target map, or shortened sequence is
substituted for a larger model that does not fit.

First materialize the original public assets. Run from the repository root:

```bash
git lfs pull --include 'checkpoints/**,data/vision/**,paper1/assets/**'
cd paper1
uv sync --frozen --python 3.14.3
```

Choose absolute data and output directories with enough disk space for the
complete checkpoint set. The large trained tensors are available through a
separate checkpoint index. They are not stored in ordinary Git history.
Full training regenerates them from the pinned original model assets.

```bash
export PAPER1_RUN=/absolute/path/to/paper1-run
export PAPER1_DATA=/absolute/path/to/paper1-inputs
export PAPER1_ASSETS=/absolute/path/to/legolm.paper
mkdir -p "$PAPER1_RUN"

uv run --frozen paper1 bootstrap \
  --asset-root "$PAPER1_ASSETS" --data-cache "$PAPER1_DATA"
uv run --frozen paper1 matrix --output "$PAPER1_RUN/registered/configs"
```

`bootstrap` verifies the frozen repository assets, copies the exact bundled mini
tokenizer and byte table, downloads four pinned ClimbMix shards with SHA-256
checks, and prepares the original raw audio/IMU datasets. It rejects mismatched
existing cache files without overwriting them. Do not retrain a tokenizer for
the frozen mini checkpoint. Hugging Face model/tokenizer revisions are pinned
in `hosts.py`; their downloads occur when those hosts are first loaded.

Select an exclusively allocated experiment GPU before model execution.
`CUDA_VISIBLE_DEVICES` selects its UUID; `PAPER1_EXPECTED_GPU` checks its model
name. Replace both examples with the allocated device's values from `nvidia-smi -L`:

```bash
export CUDA_VISIBLE_DEVICES=GPU-your-experiment-device-uuid
export PAPER1_EXPECTED_GPU='NVIDIA RTX PRO 6000 Blackwell Workstation Edition'
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

uv run --frozen paper1 prepare \
  --asset-root "$PAPER1_ASSETS" --data-cache "$PAPER1_DATA" \
  --output "$PAPER1_RUN/features-fp32" --device cuda

for paper1_host in mini lfm230m qwen08; do
  uv run --frozen paper1 prepare-text --host "$paper1_host" \
    --asset-root "$PAPER1_ASSETS" --data-cache "$PAPER1_DATA" \
    --output "$PAPER1_RUN/text"
  uv run --frozen paper1 canary --host "$paper1_host" \
    --asset-root "$PAPER1_ASSETS" --data-cache "$PAPER1_DATA" \
    --output "$PAPER1_RUN/canaries/$paper1_host-fp32.json" --device cuda
done
uv run --frozen paper1 prepare-common-text \
  --data-cache "$PAPER1_DATA" --output "$PAPER1_RUN/text"
```

Feature preparation also downloads CIFAR-10 and fixes its development split.
It records raw-source hashes, feature hashes, all train/test IDs, labels, and
source groups. The following commands use those exact materialized features:

```bash
uv run --frozen paper1 encoder-audit \
  --asset-root "$PAPER1_ASSETS" --data-cache "$PAPER1_DATA" \
  --features "$PAPER1_RUN/features-fp32" --text "$PAPER1_RUN/text" \
  --output "$PAPER1_RUN/audit" --device cpu

uv run --frozen paper1 linear-probes \
  --asset-root "$PAPER1_ASSETS" --data-cache "$PAPER1_DATA" \
  --features "$PAPER1_RUN/features-fp32" --text "$PAPER1_RUN/text" \
  --output "$PAPER1_RUN/linear-v2" --device cuda

uv run --frozen paper1 accept \
  --asset-root "$PAPER1_ASSETS" --data-cache "$PAPER1_DATA" \
  --features "$PAPER1_RUN/features-fp32" --text "$PAPER1_RUN/text" \
  --output "$PAPER1_RUN/canaries" --device cuda
PYTHONPATH=src uv run --frozen python -m unittest discover -s tests -v
```

The acceptance trajectories are excluded from scientific results. Do not
continue past a failed canary by increasing its tolerance. The source and
backend identities must remain fixed during a training matrix.

On the accepted 96 GB device, run the parallel canary and then the full matrix:

```bash
uv run --frozen paper1 schedule --parallel-canary --workers 3 \
  --asset-root "$PAPER1_ASSETS" --data-cache "$PAPER1_DATA" \
  --features "$PAPER1_RUN/features-fp32" --text "$PAPER1_RUN/text" \
  --output "$PAPER1_RUN/grid" --device cuda

uv run --frozen paper1 schedule --workers 3 --evaluate \
  --linear-dir "$PAPER1_RUN/linear-v2" \
  --asset-root "$PAPER1_ASSETS" --data-cache "$PAPER1_DATA" \
  --features "$PAPER1_RUN/features-fp32" --text "$PAPER1_RUN/text" \
  --output "$PAPER1_RUN/grid" --device cuda
```

The scheduler uses at most three workers and a 25% allocator cap per worker on
CUDA. Its parallel acceptance check is mandatory for parallel execution. A
single-worker run uses `--workers 1` and does not need the parallel canary, but
must still fit the unchanged scientific recipe and pass all host/training
checks. Do not carry a manually set 25% per-process override into isolated
resource measurements.

Each trajectory writes immutable scientific checkpoints and an atomic resume
checkpoint with optimizer/RNG state. The scheduler records each child PID and
attempt log. Re-run the same scheduling command after a recoverable interruption;
completed trajectories are checked and interrupted ones resume from their saved
state. If CUDA reports an Xid, recover the
allocated device through the machine's operating procedure before resuming.
The package does not stop production services or reset GPUs itself.

A nonfinite loss or gradient norm inside the unchanged optimizer operation is a
terminal **numerical failure**, distinct from an interrupted job. The supervisor
records the failure before an optimizer update, evaluates checkpoints already
saved, and continues the other registered trials. It does not retry seeds until
they succeed. CUDA, memory, I/O, and unknown errors still stop the queue.

The final release must account for all 411 declared trials and evaluate every
available registered checkpoint. Missing later endpoints from numerical failures
are recorded explicitly. Each primary estimate requires its complete registered
seed set; a dash does not hide an average over surviving seeds. The evidence and
condition index retain the failure, seed coverage, and available per-seed scores.

Changing numerical training source, data, precision, or model identity creates a
different matrix. Do not merge it into the published one. If scoring code changes
during a retry, multiple implementations of one cell require explicit source
reconciliation; the exporter refuses to select a result by its value.

After the evaluated matrix completes, obtain its exact path and reconcile its
scoring implementation. This is an audit-only pass when every measurement was
already made with the final scorer. Otherwise, it preserves old per-item files,
archives their catalogs, and re-evaluates the affected immutable checkpoints.
Selection depends on source/backend/batch identity, never on score. It also
fills the matching frozen raw-generation references. Then run the serial
supplement, with the GPU isolated for resource measurements:

```bash
PAPER1_MATRIX="$(uv run --frozen python -c \
  'import json,os; print(json.load(open(os.environ["PAPER1_RUN"]+"/grid/active-matrix.json"))["directory"])')"

uv run --frozen paper1 reconcile --matrix "$PAPER1_MATRIX" \
  --asset-root "$PAPER1_ASSETS" --data-cache "$PAPER1_DATA" \
  --features "$PAPER1_RUN/features-fp32" --text "$PAPER1_RUN/text" \
  --linear-dir "$PAPER1_RUN/linear-v2" \
  --output "$PAPER1_RUN/reconciliation" --device cuda

uv run --frozen paper1 supplement --matrix "$PAPER1_MATRIX" \
  --asset-root "$PAPER1_ASSETS" --data-cache "$PAPER1_DATA" \
  --features "$PAPER1_RUN/features-fp32" --text "$PAPER1_RUN/text" \
  --output "$PAPER1_RUN/supplements" --device cuda

uv run --frozen paper1 export-artifacts \
  --matrix "$PAPER1_MATRIX" --run-root "$PAPER1_RUN" \
  --output "$PAPER1_RUN/release/evidence"
uv run --frozen paper1 aggregate \
  --evidence "$PAPER1_RUN/release/evidence" --output "$PAPER1_RUN/release"
```

The supplement requires three matching frozen raw-generation references,
18 composition cases (396 operator/subset cells), and 36 resource cases. It does
not start until all 411 main trials have terminal outcomes and their available
checkpoints have completed evaluation. A missing required composition or resource
checkpoint is an explicit dependency error; it cannot be replaced by another seed
or budget.
Use the rendering command above with this release's summary and evidence.

The final evidence verifier requires the reconciliation record and a single
scoring implementation for all checkpoint measurements. It also checks that
the shipped numerical evaluator files match the recorded hashes. Changes to
that evaluator require reconciliation before a new final release.

The directory names in these commands are intentional: the exporter uses
`features-fp32/`, `text/`, `canaries/`, `linear-v2/`, `audit/`, `grid/`, and
`supplements/` beneath the explicit run root. Do not point it at an unrelated
folder that happens to contain similarly named results.

## Evaluate a downloaded checkpoint

Use the bootstrap, text-preparation, and host numerical-canary commands above
for checkpoint evaluation. Text preparation regenerates the same integer token
arrays on Linux and macOS. PyTorch container identifiers can differ across
platforms, so evaluation verifies the registered input identity and both full
token-array hashes when the serialized file hash differs. Changed tokens are
rejected. The reference hashes are in `assets/text-cache-identities.json`.
Download the exact frozen feature caches below in
place of re-encoding sensors: small arithmetic differences between encoder
devices must not silently change a checkpoint's registered input features.
Their sidecar metadata comes from the bundled evidence, whose ordinary JSON
files do not require materializing all per-item Git LFS records. The three
feature caches total about 140 MiB and preserve the stated encoder-exposure
limitations. Use a fresh output directory if other feature files already exist.

The training matrix can then be skipped. This example downloads the Qwen IMU
true-feature P bridge at 6000 steps and seed 42, then runs that checkpoint's
registered measurement suites:

```bash
uv run --frozen paper1 fetch-features \
  --index artifacts/revision-2/checkpoints.json \
  --evidence artifacts/revision-2/evidence \
  --output "$PAPER1_RUN/downloaded-features"

uv run --frozen paper1 fetch-checkpoint \
  --index artifacts/revision-2/checkpoints.json \
  --host qwen08 --cohort P --modality imu --kind bridge \
  --feature-mode true --seed 42 --step 6000 --diversity 0 \
  --output "$PAPER1_RUN/downloaded-checkpoints" \
  > "$PAPER1_RUN/selected-checkpoint.json"

PAPER1_CHECKPOINT="$(uv run --frozen python -c \
  'import json,os; print(json.load(open(os.environ["PAPER1_RUN"]+"/selected-checkpoint.json"))["checkpoint"])')"

uv run --frozen paper1 evaluate --checkpoint "$PAPER1_CHECKPOINT" \
  --asset-root "$PAPER1_ASSETS" --data-cache "$PAPER1_DATA" \
  --features "$PAPER1_RUN/downloaded-features" --text "$PAPER1_RUN/text" \
  --output "$PAPER1_RUN/checkpoint-evaluation" --device cuda --sensor-batch 4 \
  --eval-modality imu \
  --suites primary common geometry paraphrases interventions generation
```

The command selects exactly one checkpoint and verifies its size and SHA-256.
It rejects an absent/ambiguous recipe and a mismatched existing local file.
The index lists all retained checkpoints, including short endpoints and saved
trajectory milestones. The [measurement coverage map](artifacts/revision-2/checkpoint-measurement-coverage.json)
separates measured checkpoints from additional snapshots retained by the shared
saving policy; the latter have no registered score. Frozen hosts and static U adapters are shared across
modalities as declared in the index. Model/feature/text identities are checked
again before evaluation. The full archive is large; downloading a single
checkpoint does not download the other trajectories or optimizer states.

## Inspect numerical behavior during generation

The [published fixed-batch observations](artifacts/revision-2/diagnostics/g-audio-normalization.json)
reproduce the saved responses for all three Qwen G/audio seeds. In seeds 42 and
1042, the final RMS normalization overflows its FP32 mean square on the first
forward pass, returns zeros, and each decoding step has zero logits. Greedy
decoding selects token zero, "!". Seed 2042 has no normalization overflow in
this inspected batch. This artifact is separate from the scored evidence and
does not change the retained generation results.

The optional diagnostic observes the first fixed batch of eight G instructions
for one sensor. It records every decoding step's logits and the first forward
pass's RMS normalization inputs, outputs, FP32 variance, and FP64 variance
recomputation. It preserves the registered decoder and does not replace task
scores. Complete the environment, input bootstrap, evidence verification, and
frozen-feature download above first. For the published Qwen G/audio seed 42:

```bash
uv run --frozen paper1 fetch-checkpoint \
  --index artifacts/revision-2/checkpoints.json \
  --host qwen08 --cohort G --modality audio --kind bridge \
  --feature-mode true --seed 42 --step 6000 --diversity 0 \
  --output "$PAPER1_RUN/downloaded-checkpoints" \
  > "$PAPER1_RUN/selected-g-audio.json"

PAPER1_G_CHECKPOINT="$(uv run --frozen python -c \
  'import json,os; print(json.load(open(os.environ["PAPER1_RUN"]+"/selected-g-audio.json"))["checkpoint"])')"
PAPER1_G_REFERENCE="$(uv run --frozen python -c '
import json
from pathlib import Path
root=Path("artifacts/revision-2/evidence")
rows=[r for r in json.loads((root/"index.json").read_text())["evaluations"]
      if r["measurement"]=="generation" and r["step"]==6000
      and all(r["recipe"][k]==v for k,v in {
          "host":"qwen08", "cohort":"G", "modality":"audio", "kind":"bridge",
          "feature_mode":"true", "seed":42, "diversity":0.0}.items())
      and r["settings"]=={"objective":"G", "intervention":"correct"}]
assert len(rows)==1
print((root/rows[0]["path"]).resolve())')"

uv run --frozen paper1 diagnose-generation \
  --checkpoint "$PAPER1_G_CHECKPOINT" --reference "$PAPER1_G_REFERENCE" \
  --asset-root "$PAPER1_ASSETS" --data-cache "$PAPER1_DATA" \
  --features "$PAPER1_RUN/downloaded-features" \
  --output "$PAPER1_RUN/g-audio-diagnostic.json" --device cuda
```

Use a new output filename for each diagnostic. The command verifies the
checkpoint, feature, host, evaluator, decoder, and case identities, records its
backend, and reports whether the responses match the published batch. Another
backend need not match bitwise. The observations describe this fixed batch;
they do not estimate full-panel incidence, latency, or the effect of changing
normalization arithmetic. No arithmetic is changed in the model itself.

## Interpretation and historical compatibility

The larger-host runs preserve the recipe and scientific comparisons. They do
not reproduce the mini's numeric scores or isolate parameter count from changes
in architecture, tokenizer, training history, and instruction tuning. A seed
does not promise bitwise equality on another backend. Each run records its
complete target map, actual parameter counts, software versions, and precision.

The original audio encoder used all clips for normalization before splitting
folds. The release discloses that exposure and separately removes the known
source overlaps for a sensitivity analysis. This secondary analysis does not
restore a clean end-to-end audio holdout. The original classifier-head baseline
also maps ESC-50 numeric taxonomy to the task's alphabetical category order.

`extraction.json` attributes the four original architecture/encoder snapshots.
The new package imports `legolm_p01`; it never imports another paper's root
package named `src`. Shared checkpoint assets remain at the public repository
root and are pinned in `assets/inputs.json`.

To rebuild the historical manuscript into a new directory, copy the historical
`arxiv/references.bib` there before invoking its renderer: the old renderer does
not copy that file. Its original result artifacts remain historical evidence;
the corrected revision does not silently replace their measurements.
