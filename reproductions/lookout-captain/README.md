# The Lookout and the Captain

This directory reproduces the mode-conditioning and invocation experiments
with LFM2.5-VL-450M, LFM2.5-VL-3B, and Qwen3.8-27B. It contains its own
environment, saved measurements, experiment code, and paper renderer.
Other studies use other directories on the same `main` branch.

The task has three actions: greet, stop, and proceed. A detector selects
an operator-mode packet; state-conditioned receivers modify a frozen
vision-language model's residuals. The stream experiment measures skipped
model invocations and retained-action accuracy in a replay.
Read [the manuscript](paper/main.pdf) for the scope and negative results.

## Rebuild the published analysis and paper on a CPU

Install Python 3.12 or newer, [uv](https://docs.astral.sh/uv/), and
[Tectonic](https://tectonic-typesetting.github.io/). From this directory:

~~~bash
uv sync --frozen --group dev
uv run --frozen pytest -q
uv run --frozen lookout-analyze
uv run --frozen lookout-render
~~~

These commands verify all 232 evidence-file hashes, recompute individual
response metrics and paired intervals, write `analysis/summary.json`, and
build `paper/main.pdf`. No model, image download, or GPU is required.
Tectonic may fetch its TeX bundle on its first build.

Edit `paper/main.template.tex` for prose and `paper/references.bib` for
references. Numeric tables, figures, and macros come from
`src/legolm_lookout/render.py` and the audited analysis. `paper/main.tex`
and `paper/numbers.tex` are generated. The renderer also writes
`paper/source.zip`. The source archive contains a
complete LaTeX build, including its figures and bibliography.

## Files and provenance

| Path | Purpose |
| --- | --- |
| `artifacts/` | Original 232 JSON files, including individual results, training records, pooled readings, and detector diagnostics |
| `analysis/summary.json` | Recomputed shares, paired contrasts, conditional bootstrap intervals, cut censoring, and audit findings |
| `harness/tracks/vision-mode/` | Exact study scripts, dated experiment plan, and scene asset definition |
| `harness/tracks/model-of-models/` | The two local dependencies required by the study scripts |
| `experiments.json` | All 72 recorded run groups in dependency order, with their actual seed sets |
| `provenance/evidence-manifest.json` | Source commit and SHA-256 for every saved result |
| `provenance/harness-manifest.json` | Exact archived harness file identities |
| `provenance/models.json` | Host revisions and the basis for recovering each identity |
| `provenance/coco-inputs.json` | Disjoint image-ID sets and recovered hashes of all 3,812 eligible photographs |
| `provenance/rendered-inputs.json` | Scene draws, two recorded pixel versions, and the mixed-training recipe |
| `provenance/checkpoints.json` | 69 study checkpoint downloads and three upstream detector identities |
| `provenance/checkpoint-verification.json` | Verification of all 72 recorded checkpoint hashes |
| `REVISION_NOTES.md` | Corrections and limitations carried into this revision |

Historical artifact names are retained. The original `experiment` field
sometimes omits a variant suffix; use filenames and checkpoint identities
when joining results. The analysis does so. Historical pooled readings
are not its numerical source.

The harness snapshot is pinned to the source commit in its manifest.
Original records did not hash uncommitted source or record a complete
environment lock. The 27B host revision is explicit in those records.
The LFM revisions were recovered from the study machine's single cached
snapshots; this is recorded provenance, not proof that immutable revisions
were recorded at execution time.

The optional experiment environment pins the main libraries recovered
from that machine at publication time. It is a repeatable environment for
a new run, not a retroactive historical lock. Retraining and inference can
differ across devices and software. Saved-data analysis is independent
of that uncertainty.

## Prepare inputs for model experiments

Use a CUDA machine for the full suite. The original runs used an RTX PRO
6000 with 96 GB VRAM. The 27B model alone uses about 52 GiB of bfloat16
weights, before images, activations, and runtime allocations. The LFM
hosts require substantially less memory. The suite runs one host process
at a time.

~~~bash
uv sync --frozen --extra experiments --group dev
uv run --frozen --extra experiments lookout-inputs coco --download
uv run --frozen --extra experiments lookout-inputs models --host all --download
uv run --frozen --extra experiments lookout-inputs detectors --download
~~~

COCO defaults to `~/data/coco/`, containing
`annotations/instances_val2017.json` and `val2017/*.jpg`. Supply
`lookout-inputs coco --directory /your/coco/path` to verify an existing
copy, and pass `--coco-dir /your/coco/path` to individual experiment
commands. Omitting `--download` verifies inputs without fetching them.
The helper uses the dataset's HTTPS S3 endpoint and checks the selected
image and annotation hashes.

For evaluation with the original trained adapters:

~~~bash
uv run --frozen --extra experiments lookout-inputs checkpoints --download
~~~

These 69 files total about 1.37 GB and are stored in ignored
`harness/checkpoints/experiments/`. The three detectors come from
Ultralytics' release assets and must match their recorded hashes.
Training writes new checkpoints there. Use a separate checkout for fresh
training if you also want to retain an unchanged evaluation cache.
Experiment commands never overwrite the published `artifacts/`.

The mesh harness downloads and verifies one HSRD-100 archive as needed.
Original temporary rendered PNGs were not recovered. Their scene
parameters and evaluation hashes survive. Two pixel versions have
identical scene draws; rendering devices can produce different PNG bytes.
After regenerating the scenes:

~~~bash
uv run --frozen --extra experiments lookout-inputs renders \
  --kind mesh --directory /tmp/vm-r3-frames
uv run --frozen --extra experiments lookout-inputs renders \
  --kind capsule --directory /tmp/vm-r3-frames
uv run --frozen --extra experiments lookout-inputs renders \
  --kind mixed_training --directory /tmp/vm-r3-trainframes
~~~

The mesh check reports which recorded pixel version matches. A mismatch
means the new render is not pixel-identical to either recorded version;
retain its manifest when reporting a replication. On Linux without a
display, select a MuJoCo backend, for example `MUJOCO_GL=egl` with a
configured NVIDIA OpenGL environment.

## Run the complete recorded experiment set

List exact commands without executing them:

~~~bash
uv run --frozen --extra experiments lookout-suite
~~~

Run all groups after preparing inputs:

~~~bash
uv run --frozen --extra experiments lookout-suite --run all
~~~

The suite covers 171 individual seeded records plus detector diagnostics.
Its 72 groups become 86 commands. Three-seed groups use `--all-seeds`
where supported; short follow-ups use only their recorded one or two seeds.
It trains the original adapter pairs, classifier, box projections, and
alternate recipes, then runs the comparisons in dependency order.
New results are under `harness/findings/`; originals remain in `artifacts/`.

To resume after an interruption, use the printed group ID with
`lookout-suite --run ID`. Ensure its earlier checkpoint and result
dependencies exist. Completed groups do not need to be discarded.
The suite prints each command before starting and does not infer
completion from a partial output file.

Analyze a complete new run separately:

~~~bash
uv run --frozen --extra experiments lookout-analyze \
  --artifacts harness/findings --new-run --output runs/summary.json
~~~

This checks observation populations and reported metrics and computes
the same contrasts and intervals. It marks the input as a new run and
records its hashes separately from publication evidence. Historical
pooled JSON may have different seed summaries when short follow-ups
run separately; the analysis uses the individual seed records.

## Individual experiment commands

Each script has `--help`. The launcher sets pinned host revisions
without changing the archived source bytes. Examples:

~~~bash
# Original adapter training and evaluation.
uv run --frozen --extra experiments lookout-experiment vm_rung1 --host 27b --all-seeds --device cuda

# Detector switch and contrary text.
uv run --frozen --extra experiments lookout-experiment vm_rung2y --host 27b --all-seeds --device cuda
uv run --frozen --extra experiments lookout-experiment vm_rung1b --host 27b --emitter oracle --dose-scale 1.5 --all-seeds --device cuda

# Box-feature and blank-image projection training/evaluation.
uv run --frozen --extra experiments lookout-experiment vm_rung2b --host 3b --all-seeds --device cuda
uv run --frozen --extra experiments lookout-experiment vm_rung2c --host 3b --all-seeds --device cuda

# Shared 60-episode replay, including the large host.
uv run --frozen --extra experiments lookout-experiment vm_rung2t --host 27b --all-seeds --device cuda

# Rendered detector test and host transfer.
uv run --frozen --extra experiments lookout-experiment vm_rung3 --part 1 --figure mesh --device cuda
uv run --frozen --extra experiments lookout-experiment vm_rung3 --part 2 --figure mesh --host 27b --all-seeds --device cuda

# Mixed training and fixed-constant-baseline evaluation for one seed.
uv run --frozen --extra experiments lookout-experiment vm_door2x --host 27b --seed 42 --variant mx2 --device cuda
uv run --frozen --extra experiments lookout-experiment vm_rung1b --host 27b --seed 42 --door-variant mx2 --device cuda
uv run --frozen --extra experiments lookout-experiment vm_rung3 --part 2 --figure mesh --host 27b --seed 42 --door-variant mx2 --constant-variant registered --device cuda

# Crossed initialization and training-order seeds.
uv run --frozen --extra experiments lookout-experiment vm_door2x --host 27b --seed 42 --variant mx2 --order-seed 1042 --device cuda
~~~

`--host all` in the archived scripts means the two LFM hosts; it does
not include 27B. The suite names 27B explicitly. Its full manifest also
includes classifier emitters, detector sizes, the capsule host probe,
shorter mixed training, doubled photo-only training, and crossed orders.

## Interpretation and licenses

The main Text control includes a separately trained constant adapter.
The wrong-mode control is a fixed cycle. Detector gaps use costs on
actual errors. Intervals resample images or episodes with tested seeds
held fixed; they do not estimate arbitrary seed or phrase robustness.
Corrected delay accounting excludes starts and includes unresolved cuts.
Avoided invocations are not measured system energy savings.

Study code is under the included [MIT license](LICENSE). The human scan,
HSR0081-Body-004 Scan-LOD2, is from
[Digital Reality Lab's HSRD-100](https://huggingface.co/datasets/digitalrealitylab/HSRD-100),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The renderer changes placement and yaw. The archive hash and attribution
are in `harness/tracks/vision-mode/assets/README.md`.
COCO images, pretrained hosts, MobileNet weights, YOLO weights, and
Ultralytics software retain their upstream licenses.
