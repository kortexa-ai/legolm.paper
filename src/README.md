# Paper Source

This directory is the experiment layer for the manuscript. The authoritative
paper is `arxiv/main.tex`, which is **generated** by `src/render_arxiv.py` from
a bundled result artifact — never hand-edit the tex; edit the renderer
templates and re-render. `METHOD_CONTRACT.md` specifies what each experiment
must measure.

It is intentionally narrow:

- It freezes the paper-facing experiment logic in one place.
- It reuses the repo checkpoints, tokenizer assets, and datasets.

What it contains:

- Qwen mini-model runtime (`QwenModel`, LoRA shells, hyper-weight injection)
- Tokenizer loading and ClimbMix dataloader logic used by the paper runs
- Static LoRA baseline
- Bridge runs with `true`, `shuffled`, `random`, and `constant` feature modes
- IMU diversity runs using diversity on generated LoRA weights
- Additive composition, with per-example merged eval by default
- Prefix baseline (one-token-ahead objective, byte-weighted BPB eval)
- Task evaluation for audio and IMU: label-stratified subsets, labels scored
  on the natural tokenization of prompt+label
- Vision/audio/IMU encoder definitions and dataset loaders needed by the paper

What it still intentionally reuses as external assets:

- Base LM checkpoint under `checkpoints/experiments/`
- Encoder checkpoints under `checkpoints/`
- `~/.cache/autoresearch/`, which is now bootstrapped by `./setup.sh`

The paper runner now also bakes in a few paper-facing guardrails:

- Main bridge runs are per-example conditioned during both train and eval
- `shuffled` uses a persistent run-level derangement instead of a minibatch-local permute
- Audio task eval uses ESC-50 fold 5 as test, matching the encoder's held-out split
- Static LoRA and bridge runs share the same default `--lr`
- Training loops run a fixed number of optimizer steps (hardware-independent); result JSON still records elapsed time, steps/sec, and tokens seen
- Diversity runs include a held-out IMU probe with cross-input cosine and activity-pair cosine tables for Section 6.1
- Text training defaults to the fixed shard subset `00000,00001,00002` plus pinned val shard `06542`, so machines with extra cache shards do not silently drift

Example commands:

```bash
./setup.sh
uv run paper-run bridge --modality imu --steps 300 --lr 1e-3 --log-csv logs/paper-bridge-imu-rerun.csv
uv run paper-run shuffled --modality audio --steps 300
uv run paper-run diversity --steps 300 --diversity-weight 0.1 --probe-max-items-per-activity 32
uv run paper-run composition --bricks vision,audio,imu --steps-per-brick 150
uv run paper-run composition --bricks vision,audio,imu --eval-mode fixed
uv run paper-run prefix --modality vision --n-prefix 8 --steps 300
uv run paper-run task-eval --modality audio --steps 600 --max-eval-items 200
uv run paper-reproduce --suite smoke --quick
bash run_all.sh --quick
```

For a clean machine, run `./setup.sh` once to create `.venv` and populate
`~/.cache/autoresearch` with the tokenizer, ClimbMix shards, ESC-50, and
UCI HAR assets the paper code expects.
