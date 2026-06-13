# Asset Checklist

Everything the paper runner needs to reproduce the manuscript, and where it
lives. `./setup.sh` materializes the cache assets on a clean machine.

## Required Checkpoints

These are expected relative to the paper repo root:

- `checkpoints/experiments/mini-base.pt`
- `checkpoints/experiments/vision-perceiver.pt`
- `checkpoints/experiments/esc50-512d.pt`
- `checkpoints/encoders/imu.pt`

If these move, update the paths in `src/common.py`.

## Required In-Repo Data

Vision runs expect:

- `data/vision/dataset.json`
- the frame files referenced by that JSON (Big Buck Bunny keyframes,
  CC-BY 3.0 — see `data/vision/README.md` for provenance)

The loader also maps legacy `vision_data/...` paths into `data/vision/...`, but
the actual image files still need to exist.

## Required Local Cache Assets

These are populated by `./setup.sh` on a clean machine. Existing machines can
keep their current `~/.cache/autoresearch` contents; the setup script only adds
missing paper assets and does not delete anything else.

### Tokenizer

- `~/.cache/autoresearch/tokenizer/tokenizer.pkl`
- `~/.cache/autoresearch/tokenizer/token_bytes.pt`

### ClimbMix Text Data

The paper runner now defaults to a fixed text subset:

- `~/.cache/autoresearch/data/shard_00000.parquet`
- `~/.cache/autoresearch/data/shard_00001.parquet`
- `~/.cache/autoresearch/data/shard_00002.parquet`
- `~/.cache/autoresearch/data/shard_06542.parquet` (pinned val)

This makes clean and already-messy machines behave the same by default. The
runner reads these directly via the packed-text dataloader snapshot in
`src/runtime_data.py`.

### Audio Raw Data

Expected path after download/extract:

- `~/.cache/autoresearch/audio-encoder/ESC-50-master/`

The audio loader can download this itself if network access is available.

### IMU Raw Data

Expected path after download/extract:

- `~/.cache/autoresearch/sensor-fusion/UCI HAR Dataset/`

The IMU loader can download this itself if network access is available.

## Environment Knobs

These affect reproducibility and should be recorded with any serious rerun:

- `AUTORESEARCH_SEQ_LEN`
- `AUTORESEARCH_EVAL_TOKENS`
- `AUTORESEARCH_NUM_TRAIN_SHARDS`
- `AUTORESEARCH_TRAIN_SHARDS`

If unset, the paper runner defaults to sequence length `128` (matching the
mini-base checkpoint's training-time value), eval token budget `2 * 524288`,
and training shards `00000,00001,00002`.

## Sanity Commands

From the repo root:

```bash
./setup.sh
uv run paper-run --help
uv run paper-run bridge --modality imu --steps 10 --eval-tokens 4096 --sensor-limit 8
uv run paper-reproduce --suite smoke --quick --output-dir /tmp/clb-paper-smoke
```

If those work, the setup is structurally sound. Note: `paper-reproduce`
self-heals a stale `token_bytes.pt` against the shipped tokenizer on startup.

## Nice-To-Have Follow-Up

- add a tiny asset validation script that checks every checkpoint/data path and
  fingerprint up front (tracked in `to_fix.md`)
- decide whether tokenizer/ClimbMix assets should stay in `~/.cache/autoresearch`
  or become repo-configurable
