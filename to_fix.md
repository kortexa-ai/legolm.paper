# Status Ledger

Rewritten 2026-06-11 after the pre-submission review (in the development repo's history),
the bug-fix pass, the full rerun, and the prose rewrite. The original issue
list from April lives in git history.

## Resolved (highlights)

- Paper tables/figures/manuscript all render from a single bundled artifact
  (`arxiv/summary.json` + `arxiv/summary-scaling.json`); model architecture
  facts are read from the checkpoint at render time.
- Prefix-tuning off-by-one and bits/token-vs-BPB unit bugs fixed; result
  re-measured honestly.
- Task-probe label-tokenization bug (−inf labels) and ordered-subset skew
  fixed; probes now show real conditioning with controls at chance.
- token_bytes over-count fixed with self-healing cache rebuild.
- Wall-clock budgets replaced by fixed step counts; same-seed runs are
  bit-identical; optimizer recipe unified across conditions.
- Capacity-matched constant-feature control added end to end.
- Vision data replaced with Big Buck Bunny (CC-BY 3.0) and the perceiver
  retrained on it; provenance documented.
- Stale pre-rerun documents (markdown paper, old drafts, review snapshots,
  supplemental fossils) removed from the tree.

## Open (post-submission, non-blocking)

1. **Asset fingerprinting** — setup trusts existing cache files by name;
   tokenizer self-heals, but shards/checkpoints have no checksum validation.
2. **CI smoke job** — `paper-reproduce --suite smoke --quick` on push would
   catch import/path regressions early.
3. **Vendor-provenance headers** — `src/runtime_*.py` and `*_snapshot.py`
   files carry no "snapshotted from legolm @ commit" headers.
