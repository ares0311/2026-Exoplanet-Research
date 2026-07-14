# Six-Shard Launcher

`Skills/run_six_shards.py` replaces six manually managed terminal tabs with one
supervisor process. It always starts exactly six shard processes with six
workers each (36 workers total), matching the project's measured safe cadence.
This single-parent shape is the standing default for any reviewed workload that
benefits from the same safely disjoint parallelism; task-specific equivalents
must preserve the safety properties below.

## Safety contract

- Only reviewed scripts that already implement `--workers`, `--shard-index`,
  and `--shard-count` are allowed.
- The launcher must start from a clean `main` checkout whose
  `.agent-project-id` matches this repository.
- Callers cannot override the 6×6 flags.
- Storage preflight includes repo-managed data plus the shared Lightkurve cache
  and refuses a projected footprint above 100 GB.
- Each shard writes a separate timestamped log under `logs/six_shard_runs/`.
- Shard Run Report commits are serialized by a process lock; acquisition and
  processing remain fully concurrent.
- Any non-zero shard exit makes the launcher exit non-zero. Ctrl-C or SIGTERM
  terminates every active child.

Supported scripts:

- `crossmatch_tess_catalina_labels.py` (only the contract-bounded 216-TIC
  metadata/crossmatch pilot is authorized; full 2,790-TIC execution remains
  gated);
- `process_t1_kepler_batch.py` (default; its historical 6×6 workload is already
  complete, so do not rerun it without a new manifest);
- `fetch_t1_2_k2_calibration_snippets.py`;
- `star_scanner.py` (requires its normal prepared-batch execution arguments).

Scripts without repo-native shard flags must add and validate isolation before
being added to this allowlist.

## Dry run

```bash
git switch main
git pull --ff-only origin main
.venv/bin/python Skills/run_six_shards.py \
  --dry-run \
  --expected-new-gb 1 \
  -- \
  --max-targets 1000
```

This prints all six exact child commands and the storage projection without
starting a process.

## Live run

```bash
git switch main
git pull --ff-only origin main
caffeinate -i .venv/bin/python Skills/run_six_shards.py \
  --expected-new-gb 1 \
  -- \
  --manifest metadata/NEW_MANIFEST.jsonl \
  --max-targets 1000
```

`--expected-new-gb` is the estimated peak new project-managed storage, not the
total source archive size. For stream/process/evict jobs, estimate peak
in-flight data plus durable outputs. If the estimate would push total managed
data above 100 GB, the launcher refuses to start.

To select another reviewed downloader, add for example:

```bash
--script fetch_t1_2_k2_calibration_snippets.py
```

Arguments for the child script must follow `--`. The launcher prints a
30-second heartbeat with completed/active/failed counts and ETA, then writes
`launcher_summary.json` beside the six logs. Each acquisition script continues
to write its own shard-specific structured Run Report.
