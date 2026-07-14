# Representation Preprocessing Benchmark

Version 0.2.54 adds the bounded Phase 3 preprocessing gate required before any
derived representation arrays or new model training are authorized. The tool
opens a deterministic 36-product sample from the committed
`tess_cached_unlabeled_representation_v1` inventory and downloads nothing.

## Production outcome

The benchmark measures cached-FITS preprocessing throughput, process memory,
and the projected normalized-flux footprint for all 11,960 inventoried
products. It does not train or evaluate a model, and the training-only source
cannot be reused for validation, calibration, frozen evaluation, or discovery
claims.

## Execution model

`Skills/benchmark_representation_preprocessing.py` uses one supervising parent,
six Python shard subprocesses, and six FITS I/O/preprocessing threads inside
each shard. Numeric-library inner threads are capped at one. Each selected TIC
group appears once, and the deterministic selection spans sectors 1 through
98.

For each cached product the script:

1. verifies the committed inventory SHA-256, row count, role, cache-relative
   path containment, and exact file size;
2. reads `TIME`, `PDCSAP_FLUX`, and `QUALITY` from the cached SPOC FITS table;
3. retains finite cadences with `QUALITY == 0`;
4. applies per-product median and robust-MAD normalization;
5. linearly resamples to 2,048 float32 flux bins in memory; and
6. records size/timing/checksum metadata, then discards the array.

The aggregate result is written to
`artifacts/manifests/representation_preprocessing_benchmark_v1.json`. A
successful run also writes the standard report ledger at
`artifacts/manifests/run_reports/benchmark_representation_preprocessing.jsonl`.
Product failures exceed the default zero-failure allowance and fail the run.

## Commands

Validate the exact selection without opening FITS payloads or writing output:

```bash
git switch main
git pull --ff-only origin main
.venv/bin/python Skills/benchmark_representation_preprocessing.py --dry-run
```

Run the bounded benchmark from merged `main`:

```bash
git switch main
git pull --ff-only origin main
caffeinate -i .venv/bin/python Skills/benchmark_representation_preprocessing.py
```

The shard and worker counts are configurable for a documented diagnosis, but
the production default is the measured 6×6 shape. `--backend thread` exists as
a constrained-environment diagnostic only; the production default uses the six
supervised subprocesses.

## Evidence state

The merged-code 36-product artifact is intentionally pending until version
0.2.54 reaches `main`. Pre-merge validation passed eight focused tests, verified
the real committed inventory and 36-product sector-spanning selection in dry
run, and completed a read-only six-subprocess smoke over six real cached FITS
products with 6/6 successes and no persisted derived arrays.
