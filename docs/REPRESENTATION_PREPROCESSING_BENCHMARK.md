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

The version 0.2.54 merged-code run passed all 36 products with zero failures,
downloads, or persisted arrays. Measured preprocessing took 0.4197 seconds at
85.77 products/s. The sample read 81.85 MB and retained 704,704 of 807,636
cadences. Each 2,048-bin float32 vector is 8,192 bytes, projecting the full
11,960-product flux-only transform at 97,976,320 bytes (97.98 MB) and 139.44
seconds at the observed aggregate rate. Maximum shard-process high-water RSS
was 83,591,168 bytes.

The committed evidence is
`artifacts/manifests/representation_preprocessing_benchmark_v1.json` (SHA-256
`08c68fca194a6a5f515a17ab7e667cfd453d8f59abd8ba7bfccb13fcc207fa49`), with
Run Report commit `9f1b9e8`. This closes the preprocessing measurement gate.
Future representation experiments should stream the source instead of storing
a durable derived corpus. The result does not authorize training by itself;
the roadmap still requires stellar-variability labels, injection-recovery
comparison, and an external foundation-model baseline.
