# Phase 3 External Baseline Inference Smoke

## Production outcome

`Skills/smoke_representation_baseline_inference.py` is the bounded runtime gate
between source verification and any broader embedding experiment. It answers
one narrow question: can the two pinned external encoders produce finite,
correctly shaped embeddings on the validated Python 3.14 Mac runtime using one
real cached TESS product, within measured memory and time bounds?

A PASS does not authorize training, a production scorer change, or full-corpus
embedding extraction.

## Fail-closed contract

The smoke:

1. verifies the committed source contract SHA against the successful source
   evidence artifact;
2. loads the exact training-only cached TESS inventory contract;
3. selects one deterministic target-balanced SPOC product;
4. keeps at most 2,048 finite, positive-flux, `QUALITY == 0` cadences;
5. converts flux to relative magnitude with `-2.5 log10(flux / median_flux)`;
6. downloads only the exact commit/file/hash for Chronos-Bolt tiny and
   Astromer2 into `.cache/representation_models/`;
7. constructs CPU ONNX sessions with one intra-op and one inter-op thread;
8. runs each model in a separate child process so peak RSS is attributable;
9. requires a finite `(1, 1, 1, 256)` mean embedding from each model; and
10. writes one structured artifact plus the standard Run Report.

The model cache and its Hugging Face `local_dir` metadata are ignored. Expected
weight payload is 29,890,844 bytes; package wheels and transitive dependencies
are separate environment state. The smoke downloads no mission data and writes
no derived light-curve corpus.

Version 0.2.58 sets `HF_HOME` and `HF_XET_CACHE` below the ignored model cache
before importing `huggingface_hub`. The first merged 0.2.57 attempt exposed
this requirement by failing closed when Xet tried to write its log below
sandbox-blocked `~/.cache/huggingface`; it downloaded no model payload and left
only 8 KB of local-dir metadata.

## Runtime installation

The optional `representation` dependency group is pinned to the already
verified source contract. It is not part of the default project installation:

```bash
git switch main
git pull --ff-only origin main
.venv/bin/python -m pip install -e '.[representation]'
```

Before installation, use pip's dry-run/report mode and fresh storage checks to
record the actual transitive environment delta. Do not recreate `.venv`.

## Merged-code smoke

After the optional group is present:

```bash
git switch main
git pull --ff-only origin main
caffeinate -i .venv/bin/python Skills/smoke_representation_baseline_inference.py
```

The parent prints download and inference progress with elapsed time and ETA.
Use `--status-only` to read the durable artifact after a successful run.

## Parallelism decision

This gate has two models and one shared input. It runs model children
sequentially because isolated peak-memory measurement is the purpose of the
smoke and process startup would dominate a two-item run. If the smoke passes,
future per-light-curve embedding extraction is embarrassingly parallel and must
use one parent with six shards and six workers per shard, with ONNX native
threads held at one per worker.

## Remaining scientific gates

Even a successful smoke leaves these requirements open:

- stellar-variability labels;
- a frozen, target-grouped train/validation/test comparison;
- BLS/TLS/statistical and `benchmark_cnn_v1` comparators;
- top-k review-yield measurement; and
- injection-recovery comparison.

No consumed pilot-v1 test data may be reused for tuning the new experiment.
