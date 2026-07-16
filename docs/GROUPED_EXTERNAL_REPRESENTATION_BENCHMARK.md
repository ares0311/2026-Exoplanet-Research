# Grouped External-Representation Benchmark

## Status

Version 0.2.73 defined the reviewed, cache-only execution gate. Its first
merged-main run failed closed before processing: v1 incorrectly requested the
TESS-style `QUALITY` column from Kepler files, which use `SAP_QUALITY`.
Version 0.2.74 preserves v1 as failed-schema evidence and activates immutable
v2 with the corrected column. The evidence run remains pending. This gate does
not authorize training, broad embedding extraction, checkpoint promotion, or
a production-scoring change.
Its release validation passed 2,789 default tests plus Ruff and mypy as 8/8
supervised gates in 34.1 seconds under the canonical 6x6 test topology.
Version 0.2.74's corrected gate passed 2,790 default tests plus Ruff and mypy
as 8/8 supervised gates in 36.3 seconds under the same topology.

## Production outcome

This benchmark determines whether either already-verified frozen external
representation adds materially useful grouped-holdout ranking signal beyond
both `benchmark_cnn_v1` and a statistical ephemeris baseline. It is the next
Phase 3 decision gate after the external-model identity, inference, stellar-
variability, and injection-sensitivity prerequisites passed.

## Frozen population

`metadata/grouped_external_representation_contract_v1.json` is retained for
the failed attempt; active v2 pins every input by SHA-256. From the predefined
KIC-grouped master-corpus splits, it keeps the lexically first signal per KIC
and deterministically selects a balanced subset:

| Split | Negative | Positive | Total |
|---|---:|---:|---:|
| Train | 512 | 512 | 1,024 |
| Validation | 128 | 128 | 256 |
| Test | 128 | 128 | 256 |

All 1,536 KICs are unique. The exact selected population has SHA-256
`4b245113…741b76`. Its read-only Kepler cache inventory is 24,036 FITS files
and 10,398,406,656 bytes with canonical inventory SHA-256
`39bfdbc8…94648`. No archive access or cache population is permitted.
V2 also pins the exact 111 known 65,536-byte truncated products (7,274,496
bytes total, inventory SHA-256 `c9e1556e…02245`). Only those exact paths may be
skipped, and every affected KIC must still contribute at least one readable
quarter. Any other FITS or schema error fails the run.

## Preprocessing and comparators

Each KIC's cache-local `PDCSAP_FLUX` cadences are filtered to finite positive
flux with Kepler `SAP_QUALITY == 0`, folded on the committed DR25 ephemeris,
median-binned by phase, and converted to relative magnitude. Chronos-Bolt tiny
receives 2,048 bins and Astromer2 receives 200 bins. Their exact cached ONNX
weights remain frozen. Identical deterministic L2 logistic probes train only
on `train`, use validation AUC for early stopping, and select the classification
threshold by validation F1.

The required comparators are:

- the frozen, calibrated `benchmark_cnn_v1` on its existing 201-bin input;
- a seven-feature statistical ephemeris/flux-summary linear baseline;
- Chronos-Bolt tiny and Astromer2 frozen embeddings.

The test subset opens once after preprocessing, probe, threshold, and decision
rules are frozen. An external representation adds value only if it exceeds the
better of both required comparators by at least 0.01 absolute on test ROC AUC,
average precision, or top-100 positive fraction. The balanced population makes
false-discovery metrics sample-conditional, not deployment prevalence claims.

## Parallel and storage shape

The reviewed shape is one parent, six modulo-KIC shards, and six FITS workers
per shard. Each shard owns one one-thread ONNX session per frozen model and
serializes model inference. Temporary compressed embeddings live only under
ignored `logs/`; aggregate reconciliation writes small JSON evidence and then
deletes all six arrays. A successful gate requires zero downloads, failures,
duplicate KICs, or persisted embeddings.

After version 0.2.73 is merged, run from clean `main`:

```bash
git switch main
git pull --ff-only origin main
caffeinate -i .venv/bin/python Skills/run_six_shards.py \
  --script benchmark_grouped_external_representations.py \
  --expected-new-gb 0.05
```

Then reconcile the six shard outputs:

```bash
git switch main
git pull --ff-only origin main
caffeinate -i .venv/bin/python \
  Skills/benchmark_grouped_external_representations.py \
  --aggregate-only --shard-count 6
```

The agent normally executes both commands directly; they are documented for
reproducibility, not as an operator handoff.

## Acceptance boundary

A technically passing run means only that the exact contracted benchmark
completed and produced interpretable held-out evidence. The scientific outcome
may be `external_adds_value` or `no_external_added_value`. Neither outcome
authorizes broad extraction, supervised training, model promotion, or a change
to production scoring; those require a separate reviewed decision.
