# Representation Variability/Injection Benchmark

## Status

Version 0.2.70 defines the reviewed, cache-only execution gate. Version 0.2.71
also verifies every aggregate-owned ASAS-SN shard path and SHA-256 before it
loads the 48 labels. The merged-main evidence run passed on 2026-07-16 and is
preserved by version 0.2.72. This gate does not authorize training, broad
embedding extraction, model promotion, or a production-scoring change.

## Production outcome

The benchmark measures whether the two already-verified frozen external
representations respond to bounded transit injections on real stellar-
variability backgrounds, alongside the existing blind BLS search. It closes
the scientific-design prerequisite between the 48-row ASAS-SN overlap evidence
and any decision about a broader representation experiment.

## Frozen inputs and population

`metadata/representation_variability_injection_contract_v1.json` pins every
input by SHA-256. The population is all 48 exact-TIC ASAS-SN matches in the
committed 2,790-TIC TESS cache inventory. For each TIC, the benchmark selects
the largest cached product, breaking ties by lowest sector and then lexical
cache path. The selected products have measured 23.75-28.20 day baselines and
11,241-114,232 clean cadences.

ASAS-SN classes remain automated catalog outputs, not ground truth. They are
preserved only as descriptive background strata. No classifier is fitted.

## Paired benchmark

Each TIC receives four multiplicative box injections:

| Scenario | Period | Depth | Duration |
|---|---:|---:|---:|
| `short_low` | 3 days | 500 ppm | 2 hours |
| `short_high` | 3 days | 2,000 ppm | 2 hours |
| `long_low` | 10 days | 500 ppm | 4 hours |
| `long_high` | 10 days | 2,000 ppm | 4 hours |

The 192 unique TIC/scenario trials are evaluated by the bounded blind BLS
configuration in the contract. Chronos-Bolt tiny and Astromer2 then embed the
original and injected series from exact, already-cached ONNX weights. Both
series use the same original median and deterministic even thinning across the
full sector baseline. Only hashes and paired cosine/L2 distances are written;
embedding arrays and modified light curves are discarded.

## Parallel and storage shape

Run from clean merged `main` through `Skills/run_six_shards.py`: six modulo TIC
process shards with six FITS/BLS workers per shard. Each shard owns one frozen
session per model and serializes model inference, preventing 36 redundant
copies of both ONNX sessions. Numeric backends and ONNX sessions use one inner
thread. Shard artifacts and Run Reports are disjoint.

The benchmark downloads zero bytes and writes only small JSON/JSONL evidence.
It reads existing Lightkurve and representation-model caches and must remain
within the project storage supervisor's 100 GB ceiling.

## Acceptance gate

Global reconciliation must prove all of the following:

- 48 unique TICs and 192 unique injection trials;
- two exact frozen models and 384 unique model/trial rows;
- zero failed or duplicate model trials;
- zero downloaded bytes and zero persisted embeddings;
- descriptive metrics only, with `training_authorized=false` and
  `production_change_authorized=false`.

A passing run supports scientific interpretation of the paired sensitivity
metrics only. It is not evidence of classifier accuracy, survey completeness,
or production superiority.

## Merged evidence

The single-parent 6x6 run passed all six shards and global reconciliation:
48 unique TICs, 192 unique injection trials, 384 unique model rows, and zero
failures, duplicates, downloads, or persisted embeddings. First-shard-start to
last-shard-completion wall time was 15.854 seconds; shard elapsed times were
3.452-10.476 seconds. The aggregate SHA-256 is
`93ae6fb818054947ecfd485b3e74ec5cef1f88d1d5e18fd232e5aebd8303f59f`.

Both frozen models produced a larger cosine shift for the 2,000-ppm injection
than the paired 500-ppm injection for all 96 TIC/period comparisons. Median
cosine shifts increased by about 16x at 4x depth for both 3-day and 10-day
scenarios, consistent with a stable depth-sensitive response in this bounded
grid. Blind BLS recovered 13/192 trials: 4/48 short-low, 4/48 short-high, 2/48
long-low, and 3/48 long-high. These low, non-monotonic recovery counts are
descriptive evidence on variable-star backgrounds, not completeness estimates.

The next Phase 3 decision may use this result to design a separately contracted
grouped benchmark against labeled planet/false-positive data. It must retain
the frozen CNN and classical baselines, keep ASAS-SN rows training-disabled,
and obtain a new authorization before broad extraction or model fitting.

Version 0.2.72's evidence-release validation passed 2,782 default tests plus
Ruff and mypy as 8/8 supervised gates in 37.2 seconds under the canonical 6x6
test topology.

## Validation evidence before merge

The offline focused suite covers contract bounds, deterministic product and
shard selection, full-baseline thinning, cosine distance, BLS summaries, depth
ordering, and launcher allowlisting. A cache-only one-TIC runtime smoke read
12,508 clean cadences, constructed all four trials, produced three blind-BLS
candidates per trial, and returned finite 256-element outputs from both frozen
models without downloads or persisted artifacts.
