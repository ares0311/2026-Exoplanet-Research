# Representation Variability/Injection Benchmark

## Status

Version 0.2.70 defines the reviewed, cache-only execution gate. The merged-main
evidence run is pending. This gate does not authorize training, broad embedding
extraction, model promotion, or a production-scoring change.

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

## Validation evidence before merge

The offline focused suite covers contract bounds, deterministic product and
shard selection, full-baseline thinning, cosine distance, BLS summaries, depth
ordering, and launcher allowlisting. A cache-only one-TIC runtime smoke read
12,508 clean cadences, constructed all four trials, produced three blind-BLS
candidates per trial, and returned finite 256-element outputs from both frozen
models without downloads or persisted artifacts.
