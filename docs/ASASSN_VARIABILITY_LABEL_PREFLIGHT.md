# ASAS-SN Variability-Label Overlap Preflight

## Production outcome

The Catalina pilot closed with zero candidates among 216 frozen TESS TICs, so
it cannot support the Phase 3 stellar-variability benchmark. Version 0.2.68
adds a reproducible metadata-only gate for ASAS-SN Catalog X. The catalog is
all-sky, contains 378,861 rows, and exposes an exact TIC identifier for every
row through VizieR. The gate determines whether enough of the frozen 2,790-TIC
inventory overlaps to justify a separately reviewed benchmark design.

This is not a ground-truth or training gate. ASAS-SN `Class` values are random-
forest outputs produced from g-band light curves and augmented with Citizen
ASAS-SN data and publication quality control. Every matched row therefore
preserves `Class`, `Prob`, and `Discovery`, and every artifact keeps
`training_authorized=false`.

Primary sources:

- Paper: <https://doi.org/10.1093/mnras/stac3801>
- CDS/VizieR catalog: <https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/519/5271>
- Immutable contract: `metadata/asassn_variability_label_source_contract_v1.json`

## Exploratory measurement

A bounded implementation-decision probe on 2026-07-14 used 56 exact-ID TAP
requests with six workers and downloaded zero catalog payload bytes. It found
48 unique matches among all 2,790 inventory TICs in 7.86 seconds: 44 known
variables and four ASAS-SN discoveries, with probabilities from 0.902 upward
and no duplicate TIC rows. This result is explicitly recorded as exploratory,
not presented as blinded durable evidence.

The immutable reproduction floor is 24 unique matches, 20 known variables,
minimum probability 0.90, and zero duplicate TIC or ASAS-SN identifiers. The
floor is transparently set at half the exploratory match/known counts so source
drift fails closed. Passing it authorizes follow-up benchmark design only.

## Merged-main evidence

The reviewed preflight uses one parent, six modulo process shards, and six TAP
workers per shard. Each shard writes one row for every owned TIC, so missing
matches remain explicit and the aggregate can reconcile exactly 2,790 TICs.
Shard zero also verifies the compressed delivery headers, VizieR schema, row
count, TIC count, class distribution, and known/new distribution. The 64.26 MB
catalog and all ASAS-SN light curves remain undownloaded.

The merged version 0.2.68 run passed on 2026-07-14. All six shards processed
2,790 unique TICs in 58 exact-ID batches. Shard zero also passed all six source
metadata operations. The first shard started at 22:14:59.645 UTC and the last
completed at 22:15:06.407 UTC: 6.762 seconds observed wall time, or 412.6
TICs/s. Per-shard elapsed time was 1.665-5.876 seconds; shard zero is expected
to be slower because it owns the sequential source-identity checks. The
supervisor's deliberate one-second start staggering and those six source
checks explain why overall scaling from the preliminary six-worker probe is
sub-linear; there were no final errors or duplicate rows.

The durable aggregate reproduces 48 exact matches (1.7204%): 44 known
variables and four ASAS-SN discoveries. Classes are EA=26, EB=9, EW=2, ROT=10,
and SR=1; probabilities range from 0.902 upward. All five evidence checks pass,
all ASAS-SN identifiers are globally unique, catalog payload bytes remain zero,
and `training_authorized=false`. Version 0.2.69 commits and integrity-tests the
evidence. Follow-up benchmark design is authorized; training is not.
Aggregate SHA-256 is
`36de00dce1935aa70b3fdafb7f343fd6fd43b03696c49db7336a7842d83da403`;
the exact-path Run Report commit is `78b7be6`.

Reproduction command retained for audit:

```bash
git switch main
git pull --ff-only origin main
caffeinate -i .venv/bin/python Skills/run_six_shards.py \
  --script preflight_tess_asassn_labels.py \
  --expected-new-gb 0.001 \
  -- \
  --batch-size 50
```

Then reconcile all six outputs:

```bash
git switch main
git pull --ff-only origin main
.venv/bin/python Skills/preflight_tess_asassn_labels.py \
  --aggregate-only \
  --shard-count 6
```

Expected project-managed output is below 1 MB. The first command writes six
disjoint JSONL files, six summaries, and six shard Run Reports. The second
writes `artifacts/manifests/tess_asassn_preflight_aggregate_v1.json` plus one
aggregate Run Report.

## Stop conditions

- Stop on any delivery, schema, total-count, class-count, or known/new-count
  drift.
- Stop if any batch returns an unrequested or duplicate TIC.
- Stop if global reconciliation does not contain exactly 2,790 unique TICs.
- A gate failure is evidence; do not lower the probability or duplicate floors
  after seeing the result.
- A gate pass does not authorize catalog/light-curve downloads, FITS reads,
  embedding extraction, supervised training, model promotion, or a production
  scoring change.
