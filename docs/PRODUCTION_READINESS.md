# PRODUCTION READINESS

Last reviewed: 2026-07-13 (formal production-ensemble acceptance PASS: the
catalog ephemeris for pi Mensae c was recovered within 0.005% with ensemble
FPP 0.4405, while the TOI-146.01 false-positive control produced no signal.
No Tier 1 gaps remain open.)
Scope decision: T2-2 and T2-3 are permanently out of scope — see DECISION-013
Branch: `main` (90 production-critical Skills; non-production fluff removed)
Test baseline: 2,726 default tests passing; 2 `integration_live` tests excluded by
the configured marker expression (2026-07-13; 6×6 gate: 34.1s)

---

## Live-Readiness Summary

| Scorer Mode | Status | Blocker |
|---|---|---|
| `--scorer bayesian` | **PRODUCTION READY** | None — default mode, zero external dependencies |
| `--scorer xgboost` | **PRODUCTION READY** | None — trained on 7,586 Kepler KOIs, AUC=0.992 |
| `--scorer ensemble` | **PRODUCTION READY** | None — conservative XGBoost+Bayesian blend when CNN absent |
| `--scorer cnn` | **PRODUCTION BENCHMARK PROMOTED** | `benchmark_cnn_v1` registered and promoted; Kepler-domain limitations still apply |
| `--scorer full-ensemble` | **PRODUCTION READY** | None — T1-2 stacking calibration complete; weights (XGBoost=0.95/CNN=0.00/Bayesian=0.05) wired into `cli.py` |

The system is safe to deploy now for Bayesian and XGBoost scoring modes. The
master-corpus Kepler CNN checkpoint has also been promoted as the frozen
`benchmark_cnn_v1` Kepler-domain CNN benchmark after human approval. Full
ensemble production is unblocked: T1-2 stacking calibration is complete and
its calibrated weights are live in `cli.py`. Do not tune stacking weights
against training or frozen-eval data — any future recalibration needs its
own fresh held-out set, same as T1-2's K2 set was for this one.

Version note: 0.2.62 is the current patch level. 0.2.8 fixed QLP stitch
normalization and feature serialization, 0.2.9 adds raw vetting diagnostics,
fetch provenance, missing-feature names, and human-readable missing-diagnostic
reasons, 0.2.10 adds bounded retry for transient MAST/Lightkurve connection
disconnects, 0.2.11 wires real TIC catalog `stellar_radius_rsun`,
`stellar_mass_msun`, `stellar_teff_k`, and `contamination_ratio` into
`vet_signal()` for every TESS scan (previously all four were always `None`,
and `vet_signal()` did not even accept `stellar_teff_k`, so
`limb_darkening_plausibility_score` silently used the solar-default 5778 K for
every target), 0.2.12 records the TAP schema case-sensitivity fix in the
active dataset handoff contract, 0.2.13 adds the committed leakage-safe
Kepler training manifest and raw-FITS cleanup policy, 0.2.14 adds the
bounded, resumable Kepler-first processing batch tool
(`Skills/process_t1_kepler_batch.py`) that consumes that manifest, 0.2.15
adds an ETA to that tool's per-target progress output for multi-hour bounded
runs, 0.2.16 adds bounded `--workers` concurrency to the same tool
(default 1, sequential; `docs/SYSTEM_PROFILE.md` recommends 4-6 for this
external-service workload once verified) plus a correctness fix required to
support it safely: raw FITS downloads now use a per-target subdirectory
instead of a single shared scratch directory, so concurrent fetches can no
longer delete each other's in-flight files, and 0.2.17 fixes a live
`--workers 6` crash (corrected 2026-07-04: originally mislabeled `--workers
4` throughout this doc; the human confirms every live run used `--workers
6`, not 4) (`ValueError: I/O operation on closed file`) caused by
that tool calling Lightkurve's public `search.download_all()`, which mutates
process-global `sys.stdout` and is unsafe under concurrent worker threads —
the same failure class already fixed once before for `star_scanner.py` (see
run003 under T1-0). The fix reuses `fetch.py`'s already-proven-safe
`_download_collection_with_cache_repair()` download path instead. The catalog
lookup fails open (all-`None`) on any network/parse error or non-TIC target,
so it never blocks a scan. 0.2.18 adds process-level sharding
(`--shard-index`/`--shard-count`) for multi-tab throughput and a new Run
Report Policy (`Skills/run_report.py`) so every acquisition/processing script
self-reports and auto-commits/pushes its own completion record instead of
requiring pasted console output — see `docs/DISCOVERY_RUNBOOK.md` Rule 7.
0.2.19 fixes a real intra-process download-serialization bug found via the
first live 2-shard test's sub-linear scaling: `_download_one_quietly()` held
a lock around the entire download call, not just the monkey-patch needed to
force `verbose=False`, serializing every download within one process
regardless of `--workers`. Fixed with an idempotent, non-restoring wrap
instead of a lock-guarded monkeypatch-and-restore. 0.2.20 adds offline
validation (`test_four_concurrent_shards_never_collide`) for the planned
4-concurrent-shard live test, agreed with the human as one deliberate step
up from the only live data point (2 shards) rather than jumping straight to
the 6-7 tabs available -- no functional code changes, since sharding was
already N-way generalized. 0.2.21 fixes a WAL-mode race
(`sqlite3.OperationalError: database is locked`) that CI caught on that
offline test before it could hit the live 4-shard run -- concurrent shards
constructing `T1KeplerProcessingStore` for the same fresh db at nearly the
same instant could lose the one-connection-only WAL transition; fixed with
a tolerant retry, falling back to the default journal mode if it never
succeeds. 0.2.22 adds manifest-progress percent-complete to the startup
banner, final summary, and `--status-only`, and documents the real 4-shard
(~2.15s/target) and 6-shard (~2.11s/target, flat) live test results -- 4-6
shards is the practical ceiling for this workload. 0.2.23 extends the
percent-complete to the committed run report itself (`Skills/run_report.py`),
not just console output, so reading a run report later shows true global
progress without summing shard files by hand. 0.2.24 fixes a leakage-safety
gap found while confirming the Kepler-first manifest reached 100% complete:
`Skills/build_cnn_training_data.py` ignored the manifest's pre-assigned
leakage-safe split and would have silently re-shuffled it; it now respects a
predefined split when present, with a hard error on inconsistent group
assignments. 0.2.25 fixes the real root cause the human's first live run hit:
`build_cnn_training_data.py` only recognized `kepid`/`tic_id`/`original_tic_id`/
`group_id`, not the T1-1 pipeline's actual `target_id`/`group_key` fields, so
every row collapsed into one fake group and the new 0.2.24 check (correctly)
raised on the resulting cross-split conflict. Fixed by recognizing `group_key`
and falling back to `target_id`; this bug pre-dated 0.2.24 and would have
silently corrupted the split under the old random-grouping logic instead of
erroring. 0.2.26 adds the verified Kepler DR24 TCE expansion manifest builder
after live schema/label checks showed `Q1_Q17_DR24_TCE.av_training_set` has
real labels while the newer DR25 TCE label columns are empty; the expansion was
later fully processed and produced the master corpus used by the current
passing CNN. 0.2.27 promotes the human-approved `benchmark_cnn_v1` CNN
checkpoint and registers the selected production artifacts under `models/`.
0.2.28 builds the T1-2 held-out K2 calibration manifest, catalog-only
Bayesian+XGBoost scoring, and the native K2 snippet fetcher (see T1-2 below);
it also fixes two real unit bugs found live while building that manifest — a
double-counted BJD offset in `Skills/fetch_tess_k2_overlap_snippets.py` and a
wrong days-vs-hours assumption for `pl_trandur` — neither of which affected
any promoted model. 0.2.29 closes T1-2 end to end: adds process-level
sharding to the K2 snippet fetcher, adds `Skills/merge_t1_2_k2_cnn_predictions.py`
to fill in `cnn_prob` from the promoted CNN checkpoint (an explicit,
logged cross-mission application — see T1-2 below), fixes a real
`float(None)` crash bug in `calibrate_stacking_weights.py`'s loader, fixes a
severe silent-failure bug in `CnnScorer` where invoking a `Skills/*.py`
script directly (`python Skills/foo.py`, as every recipe in this project
does) left the repo root off `sys.path`, silently failing the
`Skills.cnn_inference_batcher` import and returning a constant, garbage
`cnn_prob=0.5` for every prediction with no error or warning (see T1-2 below
for the full incident record and the `_ensure_repo_root_on_sys_path()`
fix plus new `RuntimeWarning` on genuine load failures), runs the
AUC-maximising stacking calibration on the corrected data
(XGBoost=0.95/CNN=0.00/Bayesian=0.05, best AUC 0.9576 — numerically
identical to the pre-fix run, but now backed by real per-tier signal:
CNN=0.7458/XGBoost=0.9575/Bayesian=0.8656 standalone AUC), and wires the
calibrated weights into `cli.py`'s `full-ensemble` blend and
`StackingScorer`'s defaults, replacing the old uncalibrated 0.35/0.35/0.30
guess.
0.2.30 begins the next Astrometrics master-guide production phase with a
versioned, mission-neutral dataset-manifest contract: committed JSON Schema and
frozen Pydantic models, stable dataset IDs for the actual Kepler/K2 production
sources, fail-closed repository-path and SHA-256 validation, a reusable CLI,
and explicit links from the data-role registry. TESS/JWST must use the same
contract when their first row-level production manifests are created; no
placeholder datasets are fabricated.
0.2.31 adds the Astrometrics master-guide candidate-ledger contract without
breaking the legacy convenience database: strict mission-neutral Pydantic
records, an append-only SQLite table, stable source-dataset IDs, raw and
regeneration provenance, generator/model/calibration/injection context, and
structured human-review state. Production scan-path wiring remains the next
ledger task; until then, the old table must not be mistaken for the complete
scientific ledger.
0.2.32 adds the formal production-acceptance harness and records the first
catalog-backed TESS control run as FAIL rather than accepting low-FPP signals
at the wrong period. It also synchronizes the source fallback version, repairs
the stale Tier 2 reporter so it reads the promoted benchmark evidence by
default, and removes contradictory readiness/roadmap status left behind after
the completed Kepler processing and T1-2 stacking gates.
0.2.33 closes the formal production-ensemble acceptance gate with a durable
PASS report in `artifacts/manifests/formal_acceptance_v2.json`. The confirmed
control, pi Mensae c, recovered 6.2676207 d against the catalog 6.2679 d period
and produced ensemble FPP 0.4405; the TOI-146.01 false-positive control
produced no signal. The harness now evaluates the probability emitted by the
scorer under test rather than always substituting the Bayesian posterior. The
earlier TOI-700 Bayesian-only v1 FAIL remains committed evidence of a known
positive-control limitation; it is not rewritten as a pass.
0.2.34 fixes a provenance-contract blocker found before production candidate-
ledger wiring: stitched light curves may contain multiple raw archive products,
but candidate-ledger schema v1 could record only one `raw_uri`. Fetch provenance
now preserves the complete exact URI tuple exposed by Lightkurve/MAST, including
the QLP `dataURL` fallback and the selected JWST product. Candidate-ledger schema
v2 requires a non-empty `raw_uris` tuple, SQLite stores it losslessly, and v1
history remains readable. Live ledger writes remain fail-closed until a real
row-level live-search dataset manifest exists; training/calibration manifests
must not be reused or a synthetic source ID invented merely to enable wiring.
0.2.35 adopts maximum-safe parallelism as a standing directive and makes
`pytest-xdist -n auto --dist=worksteal` the default local/CI test runner.
Benchmarks on the 16-CPU M4 Max preserved all 2,630 passes while reducing the
suite from 226.45s serial to 81.57s with all 16 workers (the 8-worker result was
81.54s, statistically equivalent). The portable `auto` setting lets constrained
CI runners use their own reported CPU capacity rather than hard-coding the Mac.
0.2.36 closes the real-live-manifest prerequisite for Phase 1 candidate-ledger
wiring. `Skills/star_scanner.py --prepare-only` now performs a metadata-only,
six-worker target preparation pass: it freezes the policy queue, exact MAST QLP
product inventory, batch manifest, and checksum-validated `live_search` dataset
manifest without downloading raw light curves or invoking the scan pipeline.
Successful preparation and scan invocations now also use the required
`star_scanner.jsonl` Run Report ledger, with the git runner injectable so tests
never invoke the real commit/push path.
The first immutable bundle, `tess_live_search_v1`, contains 18 eligible TIC
targets and 103 exact QLP products totaling an estimated 0.04565088 GB; two
additional inspected targets with no products remain as explicit rejections in
the canonical queue. Selection excluded 7,753 TOIs, 3,787 CTOIs, 3,590
confirmed transiting hosts, and all 200 targets in the historical discovery
logs. The live preparation also exposed two upstream-contract changes that are
now handled fail-closed: the current CTOI export no longer supplies the ratings
column assumed by the old default filter, and the NASA Exoplanet Archive `ps`
table uses `tran_flag` while TIC values are prefixed with `TIC `. Candidate-
ledger schema v2 can now be wired to this stable source dataset and product
inventory; that wiring remains the next Phase 1 roadmap task.
0.2.37 completes that candidate-ledger wiring for the frozen batch.
`star_scanner.py --execute-prepared-batch` validates the dataset checksum,
batch identity, exact target membership, per-target product sizes, and complete
URI inventory before scanning. It writes every candidate, high-priority null,
or post-fetch preprocessing failure as a schema-v2 record whose
`source_dataset_id` is `tess_live_search_v1`; fetched URI tuples must exactly
match the frozen inventory or the write is refused. The execution path is
resume-safe, defaults to six I/O workers, and supports modulo process sharding
with automatically scoped scan logs, SQLite ledgers, and Run Report ledgers.
Runtime `data/*.sqlite*` files are now ignored and recorded in the local-
artifact ledger, keeping `git add .` safe. No live scan was executed by this
change; the 18-target evidence run remains operator-coordinated work.
0.2.38 closes an operator-visibility defect found before that run: prepared
shards previously printed a startup banner and per-target completion ETA, but
could remain silent throughout a long download/BLS interval. They now print a
flushed line when every target starts plus a flushed 30-second heartbeat with
completed, active, pending, elapsed, and ETA fields. The interval is configurable
with `--heartbeat-seconds`; tests exercise the heartbeat while work is active.
0.2.39 adds `Skills/tpf_centroid_diagnostic.py`, a bounded TESS-SPOC target-pixel
diagnostic for the remaining TIC 355651994_s02 review gap. It measures robust,
pointing-corrected in-transit versus symmetric local out-of-transit aperture-
photocenter offsets per independent event and sector. It records exact product
provenance, aperture provenance, pixel/arcsecond offsets, event-level
significance only when at least four events exist, strict JSON null otherwise,
and partial-product failures; it also prints per-product ETA and writes a Run
Report. It fails closed on inadequate transit coverage and explicitly says it
cannot localize the transit source or replace difference imaging. The tool is
offline-tested. The first bounded live run downloaded both products but found
no predicted transit coverage: sector 1 begins 4.664 days after the nearest
prior event, and sector 28 ends 10.880 days before the nearest following event.
0.2.40 makes that scientifically useful fail-closed outcome durable instead of
raising without an artifact: all-no-coverage runs now write strict structured
JSON with exact TPF time ranges, nearest predicted event centers, gap distances,
and product provenance, then append a Run Report. The cached rerun completed in
8.97 seconds and was auto-committed as `986e7bc`; its runtime JSON hash is
`af9a71c6…bf9f`. The quality-masked sector-1 coverage begins 4.664 days after
the nearest event, and sector 28 ends 10.880 days before the next. The existing
QLP review contains exactly two independently measured events, while odd/even
requires at least four. Therefore both requested diagnostics are complete to
the limit of available observations and remain unavailable—not assumed
favorable. The committed evidence is
`artifacts/manifests/tess_live_search_v1_tpf_coverage_summary.json`.
0.2.41 closes the Phase 1 canonical-regression-eval gap with a four-case,
offline suite. Catalog-backed confirmed-planet and known-false-positive
controls remain isolated from deterministic injected controls through separate
`frozen_eval` registry roles. The committed baseline records pi Mensae c PASS,
TOI-146.01 conservative rejection PASS, 2/2 deep-transit recovery, and 0/2
subthreshold recovery; future runs emit sample-level metric deltas and fail on
outcome or recovery-rate regressions.
0.2.42 adds the bounded Phase 2 production-sensitivity runner: it injects a
versioned 36-trial grid into two cached real Kepler quarter-1 frozen-eval
backgrounds, executes the production search and full-ensemble scoring path,
reports recovery curves by period/depth/duration/background, supports six
workers and collision-free process shards, and writes a structured Run Report.
During its real-cache smoke test, the full-ensemble path exposed a macOS native-
runtime collision between the XGBoost and PyTorch wheels: XGBoost-first blocked
PyTorch initialization, while PyTorch-first terminated during XGBoost load.
Full-ensemble CNN review scoring now runs in a bounded child interpreter while
XGBoost remains in the parent. The 36-trial, six-worker smoke completed in 7.1
seconds with 23 recoveries and zero failures. Durable curves are deliberately
not claimed until the merged runner performs its production run.
0.2.43 fails the first merged evidence attempt closed: the light curves were
correctly filtered to quarter 1, but provenance incorrectly retained all 17
discovered quarter paths. Paths and curves are now filtered as aligned pairs,
the regression is unit-tested, and the sensitivity config identifies the
corrected `full_ensemble_v0.2.43` runtime. The invalid artifact and report are
not retained; durable evidence still requires a corrected merged-code rerun.
0.2.44 commits that corrected merged-code run as
`artifacts/manifests/production_sensitivity_v1.json` (SHA-256
`8c2246ec…b7b16a1`) plus its successful Run Report (SHA-256
`55da9ad1…a9cc9cf`). All 36 trials completed in 6.83 seconds with 23 recoveries
and zero failures; each background now records exactly one used Q1 product.
Recovery was 12/12 at 1 day, 9/12 at 3 days, and 2/12 at 10 days; 9/18 for
2-hour and 14/18 for 8-hour injections. Depth bins were non-monotonic
(9/12 at 500 ppm, 8/12 at 2,000 ppm, 6/12 at 10,000 ppm), which is explicitly
treated as small-grid/strongest-peak competition evidence rather than a
physical completeness curve. This closes bounded short-period v1 evidence
only. TTV, single-transit, gap, stellar-variability, longer-period, and
multi-quarter coverage remain open before any general completeness claim.
0.2.45 adds the expanded v2 sensitivity contract without rewriting v1. Explicit
scenario rows cover periodic 3/90-day signals, moderate/strong sinusoidal TTV,
single-transit event-time recovery, partial/full deterministic data gaps, and
low-depth injection over 5,000-ppm stellar variability on two real Kepler Q1-Q4
backgrounds. Gap cases record source, injected, and removed cadence counts.
The 16-trial, six-worker pre-merge smoke completed in 17.6 seconds with 8
recoveries and zero failures. That smoke is tooling validation only; durable v2
curves require a merged-code rerun.
0.2.46 commits the merged-code v2 run as
`artifacts/manifests/production_sensitivity_v2.json` (SHA-256
`a7ea7eff…16d597c`) and appends its successful Run Report. All 16 trials
completed in 7.98 seconds with eight recoveries and zero failures. Moderate TTV
recovered 2/2 while strong TTV recovered 0/2; partial/full gap cases recovered
4/4 while explicitly recording 8/24 removed cadences; 3-day periodic recovered
2/2 while 90-day periodic recovered 0/2; single-transit and low-depth
variability each recovered 0/2. Exact Q1-Q4 provenance lists four products per
background. Bounded scenario coverage is now durable; these two-sample cells
are boundary evidence, not survey completeness estimates. Phase 2 calibrated
candidate context is the next active production priority.
0.2.47 adds fail-closed calibrated-candidate-context tooling. A frozen Pydantic
reference contract and Run-Report-enabled builder derive exact full-ensemble
score ranks from the 588-row `t1_2_k2pandc_calibration` predictions and
production stacking weights. Full-ensemble output gains raw score, empirical
quantile, calibration dataset ID, threshold version, observed tail-negative
fraction, and its reference counts/limitations. Because T1-2 optimized AUC—not
probability calibration or decision utility—`calibrated_score` and
`decision_threshold` remain null and the threshold version explicitly says
none exists. Schema-v2 live-search ledger records preserve the context. The
real reference build completed as a temporary smoke; the merged-code artifact
is still required before this roadmap item closes.
0.2.48 commits that merged-code reference at
`models/candidate_context_v1.json` (SHA-256 `b7d7a72d…ff1216c`) and its
successful report-only commit `24fea1b`. The artifact validates 588 sorted
scores (356 positive, 232 negative) against exact prediction and stacking-
weight hashes. Full-ensemble scans automatically load it when present; outputs
and schema-v2 ledger rows now carry empirical rank/FDR context. The null
calibrated score and null decision threshold are intentional production facts,
not missing work. Bounded Phase 2 is complete; the active roadmap priority is
the Phase 3 representation benchmark.
0.2.49 adds the bounded Phase 3 benchmark contract and tooling. A compact
masked-reconstruction Transformer pretrains only on the predefined Kepler
training split with labels hidden from the objective, freezes its encoder,
fits a linear probe, and opens the KIC-grouped frozen test split once. The
result compares grouped AUC and top-100 yield against a period/duration/flux-
summary linear baseline and frozen `benchmark_cnn_v1` AUC 0.957211. The script
uses accelerator-first `device=auto`, per-epoch progress/ETA, an ignored
checkpoint, exact hashes, and a Run Report. Tooling is complete; the merged-
code evidence run remains pending, and the full Phase 3 data/evaluation
requirements remain explicitly open.
0.2.50 commits the merged-code Phase 3 pilot result at
`artifacts/manifests/representation_pilot_v1.json` (SHA-256
`74835fc9…807fc4`). The 33.5-second MPS run processed all 15,649 predefined
rows and selected masked-reconstruction epoch 11. The frozen embedding probe
did not beat the promoted CNN: test AUC 0.832630 and F1 0.635135 versus CNN AUC
0.957211 and F1 0.834688. It did exceed the deliberately small tabular baseline
(AUC 0.823495) and raised top-100 positive yield from 6% to 72%, evidence that
the representation contains useful ranking signal but not enough to replace
the CNN. The compact architecture is rejected unchanged. Phase 3 remains open
for materially broader unlabeled data and the missing variability, injection,
and foundation-model comparisons.
0.2.51 adds a metadata-only TESS cache inventory builder for the next Phase 3
data gate. Read-only preflight found the 41 GB Kepler cache has 6,515 unique
KICs but only 11 outside the labeled master corpus, so it is not a broad
unlabeled source. The 37 GB TESS SPOC cache has 14,728 product directories and
3,033 unique TICs, 2,797 outside all local labeled TESS corpora before frozen
live-role exclusion. The builder records exact MAST URIs, cache-relative paths,
sectors, and sizes; excludes labeled/live TICs; prints progress/ETA; writes a
Run Report; and never opens FITS payloads or downloads data.
0.2.52 commits that merged-code inventory and registers it as
`tess_cached_unlabeled_representation_v1`. The 2.6-second metadata-only run
scanned 14,728 cache directories and selected 11,960 exact SPOC light-curve
products across 2,790 TICs and 84 sectors, totaling 29.79762048 GB already on
disk. It excluded 236 locally labeled TICs; none of the 18 frozen live-search
TICs overlapped this SPOC cache. The 6.1 MB row inventory has SHA-256
`38a86966…c155a`; Run Report commit `2016811`. The source is training-only and
does not close Phase 3. Derived-array creation remains gated on a bounded
streaming/preprocessing size and throughput benchmark.
0.2.53 adds the single-terminal 6×6 acquisition supervisor requested by the
operator. `Skills/run_six_shards.py` constructs exactly six native shard
commands with six workers each, permits only reviewed shard-capable Skills,
rejects caller overrides of shard flags, requires authoritative clean `main`,
preflights repo plus shared-cache storage against the 100 GB ceiling, staggers
starts, writes separate logs, emits heartbeat/ETA, terminates children on
interrupt, and fails if any shard fails. A shared file lock serializes only the
six child Run Report git transactions, preventing `.git/index.lock` races while
all acquisition work remains concurrent. No live download was executed because
the existing T1-1 and T1-2 manifests are complete. The same release adds
`Skills/run_quality_gates.py`, which assigns every test module exactly once
across six pytest shards with six xdist workers each and supervises Ruff/mypy
beside them. Per-gate logs and a combined JSON result make full validation
single-command and fail-closed. Native numeric backends are capped at one inner
thread per worker; the optimized full run passed all 2,718 default tests and
both static gates in 34.1s versus the prior 81.57s single-xdist baseline. Unit
tests that only assert injection-grid orchestration now stub the separately
covered BLS layer instead of repeating expensive real searches. This
single-parent optimized pattern is now the default for every safely
partitionable workload. The acquisition launcher's merged-code
validation remains a no-download dry run.
0.2.54 adds `Skills/benchmark_representation_preprocessing.py`, the bounded
Phase 3 gate that must run before derived arrays or broader representation
training are authorized. It validates the exact committed training-only TESS
inventory and cached-file containment/sizes, selects 36 distinct TIC groups
across sectors 1-98, and uses six supervised Python shard subprocesses with six
FITS workers each. Each product is filtered to finite `QUALITY == 0` cadences,
robustly normalized, resampled to 2,048 float32 bins in memory, hashed for
evidence, and discarded. The aggregate records throughput, child/parent memory,
and projected full-inventory normalized-flux size; it performs no downloads.
Eight focused tests, the real-inventory dry run, a six-subprocess real-cache
smoke (6/6 successful), and the full 2,726-test/Ruff/mypy 6×6 gate pass. The
merged-code run then passed 36/36 products with zero failures, downloads, or
persisted arrays in 0.4197 measured seconds (85.77 products/s). It projects the
full 11,960-product normalized-flux-only transform at 97.98 MB and 139.44
seconds. Artifact SHA-256 is `08c68fca…07fa49`; Run Report commit `9f1b9e8`.
The preprocessing measurement gate is complete. Prefer streaming for the next
experiment; broader training remains unauthorized until its plan supplies the
still-missing stellar-variability labels, injection-recovery comparison, and
external foundation-model baseline.
0.2.55 closes the external-baseline source-identity and direct-footprint
prerequisite without installing or downloading any payload. The immutable
contract selects Chronos-Bolt tiny as the general time-series foundation
baseline and Astromer2 as the astronomy-native comparator, pins exact Hugging
Face commits/files/SHA-256 values, and pins Python 3.14-compatible wheels for
`light-curve`, `onnxruntime`, and `huggingface-hub`. The two ONNX files and
three direct wheels total exactly 56,036,648 bytes. Full Chronos2 is excluded
from the first gate because its 463.8 MB ONNX file duplicates the bounded
model's role. `Skills/verify_representation_baseline_sources.py` checks current
primary metadata plus pinned-file HEAD headers, prints progress/ETA, fails
closed on drift, writes structured evidence and a Run Report, and downloads
zero package/model payload bytes. Seven offline tests cover success and drift
failures. The merged-code live metadata verification remains the immediate next
step; no dependency, model weight, inference, or training is authorized yet.
The canonical local release gate passed 2,733 default tests plus Ruff/mypy in
31.1 seconds using six disjoint test shards × six xdist workers.
The first merged 0.2.55 live metadata run failed closed after four of seven
steps, before writing an evidence artifact or Run Report and without
downloading a payload. The pinned source was unchanged; Python's URL opener had
followed Hugging Face's 302 resolver response into Xet, where the original
`x-repo-commit`, `x-linked-size`, and `x-linked-etag` headers no longer exist.
Version 0.2.56 installs a no-redirect handler for this HEAD request, treats the
authoritative 302 headers as the result, and adds an offline regression test.
A live read-only Chronos HEAD smoke returned the exact pinned commit, size, and
hash. The 0.2.56 canonical release gate passed 2,734 tests plus Ruff/mypy in
26.1 seconds. The merged full verifier then passed all 7/7 metadata operations
in 4.9396 seconds: three exact Python wheels plus two exact ONNX repositories
and pinned resolver headers, with five sources verified, 56,036,648 projected
direct bytes, and zero payload bytes downloaded. Durable artifact
`representation_baseline_source_verification_v1.json` has SHA-256
`5610bbb859463e180bd9ee65ee7317518458560421487253d272b2d3b5753042`;
Run Report commit `ae4e659`. The external-baseline source identity/footprint
gate is complete. It does not authorize dependency installation, model weight
download, inference, or training; stellar-variability labels and the
injection-recovery comparison remain open scientific prerequisites.
0.2.57 adds the bounded runtime gate that must precede any broad embedding
extraction. `Skills/smoke_representation_baseline_inference.py` validates the
source and training-only inventory contracts, selects one deterministic cached
SPOC product, keeps at most 2,048 clean positive-flux cadences, and downloads
only the two exact-revision ONNX files into ignored in-repo cache. Each model
runs in an isolated CPU child with one ONNX intra/inter-op thread; success
requires a finite `(1, 1, 1, 256)` mean embedding and records time, peak RSS,
model/input hashes, provider, and thread bounds. The `representation` optional
dependency group pins the three already-verified packages without changing the
default runtime. Nine offline tests pass. At that point the merged dependency
dry-run, installation, and smoke evidence remained next; no training, full-corpus
extraction, or production model change is authorized. The canonical local gate
passed 2,743 default tests plus Ruff/mypy in 24.2 seconds with the optional
inference packages absent, confirming default tests remain offline and
dependency-independent.
The first merged 0.2.57 runtime attempt failed closed before model payload
download: `hf-xet` tried to write its runtime log under sandbox-blocked
`~/.cache/huggingface`. It produced only 8 KB of ignored local-dir metadata,
no result artifact, and no Run Report. Version 0.2.58 configures `HF_HOME` and
`HF_XET_CACHE` below `.cache/representation_models/` before Hub import, keeping
all helper state repo-contained and ignored. The optional group is installed
and `pip check` passes; its measured venv delta is 119,648 KiB. The 0.2.58
canonical gate passed the same 2,743 tests plus Ruff/mypy in 26.2 seconds.
The merged 0.2.58 retry then passed both exact models in 26.875 seconds. Both
emitted finite `(1, 1, 1, 256)` embeddings from the same 2,048-cadence TESS
input; Chronos-Bolt tiny peak RSS was 126,058,496 bytes and Astromer2 peak RSS
was 186,204,160 bytes. Exact weight payload is 29,890,844 bytes and all ignored
cache files total 29,960,842 bytes. Durable artifact
`representation_baseline_inference_smoke_v1.json` has SHA-256
`1cc59ab32e3a8a57e7d966c7cb9b22af04185c0c45245547c853fae7e5de5d10`;
Run Report commit `f8a7207`. Version 0.2.59 records this PASS. Runtime
integration is complete, but no training or full-corpus extraction is
authorized; stellar-variability labels and injection-recovery comparison
remain the next scientific gates. The 0.2.59 evidence-release gate passed all
2,743 default tests plus Ruff/mypy under the 6×6 topology in 34.3 seconds.

Version 0.2.60 pins the first acceptable stellar-variability ground-truth
source before any crossmatch or training. The Drake et al. Catalina
CDS/VizieR table supplies 47,055 publication-backed rows across 17 classes
with explicit inspection flags; its compressed payload is only 1,166,660
bytes. `Skills/verify_stellar_variability_label_source.py` performs five
primary-source checks (delivery headers, schema, row count, class distribution,
and three sample rows), writes structured evidence and a Run Report, and
downloads zero full-catalog bytes. Eight offline tests cover success and
fail-closed drift. Gaia DR3 automated predictions are rejected as ground truth,
and gated approximately 160 GB StarEmbed data are not authorized under the
100 GB ceiling. The merged live metadata verification is next. Source PASS
will still leave the leakage-safe 2,790-TIC crossmatch and embedding-aware
injection comparison open; the independent-TIC work must use the measured
single-parent 6×6 shape. The version 0.2.60 release gate passed 2,751 default
tests plus Ruff/mypy as 8/8 supervised gates in 25.2 seconds.

The merged verifier passed all 5 primary-source operations on 2026-07-14 in
3.334 seconds: 47,055 total rows and all 17 class counts matched, required
schema and 1,166,660-byte delivery metadata were exact, and three labeled rows
validated with zero full-catalog bytes downloaded. Durable artifact
`stellar_variability_label_source_verification_v1.json` has SHA-256
`eb5d4bc6ae02065752e515fff19ed9b012d163f1d82a2be958796a65ba339b9a`;
Run Report commit `b0003bb`. Version 0.2.61 closes source identity only. The
leakage-safe 2,790-TIC metadata/crossmatch and embedding-aware injection
comparison remain the next scientific gates; training is still unauthorized.
The version 0.2.61 evidence-release gate passed the unchanged 2,751 default
tests plus Ruff/mypy as 8/8 supervised gates in 40.2 seconds. All six shards
slowed without errors, timeouts, or a single-shard imbalance; retain the 6×6
topology and do not scale test concurrency further from this run.

Version 0.2.62 implements the next leakage-safe gate without authorizing the
full corpus. `Skills/crossmatch_tess_catalina_labels.py` selects a frozen
216-TIC pilot before modulo partitioning, batches exact TIC IDs through six
MAST workers inside each of six supervised shards, shares one locked/hash-
pinned 1,166,660-byte Catalina cache, and writes disjoint outputs plus shard
Run Reports. The contract precommits 3/1-arcsecond candidate/acceptance radii,
V/TESS magnitude checks, duplicate/non-star/blend rejection, raw class-code
preservation, and global one-to-one reconciliation. Seven focused crossmatch
tests plus one launcher test pass. Every row remains training-disabled; full
2,790-TIC execution and embedding/injection evaluation await merged pilot
throughput, error, overlap, and duplicate evidence.
The version 0.2.62 release gate passed 2,759 default tests plus Ruff/mypy as
8/8 supervised gates in 34.3 seconds under the canonical 6×6 topology.

**TESS live-search v1 evidence (2026-07-11): COMPLETE / REVIEW EVIDENCE-LIMITED.** Three
shards at six workers each processed all 18 frozen targets in 72.79 seconds of
observed wall time, writing 56 schema-v2 records: 53 signal rows and three null
results, with zero preprocessing or ledger failures. Per-shard run reports are
committed and `artifacts/manifests/tess_live_search_v1_run_summary.json`
preserves the ignored SQLite/log hashes and conservative review queue. Three
signals on two targets have FPP below 0.15. The 2026-07-12 conservative review
is recorded in `artifacts/manifests/tess_live_search_v1_fp_review_summary.json`
and appended to the schema-v2 history. TIC 201251996_s03 and s05 are now
`likely_false_positive`: their 58.4% and 31.7% depths fail both large-depth and
companion-radius checks, with weak XGBoost support. TIC 355651994_s02
(P=97.1618 d, depth=1.1546%, FPP=0.0663) is `plausible_but_weak`: 14 available
checks pass, but limb-darkening plausibility fails and 17 diagnostics remain
missing, including centroid and odd/even evidence. Its QLP provenance score is
0.5625, so no `tfop_ready` or external-submission claim is authorized.
Follow-up now establishes that centroid and odd/even are observationally
unavailable: neither of the two TPF sectors covers a predicted event, and only
two QLP events exist versus four required for odd/even. The review status
remains `plausible_but_weak`; additional event-covering observations, not more
analysis of the current files, would be required to resolve those checks.

---

## Tier 1 Gaps (Blocking Live Discovery Production)

### T1-0: First Real Discovery Scan Evidence

- **What is missing**: Nothing required for the current production path. The first QLP scan evidence exists and is retained as historical provenance; no external submission is authorized.
- **Gate status**: **HISTORICAL / NO LONGER ACTIVE** — Option B1-B4 are merged; PR #143 fixed startup/target-selection failures; PR #145 added bounded workers and ETA output. `logs/discovery_run_001.json` through `logs/discovery_run_005_qlp_flux_safe.json` did **not** close this gate for the documented SPOC/no-data, corrupt-cache, stdout-race, wrong-flux-column, and no-progress/no-durable-log reasons. `logs/discovery_run_006_qlp_progress_safe.json` completed locally on 2026-06-29 with real QLP evidence: 200 entries, 192 `candidate_found`, 6 `scanned_clear`, 1 `no_data`, 1 `error`, and active target count 0. `logs/discovery_filtered_006_qlp_progress_safe.json` contains 2 candidates with FPP <= 0.15. Targeted run008 on 2026-06-30 reproduced both filtered candidates after disabling Lightkurve's implicit pre-clean stitch normalizer and serializing vetting features into `exo --output`. Version 0.2.10 regenerated candidate packets moved the two filtered candidates above the prior FPP < 0.15 escalation threshold, so the run006/run008 loop is not submission-ready and is not the active blocker.
- **Why this is not highest priority now**: The user explicitly adopted `docs/exoplanet_exomoon_dataset_handoff.md` as the path to get a trained model. Further run006/run008 review would not close the active production gap unless explicitly requested as forensic work.
- **Root cause of run 001**: `select_targets()` selected TIC catalog stars without requiring light-curve availability, while `run_pipeline()` could not override `fetch_lightcurve()` and therefore fetched only `author='SPOC', exptime='long'`.
- **Root cause of run 002**: interrupted prior QLP downloads left corrupt FITS files in the local Lightkurve MAST cache, and the shared fetch path surfaced Lightkurve's "This file may be corrupt due to an interrupted download" error as a terminal scan error instead of deleting the named cache file and retrying.
- **Root cause of run 003**: the shared fetch path still used Lightkurve public download methods. Lightkurve decorates `SearchResult.download()` and `download_all()` with `suppress_stdout`, which assigns `sys.stdout` process-wide. That is unsafe under `star_scanner.py` worker-thread downloads because the main thread prints progress while a worker can temporarily replace or close stdout.
- **Root cause of run 004**: the shared fetch path requested `pdcsap_flux`, which is a SPOC-style column. Valid QLP HLSP products do not include `PDCSAP_FLUX`; older sectors use `KSPSAP_FLUX`, newer sectors may use `DET_FLUX` or `SYS_RM_FLUX`, and `SAP_FLUX` remains a fallback. Lightkurve wrapped the missing-column `KeyError` as a misleading corrupt-download message.
- **Root cause of run 005**: `ScanLog` created no durable file until the first completed `record()`, `run_background_scan()` printed progress only after a future completed, and the Lightkurve per-product path still let Astroquery `Observations.download_products(verbose=True)` print MAST download banners from worker threads.
- **Additional preflight finding (2026-06-29)**: A one-target live smoke exposed an internal scanner bottleneck before the production batch: QLP TIC 425884922 produced 37,946 cleaned cadences over a 2,608-day baseline, and Astropy's default BLS `autoperiod()` generated ~620-652 million trial periods. The same smoke then exposed an internal argument-order bug when `run_pipeline()` called `vet_signal(signal, light_curve)` instead of the documented `vet_signal(light_curve, signal)`.
- **Run006 evidence**: SHA-256 `8ed084e39fcf1b1f7f0405208a413d4651641aba195305f3ca3b2b8bc3615dc8` for the scan log and `17630739c28bed296910512b86c63c77d952708cf84ab2fe6d8f55ae120a5fc9` for the filtered output. Filtered candidates: TIC 201252011, period 227.39056281978395 d, FPP 0.1160636155807766; TIC 257712351, period 142.95415231096942 d, FPP 0.12672985673564718.
- **Numerical guardrail after run006**: Version 0.2.6 rejects BLS peaks with non-finite/non-positive values and rejects peaks pinned to the lower or upper period-grid boundary. This directly addresses the run006 negative-duration error and the 81 period-boundary detections before any follow-up evidence run.
- **Targeted run008 evidence**: `logs/discovery_run_008_targeted_qlp_stitch_safe.json` has 2 entries, both `candidate_found`, active `{}`, SHA-256 `8626587c4fe59565132e078273763c7beac4a0a88597615f71e147a5134d1b0a`. Filtered output SHA-256 `574a4cf188faa9e273128496fcd23b27cb8369a3e9d2ad2c1b5bbaedd9effed4`; both rows remain below FPP 0.15: TIC 201252011 at P=227.39056281978395 d, FPP=0.11606180728511539; TIC 257712351 at P=142.95415231096942 d, FPP=0.12672948535351847.
- **Run008 root-cause fixes**: Lightkurve's `LightCurveCollection.stitch()` defaulted to `corrector_func=lambda x: x.normalize()`, causing QLP products to be normalized before project sigma-clipping. `fetch_lightcurve()` now calls `stitch(corrector_func=None)` so normalization happens in `clean_lightcurve()` after NaN/outlier removal. `exo --output` now includes the computed `features` dict, so `Skills/false_positive_vetter.py` evaluates real diagnostics instead of reporting all features missing. Version 0.2.9 adds raw `diagnostics`, `fetch_provenance`, `features_missing`, and missing-diagnostic explanations so reviewers can distinguish insufficient phase coverage from not-yet-run catalog/centroid checks. Version 0.2.10 retries transient MAST/Lightkurve connection disconnects in the fetch path instead of failing candidate review on one dropped remote connection.
- **TIC catalog contamination/stellar-density wiring (2026-07-02, version 0.2.11, forensic/supplementary — not the active production path)**: `run_pipeline()` was not passing any catalog kwargs to `vet_signal()` at all — every scan (including run006/run008) computed `contamination_score`, `dilution_sensitivity_score`, `stellar_density_consistency_score`, `nearby_bright_source_score`, and `companion_radius_too_large_score` from `None`/solar-default inputs, and `limb_darkening_plausibility_score` always used the solar-default 5778 K because `vet_signal()` did not even accept a `stellar_teff_k` argument. New `fetch_tic_stellar_params()` in `fetch.py` queries the TIC catalog once per TESS scan (fails open to all-`None` on any error, and is not called for Kepler/K2/JWST) and `run_pipeline()` now forwards `stellar_radius_rsun`/`stellar_mass_msun`/`stellar_teff_k`/`contamination_ratio` into `vet_signal()`. This is a correctness fix to the already-`PRODUCTION READY` Bayesian/XGBoost/ensemble scorers used by any future scan; it does not reopen T1-0 as the active path and centroid/multi-sector diagnostics remain unaddressed. Not validated against the run008 candidates specifically — that would require the user's Mac and is optional forensic work only, not a required next step.
- **Next escalation rule**: Do not submit/contact externally from run006/run008 without explicit human approval. Do not use this historical loop to block the T1-1 model-training path.

### T1-1: Production Tier 2 CNN Checkpoint

- **Status**: **COMPLETE / PROMOTED** — human approved promotion on 2026-07-09, and selected artifacts are registered as `benchmark_cnn_v1`
- **Gate status**: **PASS** — `models/cnn/benchmark_cnn_v1/best.pt` has SHA-256 `f29e6891c255289fa1e2eddad1fb6ca131c063cf11c24b8113e0e29d049441c5`; raw test AUC 0.9572, calibrated F1 0.8347, Brier 0.0580, ECE 0.0142, T=1.0
- **Downstream status**: T1-2 (stacking calibration using this promoted artifact) is now **COMPLETE** — see T1-2 below. No Tier 1 gaps remain open; do not restart old C1-C19/C20 experiments or new data pulls unless a named validation gate fails in the future.
- **Active source contract**: Verify every public data source before use; discover schemas from primary services; preserve immutable source snapshots and enough metadata to redownload raw files after cleanup; use leakage-safe manifests/splits; keep storage bounded; use no synthetic examples for supervised model training in this phase; do not use Kaggle mirrors or unverified pretrained weights when primary NASA/MAST sources are available.
- **Astrometrics policy gates for promotion**: complete. The promotion includes temperature-calibration-aware promotion tooling, model card, reproducibility manifest, data-role registry, storage/retention ledger updates, exact selected artifact scope, frozen `benchmark_cnn_v1` designation, and explicit human approval.
- **Source-contract verification tool (2026-07-02)**: `Skills/verify_dataset_sources.py` implements the "Resource Access Contract" and "Minimum access smoke test" from `docs/exoplanet_exomoon_dataset_handoff.md` exactly — queries `TAP_SCHEMA.columns` for `cumulative` and `toi` before trusting any column name (never infers a renamed column), fetches sample rows from both tables plus the ExoFOP public TOI CSV, and confirms Lightkurve can find both a Kepler and a TESS light curve for one real target pulled from those rows. Fails closed with a specific reason at the first broken step.
- **First live run found a real bug, fixed same-day (2026-07-02)**: `cumulative` schema check failed with zero missing... zero *available* columns and no error, while `toi` passed. Root cause, confirmed via direct `curl` against `TAP_SCHEMA.tables` (not guessed): the archive registers the table as `"CUMULATIVE"` (upper case); `TAP_SCHEMA.columns.table_name` is an exact-match string column, not a resolved SQL identifier, so `= 'cumulative'` matched nothing. `toi` happened to be registered lower case. Fixed by querying `UPPER(table_name) = UPPER('...')` instead of exact match — general fix, not a per-table hardcode; user re-verified the fixed query live via `curl` and got a real 2,366-byte column list back. `FROM cumulative`/`FROM toi` (the actual row-fetch queries) were not affected — those are ordinary SQL identifiers and this codebase already queries them lower case successfully elsewhere (`Skills/fetch_nea_koi_lc_index.py`).
- **Full source smoke PASS (2026-07-02)**: the exact verifier ran end-to-end from the project `.venv` and returned `Overall: PASS`: 5 KOI rows, 5 TOI rows, 8,064 ExoFOP public TOI CSV rows, sample KIC `10797460` with 17 Kepler light-curve search results, and sample TIC `182943944` with 21 TESS light-curve search results. The source-access blocker is cleared.
- **Storage/source snapshot PASS (2026-07-02)**: `Skills/plan_t1_training_batch.py` ran live with `sample_size=5` and wrote committed source snapshots plus sample MAST download metadata without downloading FITS files. Measured source counts: `cumulative` rows=9,564, `toi` rows=7,931, `pscomppars` rows=6,298, KOI label rows=7,454 across 6,515 unique KICs (2,740 confirmed / 4,714 false positive), TOI ephemeris rows=7,824 across 7,535 TICs, and ExoFOP public TOI CSV rows=8,064. MAST search metadata estimated all-KOI Kepler long-cadence raw FITS at 47,099,384,640 bytes (43.86 GiB) and all-TOI TESS raw FITS at 44,994,438,720 bytes (41.90 GiB), combined 92,093,823,360 bytes under the 100 GiB working cap.
- **Leakage-safe Kepler manifest PASS (2026-07-02)**: `Skills/build_t1_training_manifest.py` ran live against the verified `cumulative` schema and wrote committed metadata without downloading FITS files. `metadata/t1_1_kepler_training_manifest.jsonl` has 7,454 KOI rows across 6,515 target groups; all rows from the same KIC share one deterministic split. Split/label counts: train 5,155 rows (label0=3,268 / label1=1,887), val 1,143 (721 / 422), test 1,156 (725 / 431). `metadata/t1_1_kepler_manifest_summary.json` has `flag=OK` and no leakage errors. Cleanup policy requires raw FITS under `data/raw/t1_1_kepler_lc` to be deleted only after processed snippets validate, manifest summary is OK, `logs/t1_1_kepler_processing.sqlite3` has no incomplete active targets, and the operator confirms failed raw FITS are not needed for debugging.
- **Bounded Kepler-first processing batch built and live-smoked (2026-07-02, version 0.2.14)**: `Skills/process_t1_kepler_batch.py` consumes `metadata/t1_1_kepler_training_manifest.jsonl`, fetches each unique KIC target's Kepler light curve once (reusing the proven phase-fold/normalisation math from `Skills/fetch_kepler_lc_snippets.py`), phase-folds every KOI row sharing that target at its own period/epoch, and writes processed snippets to `data/processed/t1_1_kepler_snippets/kepler_snippets.jsonl`. Progress/resume state lives in SQLite at `logs/t1_1_kepler_processing.sqlite3` (per-target `active`/`done` status; a target only ever gets marked `done` after its snippets are flushed, so an interrupted run never leaves partial/duplicate output). Raw FITS downloads are scoped to `data/raw/t1_1_kepler_lc` and the directory is wiped after every target (success or failure), so local raw storage never exceeds roughly one target's data at a time. The first live run on the user's Mac processed 25 targets in 1,216s, wrote 26 snippets, failed 0 rows, left SQLite summary `done|25|26|0`, and left `data/raw/t1_1_kepler_lc` at `0B`.
- **ETA added to per-target progress (2026-07-02)**: at the observed live rate (~49s/target), a 250-target run takes roughly 3.4 hours, so the per-target progress line now prints `elapsed=Xs ETA=YmZZs` instead of elapsed time alone — a multi-hour batch must never look hung. 8 new tests (35 total for this Skill).
- **Second live run PASS (2026-07-02)**: `--max-targets 250` completed on the user's Mac in 7,647s (2h7m), 250 targets processed, 268 snippets written, 0 failed rows, SQLite summary consistent throughout, `data/raw/t1_1_kepler_lc` empty after completion. 277/6,515 targets now done, 6,238 remaining. Confirms the tool is reliable at scale, not just on the initial 25-target smoke.
- **Bounded worker concurrency added (2026-07-02, version 0.2.16)**: `Skills/process_t1_kepler_batch.py` now accepts `--workers` (default 1, sequential) using the same `ThreadPoolExecutor` pattern already proven in `Skills/fetch_kepler_lc_snippets.py`. `docs/SYSTEM_PROFILE.md` recommends 4-6 workers for this kind of external-service/live-catalog workload. Fixed a real concurrency-correctness issue in the process of adding this: the raw-FITS fetcher previously wiped one shared scratch directory after every target, which would have let one worker's cleanup delete a different worker's in-flight download; each target now gets its own `raw_dir/target_<id>` subdirectory, deleted independently. 6 new tests, including one that runs three fetches on real threads and asserts their download directories never collide.
- **Live `--workers 6` crash, fixed same-day (2026-07-02/03, version 0.2.17)**: the user's first concurrent live run crashed with `ValueError: I/O operation on closed file`. Root cause: `make_default_lc_fetcher()` called Lightkurve's public `search.download_all()`, which is decorated with `suppress_stdout` and mutates process-global `sys.stdout` — unsafe while other worker threads print progress. This is the exact same failure class already diagnosed and fixed once before in this project (see run003 under T1-0 for `star_scanner.py`), which should have been checked against before reusing the older `download_all()` pattern. Fix: extended `fetch.py`'s already-proven-safe `_download_collection_with_cache_repair()` / `_download_one_with_cache_repair()` helpers with an optional `download_dir` parameter (backward compatible; the existing `fetch_lightcurve()` call site is unaffected), and rewired `make_default_lc_fetcher()` to call that path instead of `download_all()`. That helper never touches `sys.stdout`; it monkey-patches `Observations.download_products(verbose=False)` under a module-level lock. Rewrote the affected tests to mock at that function boundary and added an explicit regression test, `test_never_calls_download_all`, asserting the unsafe method is never invoked. The interrupted run's SQLite progress is safe: mid-flight targets were left `active` (not `done`) and are retried automatically on resume — no partial or duplicate output was written. (Corrected 2026-07-04: this entry and the ones below originally mislabeled every live run as `--workers 4`; the human confirms `--workers 6` was used throughout.)
- **First post-fix `--workers 6` run PASS (2026-07-03)**: the human re-ran `--max-targets 250 --workers 6` on version 0.2.17. No crash. The pasted console output confirms the startup banner prints `(workers=6)` and every per-target completion line prints `elapsed=...s ETA=...`, exactly as designed. Result: 250 targets processed, 288 snippets written, 0 rows failed, 6,962s (1h56m) elapsed. Cumulative: 530/6,515 targets done, 5,985 remaining. The 6-worker run beat the prior sequential 250-target run (7,647s) by only ~9%, confirming this workload is MAST-network-bound rather than CPU-bound, so `docs/SYSTEM_PROFILE.md`'s "modest, not linear" worker-count guidance holds in practice.
- **Second consecutive clean `--workers 6` run PASS (2026-07-03)**: same recipe: 250 targets processed, 278 snippets written, 0 rows failed, 6,583s (1h50m) elapsed. Cumulative: 780/6,515 targets done, 5,735 remaining. No crash or throttling across two consecutive concurrent runs.
- **Process-level sharding + Run Report Policy added (2026-07-03, version 0.2.18)**: user asked for real multi-tab throughput beyond one process's `--workers` ceiling. New `--shard-index`/`--shard-count` flags partition pending targets by `target_id % shard_count`; the shared `--db-path` (now WAL-mode SQLite, 30s busy timeout) is the single source of truth for global done/active state, and each shard gets its own auto-suffixed output file and raw-download subdirectory, so concurrent shards can never collide the way the 0.2.16→0.2.17 crash did. Separately, a new Run Report Policy (`Skills/run_report.py`) has every run auto-commit and push a small structured completion record after finishing, so cumulative progress is checkable from git without pasted console output (`--status-only` or `run_report.py <script>`). 32 new tests (14 sharding/status-only/run-report-CLI in `test_process_t1_kepler_batch.py`, 18 in `test_run_report.py`). See `docs/DISCOVERY_RUNBOOK.md` Rule 7 and `CLAUDE.md`'s Run Report Policy for the full retrofit scope across other acquisition scripts (not yet done).
- **First live 2-shard test PASS, self-reported (2026-07-03)**: the human ran 2 concurrent shards (`--shard-count 2 --workers 6` each). Both auto-committed their own run reports: shard 0 processed 250 targets (273 snippets) in 8,541s; shard 1 processed 250 (277 snippets) in 8,573s, 0 failures in either. Combined wall-clock ~8,573s for 500 targets = ~17.1s/target, versus the best single-tab rate of 26.3s/target — a real ~35% combined gain, though short of full 2x scaling. Cumulative: 1,280/6,515 targets done, 5,235 remaining.
- **Root cause of sub-linear scaling found and fixed same-day (version 0.2.19)**: `exo_toolkit/fetch.py`'s `_download_one_quietly()` held `_DOWNLOAD_PRODUCTS_LOCK` around the *entire* download call, not just the monkey-patch needed to force Astroquery's `verbose=False` — fully serializing every download within one process regardless of `--workers`, and explaining both the earlier disappointing `--workers 6` results and the sub-2x shard scaling (each shard's own 6 workers were still queuing on that shard's own lock). Fixed by replacing the lock-guarded monkeypatch-and-restore with an idempotent, non-restoring wrap (`_ensure_download_products_quiet()`) — no lock held during any download. 4 new tests, including a real 2-thread test proving concurrent calls now overlap. Removing this artificial brake could reveal MAST's real concurrency ceiling for the first time (2 shards × 6 workers = 12 concurrent connections, already past `docs/SYSTEM_PROFILE.md`'s 4-6 external-service guidance) — the next shard run should be watched for new errors/throttling, not assumed clean by default.
- **4-shard test planned (2026-07-03, prep in version 0.2.20)**: agreed with the human to step up from 2 to 4 concurrent shards next (24 concurrent MAST connections at `--workers 6` each), not jump straight to the 6-7 tabs available, per the Measure-then-scale cadence. New `test_four_concurrent_shards_never_collide` validates the 4-way partition/isolation logic offline (disjoint raw dirs, exactly-once target coverage, correct global `done` state) before the live run; no functional code changes were needed since `shard_index`/`shard_count` were already N-way generalized.
- **Real bug found by that offline test, fixed same-day (version 0.2.21)**: CI failed on the new 4-shard test with `sqlite3.OperationalError: database is locked`. Root cause: `T1KeplerProcessingStore._connect()` runs `PRAGMA journal_mode=WAL;` on every connection, but SQLite only permits one connection to perform that mode transition at a time — 4 threads (or 4 real shard processes) constructing the store for the same fresh `db_path` at nearly the same instant race for that exclusivity and lose immediately, not smoothed over by the connection's `timeout=30.0` parameter (the transition needs momentary exclusivity, not just ordinary write-lock waiting). Fixed with `_ensure_wal_mode()`: a small retry loop (10 attempts, 0.2s apart), falling back to the default journal mode if it never succeeds (WAL is a concurrency nicety here, not a correctness requirement). 2 new tests. This would have been a real risk for the planned 4-tab live test if the tabs were started close together in time — caught by the offline test before it could happen live.
- **4-shard and 6-shard live tests both PASS, self-reported via Run Report Policy (2026-07-03/04)**: 4-shard run (`--workers 6` each — 24 total concurrent connections): 1,000 targets (250/shard) in ~2,149s combined wall-clock = **~2.15s/target** — a dramatic jump from the pre-lock-fix 17.1s/target baseline, confirming 0.2.19 unlocked real intra-shard concurrency, not just inter-shard. The very next 6-shard run (`--workers 6` each — 36 total concurrent connections): 1,500 targets (250/shard, 1 row failure) in ~3,161s = **~2.11s/target** — essentially flat versus the 4-shard run. Per the Measure-then-scale cadence, this flat result is the stop signal: **24-36 total concurrent connections (4-6 shards × 6 workers each) is the practical ceiling for this workload**; this is the human's real standing cadence, not a conservative default — do not recommend fewer shards/workers than this without new evidence it's necessary. Cumulative done, computed from these reports plus the 877-before-2-shard baseline: 877 + 500 + 1,000 + 1,500 = **3,877/6,515 (59.5%)**.
- **DR24 TCE expansion manifest fully processed in one pass, self-reported (2026-07-04)**: the human ran 6 shards × 6 workers (36 total connections) against the new 4,760-target expansion manifest and it completed entirely in a single invocation (no repeat runs needed): 4,760/4,760 targets processed (100%), 8,207 snippets written, 6 rows failed (99.93% row success). Combined wall-clock ~5,768s (96 min, bounded by the slowest shard) = **~1.21s/target** — faster than the corrected 6-shard KOI rate above, consistent with the 6×6 cadence being solid on this workload rather than something to second-guess.
- **Master-corpus Kepler CNN checkpoint is promoted as `benchmark_cnn_v1` (2026-07-09)**: the new DR24 snippets were merged with the existing corpus (7,442 + 8,207 = 15,649 rows, cross-checked exactly) and CNN splits rebuilt (`data/t1_1_kepler_master_cnn_splits/`, validator PASS, train/val/test = 10,800/2,393/2,456, ~21.8% positive overall — noticeably more class-imbalanced than the original ~37%). `checkpoints/cnn_t1_1_kepler_master/best.pt` trained from scratch on this larger corpus and **strictly superseded `checkpoints/cnn_t1_1_kepler/` on every metric** despite the harder class balance: raw test AUC 0.9572 (vs 0.9252), calibrated F1 0.8347 (vs 0.8281), Brier 0.0580 (vs 0.1052, nearly halved), ECE 0.0142 (vs 0.0441, nearly a third), temperature T=1.0. `Flag: PASS`. Human approved promotion on 2026-07-09; selected artifacts are under `models/cnn/benchmark_cnn_v1/` and registered in `models/registry.json`.
- **Kepler/TESS label-source completeness investigation (2026-07-05)**: the human directly challenged whether all labeled Kepler/TESS data had been found, then dropped `docs/seti_labeled_hit_data_research.md` — a research note whose "Comprehensive Protocol" section is a reusable TAP-schema-based discovery + VizieR/literature-audit methodology, now wired into `CLAUDE.md`'s new Label-Source Discovery Protocol as a standing directive (the note's SETI/Breakthrough Listen section is a separate calibration track with its own "no usable per-hit labeled table found" conclusion, unrelated to T1-1). Running the protocol's broader table-discovery query (not just `%tce%`) surfaced the full `_KOI` table family and `k2pandc`, but confirmed no additional new usable Kepler data exists beyond what is already in use — DR25's ~6,382 kepids not in DR24 mostly have no usable label anywhere, and the 535 that do were already captured by the KOI-based manifest independent of any TCE table. For TESS, three real leads were found but remain open/unresolved: the TEV TCE catalog's data API was not located (JS SPA), Planet Hunters TESS/NotPlaNET (`github.com/vtardugno/TESS-CNN`) has real human-vetted labels confirmed not publicly downloadable (contact-only), and the very recent T16 Planet Hunt paper (arXiv:2604.18579, ~11,554 candidates) has an unverified data-availability statement (arXiv/IOPscience blocked automated fetches). No new usable data resulted from this investigation itself.
- **Cross-mission CNN scoring guard added (version 0.2.27, 2026-07-05)**: before promoting `checkpoints/cnn_t1_1_kepler_master/`, the human asked for the code to be hardened against misuse first — a live grep confirmed zero mission-aware gating existed anywhere in the CNN scoring path. `CnnTrainingConfig` gains an optional `training_mission` field (backward compatible with pre-existing `config.json` files, which load it as `None`); `train_cnn.py --mission TESS|Kepler|K2|JWST` stamps it; `CnnScorer.training_mission` exposes it (read from the checkpoint's sibling config, with an explicit override for retrofitting legacy checkpoints). `run_pipeline()`/`exo scan` now refuse by default to apply a CNN checkpoint to a mission it wasn't declared trained on — including checkpoints with an undeclared/`None` mission — raising a clear error; `allow_cross_mission_cnn=True` / `--allow-cross-mission-cnn` is the explicit override for deliberate out-of-domain testing. 18 new tests; full 2,544-test suite re-run clean with zero regressions.
- **Manifest-progress percent-complete added (version 0.2.22)**: user asked to see completion percent on every run without a separate `--status-only` call. The startup banner, final "Done in Xs" line, `format_batch_summary()`, and `--status-only` (when the manifest is available) now print `n_done/n_total (XX.X%)` against the full 6,515-target manifest, computed via a fresh SQLite query at completion so it reflects other concurrently-running shards too. 8 new tests.
- **Percent-complete extended to the committed run report itself (version 0.2.23)**: 0.2.22 only reached console output; the git-committed run report (the "macro" record checked via `git pull` later) still required manually summing multiple shard files against a remembered baseline. `Skills/run_report.py`'s `RunReport` gains optional `items_done_total`/`items_total`/`percent_done` fields (generic and opt-in, `None` for scripts with no "total universe" concept), rendered by `format_run_report()` when present; `process_t1_kepler_batch.py` now populates them. Reading the latest run report for any shard shows the true global percent directly. 5 new tests.
- **T1-1 Kepler-first manifest processing reached 100% (2026-07-04)**: computed from every self-reported run report (877 historical baseline + 5,638 summed `items_processed` = 6,515, matching the manifest's own unique target count exactly). Shards 0-3 of the last 6-way split each explicitly confirmed with a `0 processed` round; shards 4-5 had not yet shown that confirming round as of this check — treat as "at or essentially at 100%," confirm with one more `--status-only` or run before declaring the data-acquisition phase of T1-1 fully closed.
- **Leakage-safety gap found and fixed while confirming completion (version 0.2.24)**: `Skills/build_cnn_training_data.py` (the tool that builds train/val/test splits before CNN training) ignored the manifest's pre-assigned leakage-safe `split` field entirely and did its own random group-based split — running it as-is on the finished corpus would have silently discarded the leakage-safety guarantee the manifest was built to provide. Fixed: it now respects a predefined split when every example carries one, with a hard `ValueError` if a group's examples disagree (itself a leakage bug, not something to silently resolve), falling back to the existing random split for corpora without one. New `split_source` field in the written manifest/summary. 7 new tests.
- **Real field-name bug found by the human's first live run of that fix (version 0.2.25)**: crashed immediately with `ValueError: group 'tic:0' has inconsistent predefined splits ('train' vs 'test')` on the actual 7,442-row combined corpus. Root cause, verified from source: `_group_id()`/`_tic_id()` only recognized `kepid`/`tic_id`/`original_tic_id`/`group_id` (the old ad hoc corpora's fields) — `process_t1_kepler_batch.py`'s `_snippet_record()` writes `target_id`/`group_key` instead, a schema this tool had never been updated to recognize. Every row fell into the same fake group `"tic:0"`; the 0.2.24 check correctly flagged the resulting conflict rather than silently corrupting the split (which is what the old random-grouping logic would have done with this same bug, unnoticed). Fixed: `_group_id()` recognizes `group_key` (used verbatim), `_tic_id()` falls back to `target_id`, and the fallback example-ID no longer mislabels Kepler targets as `"TIC..."`. 4 regression tests reproduce the exact live schema and crash.
- **Manifest confirmed at 100% and Kepler CNN checkpoint PASSED production gates (2026-07-04)**: shards 4-5 confirmed exhausted (877 + 5,638 = 6,515, exact match to the manifest's target-group count). `data/processed/t1_1_kepler_snippets/combined.jsonl` (7,442 rows) and `data/t1_1_kepler_cnn_splits/` were built and validated PASS on `build_cnn_training_data.py` 0.2.25 (`split_source: predefined`; train/val/test = 5,148/1,141/1,153). `checkpoints/cnn_t1_1_kepler/best.pt` was trained from scratch on these splits (best epoch 22, val AUC 0.9148) and evaluated: raw test AUC 0.9252, calibrated F1 0.8281, temperature T=1.0 (no calibration degradation) — the first Kepler CNN checkpoint ever to clear the 0.85 AUC / 0.80 F1 production gates. Promotion to `models/` is paused pending explicit human approval.
- **Kepler DR24 TCE expansion source found and verified live (2026-07-04, version 0.2.26)**: with the Kepler gate already passed, the user asked to grow the corpus further if a real source exists. `Q1_Q17_DR24_TCE` (NASA Exoplanet Archive TAP) carries genuine human/robovetter `av_training_set` labels (`PC`=3,600, `AFP`=9,596, `NTP`=2,541, `UNK`=4,630 excluded as ambiguous) across 20,367 rows / 12,669 unique KICs — confirmed live, not assumed (the same field is present but entirely empty in the newer `Q1_Q17_DR25_TCE` delivery, a real dead end ruled out before committing to DR24). A live overlap check against the committed manifest found **7,091 genuinely new KIC targets** (more than the entire current 6,515-target manifest). New `Skills/build_t1_dr24_tce_expansion_manifest.py` reuses `build_t1_training_manifest.py`'s leakage-safe grouping/split helpers, excludes any KIC already in `metadata/t1_1_kepler_training_manifest.jsonl`, and maps PC→1/AFP→0/NTP→0. `Skills/process_t1_kepler_batch.py` needs no changes — its `--manifest` flag already accepts an arbitrary manifest path, and `_snippet_record()` is schema-generic. 7 new offline tests. **Next `[HUMAN]` action**: run the new builder live (catalog-only, no FITS download) and confirm `flag: OK`.
- **Code status**: Training and state-dict inference paths are operational; the package scorer reconstructs the trained architecture and fails closed when loading fails
- **Prior local corpus status**: **VALID as of 2026-06-12** — 2,037 snippets (1,012 positive CP+KP, 1,025 negative FP+FA, ratio 0.99); zero-epoch corpus retired and rebuilt from scratch with valid BJD epochs; label bug fixed (KP→1); MAST throttling fix applied (`bbb0877`)
- **Local corpus status**: **KEPLER LOCAL VALIDATED** — TESS v2 complete at 2,619 snippets; Kepler finite rebuild has 6,837 parseable snippets with zero non-finite flux rows, zero duplicate resume keys, labels negative=4,280 and positive=2,557; `data/kepler_cnn_splits` validator PASS with train/val/test = 4,741 / 1,060 / 1,036
- **Data policy**: Local CNN corpora, splits, checkpoints, and training logs remain uncommitted unless a future promotion decision explicitly commits a production checkpoint, calibration metadata, registry entry, and reproducibility manifest under `models/`; production-relevant local artifact state must remain visible in `docs/LOCAL_ARTIFACT_LEDGER.md` and `artifacts/manifests/local_artifacts.json`
- **Retired splits**: The seed-42 1,837 / 392 / 394 split and temporary 1,492 / 369 / 368 replacement split were both derived from the invalid zero-epoch corpus and must not be reused
- **Rejected candidate 1**: SHA-256 `e02af3903ab65f4af4f3f05f95dd6da8815a6746fea1bf2eac67bbba3555d6c6`, trained on Python 3.14.3 with PyTorch 2.12.0 on the invalid zero-epoch corpus; best epoch 5, validation AUC 0.7476; test raw AUC 0.7404, F1 0.5804, Brier 0.2131, ECE 0.0716; Platt calibration threshold 0.503 produced test F1 0.6297, Brier 0.2295, ECE 0.1273; **REJECTED** — AUC and F1 below targets, calibration worsened both Brier and ECE
- **Rejected candidate 2**: Trained on Python 3.14.3 with PyTorch 2.12.0 on the valid 2026-06-12 corpus (2,037 snippets); splits: 1,425 train / 306 val / 306 test; default config (LR=1e-3, weight_decay=1e-4, dropout 0.5/0.3, aug_noise=0.02); best epoch 4, validation AUC 0.8177, F1 0.7711; test raw AUC 0.7180, F1 0.6998, Brier 0.2153, ECE 0.0646; Platt calibration A=1.5546, B=−0.7152, threshold 0.43 produced test F1 0.6998, Brier 0.2237, ECE 0.0730; **REJECTED** — test AUC 0.7180 < 0.85 gate, test F1 0.6998 < 0.80 gate, calibration worsened both Brier and ECE; val→test AUC gap of 10 points (0.8177→0.7180) indicates insufficient regularization
- **Root cause (candidate 2)**: Model overfit before early stopping (best epoch 4 out of 50); `train_loss=0.4409` vs `val_loss=0.6200` at epoch 14; 10-point AUC gap confirms under-regularization for 1,425-example corpus
- **Rejected candidate 3**: seed-42 splits, `cnn_retrain_v1.json` (LR=3e-4, weight_decay=1e-3, dropout 0.5/0.5, aug_noise=0.05); best epoch 13, val AUC=0.8235; test AUC=0.7283; val→test gap 9.5 pts; **REJECTED** — test AUC below gate; root cause: seed-42 split assigned harder examples to test partition
- **Rejected candidate 4**: seed-7 splits (1,425/306/306), `cnn_retrain_v1.json`; best epoch 32, val AUC=0.7914; test AUC=0.7682; val→test gap 2.3 pts; **REJECTED** — test AUC below gate; gap resolved but model generalization insufficient
- **Rejected candidate 5**: seed-7 splits, `cnn_retrain_v1.json`, ETA-enabled training; best epoch 33, val AUC=0.8083; test AUC=0.7758, F1=0.7268; val→test gap 3.3 pts; **REJECTED** — AUC 0.7758 < 0.85 gate, F1 0.7268 < 0.80 gate; gains plateaued without architectural improvements
- **Rejected candidate 6**: `cnn_retrain_v2.json` (use_batch_norm=true, flip=true, shift=20); best epoch 1 (val AUC=0.7344); val loss exploded 0.68→1.07; val AUC below 0.5 by epoch 3; early stopping epoch 11; **REJECTED** — BatchNorm1d running stats do not converge in 22 mini-batches/epoch; stale stats misnormalize val set in eval mode; BN incompatible with this dataset size
- **Rejected candidate 7**: seed-7 splits, `cnn_retrain_v2b.json` (use_batch_norm=false, flip=true, shift=20); best epoch 23, val AUC=0.7887; test raw AUC=0.7527, F1=0.7214, Brier=0.2070, ECE=0.0990; Platt A=1.4732, B=−0.6896, threshold=0.50; calibrated test F1=0.7202, Brier=0.2200, ECE=0.1168; val→test gap 3.6 pts; **REJECTED** — test AUC 0.7527 < 0.85 gate, F1 0.7202 < 0.80 gate; root cause: model has ~435K parameters against 1,425 examples (massively overparameterized); augmentation reduced gap slightly but also depressed val AUC relative to candidate 5
- **Root cause (systematic, candidates 2–7)**: Dense(256) layer alone has 410K parameters; with 1,425 training examples the model cannot generalize past ~0.78 test AUC regardless of regularization; model capacity must be reduced to match the dataset size
- **Rejected candidate 8**: seed-7 splits, `cnn_retrain_v3.json` (Conv 8/16/32, Dense 64, dropout 0.3, flip+shift); best epoch 11, val AUC=0.7734; test raw AUC=0.7094, F1=0.6805, Brier=0.2238, ECE=0.0992; Platt A=1.3045, B=−0.6584; calibrated test F1=0.6792, Brier=0.2307, ECE=0.1282; val→test gap 6.4 pts; **REJECTED** — test AUC 0.7094 < 0.85 gate; root cause: halved conv channels reduced feature extraction capacity; flip+shift augmentation net harmful across all candidates where it was tested
- **Rejected candidate 9**: seed-7 splits, `cnn_retrain_v3b.json` (Conv 16/32/64, Dense 128, dropout 0.4, no flip/shift); best epoch 18, val AUC=0.7807; test raw AUC=0.7573, F1=0.7257, Brier=0.2051, ECE=0.0867; Platt A=1.5114, B=−0.7136; calibrated test F1=0.7207, Brier=0.2155, ECE=0.1230; val→test gap 2.3 pts; **REJECTED** — test AUC 0.7573 < 0.85 gate, F1 0.7207 < 0.80 gate; reducing Dense(256→128) slightly hurt val AUC without improving test AUC vs C5
- **Rejected candidate 10 (ensemble)**: 3-seed ensemble of `cnn_retrain_v1.json` (seeds 7, 13, 99); individual val AUCs: 0.7914, 0.7848, 0.8022; ensemble val AUC=0.8022; test raw AUC=0.7670, F1=0.7317, Brier=0.2057, ECE=0.1260; Platt A=1.5945, B=−0.7796; calibrated test F1=0.7317; **REJECTED** — test AUC 0.7670 < 0.85 gate, F1 0.7317 < 0.80 gate; ensemble is *worse* than best single model (C5: 0.7758); members too correlated on 1,425 examples to provide diversity gain
- **Systematic ceiling confirmed (candidates 2–10)**: All 10 candidates (single-model and 3-seed ensemble) produced test AUC 0.71–0.78; ceiling is a data-size constraint; 1,425 training examples cannot drive this architecture to 0.85 AUC regardless of tuning, regularization, augmentation, or ensembling strategy
- **Rejected candidate 11 (Kepler->TESS transfer)**: `checkpoints/cnn_tess_finetuned/best.pt`, SHA-256 `3fc115b3623b2485373aefef30a7aa901e1183cc77ef4b57ce6c1f2219f49214`; trained on Python 3.14.3 with PyTorch 2.12.0 using `device=mps`; initialized from Kepler pretrain SHA `c782d7af61171b3f58447f7a49343c86618c447292a71bd28d540807835787c7`; TESS splits train/val/test = 1,477 / 318 / 315; `configs/cnn_tess_finetune.json` with LR=1e-4, weight_decay=1e-3, batch=64, seed=7, frozen conv layers for 15 epochs; best epoch 22, validation loss 0.5255, validation AUC 0.8408; test raw AUC=0.8115, F1=0.7523, Brier=0.1818, ECE=0.0854; Platt A=1.80214901, B=-0.77900211, threshold=0.45; calibrated test F1=0.7508, Brier=0.1966, ECE=0.1152; **REJECTED** — test AUC 0.8115 < 0.85 gate, calibrated F1 0.7508 < 0.80 gate, and calibration worsened both Brier and ECE
- **Rejected candidate 12 (full-unfreeze Kepler->TESS transfer)**: `checkpoints/cnn_tess_c12/best.pt`, SHA-256 `cc8fbd2004e0fd41dc48bf7f48e3d6b552c75164c62556c3a016af3ca1642ff0`; trained on Python 3.14.3 with PyTorch 2.12.0 using `device=mps`; initialized from Kepler pretrain SHA `c782d7af61171b3f58447f7a49343c86618c447292a71bd28d540807835787c7`; TESS splits train/val/test = 1,477 / 318 / 315; `configs/cnn_tess_finetune_c12.json` with full unfreeze from epoch 1, LR=3e-5, weight_decay=1e-3, batch=32, seed=7; best epoch 20, validation loss 0.5302, validation AUC 0.8356; test raw AUC=0.8124, F1=0.7542, Brier=0.1780, ECE=0.0556; Platt A=1.73886384, B=-0.82141269, threshold=0.47; calibrated test F1=0.7516, Brier=0.1979, ECE=0.1283; **REJECTED** — validation AUC missed the 0.85 continuation threshold, test AUC and calibrated F1 remain below production gates, and calibration worsened both Brier and ECE
- **Transfer-learning result (C11/C12)**: Kepler pretraining lifted held-out test AUC above the prior TESS-only/ensemble range, but both C11/C12 transfer candidates plateaued near test AUC 0.812 and calibrated F1 0.75 on 1,477 training examples. Root cause was data-size ceiling.
- **Rejected candidate 13 (C13 — combined corpus, default LR)**: `checkpoints/cnn_tess_c13/best.pt`; trained on Python 3.14.3 with PyTorch 2.12.0 using `device=mps`; initialized from Kepler pretrain SHA `c782d7af61171b3f58447f7a49343c86618c447292a71bd28d540807835787c7`; combined splits train/val/test = 4,892 / 1,049 / 1,033 from `data/tess_combined_cnn_splits`; default config LR=1e-3 with half-LR schedule; best epoch 8, val_loss=0.8197, val_auc=0.8195; test raw AUC=0.8342, F1=0.7960, Brier=0.1664, ECE=0.0625; Platt A=1.78234347, B=-0.76273353, threshold=0.47; calibrated test F1=0.7960, Brier=0.1828, ECE=0.1334; **REJECTED** — test AUC 0.8342 < 0.85 gate, calibrated F1 0.7960 < 0.80 gate, and calibration worsened both Brier and ECE. Root cause: LR=1e-3 too high for pretrained init — val_loss spiked at epoch 5 (0.8421) then diverged to 1.3230 while train_loss fell to 0.2247; model overfit despite large dataset. Positive signs: test AUC improved +2.2 pts over C12; test raw F1 improved +4.2 pts; first candidate where test AUC exceeds val AUC (model generalizes when not derailed by LR).
- **C13 corpus expansion effect confirmed**: 4,892 vs 1,477 training examples drove AUC from 0.8124 to 0.8342 and F1 from 0.7542 to 0.7960. The data-size ceiling is broken; LR is now the limiting factor.
- **Path A inventory result**: Completed locally on 2026-06-18 against `data/tess_snippets_v2.jsonl`; ExoFOP TOI live counts CP=733, KP=591, FP=1,244, FA=100 (positive=1,324; negative=1,344; total=2,668); expansion inventory found only 56 new labeled TIC IDs (16 positive, 40 negative; 33 TOI, 23 CTOI). **Do not run the long MAST snippet fetch as a production-closing attempt**; even 100% fetch success would not materially move the CNN from 2,110 usable examples toward the ≥5,000 target.
- **TESS TCE source probe**: `Skills/tess_tce_fetcher.py` now fails closed with `Flag: UNAVAILABLE` for the stale historical ExoMAST TCE endpoint, which returned HTTP 404 on 2026-06-18. Do not treat that endpoint as the next large TESS-domain label source unless a current provider contract is found and documented.
- **Path B corpus result (2026-06-20)**: `Skills/fetch_tess_kepler_overlap_snippets.py` completed across multiple sessions (~11.8 h total, 4 workers, polite 0.25–1.0 s/worker request delay); `data/tess_kepler_overlap_snippets.jsonl` has **4,864 snippets**; `data/tess_kepler_overlap_snippets.jsonl.failures.jsonl` has ~2,716 terminal failures (NO_DATA/SHORT/NONFINITE/NO_LIGHTKURVE — correctly excluded). Combined corpus projection: TESS v2 (2,619) + overlap (4,864) = **~7,483 total snippets before dedup/filtering**, ~5× the 1,477 training examples that caused the systematic AUC ceiling.
- **Rejected candidate 14 (C14 — combined corpus, low LR)**: `checkpoints/cnn_tess_c14/best.pt`, SHA-256 `12fe6fe1004e1ea75b6fa5f244512cbe93e1b176bf3ec822ef5bd6df861d753d`; trained on Python 3.14.3 with PyTorch 2.12.0 using `device=mps`; combined splits train/val/test = 4,892 / 1,049 / 1,033; `configs/cnn_tess_finetune_c12.json` (LR=3e-5, batch=32, patience=20, full-unfreeze from epoch 1); best epoch 61, val_loss=0.7744, val_auc=0.8116; early stop epoch 81; test raw AUC=0.8319, F1=0.7859, Brier=0.1663, ECE=0.0273; Platt A=1.69593273, B=-0.73468346, threshold=0.47; calibrated test F1=0.7860, Brier=0.1932, ECE=0.1441; **REJECTED** — AUC 0.8319 < 0.85, cal F1 0.7860 < 0.80, Platt worsened Brier/ECE. Root cause: LR=3e-5 too conservative — LR scheduler decayed to 1.17e-7 by epoch 79, locking model in local optimum below C13 ceiling. Note: raw ECE=0.0273 is excellent (model well-calibrated without Platt); Platt is confirmed as systematic overcorrection (A≈1.7 sharpens already-calibrated probabilities) — 4 consecutive candidates affected.
- **Systematic calibration problem identified (C11–C14)**: Platt scaling has worsened Brier and ECE across all 4 Kepler→TESS transfer candidates. Raw ECE for C12=0.0556, C13=0.0625, C14=0.0273 — model probabilities are already well-calibrated. Platt A≈1.7–1.8 overcorrects by sharpening predictions that do not need sharpening. A future [AGENT] task should replace Platt with temperature scaling (single parameter) or skip calibration when raw ECE ≤ 0.05. This does not require a gate change, only a calibration method change in `evaluate_cnn_checkpoint.py`. Requires explicit human approval before modifying the production gate definition.
- **Rejected candidate 15 (C15 — combined corpus, intermediate LR)**: `checkpoints/cnn_tess_c15/best.pt`, SHA-256 `34f50183d19b73cdee48bbd1cc3a3680173c802faf5c9d4227369c75c772128c`; trained on Python 3.14.3 with PyTorch 2.12.0 using `device=mps`; combined splits train/val/test = 4,892 / 1,049 / 1,033; `configs/cnn_tess_c15.json` (LR=1e-4, min_lr=1e-6, lr_scheduler_patience=10, weight_decay=1e-3, use_batch_norm=false, augment=true, batch=32, patience=20, full-unfreeze from epoch 1); best epoch 16, val_auc=0.8162, val_loss=0.7663; early stop epoch 36; test raw AUC=0.8353, F1=0.7949, Brier=0.1642, ECE=0.0427; Platt A=1.73766998, B=-0.73987247, threshold=0.52; calibrated test F1=0.7938, Brier=0.1888, ECE=0.1389; **REJECTED** — test AUC 0.8353 < 0.85 gate, cal F1 0.7938 < 0.80 gate, Platt worsened Brier/ECE. Root cause: **LR tuning exhausted** — C13 (1e-3), C14 (3e-5), C15 (1e-4) all plateau at test AUC 0.83–0.84. Model peaked at epoch 16 (train_loss=0.46) then overfit hard (val_loss 0.77→1.03 while train_loss fell to 0.28 over 20 patience epochs). Primary bottleneck is insufficient regularization (weight_decay=1e-3, no batch norm), not LR. Platt A=1.74 is 5th consecutive calibration failure across C11–C15.
- **LR tuning trajectory**: C13 (LR=1e-3) → test AUC 0.8342; C14 (LR=3e-5) → test AUC 0.8319; C15 (LR=1e-4) → test AUC 0.8353. All three candidates converge at 0.83–0.84. Continued LR search is not productive.
- **Rejected candidate 16 (C16 — BN + strong L2)**: `checkpoints/cnn_tess_c16/best.pt`; trained on Python 3.14.3 with PyTorch 2.12.1 using `device=mps`; `configs/cnn_tess_c16.json` (LR=1e-4, weight_decay=1e-2, use_batch_norm=true, patience=25, augment=true); pretrain load: **8 tensors matched, 4 skipped** (shape mismatch) — BatchNorm layers shift Sequential indices so only `conv.0` and FC layers transferred; 2nd and 3rd conv layers trained from random init; best epoch 10, val_auc=0.6650, val_loss=1.0478; val_loss exploded 0.78→3.83 over 35 epochs while train_loss fell 1.77→0.33; early stop epoch 35; evaluator not run (val AUC far below 0.85 gate). **REJECTED** — val AUC 0.6650 < 0.85 gate; catastrophically worse than C13–C15. Root cause: BN index shift causes severely partial pretrain transfer (only first conv layer out of three); random-init 2nd+3rd conv layers combined with aggressive weight_decay=1e-2 caused immediate catastrophic overfitting.
- **Strategy exhaustion summary**: LR tuning (C13–C15, three orders of magnitude) → 0.83–0.84 ceiling. BN+WD regularization (C16) → catastrophic failure at 0.67. Both approaches exhausted. The 0.83–0.84 ceiling in C13–C15 is most likely a data ceiling, not a tuning problem.
- **C17 REJECTED (2026-06-21)** — joint Kepler+TESS fine-tuning:
  - **Result**: best val AUC 0.7859 (epoch 16), early stop epoch 46; val_loss 0.79→1.42 while train_loss 0.60→0.20
  - **Root cause**: domain mismatch. Kepler (30-min cadence) and TESS (2-min cadence) transit morphologies differ in noise profile, cadence aliasing, and phase-fold artifacts. Joint training caused the conv layers to drift toward mixed-domain representations that do not generalize to the TESS-only val set.
  - **Do not retry joint training**. Retain `data/joint_cnn_splits/` for reproducibility.
- **C18 REJECTED (2026-06-21)** — FC head warm-up with `freeze_conv_epochs=10`:
  - **Result**: SHA-256 `d33c15f45bd369d5eba4b87da3aa1908decc3baef5231dcff8544dd70987d496`; best epoch 22, val AUC 0.8262; early stop epoch 47. Test raw AUC=0.8439, F1=0.7979, Brier=0.1593, ECE=0.0301. Temperature T=1.61363521. Calibrated: threshold=0.46, Brier=0.1632, ECE=0.0667. **Flag: FAIL**.
  - **Best candidate of all 19**: test AUC improved from the 0.83–0.84 plateau (C13–C15) to 0.8439. `freeze_conv_epochs` confirmed as the right direction.
  - **Why it failed**: (1) raw AUC 0.8439 < 0.85 gate (short by 0.006). (2) T=1.61 fitted on overconfident val then applied to already-well-calibrated test (raw ECE=0.0301), worsening ECE to 0.0667.
  - **Do not rerun C18 unchanged.**
- **C19 REJECTED (2026-06-22)** — Extended FC head warm-up with `freeze_conv_epochs=20`:
  - **Result**: SHA-256 `65f3721fac577807f35e4edaeaa9cc0cd0f50959441344487f7c77f35a570436`; best epoch 29 (8 epochs after unfreeze at epoch 21); early stop epoch 54. Test raw AUC=0.8420, F1=0.7951, Brier=0.1606, ECE=0.0377. Temperature T=1.8785927. Calibrated: threshold=0.40, Brier=0.1658, ECE=0.0760. **Flag: FAIL**.
  - **Regressed from C18 in every metric**. Root cause: LR scheduler fires on val_auc plateaus; during 20 frozen epochs val_auc improved monotonically, so LR never decayed. Conv unfroze at epoch 21 with LR still at 1e-4 — same as C18 at epoch 11. Longer frozen phase over-adapted the FC head (T=1.88 vs T=1.61), producing worse calibration and lower test AUC.
  - **freeze_conv strategy exhausted.** Do not retry without a materially different corpus or training schedule.
- **Strategic decision (2026-06-22)**: Human chose Option C — more data. All training-side approaches exhausted on 4,892 examples. Next authorized corpus: K2 EPIC overlap (K2 KOI confirmed planets/FPs with TESS re-observations folded at K2 ephemerides). See runbook Step 7g.
- **ECE-skip gate fix (2026-06-22)**: `evaluate_cnn_checkpoint.py` now skips temperature scaling when raw test ECE < 0.05. Root cause of C11–C19 calibration doom loop confirmed: val is overconfident due to early-stopping selection bias; T > 1 applied to already-calibrated test structurally worsened ECE. With the fix, C20 gate is: raw AUC ≥ 0.85 AND raw F1 ≥ 0.80 (when ECE < 0.05, cal==raw). `Skills/fetch_tess_k2_overlap_snippets.py` and `configs/cnn_tess_c20.json` committed.
- **Current data gate**: TESS combined splits VALIDATED; Kepler splits VALIDATED; C13–C19 all rejected; no CNN checkpoint approved for promotion. K2 overlap corpus is complete at 2,086 snippets. C20-style assembly is not blocked by T1-0 anymore, but it must be re-planned under the dataset handoff brief with source/schema verification and leakage-safe manifests before any local training request.
- **Current authorized runbook**: `docs/CNN_PRODUCTION_RUNBOOK.md`
- **Current promotion gate**: raw held-out test AUC ≥ 0.85; calibrated held-out test F1 ≥ 0.80; temperature scaling calibration must not worsen held-out test Brier score or ECE
- **Calibration note**: Temperature scaling (T fitted via NLL on val split) replaced Platt scaling on 2026-06-21. Platt A≈1.7–1.8 consistently worsened calibration because raw predictions were already well-calibrated (ECE 0.02–0.06). Temperature scaling is the identity at T=1 and will not artificially sharpen probabilities.
- **Kepler pretraining gate**: **LOCAL PRETRAINED ON MPS** — `checkpoints/cnn_kepler_pretrain/best.pt`, SHA-256 `c782d7af61171b3f58447f7a49343c86618c447292a71bd28d540807835787c7`; Python 3.14.3 venv, PyTorch 2.12.1; startup banner `device=mps`; best epoch 19, best validation loss 0.3905, best validation AUC 0.9186; final epoch 34 val AUC 0.9123; retain as transfer-learning source
- **Architecture spec**: `docs/CNN_SPEC.md`
- **Artifact policy**: Keep `git add .` safe through `.gitignore`; commit local artifact status in the artifact ledger; commit the validated production checkpoint, calibration metadata, model registry entry, and reproducibility manifest under `models/` only after all production-readiness checks pass and the human approves promotion

### T1-2: Stacking Tier 3 Production Weight Calibration

- **Status**: **COMPLETE (2026-07-10)** — held-out K2 calibration set built, native snippets fetched live (sharded), CNN predictions merged, stacking weights calibrated, and the calibrated weights are wired into production (`src/exo_toolkit/cli.py`'s `full-ensemble` blend and `StackingScorer.from_model_paths()` defaults).
- **Live fetch complete (2026-07-10)**: the human ran `fetch_t1_2_k2_calibration_snippets.py` sharded 4-way (`--shard-count 4`, `--workers 6` each, per `CLAUDE.md`'s "always shard when it applies" standing rule) — all 4 shards self-reported success via the Run Report Policy: 596/596 targets processed (142/132/168/154 per shard), 588 snippets written, 8 terminal failures, ~3.5 minutes combined wall-clock (bounded by the slowest shard at ~211s). The 4 shard output files were concatenated into `data/t1_2_k2_calibration_snippets.jsonl` (588 lines, verified zero duplicate EPIC IDs).
- **CNN merge (2026-07-10)**: new `Skills/merge_t1_2_k2_cnn_predictions.py` scores every row with a fetched snippet using `CnnScorer.from_checkpoint("models/cnn/benchmark_cnn_v1/best.pt", ...)`, explicitly declaring `training_mission="Kepler"` (the checkpoint predates the `training_mission` config field). This is a deliberate **cross-mission** application — `benchmark_cnn_v1` was trained exclusively on Kepler prime-mission targets; K2 was chosen for this calibration set specifically because it is leakage-safe from that training data, not because it is the CNN's native domain. The script annotates every row with `cnn_training_mission`/`cnn_cross_mission` for transparency and drops (with an explicit counted line, never silently) the 8 rows with no fetched snippet before writing `metadata/t1_2_k2_calibration_completed_predictions.jsonl` (588 rows). Found and fixed a real integration bug in the process: `calibrate_stacking_weights.py`'s loader did `float(rec["cnn_prob"])` unconditionally and would have crashed (`TypeError`) on any row with a null `cnn_prob` — fixed with defensive skip-and-continue plus a new regression test, and the merge script filters nulls before writing as defense-in-depth.
- **SEVERE bug found and fixed while running the CNN merge live (2026-07-10)**: the first live run of `merge_t1_2_k2_cnn_predictions.py` produced a `cnn_prob` of exactly `0.5` for all 588 rows — a flat constant, not real predictions. Root cause: `CnnScorer._ensure_model()`/`_load()` (`src/exo_toolkit/ml/cnn_scorer.py`) do the absolute import `import_module("Skills.cnn_inference_batcher")`, which requires the **repo root** (not just `src/`) on `sys.path`. When a `Skills/*.py` script is invoked directly as `python Skills/foo.py`, Python auto-prepends only the script's own containing directory (`Skills/`) to `sys.path` — never the repo root — so the import silently raised `ModuleNotFoundError`, caught by a bare `except Exception: self._available = False`, with **no error, warning, or other signal**. Every `predict_proba`/`predict_proba_batch` call then returned the neutral fallback `0.5` for every input, indistinguishable from a real (if uninformative) prediction. This is exactly the failure class the merge script's own module docstring warns about for `calibrate_stacking_weights.py`'s null-handling, except worse: nulls are visible and crash loudly; a constant `0.5` looks like valid data and flows silently downstream. It was only caught by a manual sanity check (`mean(cnn_prob | label=1) == mean(cnn_prob | label=0) == 0.5000`, `cnn_prob` had exactly 1 distinct value) run *after* the first (bogus) `calibrate_stacking_weights.py` pass had already produced `CNN=0.000` — the same numeric weight the corrected run below also produced, which is what makes this dangerous: a plausible-looking, coincidentally-identical wrong answer that would have been indistinguishable from the right one without independently checking the intermediate `cnn_prob` distribution. **Fixed** in `src/exo_toolkit/ml/cnn_scorer.py`: a new `_ensure_repo_root_on_sys_path()` helper inserts the repo root onto `sys.path` (idempotent) before every `import_module("Skills....")` call site (three sites: calibration load, model load, calibration apply), and `_ensure_model()` now emits a `RuntimeWarning` naming the real exception whenever `Skills.cnn_inference_batcher` fails to import or the checkpoint fails to load for a reason *other* than the documented "PyTorch not installed" case (which remains silent by design — see the module docstring). 9 new regression tests in `tests/test_cnn_scorer.py`, including one that reproduces the exact incident (repo root removed from `sys.path`, checkpoint still loads) and two that assert the warn/silent split is correct. After the fix, `cnn_prob` had 576 distinct values with real discrimination (mean 0.370 for label=1 vs 0.111 for label=0) — see the corrected calibration result below.
- **Stacking weights calibrated (2026-07-10, corrected)**: `Skills/calibrate_stacking_weights.py` ran an AUC-maximising grid search over the 588 completed rows (356 positive / 232 negative), re-run after the bug above was fixed, and found **XGBoost=0.95, CNN=0.00, Bayesian=0.05** (best AUC 0.9576) — numerically identical to the pre-fix (bogus) run, but now backed by real data. Standalone per-tier AUC on this same held-out set: **CNN=0.7458, XGBoost=0.9575, Bayesian=0.8656**. This is the important correction to the pre-fix narrative: the CNN weight is zero **not** because the CNN carries no cross-mission (Kepler→K2) signal — 0.7458 is real, well above chance — but because XGBoost alone already sits almost exactly at the blend's ceiling AUC (0.9575 vs 0.9576 blended), so a coarse (step=0.05) pure-AUC grid search has no incentive to dilute it with a weaker signal. The CNN's own frozen-eval AUC in its native Kepler domain is 0.9572, consistent with 0.7458 being a real (if degraded) cross-mission transfer, not noise. Saved to `models/stacking_weights.json`.
- **Calibrated weights wired into production (2026-07-10)**: `src/exo_toolkit/cli.py` gains named `FULL_ENSEMBLE_XGB_WEIGHT`/`FULL_ENSEMBLE_CNN_WEIGHT` constants (0.95/0.00, replacing the old uncalibrated 0.35/0.35 guess from `CNN_SPEC.md`) used by the real `--scorer full-ensemble` blend; `StackingScorer.from_model_paths()`'s defaults are updated to match. Applied **globally** (not conditioned on mission-match) as a deliberate conservative choice: the cross-mission scoring guard (PR #193) already blocks the CNN by default for any mission that doesn't match its declared `training_mission`, so the only scenario where these constants apply without an explicit `--allow-cross-mission-cnn` override is genuine same-domain Kepler scoring — a case this calibration did not measure and for which the old 0.35/0.35 numbers had no calibration evidence either. A true same-domain stacking calibration would require its own dedicated held-out Kepler set (separate from the existing training/validation/frozen-eval roles) — out of scope unless explicitly requested. See `data_selection/data_selection_decision_log.md`'s 2026-07-10 entry for the full decision record.
- **Prior state (superseded)**: Conservative fallback weights were XGBoost 0.35 + CNN 0.35 + Bayesian 0.30; when CNN was absent, blend fell back to XGBoost 0.538 + Bayesian 0.462.
- **Gate**: ~500 labeled held-out examples — **satisfied**: 596 rows (356 CONFIRMED / 240 FALSE POSITIVE) selected 2026-07-10; 588 fully scored after the live fetch
- **Held-out calibration set built (2026-07-10)**: `benchmark_cnn_v1` and `models/xgboost_koi.json` were both trained/split entirely from Kepler prime-mission KIC targets, and the existing training/validation/frozen-eval roles in `data_selection/data_role_registry.yaml` explicitly forbid reuse for stacking calibration. `Skills/build_t1_2_k2_calibration_manifest.py` builds a genuinely disjoint calibration set from the NASA Exoplanet Archive K2 planets-and-candidates table (`k2pandc`) — a different EPIC target catalog observed during different campaigns/sky fields than Kepler prime, so `k2:epic:<id>` group keys can never collide with the `kepler:kic:<id>` namespace. Live-verified 2,261 usable CONFIRMED/FALSE POSITIVE rows; committed manifest at `metadata/t1_2_k2_calibration_manifest.jsonl` (596 rows, seed 42, all available false positives plus a seeded sample of confirmed rows).
- **Two real unit bugs found and fixed while building this manifest (2026-07-10)**: (1) `pl_tranmid` is already full BJD_TDB — a pre-existing script, `Skills/fetch_tess_k2_overlap_snippets.py`, wrongly assumed BKJD and added the 2454833-day offset a second time, corrupting the phase-fold epoch of every one of its 2,086 already-fetched TESS-domain K2-overlap snippets (`data/tess_k2_overlap_snippets.jsonl`, local-only, not yet used in any trained model — no promoted checkpoint is affected, but this file must be re-fetched before any future C20-style TESS combined-corpus attempt). Both the existing script and the new manifest builder are fixed and regression-tested. (2) `pl_trandur` is empirically already in hours despite its own `tap_schema.columns` metadata claiming days — confirmed by cross-checking published transit durations for K2-18 b and K2-3 b. See `data_selection/data_selection_decision_log.md`'s 2026-07-10 entry for full detail.
- **Catalog-only scoring complete (2026-07-10)**: `Skills/score_t1_2_k2_calibration.py` maps each manifest row's k2pandc columns into `CandidateFeatures` using the exact same transform (`Skills/build_training_data.py`'s `row_to_features`) the XGBoost KOI model was trained with, then scores every row with both `models/xgboost_koi.json` and the Bayesian log-score model — no light curve required for these two tiers. Ran live on the full 596-row manifest; xgb_prob correctly discriminates in the right direction (mean 0.077 for label=1 vs 0.002 for label=0). Output: `metadata/t1_2_k2_calibration_partial_predictions.jsonl` (cnn_prob still `null` pending the snippet fetch below).
- **Native K2 snippet fetcher built (2026-07-10)**: `Skills/fetch_t1_2_k2_calibration_snippets.py` fetches the **native K2** light curve (mission="K2", not TESS) for each manifest EPIC target via `exo_toolkit.fetch.fetch_lightcurve()` (the same production, cache-repair-aware, retry-aware path used by `process_t1_kepler_batch.py` — not the older, concurrency-unsafe `download_all()` pattern), phase-folds at the manifest's period/epoch, and writes 201-bin normalised snippets with resume, terminal-failure logging, bounded `--workers`, ETA console output, and Run Report Policy integration.
- **Process-level sharding added (2026-07-10)**: per `CLAUDE.md`'s Parallelism-First Recipe Policy ("always shard when it applies" — elevated to a standing rule this same day), `Skills/fetch_t1_2_k2_calibration_snippets.py` gained `--shard-index`/`--shard-count` (mirroring `process_t1_kepler_batch.py`'s pattern) before the live 596-target fetch — partitions by `epic_id % shard_count`, auto-suffixed per-shard output/failure files so concurrent tabs never collide. 18 offline tests pass (10 new across both features).

---

## Tier 2 Gaps (Improvements, Not Blocking Deployment)

### T2-1: TESS-Specific XGBoost Model — COMPLETE

- **Status: COMPLETE as of 2026-06-11**
- **Model**: `models/xgboost_toi.json` + `models/xgboost_toi.xgb.json` (committed `882b838`)
- **Training set**: 1,960 examples from ExoFOP TOI (CP/KP → positive, FP/FA → negative)
- **Performance**: AUC=0.884, F1=0.729 on held-out fold; Platt calibration A=5.3061, B=−2.7153
- **Usage**: `exo <TIC-ID> --scorer xgboost --model-path models/xgboost_toi.json`

### T2-2: Expert Vetting and Methodology Review — N/A

**Status: Out of scope (DECISION-013).** This is a citizen science project operating independently.
Conservative scoring guardrails enforced in code serve as the substitute:
- Never output "confirmed planet"
- Always expose false-positive evidence
- Suppress `tfop_ready` when key diagnostics are missing
- No external submission without explicit human approval

### T2-3: Peer Review Before Publishing — N/A

**Status: Out of scope (DECISION-013).** This is a citizen science project operating independently.
The scientific guardrails in `docs/SCORING_MODEL.md §15` and `src/exo_toolkit/pathway.py` are the
conservative substitute for formal peer review. All outputs are labeled "candidate signal" or
"follow-up target" — never "confirmed planet".

---

## What Is Complete

Full module inventory: `docs/PROJECT_STATUS.md §What Is Complete`

| Area | Status |
|---|---|
| Core pipeline: Fetch → Clean → Search → Vet → Score → Classify | ✅ |
| Bayesian log-score model (6 hypotheses, 35+ feature functions) | ✅ |
| XGBoost Tier 1 scorer + Kepler training pipeline | ✅ |
| Stacking Tier 3 scorer (conservative fallback) | ✅ |
| CNN Tier 2 benchmark (training loop, promoted checkpoint, calibration) | ✅ promoted as `benchmark_cnn_v1` |
| CLI: `exo <TIC-ID>` + all `background-*` subcommands | ✅ |
| Background automation (SQLite, priority, reports, approval gate) | ✅ |
| Calibration module (Platt scaling, isotonic PAVA, Brier metrics) | ✅ |
| Canonical sample-level regression suite (real + isolated injected controls) | ✅ |
| Bounded short-period real-background production sensitivity v1 | ✅ 23/36 recovered; zero failures |
| Expanded Q1-Q4 production sensitivity v2 | ✅ 8/16 recovered; zero failures |
| Empirical full-ensemble candidate context | ✅ 588-row K2 reference; no invented threshold |
| 90 production-critical Skills/ | ✅ |
| 2,726 default tests, ruff clean, mypy clean | ✅ |
| All scientific guardrails enforced in code | ✅ |

---

## Pre-Deployment Compliance Checklist

Run these before any live deployment or public announcement:

- [x] `.venv/bin/python Skills/run_quality_gates.py` — six test shards × six xdist workers plus concurrent Ruff/mypy all pass (2026-07-14: 2,743 passed; 2 `integration_live` tests excluded; 34.3s wall time on 0.2.59 with optional representation dependencies installed)
- [x] `exo background-run-once --dry-run` — no config errors (2026-07-10: installed entry point exercised successfully; dry run wrote no ledger/outcome data)
- [x] `.venv/bin/python Skills/tier2_progress_reporter.py` — 2026-07-11 reports READY from 15,649 committed-evidence examples/snippets, promoted checkpoint, calibration, and registry entry
- [x] Verify `configs/background_search_v0.json` fingerprint matches expected value (2026-07-10: `exo sqlite-integrity` returned `ok: true` and `missing_config_fingerprint_count: 0`)
- [x] Verify `models/xgboost_koi.json` and `models/xgboost_koi.xgb.json` exist for XGBoost scorer (stale `xgboost_koi_meta.json` name corrected 2026-07-10 — the actual companion-file convention is `.xgb.json`, see `src/exo_toolkit/ml/xgboost_scorer.py`; both files confirmed present 2026-07-10)
- [x] Formal production-ensemble confirmed control — pi Mensae c / TIC 261136679 recovered 6.2676207 d against the catalog 6.2679 d period (0.005% relative error) with ensemble FPP 0.4405 on 2026-07-11; PASS. The separate Bayesian-only TOI-700 v1 control remains a documented FAIL and regression target.
- [x] Formal production-ensemble false-positive control — TOI-146.01 / TIC 355636844 produced no signal above the production detection threshold on 2026-07-11; conservative rejection PASS.
- [x] Canonical offline regression suite — four sample-level controls PASS against `canonical_regression_eval_v1`; real-only and synthetic-inclusive frozen-evaluation roles remain separate (2026-07-12).

**2026-07-10 CLI routing bug found and fixed while running this checklist**: `exo background-run-once --dry-run`, `exo run-summary`, and `exo sqlite-integrity` — the exact invocations this checklist and `AGENTS.md`/`CLAUDE.md` document — did not work as documented. Root cause: `pyproject.toml`'s `[project.scripts]` registered `exo` to `exo_toolkit.cli:app`, a Typer app that only implements the `exo <TARGET-ID>` transit-scan command; the 17 background-automation subcommands are implemented separately by `exo_toolkit.cli`'s argparse `main()`/`build_parser()`, which was never wired to the installed console script at all (only reachable via `python -m exo_toolkit.cli <subcommand>`, the form `docs/SCHEDULER.md` correctly documents for cron/systemd). Invoking `exo background-run-once` therefore silently misparsed `"background-run-once"` as a scan TARGET_ID and failed with a Typer usage error, not a config error. Fixed: `src/exo_toolkit/cli.py` gains `cli_entry()`, a small dispatcher that routes to `main()` when the first argument names a background subcommand (`_background_command_handlers()` is now the single source of truth for that name list, shared with `build_parser()`) and falls through to the Typer `app` otherwise; `pyproject.toml`'s `exo` script now points to `exo_toolkit.cli:cli_entry`. `Skills/mcp_bootstrap_server.py`'s `_exo_command()` helper had the same bug from the other direction (it preferred a bare `exo` found on PATH over the module-invocation form for these same three subcommands) and is fixed the same way. 5 new regression tests (`TestCliEntry` in `tests/test_cli.py`, `test_background_subcommands_use_module_invocation_not_bare_exo` in `tests/test_mcp_bootstrap_server.py`); full suite re-run clean (2,611 passed). The editable install was resynchronized in the repository `.venv`, and all three installed entry-point commands now run successfully; `sqlite-integrity` reports `ok: true`.

---

## Scientific Guardrails (Non-Negotiable)

These are enforced in code and must never be bypassed:

1. Never output "confirmed planet" — only "candidate signal" or "follow-up target"
2. Always expose false-positive evidence alongside positive evidence
3. Suppress `tfop_ready` pathway when key diagnostics are missing (conservative gate)
4. No external submission or discovery contact without explicit human approval
5. Background automation draft reports require human approval before any external action
6. Conservative priors by default; mission-specific prior profiles are opt-in
7. `provenance_score` gates `tfop_ready` — 2-min SPOC with ≥2 sectors required

---

## Outside Blockers (Require Human Action — Cannot Be Automated)

| Blocker | What Is Needed | Who |
|---|---|---|
| Dataset/model source contract | Verify source URLs/schemas, storage estimates, manifests, and leakage controls from `docs/exoplanet_exomoon_dataset_handoff.md` before asking the human to run bulk downloads or training | Agent |
| Source-access smoke test | **Complete** — `Skills/verify_dataset_sources.py` passed end-to-end on 2026-07-02 with TAP schemas/rows, ExoFOP CSV, and Lightkurve Kepler/TESS searches verified | Agent |
| Storage/runtime/source snapshot plan | **Complete** — live sample metadata estimates are committed under `metadata/`; combined Kepler-long-cadence plus TESS estimate is 92,093,823,360 bytes under the 100 GiB cap | Agent |
| Leakage-safe training manifest and cleanup path | **Complete** — `metadata/t1_1_kepler_training_manifest.jsonl` and `metadata/t1_1_kepler_manifest_summary.json` are committed with 0 leakage errors | Agent |
| Bounded Kepler-first processing batch | **Complete and exhausted live** — committed run-report ledgers sum to all 6,515 target groups processed; the master corpus and promoted checkpoint were built from the validated outputs | Agent + human live runs complete |
| CNN production training run | Build/train/evaluate only after the source contract, manifests, and local artifact ledger are updated; use the local M4 Max GPU path by default | Agent + human approval for long local runs |
| CNN promotion readiness package | **Complete** — temperature-calibration promotion tooling, model card, reproducibility manifest, data-role registry, benchmark designation, storage/retention ledger updates, and exact selected artifact scope are GitHub-visible | Agent |
| CNN production promotion | **Complete in promotion PR** — human approved checkpoint SHA `f29e6891c255289fa1e2eddad1fb6ca131c063cf11c24b8113e0e29d049441c5`; selected artifacts are copied to `models/cnn/benchmark_cnn_v1/` and registered in `models/registry.json` | Agent + human approval |
| Stacking weight calibration | **Complete** — calibrated on 588 held-out K2 examples (XGBoost=0.95/CNN=0.00/Bayesian=0.05, AUC 0.9576) and wired into `cli.py` | Agent + human live snippet fetch |

---

## Planning Compliance Note

Any plan proposed in a session must:

1. Identify the highest-impact unresolved production blocker, failing readiness check, roadmap item, defect, or validation need (Tier 1 and Tier 2 are priority signals, not an authorization whitelist)
2. Show how each proposed step materially closes, unblocks, or improves a concrete production outcome
3. Include outside blockers as explicit named steps with responsible party
4. Never propose log modules, schemas, or scaffolding without a concrete production need
5. Never repeat work listed under "What Is Complete" above
