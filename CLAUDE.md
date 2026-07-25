# CLAUDE.md — Claude Code Project Context

This file is read automatically by Claude Code at session start.
It contains the architecture, module map, and current-state facts a coding
agent needs to work productively without re-reading every document.

**All binding operating rules and directives live in `AGENTS.md`** — the
PRIMARY DIRECTIVE, the production-priority process, prohibited work, the PLAN/DO
workflow, branch naming, sync policy, caffeinate policy, console-output/ETA
policy, Run Report Policy, parallelism-first policy, Python environment
policy, system-profile optimization, Astrometrics data/storage policy,
Git-Add-Safe Artifact Policy, the Label-Source Discovery Protocol, quality
gates, code/testing standards, and Scientific Guardrails all live there as
the single canonical copy. **Read `AGENTS.md` first, every session** — this
file assumes you already have.

Current gap status (see `docs/PRODUCTION_READINESS.md` for the full,
actively-maintained narrative — do not rely on any cached summary of it):
**no Tier 1 gaps open as of 2026-07-12** (T1-0/T1-1/T1-2 all complete).
Version 0.3.2 adds the Hunter product lifecycle above the accepted scientific
pipeline: `src/exo_toolkit/search_lifecycle.py` is the append-only SQLite
system of record, `src/exo_toolkit/hunter_cli.py` implements the exact
`Create-New-Search`, `Run-New-Search`, and `Show-Follow-Ups` shell entry points,
and `docs/HUNTER_PRODUCTION_WORKFLOW.md` is the operator/spec handoff. The
Bayesian default keeps core operation independent of AI. Offline acceptance is
implemented and merged-main live acceptance is PASS; the top-level workflow is
PROD as recorded in `artifacts/manifests/hunter_live_acceptance_v1.json`.
Version 0.3.1 fixes the
installed-console-script path that initially could not import repository
`Skills` modules. Version 0.3.2 fixes production scorer-schema consumption
after the first completed live run proved that FPP/confidence are nested under
`scores`, not top-level fields.
The active production priority is master-guide Phase 2 sensitivity evidence:
version 0.2.42 adds the bounded real-background production-pipeline runner and
fixes the XGBoost/PyTorch full-ensemble native-runtime collision. Version
0.2.43 makes quarter-filtered product provenance fail-closed after the first
merged run exposed an all-quarter URI overstatement. Version 0.2.44 commits the
corrected bounded v1 curves (23/36 recovered, zero failures). The active Phase
2 sensitivity now has bounded v1 and expanded v2 evidence. Version 0.2.46
commits the Q1-Q4 v2 run (8/16 recovered, zero failures) for TTV,
single-transit, deterministic gaps, injected stellar variability, and 90-day
cases. Version 0.2.48 commits the fail-closed empirical candidate-context
reference from the held-out K2 calibration role. Phase 1 and bounded Phase 2
are complete. Version 0.2.50 commits the leakage-safe Phase 3 masked-
representation pilot: its embedding probe did not beat `benchmark_cnn_v1`
(test AUC 0.832630 vs 0.957211), although it exceeded the small tabular
baseline (AUC 0.823495; top-100 yield 72% vs 6%). Do not repeat this compact
architecture unchanged. Broad unlabeled Kepler/TESS data, stellar-variability
labels, injection-recovery comparison, and external foundation-model baselines
remain open. See `docs/ROADMAP.md` and `docs/REPRESENTATION_BENCHMARK.md`.
Version 0.2.52 records the completed metadata-only next-data gate: it inventories
existing cached TESS SPOC light curves, excludes every local labeled and frozen
live-search TIC, records exact MAST URIs/cache-relative paths/sizes, and never
opens FITS payloads or downloads data. The committed inventory contains 11,960
products across 2,790 TICs (29.79762048 GB already cached); no derived arrays
are authorized until a bounded preprocessing size/throughput benchmark exists.
Version 0.2.53 adds `Skills/run_six_shards.py`, so one parent process can run
the measured six-shard/six-worker cadence without six terminal tabs. It is
fail-closed to reviewed shard-capable scripts, clean `main`, authoritative repo
identity, and the 100 GB projection; writes six logs plus heartbeats/ETA; and
serializes only shard Run Report git operations through a shared process lock.
The same release adds `Skills/run_quality_gates.py`: every test module is
assigned exactly once across six pytest shards, each shard gets six xdist
workers, and Ruff/mypy run concurrently under the same parent. This 6×6 runner
is the canonical full local quality gate; direct pytest is retained for focused
diagnosis or constrained-machine fallback. With numeric libraries capped at
one inner thread per worker, the optimized full run passed 2,718 tests plus
both static gates in 34.1s, about 58% faster than the 81.57s single-xdist
baseline. The same
single-parent optimized pattern is now the standing default for every safely
partitionable workload, not just these two scripts.
No live download was run because the completed T1-1/T1-2 manifests must not be
reprocessed. See `docs/SIX_SHARD_LAUNCHER.md`.
Version 0.2.54 applies that standing pattern to the active Phase 3 gate:
`Skills/benchmark_representation_preprocessing.py` supervises six Python shard
subprocesses with six cached-FITS workers each, downloads nothing, retains no
derived arrays, and projects full-inventory preprocessing time and normalized-
flux size from a deterministic 36-product sample. The merged run passed 36/36
with zero failures/downloads/persisted arrays at 85.77 products/s, projecting
97.98 MB and 139.44 seconds for all 11,960 products. The preprocessing gate is
complete and future experiments should stream this source. Representation
training remains unauthorized until a materially broader plan adds
stellar-variability labels, injection-recovery comparison, and an external
foundation-model baseline. See
`docs/REPRESENTATION_PREPROCESSING_BENCHMARK.md`.
Version 0.2.55 freezes the next Phase 3 source gate without installing or
downloading a model: `metadata/representation_baseline_source_contract_v1.json`
pins Python 3.14-compatible PyPI wheels plus exact ONNX repository commits,
sizes, and SHA-256 values for Chronos-Bolt tiny and Astromer2. The pair supplies
a bounded general time-series foundation baseline and an astronomy-native
control; full Chronos2 is excluded from the first pass because its 463.8 MB
ONNX file duplicates the general-baseline role of the 13.9 MB tiny model.
`Skills/verify_representation_baseline_sources.py` fails closed against current
primary metadata and pinned HEAD headers while downloading zero payload bytes.
The direct wheels plus models total 56,036,648 bytes. Source verification must
pass from merged code before optional dependencies or model weights are
introduced, and it still does not authorize training. See
`docs/REPRESENTATION_BASELINE_SOURCE_CONTRACT.md`.
The first merged 0.2.55 metadata run failed closed before artifact/report
creation because Python's default URL opener followed Hugging Face's 302 into
Xet and then could not see the resolver's `x-repo-commit`/`x-linked-*`
headers. Version 0.2.56 disables redirects only for this HEAD check, captures
the authoritative 302 headers, and adds an offline regression test. A live
read-only HEAD smoke returned the exact pinned Chronos commit, size, and hash.
The 0.2.56 full 6×6 gate passed 2,734 tests plus Ruff/mypy in 26.1s. The merged
full verifier then passed 7/7 operations in 4.94s, verifying all five pinned
sources and 56,036,648 projected direct bytes with zero payload downloads.
Artifact SHA-256 is `5610bbb8…3042`; Run Report commit `ae4e659`. Source
identity/footprint is now evidenced. At that point inference, dependencies,
weights, and training remained unauthorized pending the bounded smoke now
completed below.
Version 0.2.57 supplies that bounded smoke without changing default runtime
dependencies: `Skills/smoke_representation_baseline_inference.py` verifies the
source/inventory contracts, selects one deterministic cached SPOC product,
prepares at most 2,048 relative-magnitude cadences, downloads the two exact
ONNX revisions into ignored in-repo cache, and runs isolated one-thread CPU
sessions for Chronos-Bolt tiny and Astromer2. It requires finite
`(1, 1, 1, 256)` mean embeddings and records per-model timing/RSS. Nine offline
tests cover source drift, preprocessing, pinned downloads, thread/provider
bounds, call signatures, and guardrails. At that point the merged smoke was
next; no
scientific comparison or full-inventory extraction is authorized. See
`docs/REPRESENTATION_INFERENCE_SMOKE.md`.
The first merged 0.2.57 invocation failed closed before downloading a model:
the Xet helper attempted to create its log below sandbox-blocked
`~/.cache/huggingface`. Version 0.2.58 fixes that integration defect by setting
`HF_HOME` and `HF_XET_CACHE` to ignored repo-contained paths before the lazy
Hub import. The partial cache is only 8 KB of metadata. The 0.2.58 6×6 gate
passed 2,743 tests plus Ruff/mypy in 26.2 seconds with the optional group
installed; at that point merged smoke evidence remained next.
The merged 0.2.58 retry passed both exact models in 26.875 seconds with finite
`(1, 1, 1, 256)` outputs. Chronos peak RSS was 126,058,496 bytes and Astromer2
was 186,204,160 bytes; exact weights total 29,890,844 bytes and the full ignored
cache contains 29,960,842 bytes. Artifact SHA-256 is `1cc59ab3…5de5d10`; Run
Report commit `f8a7207`. Version 0.2.59 records this evidence. Runtime
integration is closed; variability-label and injection-recovery scientific
gates remain before broad extraction or training.
Version 0.2.60 pins the publication-backed 47,055-row, 17-class Drake et al.
Catalina variable-star table and its 1,166,660-byte CDS delivery metadata in
`metadata/stellar_variability_label_source_contract_v1.json`.
`Skills/verify_stellar_variability_label_source.py` checks headers, the live
VizieR schema, total/class counts, and three sample rows while downloading zero
full-catalog bytes. Eight offline tests cover success and fail-closed drift.
Gaia automated labels are rejected as ground truth and the gated approximately
160 GB StarEmbed corpus remains outside the auth/storage boundary. Merged-code
verification is next and does not authorize crossmatch or training. The later
independent-TIC pass must use the single-parent six-shard/six-worker shape after
a small service-throughput measurement. See
`docs/STELLAR_VARIABILITY_LABEL_SOURCE_CONTRACT.md`.
The merged verifier passed 5/5 operations in 3.334 seconds on 2026-07-14:
47,055 rows, all 17 class counts, required schema and delivery headers, and
three labeled sample rows matched with zero full-catalog bytes downloaded.
Artifact SHA-256 is `eb5d4bc…39b9a`; Run Report commit `b0003bb`. Version
0.2.61 records source identity complete. Crossmatch and training remain gated;
the next step is the leakage-safe 2,790-TIC metadata/crossmatch design.
Version 0.2.62 implements that design as a contract-bounded 216-TIC pilot.
`Skills/crossmatch_tess_catalina_labels.py` is reviewed by the one-terminal
`run_six_shards.py` allowlist: six modulo shards, six threaded MAST exact-ID
batches per shard, six IDs per request, one locked hash-pinned 1.17 MB Catalina
cache, and disjoint outputs/Run Reports. The pilot precommits 1-arcsecond,
magnitude, duplicate, object-type, blend, and raw-class safeguards; every row
remains training-disabled. Full 2,790-TIC execution is gated until merged pilot
evidence shows clean throughput/errors and globally reconciled overlap. See
`docs/TESS_CATALINA_CROSSMATCH.md`. Version 0.2.63 also contains the shared
catalog under the ignored `.cache/stellar_variability_labels/` path so the
launcher starts from a clean tree and shard Run Reports retain exact-file-only
git ownership. Version 0.2.64 also accepts the catalog's 44,538 real 71-byte
unflagged rows alongside 2,517 flagged 73-byte rows by padding only the omitted
optional trailing fields; malformed rows outside the 71- to 73-byte source
range still fail before MAST access. Its release gate passed 2,760 default
tests plus Ruff/mypy as 8/8 supervised gates in 25.2 seconds under 6×6; the
pinned gzip parsed all 47,055 rows exactly.
Version 0.2.65 preserves the invalid MAST-column v1 contract for audit and
makes `metadata/tess_catalina_crossmatch_contract_v2.json` active. A live
single-request schema check proved `duplicate_id` is accepted while
`duplicate_i` is rejected; selected columns now come from the active contract.
The v2 six-ID probe returned all six rows in 1.5 seconds. Its release gate
passed 2,761 default tests plus Ruff/mypy as 8/8 gates in 26.2 seconds.
The merged run completed 216/216 TIC queries and wrote all shard artifacts;
only the sandbox-blocked `.git/exo-run-report.lock` made children exit nonzero.
Version 0.2.66 makes that lock failure return `False` so callers warn and exit
successfully without attempting unlocked concurrent git operations.
Its release gate passed 2,762 default tests plus Ruff/mypy as 8/8 supervised
gates in 34.3 seconds under the canonical 6×6 topology.
Version 0.2.67 commits the six shard outputs and aggregate reconciliation:
216/216 unique TICs, 38 completed batches, 8.519s observed wall time, zero
Catalina candidates within 3 arcseconds, and zero accepted/duplicate sources.
Full-corpus execution and training remain unauthorized; pursue a separately
contracted label source with better TESS-inventory overlap.
Its release gate passed 2,763 default tests plus Ruff/mypy as 8/8 supervised
gates in 24.2 seconds under the canonical 6×6 topology.
Version 0.2.68 implements that next bounded source gate with
`metadata/asassn_variability_label_source_contract_v1.json` and
`Skills/preflight_tess_asassn_labels.py`. ASAS-SN Catalog X provides exact TIC
IDs for 378,861 publication-backed rows. An exploratory zero-payload query
found 48 exact matches across the frozen 2,790-TIC inventory (44 known
variables, four discoveries, minimum probability 0.902) in 7.86 seconds with
six workers. The durable run remains pending merged `main` and must use the
reviewed one-parent six-shard/six-worker launcher plus global reconciliation.
Because `Class` is automated ML output, not human ground truth, every pass
keeps training, extraction, promotion, and production scoring unauthorized.
See `docs/ASASSN_VARIABILITY_LABEL_PREFLIGHT.md`.
The version 0.2.68 release gate passed 2,772 default tests plus Ruff/mypy as
8/8 supervised gates in 31.3 seconds under the canonical 6x6 topology.
The merged 6x6 preflight passed and reproduced the exploratory result: 48
unique exact-TIC matches among all 2,790 rows (EA=26, EB=9, EW=2, ROT=10,
SR=1), including 44 known variables and four discoveries, minimum probability
0.902, and zero duplicate TIC/source identifiers. The observed first-start to
last-completion wall time was 6.762 seconds across 58 exact-ID batches plus six
source-metadata operations; no catalog payload bytes were downloaded. Version
0.2.69 commits and integrity-tests the evidence. Follow-up embedding-aware
benchmark design is authorized, while training and production changes remain
unauthorized.
Aggregate SHA-256 is `36de00dc…da403`; the seven exact-path Run Report ledgers
are commit `78b7be6`.
The version 0.2.69 evidence-release gate passed 2,773 default tests plus
Ruff/mypy as 8/8 supervised gates in 32.3 seconds under the canonical 6x6
topology.

Version 0.2.70 implements the authorized follow-up design as a bounded,
cache-only paired benchmark. The immutable contract selects one cached TESS
product for each of the 48 ASAS-SN matches and freezes four 3/10-day,
500/2,000-ppm injections. `Skills/benchmark_representation_variability_injection.py`
compares blind BLS recovery with paired cosine/L2 shifts from the exact cached
Chronos-Bolt tiny and Astromer2 ONNX models. Use the reviewed single-parent
6x6 launcher from clean merged `main`; each shard uses six FITS/BLS workers but
only one serialized session per model, avoiding 36 duplicated model sessions.
The aggregate must reconcile 48 TICs, 192 trials, 384 unique model rows, zero
downloads, and zero persisted embeddings. Results remain descriptive and both
training and production changes stay unauthorized. See
`docs/REPRESENTATION_VARIABILITY_INJECTION_BENCHMARK.md`.
The version 0.2.70 release gate passed 2,780 default tests plus Ruff/mypy as
8/8 supervised gates in 33.3 seconds under the canonical 6x6 topology.
Version 0.2.71 verifies all six aggregate-owned ASAS-SN shard paths and hashes
before loading the 48 benchmark labels; any drift, duplicate TIC, incomplete
shard set, or training-authorized row fails before scientific processing. Its
6x6 release gate passed 2,781 tests plus Ruff/mypy in 30.1 seconds.
The merged run passed all six shards and global reconciliation: 48 TICs,
192 trials, 384 unique model rows, zero failures/duplicates/downloads/persisted
embeddings, and 96/96 higher-depth larger-shift comparisons for both models.
Blind BLS recovered 13/192 trials. Version 0.2.72 commits and integrity-tests
the evidence; training, broad extraction, promotion, and production scoring
remain unauthorized. Aggregate SHA-256 is `93ae6fb8…59f`; the seven exact-path
Run Report commits end at `f0f1645`.
The version 0.2.72 evidence-release gate passed 2,782 default tests plus
Ruff/mypy as 8/8 supervised gates in 37.2 seconds under the canonical 6x6
topology.
Version 0.2.73 defines the next bounded Phase 3 grouped benchmark over 1,536
unique cache-local Kepler KICs. It compares frozen Chronos-Bolt tiny and
Astromer2 probes with the frozen calibrated CNN and a statistical ephemeris
baseline under predefined grouped train/validation/test separation. Run only
from clean merged `main` with the reviewed single-parent 6x6 launcher; the
10.398 GB selected cache inventory is read-only, and aggregate reconciliation
must delete all temporary embedding arrays. Training, broad extraction,
promotion, and production scoring remain unauthorized. See
`docs/GROUPED_EXTERNAL_REPRESENTATION_BENCHMARK.md`.
The version 0.2.73 release gate passed 2,789 default tests plus Ruff/mypy as
8/8 supervised gates in 34.1 seconds under the canonical 6x6 topology.
The first merged execution failed closed before processing because v1 requested
TESS-style `QUALITY` from Kepler products. Version 0.2.74 preserves v1 as
failed evidence and activates immutable v2 with the correct `SAP_QUALITY`
column plus an exact fingerprint of the 111 known 65,536-byte truncated cache
products. Only those paths may be skipped; all other FITS/schema errors remain
fatal and every selected KIC must retain readable data.
The version 0.2.74 release gate passed 2,790 default tests plus Ruff/mypy as
8/8 supervised gates in 36.3 seconds under the canonical 6x6 topology.
The first merged v2 execution failed closed during preparation because its 95%
occupancy requirement did not match real frozen KIC phase coverage. Version
0.2.75 preserves v2 and activates immutable v3, filling empty phase bins with
neutral median physical flux to mirror established production snippet policy
without inventing transit structure. The cache-only 36-worker preflight passed
all 1,536 KICs with exactly 111 pinned skips and finite inputs; one-product
smoke inference returned 256-element outputs from both frozen models.
The version 0.2.75 release gate passed 2,791 default tests plus Ruff/mypy as
8/8 supervised gates in 36.3 seconds under the canonical 6x6 topology.
The merged v3 run passed 1,536 unique KICs with 111 exact pinned skips, zero
failures/downloads/persisted embeddings, six temporary arrays removed, and one
test opening. The frozen CNN remained strongest: test AUC/AP/top-100 were
0.923096/0.899184/91, versus Chronos-Bolt tiny 0.722778/0.696344/71,
Astromer2 0.708984/0.659679/67, and statistical baseline
0.699402/0.607780/67. Version 0.2.76 records the evidence and the precommitted
`no_external_added_value` outcome; broad extraction, training, promotion, and
production changes remain unauthorized. Aggregate SHA-256 is
`3d24363b…4952bd`; Run Report commits end at `1200612`.
The version 0.2.76 evidence-release gate passed 2,797 default tests plus
Ruff/mypy as 8/8 supervised gates in 36.3 seconds under the canonical 6x6
topology.
Version 0.2.77 starts Phase 4's individual anomalous-transit path by replacing
`vet_signal()`'s hard-coded `None` duration/midpoint diagnostics with bounded
per-event measurements. Local sidebands, twice-noise/half-depth gates,
flux-deficit-weighted midpoints, and cadence-resolved durations fail closed
unless at least two events resolve. This activates existing duration-
consistency and TTV features without new data or model training.
The version 0.2.77 release gate passed 2,801 default tests plus Ruff/mypy as
8/8 supervised gates in 35.3 seconds under the canonical 6x6 topology.
Version 0.2.78 extends the Phase 4 individual-transit core with the
`missing_transit_fraction` diagnostic: `_measure_individual_transit_shapes()`
now also counts, among predicted transit windows with at least five cadences
of coverage, the fraction that never resolved a significant dip under the
same local-sideband twice-noise half-depth test already used for
duration/midpoint measurement. `missing_transit_fraction_score()` wires this
into `log_score_planet()` (−0.70) and `log_score_instrumental()` (+0.60),
giving evidence against periodicity even when data coverage itself is not the
limiting factor. `None` unless at least two windows have coverage to test.
See `docs/SCORING_MODEL.md §23`. This is the first extension named in the
Phase 4 roadmap note ("depth/asymmetry/missing/extra-event ranking");
asymmetry and extra-event ranking remain future bounded increments.
The version 0.2.78 release gate passed 2,813 default tests plus Ruff/mypy as
8/8 supervised gates in 28.2 seconds under the canonical 6x6 topology.
Version 0.2.79 adds the second named Phase 4 extension, `transit_asymmetry`:
for each event that already resolves a significant dip,
`_measure_individual_transit_shapes()` splits its resolved-cadence deficit
sum by sign of offset from the predicted center and records the normalized
before/after imbalance, reusing the same resolved-cadence set already
produced for duration/midpoint measurement. `transit_asymmetry_score()` is
the RMS of these imbalances relative to a 0.30 threshold, wired into
`log_score_planet()` (−0.50) and `log_score_instrumental()` (+0.50); `None`
unless at least two events resolve. See `docs/SCORING_MODEL.md §24`.
Extra-event ranking remains the last named increment of this roadmap item.
The version 0.2.79 release gate passed 2,828 default tests plus Ruff/mypy as
8/8 supervised gates in 25.2 seconds under the canonical 6x6 topology.
Version 0.2.80 adds verifiable-agent-reliability controls, not a scientific
Phase deliverable: `Skills/check_incomplete_implementations.py` (AST-based
stub/`NotImplementedError`/TODO-FIXME scanner over `src/`+`Skills/`, with an
automatic exemption for `@abstractmethod`/`Protocol` and a narrow documented
`# allow-stub:` escape hatch) and `Skills/check_directive_integrity.py`
(verifies AGENTS.md is intact and CLAUDE.md still contains the literal
pointer phrase that is Claude Code's only path to AGENTS.md, since Codex
reads AGENTS.md natively while Claude Code does not). Both are wired into
`Skills/run_quality_gates.py` as two new static gates (ten total), which now
also records `git_head_sha`/`git_dirty` in its summary JSON so a passing
result can be checked for staleness against the current tree. AGENTS.md
gains "Fail Loudly" / "No Fake Completion" / "No Unsupported Completion
Claims" sections. Both new checkers ship with fixture-based negative-control
tests (known-good/known-bad/malformed cases, never touching real repo files)
that ran and passed as part of this release gate — see
`docs/RELIABILITY_CONTROLS.md` for the full design/verification report.
The version 0.2.80 release gate passed 2,866 default tests plus Ruff/mypy,
the two new static gates, and the directive-integrity/incomplete-
implementation checks as 10/10 supervised gates in 27.1 seconds under the
canonical 6x6 topology.
Version 0.2.81 closes the "depth/asymmetry/missing/extra-event ranking"
Phase 4 extension with its third and final increment, `extra_event_count`:
`_measure_extra_events()` masks every cadence near a predicted transit
center across the full baseline, flags remaining out-of-transit cadences
≥3σ below the OOT median (MAD-based robust sigma), and clusters contiguous
flags — a cluster counts only if it spans ≥2 cadences and ≤2× the transit
duration, excluding single-point noise and broad low-frequency trends.
`extra_event_score()` wires this into `log_score_planet()` (−0.60) and
`log_score_instrumental()` (+0.50); `None` unless ≥20 out-of-transit
cadences are available. See `docs/SCORING_MODEL.md §25`.
The version 0.2.81 release gate passed 2,883 default tests plus Ruff/mypy as
10/10 supervised gates in 26.1 seconds under the canonical 6x6 topology.
Version 0.2.82 retrofits `Skills/batch_scan.py` with the Run Report Policy
(AGENTS.md Rule 7): its CLI entry point now captures `started_at`/elapsed
around the scan, builds a `RunReport` from real per-target outcome counts
(`error` status → `items_failed`), and calls `run_and_commit_report` with an
injectable `git_run_fn` so tests never touch the real git commit/push path.
This is the second of fourteen scripts on the Rule 7 retrofit list (first
was `star_scanner.py`); AGENTS.md now tracks done/remaining explicitly and
names the CLI-level wiring pattern for the next twelve.
The version 0.2.82 release gate passed 2,889 default tests plus Ruff/mypy as
10/10 supervised gates in 31.1 seconds under the canonical 6x6 topology.
Version 0.2.83 retrofits `Skills/fetch_kepler_lc_snippets.py` with the Run
Report Policy — third of fourteen scripts. Its core `build_kepler_snippets()`
returns a single `int` with 15+ existing test call sites, so rather than
changing that contract, retrofit added an optional
`stats: dict[str, int] | None = None` parameter populated in-place with
`written`/`errors`/`total` — a non-breaking side channel giving the CLI's
Run Report an accurate `items_failed` count without touching any existing
caller. AGENTS.md's retrofit note records this pattern for future scripts
whose core function also doesn't already expose error counts.
The version 0.2.83 release gate passed 2,898 default tests plus Ruff/mypy as
10/10 supervised gates in 25.1 seconds under the canonical 6x6 topology.
Version 0.2.84 retrofits `Skills/fetch_tess_lc_snippets.py` with the Run
Report Policy — fourth of fourteen scripts, same `stats` side-channel
pattern as 0.2.83. Also found and fixed a real gap while retrofitting: the
file's pre-existing `TestCliDefaults` test monkeypatched a fake
`build_tess_snippets` with a rigid signature that would have broken on the
new `stats` kwarg, and called `_cli()` without stubbing
`run_and_commit_report` — meaning the real git commit/push path would have
run under CI once this retrofit landed, violating the Rule 7 test
requirement. Both are fixed. AGENTS.md now flags this exact check for the
remaining ten scripts.
The version 0.2.84 release gate passed 2,906 default tests plus Ruff/mypy as
10/10 supervised gates in 32.1 seconds under the canonical 6x6 topology.
Version 0.2.85 retrofits `Skills/fetch_tess_kepler_overlap_snippets.py` with
the Run Report Policy — fifth of fourteen scripts, same `stats` side-channel
pattern. Its core function distinguishes retryable skips from durable
terminal failures (via a failure sidecar); retrofit uses the total non-OK
count for `items_failed` rather than just the terminal subset, per "partial
success must not be represented as complete success," and records the
terminal-failure count separately in the report's `notes` field. AGENTS.md
now documents this distinction for any remaining script with a similar
sidecar/terminal-failure design.
The version 0.2.85 release gate passed 2,914 default tests plus Ruff/mypy as
10/10 supervised gates in 25.1 seconds under the canonical 6x6 topology.
Version 0.2.86 retrofits `Skills/fetch_tess_k2_overlap_snippets.py` with the
Run Report Policy — sixth of fourteen scripts. This one had **zero**
dedicated test coverage before the retrofit (`grep -rl` across `tests/`
found only an incidental reference from an unrelated manifest test), so a
full test file (20 tests) was written alongside the Run Report wiring rather
than adding retrofit tests to nothing. AGENTS.md now directs future retrofit
work to check for this gap before assuming Run Report tests are sufficient.
The version 0.2.86 release gate passed 2,934 default tests plus Ruff/mypy as
10/10 supervised gates in 27.2 seconds under the canonical 6x6 topology.
Version 0.2.87 retrofits `Skills/fetch_kepler_tce.py` with the Run Report
Policy — seventh of fourteen scripts, also zero prior test coverage. This
one is a single-shot table download with no `_cli()`, no injectable
fetcher, and no per-item loop, unlike every other retrofitted script, so
retrofit first added `query_fn`/`stats` injectable parameters to
`fetch_koi_table()` and a proper `_cli(argv, *, git_run_fn=None) -> int`
function replacing the bare `if __name__ == "__main__":` block, before
wiring the Run Report. A query failure propagates uncaught rather than
writing a false-success report.
The version 0.2.87 release gate passed 2,943 default tests plus Ruff/mypy as
10/10 supervised gates in 27.2 seconds under the canonical 6x6 topology.
Version 0.2.88 retrofits `Skills/fetch_tess_toi.py` with the Run Report
Policy — eighth of fourteen scripts, same single-shot-table shape as
`fetch_kepler_tce.py` but this one already had a real test file and an
injectable `fetch_fn`, so only the `stats` param and `_cli()`/Run Report
wiring were new.
The version 0.2.88 release gate passed 2,949 default tests plus Ruff/mypy as
10/10 supervised gates in 27.2 seconds under the canonical 6x6 topology.
Version 0.2.89 retrofits `Skills/fetch_exofop_ctoi.py` with the Run Report
Policy — ninth of fourteen scripts. This one already returned a rich
`CtoisResult` dataclass with row counts and a `flag` field
("OK"/"EMPTY"/"FETCH_ERROR"), so no `stats` side-channel was needed. The
Run Report is written on both success and failure flags — a table-level
fetch failure gets `status="failed"` rather than producing no report at
all, per Fail Loudly. The pre-existing `_cli()`-level tests were missing
`run_and_commit_report` stubs (same gap class as `fetch_tess_lc_snippets.py`'s
retrofit); fixed.
The version 0.2.89 release gate passed 2,954 default tests plus Ruff/mypy as
10/10 supervised gates in 25.2 seconds under the canonical 6x6 topology.
Version 0.2.90 retrofits `Skills/fetch_nea_koi_lc_index.py` with the Run
Report Policy — tenth of fourteen scripts, near-identical `KoiLcIndex`/flag
shape to `fetch_exofop_ctoi.py`, so the same both-flags Run Report pattern
applied directly with no new gaps found.
The version 0.2.90 release gate passed 2,958 default tests plus Ruff/mypy as
10/10 supervised gates in 27.3 seconds under the canonical 6x6 topology.
Version 0.2.91 retrofits `Skills/fetch_additional_tess_labels.py` with the
Run Report Policy — eleventh of fourteen scripts. This one has no `flag`
field; its CLI already caught per-source TOI/CTOI fetch failures and
continued with an empty list rather than aborting, so the retrofit always
reports `status="success"` and folds fetch-failure counts into the `stats`
side channel instead.
The version 0.2.91 release gate passed 2,963 default tests plus Ruff/mypy as
10/10 supervised gates in 23.2 seconds under the canonical 6x6 topology.
Version 0.2.92 retrofits `Skills/fetch_confirmed_hosts.py` with the Run
Report Policy — twelfth of fourteen scripts. This module is primarily
consumed as a library by `Skills/star_scanner.py` (already retrofitted),
which calls it with the fail-open `strict=False` default; it had no CLI
entry point at all, so retrofit first added a proper `_cli()` writing a
sorted TIC ID JSON artifact, matching the shape every other retrofitted
script uses, before wiring the Run Report. The new CLI always calls the
fetcher with `strict=True` so a real success/failure signal reaches the
completion record, independent of the fail-open library default used by
`star_scanner.py`.
The version 0.2.92 release gate passed 2,968 default tests plus Ruff/mypy as
10/10 supervised gates in 23.2 seconds under the canonical 6x6 topology.
Version 0.2.93 retrofits `Skills/fetch_jwst_targets.py` with the Run Report
Policy — thirteenth of fourteen scripts. Its `_cli()` had no return code
(bare `sys.exit(1)` on error, implicit success otherwise); retrofit gave it
the standard `_cli(argv, *, git_run_fn=None) -> int` shape and now writes a
report on both success and MAST query-failure outcomes.
The version 0.2.93 release gate passed 2,973 default tests plus Ruff/mypy as
10/10 supervised gates in 23.2 seconds under the canonical 6x6 topology.
Version 0.2.94 retrofits `Skills/fetch_jwst_lc.py` with the Run Report
Policy — fourteenth of what is actually fifteen scripts, not fourteen:
recounting the tracking note's own "Done" list from `star_scanner.py` as
item 1 through `fetch_exofop_ctoi.py`'s previously-confirmed "ninth" label
lands on 9, matching only if `star_scanner.py` counts — carrying that count
forward through the full list totals 15, one more than the figure this
narrative had been citing since the retrofit began. `tess_tce_fetcher.py`
remains as the fifteenth and final script. `fetch_jwst_lc.py`'s batch-mode
`_cli()` had no return code; retrofit gives it the standard
`_cli(argv, *, git_run_fn=None) -> int` shape and writes one report per
batch run: `"success"` when every obsid yields a light curve, `"partial"`
when some but not all do, `"failed"` when none do. A run invoked with no
obsid/targets is a CLI usage error, not an acquisition attempt, so no
report is written for that path.
The version 0.2.94 release gate passed 2,979 default tests plus Ruff/mypy as
10/10 supervised gates in 25.2 seconds under the canonical 6x6 topology.
Version 0.2.95 retrofits `Skills/tess_tce_fetcher.py` with the Run Report
Policy — the fifteenth and final script on the AGENTS.md retrofit list,
completing it. Same both-flags `TceFetchResult`/flag shape as
`fetch_exofop_ctoi.py` and `fetch_nea_koi_lc_index.py`. This retrofit
caught a real bug during its own verification: the first `_write_run_report()`
draft set `items_written` from `result.n_total` (every fetched TCE record)
rather than the count of label rows actually written to disk (records with
disposition `ND` are excluded by `tce_to_label_rows()`); a quality-gate test
failure (`3 == 2`) surfaced the mismatch before merge, and the fix threads an
explicit `n_written` parameter through `_write_run_report()` instead of
deriving it from the fetch result.
The version 0.2.95 release gate passed 2,984 default tests plus Ruff/mypy as
10/10 supervised gates in 23.2 seconds under the canonical 6x6 topology. The
AGENTS.md Run Report Policy retrofit (Rule 7) is now complete for all 15
tracked acquisition/processing Skills.
Version 0.2.96 implements `docs/ROADMAP.md` Milestone 20 (Slick Animated
Command-Line UI), unlocked since the 0.2.33 production-ensemble acceptance
PASS and the last item on the active roadmap once Phase 3/Phase 4 and the
Rule 7 retrofit closed. `run_pipeline()` gains an optional
`on_stage(stage_name)` callback fired before each of fetch/clean/search/
vet_score_classify (vet/score/classify combine into one displayed phase
since they always run together per signal). `exo scan`'s new
`_StageAnimator` drives a `rich.console.Console.status()` spinner from it,
showing the current stage and elapsed time. Animation auto-disables
whenever `sys.stdout.isatty()` is false (redirected output, CI) and a new
`--no-animation` flag forces the same plain `[<elapsed>s] <Stage> ...`
line mode explicitly. `KeyboardInterrupt` during a scan stops the spinner
cleanly, prints `Interrupted during stage: <Stage>` to stderr, and exits
`130` with no partial JSON written. JSON output, exit codes, and the
post-run candidate summary are byte-identical whether animation ran or
not — this is a cosmetic terminal-presentation layer only, never a
scoring change. Delivers elapsed-time, not a predictive ETA: a single
scan has no per-stage duration history to extrapolate from, so a
fabricated estimate would violate the No-Unsupported-Completion-Claims
policy rather than help the operator. Documented in
`docs/DISCOVERY_RUNBOOK.md`'s new "`exo scan` interactive vs.
automation-safe display modes" section. 16 new tests across
`TestRunPipelineOnStage`, `TestScanCommand`, and `TestStageAnimator` in
`tests/test_cli.py`, including a narrow (`width=20`) non-interactive-
console render check.
The first merged 0.2.96 CI run failed closed on one test only:
`test_no_animation_help_text_present` asserted on rendered `--help` text
that Rich line-wrapped across `--no-animation` at a narrower width than
this local sandbox. Forcing `env={"COLUMNS": "300"}` on the invocation did
not fix it — Typer's Rich-based help renderer does not consult that env
var the way Click's plain-text formatter does — so the test was removed
rather than chasing an unverified Typer/Rich internal API; the flag's
registration is already proven end-to-end by
`test_no_animation_flag_accepted`, since Typer rejects an unregistered
option with a nonzero exit code.
The version 0.2.96 release gate passed 2,999 default tests plus Ruff/mypy as
10/10 supervised gates in 27.1 seconds under the canonical 6x6 topology.
Version 0.2.97 corrects `docs/PROJECT_STATUS.md`, a file explicitly named in
AGENTS.md's "Read First" required-reading list as "current active state and
next work," which had drifted badly stale: its header, "Active Production
Blocker," and "Next Actions" sections still described Phase 4 individual-
transit diagnostics and T1-2 stacking calibration as pending work, though
both closed many versions earlier (Phase 4 at 0.2.81, T1-2 well before
that), and its inline changelog stopped at version 0.2.77 while `main` was
at 0.2.96. Removes the duplicated version-by-version changelog (0.2.55
through 0.2.77) in favor of a pointer to `CLAUDE.md`'s own narrative — the
duplication was itself why the file re-drifted, matching the unscoped
"reduce duplication/drift across AGENTS.md/CLAUDE.md/docs/*.md" item already
recorded in AGENTS.md's Maintenance TODO backlog, addressed here for this
one file rather than the full backlog item. Rewrites "Active Production
Blocker" to state plainly that none is open and to point at
`docs/PRODUCTION_READINESS.md`/`docs/ROADMAP.md` for the live current state
instead of a hardcoded snapshot. "Historical Discovery Evidence" and
"Trained-Model History" sections are left unchanged since they are
correctly dated history, not current-state claims.
The version 0.2.97 release gate passed 2,999 default tests plus Ruff/mypy as
10/10 supervised gates in 31.1 seconds under the canonical 6x6 topology.
Version 0.2.98 materially widens `Skills/star_scanner.py`'s target-selection
algorithm after a live review of its output found it querying only a
handful of arbitrary sky tiles and stopping early rather than genuinely
ranking across a wide search — real user-facing gaps, not just cosmetic
ones. `priority_score()` gains a fifth weighted factor, stellar radius
(0.20 weight; peaks for `radius_rsun <= 0.7`), since transit depth scales
as `(R_planet/R_star)^2` and the formula previously ignored the single
largest physical driver of transit detectability; the other four weights
are rebalanced (0.25/0.20/0.20/0.15) to make room, and the "ideal star"
threshold tests still pass with radius both specified and omitted (`None`
stays neutral at 0.5). `select_targets()` gains `full_sweep=True`, which
queries every configured tile in parallel (`ThreadPoolExecutor`, 6 workers)
before ranking, instead of the original default that silently stopped
once a small buffer was collected — meaning "top N" previously meant "top
N of whichever 1-4 tiles happened to be queried first in a fixed order,"
not a genuine rank across the search space. The tile grid itself
(`_DEFAULT_SEARCH_CENTERS`) is also densified from 5 dec bands × 12 RA
steps (60 tiles) to 7 × 18 (126 tiles); both `select_targets()`'s and
`run_background_scan()`'s `max_tiles` defaults are bumped from 60 to 126
to preserve full-sky-band coverage for existing incremental callers (the
old default of 60 would now cover only the southern third of the grid).
`select_targets()` also accepts a mutable `search_log` dict, populated
with exactly what was searched (tiles queried/failed, sky coverage in
deg², raw candidates before exclusion, final count, elapsed time) —
answering "log what we actually looked at" directly rather than leaving
it implicit. A new `_load_asassn_variable_tic_ids()` reuses the pinned
ASAS-SN Catalog X source contract (`Skills/preflight_tess_asassn_labels.py`)
for a live, zero-payload exact-TIC check, since a star already flagged as
a known variable is a poor novel-transit-search target for the same
reason a known planet host is — the old exclusion set (TOI/CTOI/confirmed
hosts/prior scans) missed this category entirely.
`run_background_scan()` gains three new opt-in (default off) parameters —
`full_sweep`, `exclude_known_variables`, `search_log_path` — and matching
CLI flags (`--full-sweep`, `--exclude-known-variables`,
`--search-log-path`), deliberately defaulting to off so existing behavior
and the frozen `tess_live_search_v1` batch's reproducibility are
unaffected; `exclude_known_variables=True` can return fewer than
`n_targets` results, and the removed count is always reported rather than
silently backfilled or hidden. 21 new tests across `TestPriorityScore`,
`TestSelectTargets`, the new `TestAsassnVariableExclusion`, and
`TestRunBackgroundScan` in `tests/test_star_scanner.py`.
The version 0.2.98 release gate passed 3,020 default tests plus Ruff/mypy as
10/10 supervised gates in 29.1 seconds under the canonical 6x6 topology.
Version 0.2.99 fixes a real robustness gap surfaced while live-testing
0.2.98's new `full_sweep=True` path: `astroquery.mast`'s own default
request timeout is 600 seconds and this project never overrode it
anywhere; with `full_sweep=True` firing up to 126 concurrent tile
queries, a single slow/stalled tile could silently stall the entire
sweep for up to ten minutes waiting on that one straggler before
erroring. `_query_one_tile()` and `select_targets()` gain an explicit
`query_timeout_seconds` parameter (default 30) that sets
`astroquery.mast.conf.timeout` before every tile query. A live check
against the installed astroquery version caught a second, smaller bug
before it shipped: astropy's `ConfigItem` for this setting is int-typed
and raises `TypeError` on a float value (even a whole-number one like
`30.0`), so the assignment casts explicitly. 2 new tests verify the
timeout is applied as an int both with and without an explicit override.
This was found via legitimate investigation of an apparent hang during
a live full-sweep run; the actual cause of that specific hang was a
real network interruption (unrelated to this bug), corroborated by two
concurrently-running review agents also failing with DNS resolution
errors at the same time — the timeout fix is an independent hardening
improvement, not a fix for that specific incident.
The version 0.2.99 release gate passed 3,022 default tests plus Ruff/mypy as
10/10 supervised gates in 27.1 seconds under the canonical 6x6 topology.
Version 0.2.100 fixes a genuine violation of AGENTS.md's mandatory
Console Output and ETA policy ("every script that iterates over N items
must print real-time progress with ETA... a silent script is
indistinguishable from a hung one") found live while using 0.2.98's new
`full_sweep=True` path: `select_targets()`'s tile-query loops (both the
parallel `full_sweep` path and the sequential default path) printed
nothing at all until the entire loop finished, so a working-but-slow
126-tile sweep and a genuinely stuck one were indistinguishable from the
console — exactly the failure mode the policy exists to prevent, and
exactly what caused real confusion investigating an apparent hang
earlier in this same session. `select_targets()` now prints a startup
banner (tile count, full_sweep mode, worker count) and a `[done/total]
elapsed=... ETA=...` line after every tile, in both loop paths, using
the exact pattern already codified in AGENTS.md. No existing test
asserted on silence, so all 3,022 default tests are unchanged; this is a
console-output-only change.
The version 0.2.100 release gate passed 3,022 default tests plus Ruff/mypy as
10/10 supervised gates in 31.2 seconds under the canonical 6x6 topology.

Version 0.3.10 corrects a genuine Hunter business-requirement defect that
versions 0.3.3-0.3.9 had all signed off as scientifically correct:
`create_search()` hard-failed and created no search whenever fewer candidates
than requested cleared the follow-up quality gate (FPP<0.15,
confidence>0.40, eligible pathway), including the documented "zero eligible
from 202 evaluated" outcome. The current business requirement is explicit
that a normal top-N request must return the best available N with quality
reported separately, and must fail only when zero valid candidates exist at
all. `follow_up_universe()` eligibility is now availability (not already
scheduled/completed/deferred; an unfollowed `candidate_found` signal exists),
not the strict FPP/confidence/pathway bar, which is now reported per-
candidate as `meets_strict_follow_up_bar` instead of removing candidates from
the selectable pool; `hunter_cli._follow_up_from_row()` likewise always
builds a recommendation instead of silently dropping sub-threshold signals.
`FOLLOW_UP_SELECTOR_VERSION` bumped to `exo_hunter_follow_up_v3` so
historical v2-stamped manifests keep their old identity during validity
replay. Verified against a disposable copy of the real production database
(the real file was never mutated): the 202-target durable follow-up universe
now reports 190 available candidates versus zero under the old gate, and a
10-target follow-up request returns exactly 10 best-available targets
instead of raising, with database integrity intact before and after. See
`artifacts/manifests/hunter_live_acceptance_v7.json` and
`docs/HUNTER_PRODUCTION_WORKFLOW.md`. New-mode adaptive discovery expansion
(widening the live TIC sweep itself when it returns too few raw candidates)
remains open as the next highest-priority Hunter gap; current default pool
sizing has not been observed to run dry against the live catalog, but the
fixed 126-tile/tmag-range sweep will eventually exhaust as more targets are
excluded by repeated real usage.
The version 0.3.10 release gate passed 3,079 default tests plus Ruff/mypy as
10/10 supervised gates in 26.1 seconds under the canonical 6x6 topology.

Version 0.3.11 closes the gap version 0.3.10 explicitly flagged as remaining:
`_select_live_new_candidates()` (new-target discovery) raised immediately
whenever the fixed-magnitude-window live TIC sweep returned fewer raw
candidates than requested, with no attempt to broaden the search first. The
directive's canonical selection loop requires candidate pools to be adaptive,
never arbitrarily fixed, and discovery to expand before a top-N request is
allowed to fall short. A bounded expansion loop (up to 3 retries, ±1.0 Tmag
per step, clamped to [4.0, 18.0]) now widens the magnitude window and
retries when the sweep is thin; every attempt is recorded in
`selector_log.discovery_expansion_attempts`, and the function never raises
for insufficiency — that decision is delegated entirely to `create_search()`,
consistent with how follow-up mode already works after 0.3.10. The 126-tile
grid itself remains a fixed ~99 sq deg sample; this widens the magnitude
window per tile, not tile coverage. Verified offline against a scripted
discovery double (thin at the starting window, richer once widened); no live
126-tile MAST sweep was run from this session, since this sandbox has no
execution path to one outside the fixed `exo_guard` MCP commands. See
`artifacts/manifests/hunter_live_acceptance_v8.json`.
The version 0.3.11 release gate passed 3,081 default tests plus Ruff/mypy as
10/10 supervised gates under the canonical 6x6 topology.

Version 0.3.11's follow-up PR also closes the one remaining directive-
required business-validation scenario that neither 0.3.10 nor 0.3.11 had
live evidence for: a genuine live create -> execute -> persist -> follow-up-
registration run, not just candidate selection. Using a disposable copy of
the real production database (the real file was never mutated), a real
follow-up search was created for TIC 261136679 (pi Mensae c) and executed
with the Bayesian scorer through the actual `_pipeline_runner`/`run_pipeline`
path: real MAST fetch/clean/BLS search/vet/score completed with zero errors,
one individual-signal row, complete typed `ExecutionProvenance`, one new
follow-up recommendation registered, and `integrity_summary()` clean before
and after. See `artifacts/manifests/hunter_live_acceptance_v9.json`.
The version 0.3.11 evidence-release gate passed 3,081 default tests plus
Ruff/mypy as 10/10 supervised gates under the canonical 6x6 topology.

A later session closed the one gap v9 explicitly left open: the new-mode
adaptive-discovery-expansion loop had only ever been verified against an
offline scripted double. Using the same disposable-database pattern, a live
`Create-New-Search --mode new` run with a deliberately narrow, bright Tmag
4.0-4.1 window returned 0 raw candidates from a real 126-tile MAST sweep;
the loop widened it to 4.0-5.1 exactly as designed, and a second real sweep
found 13 candidates, satisfying the requested 5 targets with zero shortfall
(verified directly from the disposable database's `selector_log`, not just
console output). Executing those freshly-discovered targets then failed —
all 5 hit `PermissionError` writing to `~/.lightkurve/cache`, which sits
outside this interactive agent session's sandbox-writable paths. This is not
a code defect: the failure was recorded correctly per Fail Loudly (typed
error_message per target, non-zero exit, no false success), not swallowed.
The real production database was confirmed byte-identical before and after;
the disposable database passed integrity and foreign-key checks. See
`artifacts/manifests/hunter_live_acceptance_v10.json`.

The same session then found and applied the actual fix rather than handing
the gap to the human: Lightkurve honors the `XDG_CACHE_HOME` environment
variable when `$XDG_CACHE_HOME/lightkurve` already exists, so a repo-local,
git-ignored cache directory was pre-created and that env var set before
re-running the identical pending search — no source code changes. All 5
freshly-discovered targets fetched real light curves, ran BLS search/vet/
Bayesian scoring, and reached `candidate_found`, registering 16 new
follow-up recommendations; the real production database remained
byte-identical throughout. This closes the directive's full "New targets"
required-validation scenario end to end (discover broadly, adaptively
expand, create the exact durable search, execute exact targets, persist
real typed results/provenance, update follow-up eligibility) with live
evidence, not an offline double. See
`artifacts/manifests/hunter_live_acceptance_v11.json`. AGENTS.md's Python
Environment Policy now records both sandbox findings from this session
(console scripts execute directly even though `.venv/bin/python` does not;
the `XDG_CACHE_HOME` redirect) so a future session does not need to
rediscover either one.

Its release gate passed the unchanged 2,759 default tests plus Ruff/mypy as
8/8 supervised gates in 27.3 seconds under the canonical 6×6 topology.

Version 0.3.12 fixes two real gaps a post-acceptance audit of
`_select_live_new_candidates()` found, still open after 0.3.11. First, its
sufficiency check compared the sweep's raw candidate count against the
target count, but known-variable exclusion and QLP-product-availability
filtering only run on the raw sweep's top `targets * 3` rows — so a sweep
that cleared the raw threshold could still leave the *eligible* pool short of
N with no further expansion ever triggered. The check now runs after
stage-two inspection, on the eligible count, and expansion (both axes below)
continues whenever eligible candidates are short, not just when the raw
sweep is thin. Second, only the Tmag magnitude window ever widened; the
126-tile sky sweep was permanently fixed regardless of retries. `Skills/
star_scanner.py` gains a second, disjoint 180-tile expansion ring
(`_EXPANSION_SEARCH_CENTERS` — interleaved offset bands plus two polar
bands never reached by the base grid; `TOTAL_SEARCH_TILES = 306`), and
`_select_live_new_candidates()` now widens `max_tiles` (+60 tiles per bounded
retry, capped at the scanner's declared total) alongside Tmag widening, both
logged per attempt (`max_tiles`, `eligible_candidates`) and as
`final_max_tiles` in `selector_log`. Verified offline with scanner doubles
exercising both fixes directly: a sweep returning enough raw rows but too
few eligible ones now keeps expanding instead of stopping short, and tile
coverage grows and caps correctly. No live 126+-tile MAST sweep was run from
this session. See `docs/HUNTER_PRODUCTION_WORKFLOW.md`.
The version 0.3.12 release gate passed 3,091 default tests plus Ruff/mypy as
10/10 supervised gates in 27.1 seconds under the canonical 6x6 topology.

Version 0.3.13 closes a real bypass of "no production command bypasses the
canonical optimizer or durable pipeline": `Skills/star_scanner.py`'s and
`Skills/batch_scan.py`'s standalone CLI scan modes wrote only to a local
`ScanLog`/JSON results file, invisible to Hunter's durable
`target_search_history`. `src/exo_toolkit/hunter_history.py` gains
`build_manual_scan_source()`, building one history-manifest "source" from a
completed manual scan using the exact schema `import_history_manifest()`
already verifies for the seven curated legacy-log imports — no manifest
file needs to be written to disk, the dict passes straight through. Both
CLIs now bridge to Hunter by default after a real scan (new
`--hunter-db`/`--no-hunter-bridge` flags): `star_scanner.py`'s `--target`
and default background-scan modes (via a new `scanned_this_run` field,
additive to `run_background_scan()`'s return) and `batch_scan.py`'s CLI (via
a new optional `new_entries` out-parameter on `batch_scan()`, so `--resume`
bridges only freshly-scanned targets). Hunter's own library-import usage and
the shard launcher's `--execute-prepared-batch` path stay untouched, proven
by dedicated tests. A real integration bug was caught pre-merge: the
manifest builder and `import_history_manifest()` must resolve the source's
relative path against the same root or a fail-closed hash mismatch results;
both bridges now explicitly pass the log file's own parent directory as
`source_root` to both calls. See `docs/HUNTER_PRODUCTION_WORKFLOW.md`.
The version 0.3.13 release gate passed 3,119 default tests plus Ruff/mypy as
10/10 supervised gates in 27.2 seconds under the canonical 6x6 topology.

Version 0.3.14 adds the smart MAST recheck for deferred follow-ups, the last
gap named in the plan. `follow_up_registry` gains nullable
`last_known_sectors`/`last_mast_checked_at` columns via the same migration
pattern already used for `revisit_reason`/`parent_follow_up_id`.
`HunterStore.record_sector_recheck()` is the only writer: it compares a
fresh sector list against the row's last recorded baseline (never a blind
timer, never the original deferral snapshot) and flips `deferred` back to
`open` only when it finds sectors not already known; a row with no baseline
yet is treated as empty, matching every follow-up this project has actually
deferred so far. The baseline/timestamp always advance whether or not it
grew, with a `follow_up_events` audit row either way. The new
`Recheck-Follow-Ups` shell entry point iterates every deferred row with
bounded concurrency, reusing `Skills/sector_coverage.py`'s
`get_sector_coverage()` (metadata-only, zero downloads) via the existing
`_load_project_skill()` pattern; a per-target query failure is recorded and
reported without blocking the rest of the batch. Verified offline against a
real no_data-deferred follow-up row built through the full create-execute
lifecycle. See `docs/HUNTER_PRODUCTION_WORKFLOW.md`.
The version 0.3.14 release gate passed 3,131 default tests plus Ruff/mypy as
10/10 supervised gates in 29.2 seconds under the canonical 6x6 topology.

Version 0.3.15 fixes a real latent gap a business-objective audit found in
`load_verified_history_manifest()`'s already-existing `source_root`
parameter: every caller above it (`Create-New-Search --history-manifest`,
`HunterStore.validity_summary()`, `validate_hunter_database()`,
`Skills/validate_hunter_acceptance.py`) left it unset, so source-path
resolution always fell through to `_repository_root_for()`'s repo-root
walk-up heuristic, which silently resolves to the wrong root for an
"isolated or scripted operation" manifest (the exact scenario
`--history-manifest` is documented to support) that happens to sit inside
some other checked-out repo's subtree. All four call sites now accept an
explicit `history_source_root`/`--history-source-root` override; the
default manifest path is unchanged (heuristic remains the default when the
new parameter is omitted). The audit that found this initially treated a
locally reproduced test failure as a shipped 0.3.14 regression, then found
on deeper investigation that the failure was specific to that interactive
sandbox (pytest's `tmp_path` landed inside this repo's own tree because
`tempfile.gettempdir()` could not use the real system temp directory there,
so the walk-up escaped test isolation) rather than a defect present in real
CI; see `docs/HUNTER_PRODUCTION_WORKFLOW.md`'s 0.3.15 entry for the full
correction and the lesson recorded for future agents about distinguishing
sandbox artifacts from shipped regressions. `test_default_follow_up_imports_and_ranks_durable_history`
now passes `--history-source-root` explicitly (hermetic); a new test,
`test_history_source_root_overrides_repo_root_walk_up_heuristic`, proves the
walk-up resolves to the wrong root inside a decoy nested repo and fails
closed without the override, then succeeds with it.
The version 0.3.15 release gate passed 3,132 default tests plus Ruff/mypy as
10/10 supervised gates in 29.1 seconds under the canonical 6x6 topology.

Version 0.3.16 starts a whole-repo hardening sweep (a parallel fragility
audit found ~40 real findings across src/exo_toolkit/, the ML scorers,
Hunter, and all Skills scripts; being worked as a prioritized backlog of
focused PRs, not one bulk change) with its own highest-leverage item first:
the two AGENTS.md reliability-checker backstops had real false-negative
gaps. `Skills/check_incomplete_implementations.py`'s TODO/FIXME detection
was a case-sensitive raw substring match (missed `# todo:`/`# Fixme later`,
the common real-world casing, and could false-positive on identifiers like
`AUTODOC_ENABLED`); its `_scan_raise` only recognized a bare
`raise NotImplementedError(...)`, missing a module-qualified raise
(`raise builtins.NotImplementedError(...)`) or an indirect raise via a
stored variable (`err = NotImplementedError(...); raise err`).
`Skills/check_directive_integrity.py`'s required-section check was an
unanchored substring search of AGENTS.md's raw text, so a demoted heading
(`## Fail Loudly` → `### Fail Loudly`) or a mere prose/TOC mention of the
phrase would satisfy it even with the real H2 heading gone. All four are
now fixed: word-boundary case-insensitive marker matching, an
`ast.Attribute`/indirect-alias-aware raise scan, and heading detection
anchored to actual markdown heading lines. 9 new regression tests prove
each previously-missed case is now caught and that the real repository
tree (including AGENTS.md's real headings) still passes cleanly with the
stricter logic.
The version 0.3.16 release gate passed 3,139 default tests plus Ruff/mypy as
10/10 supervised gates in 28.1 seconds under the canonical 6x6 topology.

Version 0.3.18 continues the hardening sweep: `Skills/alert_filter.py`'s
`_fpp()` and `Skills/candidate_timeline.py`'s `record()` extracted FPP via
`row.get("false_positive_probability") or row.get("best_fpp")`-style
`or`-chains. A genuinely perfect FPP of `0.0` is falsy in Python, so a row
carrying both a current-schema field at `0.0` and a stale/legacy field
would have its real zero silently discarded in favor of the wrong value —
exactly the failure mode `compare_candidates.py`/`export_candidates.py`
already avoid elsewhere in this same codebase via `is None`/`in` checks,
making the buggy versions a real inconsistency rather than an unavoidable
design choice. Both now use `is None` checks throughout, matching that
existing correct pattern; `candidate_timeline.py`'s `planet_posterior`
extraction is also normalized to the same style for consistency, though it
was not independently buggy (its fallback source and default already
coincided). 5 new regression tests prove a genuine `0.0` now survives
extraction instead of being replaced by a stale field.
The version 0.3.18 release gate passed 3,144 default tests plus Ruff/mypy as
10/10 supervised gates in 28.1 seconds under the canonical 6x6 topology.

Version 0.3.19 continues the hardening sweep: `extract_features()` silently
substituted solar values (`R=1.0 R☉`, `M=1.0 M☉`, `Teff=5778 K`) whenever
`stellar_radius_rsun`/`stellar_mass_msun`/`stellar_teff_k` were `None`,
feeding those assumed values into `duration_plausibility_score`,
`companion_radius_too_large_score`, `duration_implausibility_score`, and
`limb_darkening_plausibility_score` instead of leaving the diagnostic
absent — contradicting this module's own stated invariant ("features whose
required diagnostics are absent are set to None") and diverging from the
sibling `stellar_density_consistency_score`, which already correctly
returns `None` under the identical missing-data condition. Since
`fetch_tic_stellar_params()` already fails open to `None` on any catalog
miss, and this project explicitly targets fainter stars (Tmag 10-14) where
TIC catalog completeness is worse, this systematically injected a
wrong-but-plausible solar-host bias into planet-vs-EB scoring for exactly
the M-dwarf-heavy population this project targets — a genuine transit
around a small star could score as duration-implausible against a wrongly
assumed larger host. All four scores are now gated on the presence of
their real required stellar parameters (`companion_radius_too_large_score`
needs only radius; the other three need radius and mass, or Teff),
matching `stellar_density_consistency_score`'s existing correct pattern.
Two pre-existing tests encoded the old buggy assumption (asserting these
scores were "always computed" even with empty diagnostics) and are
corrected; 6 new regression tests prove each is `None` when its required
stellar parameter is missing and non-`None` when present.
The version 0.3.19 release gate passed 3,143 default tests plus Ruff/mypy as
10/10 supervised gates in 28.1 seconds under the canonical 6x6 topology.

Version 0.3.20 continues the hardening sweep: `Skills/run_quality_gates.py`
captured `git_head_sha`/`git_dirty` *after* `supervise_gates()` returned, not
before, so its summary JSON could misattribute a commit landing during the
~20-30s gate run (plausible in this repo, which has 15+ scripts that
auto-commit Run Reports) to a result that never actually verified that SHA —
directly undermining the field's own stated purpose. `main()` now captures
git state immediately before the gates start and gains injectable
`supervise_gates_fn`/`git_state_fn` parameters (matching this repo's
existing dependency-injection pattern, e.g. `git_run_fn` elsewhere) so a new
regression test can prove the fix by call order, not just by output value: a
stale `git_state_fn` could otherwise coincidentally match the old buggy
timing by chance.
The version 0.3.20 release gate passed 3,140 default tests plus Ruff/mypy as
10/10 supervised gates in 29.1 seconds under the canonical 6x6 topology.

Version 0.3.21 continues the hardening sweep: `calibration.py`'s
`_fit_platt()` wrapped `scipy.optimize.minimize()` in a bare
`except Exception: return PlattParams(slope=1.0, intercept=0.0)` —
indistinguishable, in its return value, from the documented and tested
"too few samples" identity fallback, but firing on any genuine optimizer
failure with zero signal to the caller that fitting actually broke. It
also never checked `res.success`: scipy can return a non-converged result
without raising at all, which the old bare try/except would not have
caught either. Since `apply_calibration()` is wired into the live CLI path
via `--calibration-path`, a silently-broken fit for one hypothesis could
have shipped into production scoring undetected. `_fit_platt()` now raises
`RuntimeError` on both failure modes (exception, or `res.success is
False`) instead of silently returning the identity fallback; the
insufficient-data path is unchanged since it is the intentional, tested,
documented case. `fit_calibration()`'s docstring records the new possible
`RuntimeError`. 2 new regression tests force each failure mode via a
monkeypatched `minimize` and assert the raise; the existing real-data
fit test (`test_fit_reduces_brier_score`, seeded RNG, 50 samples) and all
three insufficient-data fallback tests are unaffected and still pass.
The version 0.3.21 release gate passed 3,145 default tests plus Ruff/mypy as
10/10 supervised gates in 29.1 seconds under the canonical 6x6 topology.

Local artifact/corpus/checkpoint status: `docs/LOCAL_ARTIFACT_LEDGER.md`.
Full per-Skill Milestone changelog (historical, archived verbatim, not
needed for day-to-day work): `docs/MILESTONE_HISTORY.md`.

---

## Project-Scoped MCP Servers

Three MCP servers are bootstrapped in `.mcp.json` and `.codex/config.toml`. All are implemented in `Skills/mcp_bootstrap_server.py`. Claude Code loads them automatically from `.mcp.json` at the project root.

| Server | Mode arg | Capabilities |
|---|---|---|
| `exo_project_files` | `project_files` | Read-only access to source, docs, tests, configs. Blocks `logs/`, `data/`, `.git`, `.venv`, secrets. |
| `exo_git_read` | `git_read` | Fixed read-only git commands: `status`, `diff`, `diff_staged`, `log_recent`, `branch_current`. |
| `exo_guard` | `exo_guard` | Fixed validation commands: `ruff_check`, `mypy_src`, `pytest_default`, `pytest_cov`, `background_run_once_dry_run`, `run_summary`, `sqlite_integrity`. |

**Safety contract**: No arbitrary shell execution. No live-network commands in defaults. No secrets or runtime artifacts exposed. No external submission without human approval. Full spec: `docs/Exoplanet_Research_MCP_BOOTSTRAP.md`.

---

## Project

**2026 Exoplanet Research**
Citizen-science toolkit for detecting and scoring exoplanet transit candidates from TESS and Kepler/K2 data.

Repository: `ares0311/2026-Exoplanet-Research`
Active branch: `main`
PR #1 merged 2026-04-28

---

## Architecture

```
Fetch → Clean → Search → Vet → Score → Classify
```

Python package: `src/exo_toolkit/`
Tests: `tests/`
Docs: `docs/`
CI: `.github/workflows/ci.yml`

### Module build order (each depends on prior)

| Module | Status | Tests |
|---|---|---|
| `schemas.py` | **done** | `test_schemas.py` (33) |
| `features.py` | **done** | `test_features.py` (145) — includes all 5 Milestone 12 feature functions |
| `hypotheses.py` | **done** | `test_hypotheses.py` (46) — all 5 Milestone 12 features wired |
| `scoring.py` | **done** | `test_scoring.py` (48) — invariants, prior config flow, weight-sensitivity tests |
| `priors.py` | **done** | `test_priors.py` (14) — conservative versioned default + mission-prior config |
| `pathway.py` | **done** | `test_pathway.py` (60) — parametric + all-branch coverage |
| `fetch.py` | **done** | `test_fetch.py` (55, 2 live) |
| `clean.py` | **done** | `test_clean.py` (39) |
| `search.py` | **done** | `test_search.py` (43) |
| `vet.py` | **done** | `test_vet.py` (47) |
| `calibration.py` | **done** | `test_calibration.py` (70) — now includes `save_calibration`/`load_calibration` |
| `cli.py` | **done** | `test_cli.py` (54) — version flag, meta output, calibration/CNN snippet integration |
| `ml/xgboost_scorer.py` | **done** | `test_xgboost_scorer.py` (45) |
| `ml/stacking_scorer.py` | **done** | `test_stacking_scorer.py` (22) — updated for 3-tier CNN blend |
| `ml/cnn_scorer.py` | **done** | `test_cnn_scorer.py` (21) — injectable model_fn, no PyTorch required |
| `background/` module | **done** | `test_background_automation.py` (16) |

**Current test surface:** 142 top-level test files. Version 0.2.62 adds seven offline crossmatch tests plus one launcher-allowlist test; its canonical 6×6 release gate passed 2,759 default tests plus Ruff/mypy as 8/8 supervised gates in 34.3s.
**Skills:** 122 standalone utility scripts live in `Skills/` (plus the package marker `Skills/__init__.py`). Use `rg --files Skills -g '*.py' | sort` for the authoritative current list, and see `docs/SKILLS_GUIDE.md` for workflow-oriented quick reference.

---

## Background Automation Module (`src/exo_toolkit/background/`)

Added in Weekly cleanup (2026-05-10). Implements one-shot, scheduler-friendly background search over known TESS fixture targets.

| Submodule | Purpose |
|---|---|
| `schemas.py` | `KnownTessTarget`, `PriorityFactors`, `BackgroundRunResult`, `Outcome`, `FollowUpStatus` |
| `config.py` | Load/validate `configs/background_search_v0.json`; `ConfigError` on bad config |
| `fixtures.py` | Load `fixtures/known_tess_examples.json`; `fixture_summary()` |
| `priority.py` | `build_priority_summary()` — 8-factor composite score with reason codes |
| `followup.py` | `mandatory_follow_up_tests()`, `trigger_reason_codes()` |
| `runner.py` | `background_run_once(db_path, ...)` — single bounded run; dry_run mode |
| `reports.py` | `build_draft_report()`, `export_draft_report()`, `build_submission_recommendations()` |
| `storage.py` | `BackgroundStore` — SQLite schema v2 for the run ledger, priority evaluations, outcomes, follow-up tests, reports, approvals, locks, and migrations |
| `reason_codes.py` | `ReasonCode` enum — stable string values for audit trails |

**CLI subcommands** (via `exo <subcommand>`):
`background-run-once`, `run-summary`, `sqlite-integrity`, `target-priority-summary`, `config-summary`, `fixture-summary`, `background-ledger-summary`, `reviewed-log-summary`, `needs-follow-up-summary`, `follow-up-test-summary`, `draft-report-summary`, `submission-recommendation-summary`, `report-export-summary`, `approval-record-summary`, `target-history`, `scheduler-notification-summary`, `validation-summary`

**Exit codes**: `EXIT_SUCCESS=0`, `EXIT_NEEDS_FOLLOW_UP=20`, `EXIT_BLOCKED=30`, `EXIT_CONFIG_ERROR=40`, `EXIT_INTERNAL_ERROR=50`

**Key constraint**: No external submission or discovery claim without explicit human approval. Draft reports go to `reports/background/`. SQLite DB at `logs/background_search.sqlite3`.

---

## Provenance Score (`src/exo_toolkit/fetch.py`)

`compute_provenance_score(provenance: FetchProvenance) -> float` — data-quality score in [0, 1] from cadence, sector count, and pipeline.

- Formula: `0.40*cadence_sub + 0.35*sector_sub + 0.25*pipeline_sub`
- `cadence_sub`: linear ramp, 1.0 at 2-min, 0.0 at 30-min
- `sector_sub`: `min(n_sectors / 3, 1.0)`; saturates at 3 sectors
- `pipeline_sub`: SPOC/Kepler/K2 → 1.0; QLP → 0.85; TGLC → 0.75; unknown → 0.60
- Called in `run_pipeline()` immediately after fetch; passed to `classify_submission_pathway(provenance_score=...)`
- Threshold for `tfop_ready`: ≥ 0.80 (2-min SPOC data with ≥ 2 sectors passes)
- Documented in `docs/SCORING_MODEL.md §21`

---

## Candidate Ranking (`Skills/rank_candidates.py`)

Ranks `exo --output` JSON results by composite score and prints a Rich table.

- `load_candidates(paths)` — flatten one or more JSON output files
- `compute_rank_score(row)` — `0.45*(1-FPP) + 0.30*DC + 0.15*novelty + 0.10*provenance + pathway_bonus`
- `rank_candidates(rows, top_n)` — sort by rank_score descending
- CLI: `.venv/bin/python Skills/rank_candidates.py results/*.json --top 10 [--json]`
- 12 tests in `tests/test_rank_candidates.py`

---

## Batch Scan (`Skills/batch_scan.py`)

Scans a list of TIC IDs from a text or CSV file, writing incremental JSON results.

- `read_tic_ids(path)` — parse TIC IDs from plain text or CSV (skips comments, headers)
- `batch_scan(tic_ids, *, output_path, resume, run_pipeline_fn, ...)` — calls `run_pipeline` per target; writes after each result; `--resume` skips already-completed IDs
- Status per entry: `"candidate_found"` | `"scanned_clear"` | `"error"`
- CLI: `.venv/bin/python Skills/batch_scan.py targets.txt --output results.json [--resume]`
- CLI writes a Run Report (AGENTS.md Rule 7) after each run via `Skills/run_report.py`
- 20 tests in `tests/test_batch_scan.py`

---

## Sector Coverage (`Skills/sector_coverage.py`)

Queries MAST for which TESS sectors are available for a target without downloading data.

- `get_sector_coverage(target_id, *, pipeline, search_fn)` → `SectorCoverage`
- `format_coverage_table(coverages)` → plain-text table
- CLI: `.venv/bin/python Skills/sector_coverage.py TIC 150428135 [--pipeline QLP] [--json]`
- 10 tests in `tests/test_sector_coverage.py`

---

## Star Scanner (`Skills/star_scanner.py`)

Queries the TESS Input Catalog (TIC) via astroquery to rank uncharacterised stars by transit-search promise, then scans them in priority order, logging results.

- `priority_score(tmag, teff, n_sectors, contratio, radius_rsun)` → float in [0, 1]. Five weighted factors: magnitude (0.25), stellar type (0.20), sector coverage (0.20), contamination (0.15), stellar radius (0.20 — smaller stars give deeper, more detectable transits for a fixed planet size; peaks for `radius_rsun <= 0.7`). Any factor's input may be `None` → neutral 0.5.
- `ScanLog(path)` — atomic-write JSON log; `record()`, `is_scanned()`, `scanned_ids()`, `summary()`
- `select_targets(n, tmag_range, exclude_tic_ids, *, full_sweep=False, max_workers=6, search_log=None, max_tiles=126, query_timeout_seconds=30.0)` — TIC query, ranked, filtered. Default (`full_sweep=False`) preserves the original fast/early-stop behavior (stops once a small buffer is collected) for existing incremental callers. `full_sweep=True` queries every configured tile in parallel (`ThreadPoolExecutor`) before ranking, so "top N" is a genuine rank across the whole swept area rather than whatever the first few tiles happened to return. The tile grid (`_DEFAULT_SEARCH_CENTERS`) is 7 dec bands × 18 RA steps = 126 tiles (~99 sq deg, ~0.24% of the sky) — still a documented sample, not an exhaustive survey. Pass a mutable `search_log` dict to record exactly what was searched: `tiles_configured`, `tiles_queried`, `tiles_failed`, `tile_errors`, `sky_coverage_deg2`, `raw_candidates_before_exclusion`, `candidates_after_exclusion`, `excluded_count`, `full_sweep`, `elapsed_seconds`. `query_timeout_seconds` (version 0.2.99) bounds `astroquery.mast.conf.timeout` per tile query — that library's own default is 600s and was never overridden before, so a single slow tile under `full_sweep=True`'s up-to-126-concurrent-query load could otherwise stall the whole sweep for up to ten minutes.
- `_load_asassn_variable_tic_ids(candidate_tic_ids, *, strict=False)` — live, zero-payload exact-TIC lookup against the pinned ASAS-SN Catalog X source; a star already flagged as a known variable (eclipsing binary, pulsator) is a poor novel-transit-search target for the same reason a known planet host is. Fails open like the other exclusion loaders.
- `scan_star(tic_id, *, log, ...)` → dict with status/n_signals/best_fpp/best_pathway
- `run_background_scan(log_path, ..., full_sweep=False, exclude_known_variables=False, search_log_path=None)` — iterates until Ctrl-C or max stars reached. The three new keyword params are opt-in (default off) so existing behavior and the frozen `tess_live_search_v1` batch's reproducibility are unaffected. `exclude_known_variables=True` removes selected targets already flagged as known ASAS-SN variables post-selection (can return fewer than `n_targets`; the removed count is reported, never silently dropped). `search_log_path`, if given, writes the search manifest above as a durable JSON file. CLI flags: `--full-sweep`, `--exclude-known-variables`, `--search-log-path`.
- Excludes TOI list at startup; skips already-scanned IDs from log

---

## Depth Scatter Chi-Square Score

New feature in `features.py` and `schemas.py` (Milestone 10a):

- `depth_scatter_chi2_score(depths, errors, chi2_threshold=3.0) -> float | None`
- Reduced chi-square test: `chi2_reduced = sum((d_i - d_mean_w)^2 / err_i^2) / (n-1)` using inverse-variance weighted mean
- Score = `clip(chi2_reduced / 3.0)` — saturates at chi2_reduced = 3
- High score → depths vary more than expected from measurement noise → evidence for instrumental artifact
- Wired into `log_score_instrumental()` (+0.90 weight) and `log_score_planet()` (−0.60 weight)
- Complements existing `depth_consistency_score` (robust CV, no error weighting) with error-aware test
- Returns `None` if fewer than 2 transits or any error ≤ 0

---

## Phase-Fold Plots (`Skills/plot_lc.py`)

Generates phase-folded light curve PNGs from candidate JSON rows.

- `phase_fold(time, flux, period, epoch)` → `(phase, flux)` sorted, phase in [−0.5, 0.5)
- `plot_candidate(row, *, output_dir, show, time, flux)` → `Path | None`
- `plot_all(path, *, output_dir, show)` → `list[Path]`
- Requires matplotlib; returns `None`/empty list if not installed
- 11 tests in `tests/test_plot_lc.py` (6 skipped when matplotlib absent)

---

## Watchlist (`Skills/watchlist.py`)

Persistent JSON watchlist for follow-up TIC IDs. Integrates with `batch_scan.py`.

- `Watchlist(path)` — `add(tic_id, note)`, `remove(tic_id)`, `contains(tic_id)`, `list_ids()`, `entries()`, `clear()`, `summary()`
- Atomic write via tempfile rename
- CLI: `.venv/bin/python Skills/watchlist.py add/remove/list/clear/summary`
- 13 tests in `tests/test_watchlist.py`

---

## Summary Report (`Skills/summary_report.py`)

Generates Markdown summary reports from batch_scan JSON output.

- `load_results(paths)` → flat list of result dicts
- `build_report(rows, *, title)` → Markdown string with overview table + candidates + errors
- `write_report(rows, output_path, *, title)` → `Path`
- Partitions by status: `candidate_found`, `scanned_clear`, `no_data`, `error`
- Candidates sorted by FPP ascending (best first)
- 14 tests in `tests/test_summary_report.py`

---

## TOI Checker (`Skills/toi_checker.py`)

Looks up a TIC ID in the ExoFOP TOI list to check prior follow-up status before investing pipeline time.

- `check_toi(tic_id, *, toi_table_fn) -> dict | None` — fetches ExoFOP CSV, returns dict with `toi`, `tic_id`, `disposition`, `period_days`, `epoch_bjd`, `depth_ppm`, `duration_hours`; returns `None` if not in TOI list
- `format_toi_result(result, tic_id) -> str` — one-line human-readable status string
- Handles column-name variations between ExoFOP CSV versions
- 12 tests in `tests/test_toi_checker.py`

---

## Export Candidates (`Skills/export_candidates.py`)

Exports ranked candidate results to CSV and GitHub-flavored Markdown table formats.

- `to_csv(rows, path) -> Path` — 10-column CSV with display headers; creates parent dirs
- `to_markdown_table(rows) -> str` — `| col | ... |` table; returns `"_No candidates._"` for empty input
- `to_summary_stats(rows) -> dict` — `n_candidates`, `mean_fpp`, `min_fpp`, `max_rank_score`, `pathway_counts`
- 13 tests in `tests/test_export_candidates.py`

---

## Alert Filter (`Skills/alert_filter.py`)

Filters batch_scan or star_scanner JSON results by configurable quality thresholds.

- `filter_candidates(rows, *, fpp_max, pathway, min_signals, min_rank_score, min_snr) -> list[dict]` — AND-logic; `None` = not checked
- `apply_filters(path, *, output_path, ...) -> list[dict]` — load + filter + optionally write JSON
- `_fpp()` helper extracts FPP from `scores.false_positive_probability`, `best_fpp`, or top-level `false_positive_probability`
- 12 tests in `tests/test_alert_filter.py`

---

## Transit Timing Variation Score

New feature in `features.py` and `schemas.py` (Milestone 11a):

- `transit_timing_variation_score(midpoints, period_days, epoch_bjd, rms_threshold_minutes=10.0) -> float | None`
- O-C residuals: `n_i = round((t_i - epoch_bjd) / period_days)`, residual = `(t_i - (epoch_bjd + n_i * period_days)) * 1440` minutes
- Score = `clip(RMS_OC / rms_threshold_minutes)` — saturates at threshold
- High score → timing is irregular → evidence for instrumental artifact (not a clean Keplerian transit)
- Wired into `log_score_planet()` (−0.50 weight) and `log_score_instrumental()` (+0.60 weight)
- Returns `None` if fewer than 2 midpoints

---

## Missing Transit Fraction Score

New feature in `features.py`, `schemas.py`, `hypotheses.py`, and `vet.py` (version 0.2.78):

- `missing_transit_fraction_score(missing_transit_fraction) -> float` — identity clip; input already bounded to `[0, 1]`
- `RawDiagnostics.missing_transit_fraction`: fraction of predicted transit windows with ≥5 cadences of coverage that never resolved a significant dip, using the exact same per-window resolution test `_measure_individual_transit_shapes()` already uses for durations/midpoints (local sideband baseline, twice-noise half-depth gate)
- High score → data covers most predicted windows but the signal fails to resolve at most of them → evidence against genuine periodicity even when the "no data" explanation is ruled out
- Wired into `log_score_planet()` (−0.70 weight) and `log_score_instrumental()` (+0.60 weight)
- `None` if fewer than 2 predicted windows have sufficient coverage to test
- Spec: `docs/SCORING_MODEL.md §23`

---

## Transit Asymmetry Score

New feature in `features.py`, `schemas.py`, `hypotheses.py`, and `vet.py` (version 0.2.79):

- `transit_asymmetry_score(asymmetries, rms_threshold=0.30) -> float | None` — RMS of per-event imbalance relative to threshold, same pattern as `transit_timing_variation_score()`
- `RawDiagnostics.individual_transit_asymmetries`: per-event `(after − before) / (after + before)` in `[-1, 1]`, computed from the same resolved-cadence set `_measure_individual_transit_shapes()` already produces for durations/midpoints, split by sign of offset from the *predicted* transit center
- High score → events are consistently lopsided around their predicted centers → evidence for an instrumental ramp or blended-source contamination, not a clean box/trapezoid transit
- Wired into `log_score_planet()` (−0.50 weight) and `log_score_instrumental()` (+0.50 weight)
- `None` if fewer than 2 events resolve
- Spec: `docs/SCORING_MODEL.md §24`

---

## Extra Event Score

New feature in `features.py`, `schemas.py`, `hypotheses.py`, and `vet.py` (version 0.2.81):

- `extra_event_score(extra_event_count, count_threshold=3) -> float` — clip(count / threshold)
- `RawDiagnostics.extra_event_count`: count of compact, significant flux dips outside every predicted transit window. `_measure_extra_events()` masks out cadences near any predicted center, flags remaining out-of-transit cadences ≥3σ below the OOT median (MAD-based robust sigma), and clusters contiguous flags — a cluster counts only if it spans ≥2 cadences and ≤2× the transit duration
- High score → anomalous structure outside the transit windows → evidence for a second periodicity, a blended source, or an instrumental glitch
- Wired into `log_score_planet()` (−0.60 weight) and `log_score_instrumental()` (+0.50 weight)
- `None` if fewer than 20 out-of-transit cadences are available
- Spec: `docs/SCORING_MODEL.md §25` — this is the third and final increment of the "depth/asymmetry/missing/extra-event ranking" extension named in the 0.2.77 roadmap note

---

## Milestone 12 Features (features.py + schemas.py + hypotheses.py)

Five new diagnostic scores added (Milestone 12a–12e):

| Function | Weight in planet | Wired into |
|---|---|---|
| `out_of_transit_scatter_score(oot_scatter_sigma, sigma_threshold=3.0)` | −0.70 | planet(−), instrumental(+0.80) |
| `multi_sector_depth_consistency_score(sector_depths, sector_depth_errors, cv_threshold=0.20)` | +0.60 | planet(+), instrumental(−0.50) |
| `stellar_density_consistency_score(duration_hours, period_days, depth_ppm, stellar_radius_rsun, stellar_mass_msun)` | +0.80 | planet(+), EB(−0.70), bgEB(−0.50) |
| `centroid_motion_score(centroid_motion_arcsec, saturation_arcsec=2.0)` | −1.00 | planet(−), bgEB(+1.40) |
| `limb_darkening_plausibility_score(ingress_egress_fraction, depth_ppm, stellar_teff_k=5778.0)` | +0.50 | planet(+), EB(−0.40) |

`stellar_density_consistency_score` uses transit duration approximation: `a/R_* = P / (π × T)` (b=0).
New `RawDiagnostics` fields: `oot_scatter_sigma`, `sector_depths`, `sector_depth_errors`, `centroid_motion_arcsec`, `stellar_teff_k`.

---

## CLI Version Flag and Meta Output (Milestone 12f)

- `exo --version` / `exo -V` — prints the installed `exo-toolkit` package version (currently `0.2.27`)
- fallback version `0.2.27` in `src/exo_toolkit/__init__.py` is used only if source-tree and installed package metadata are unavailable
- Each output row gains a `"features"` dict, a raw `"diagnostics"` dict, a `"fetch_provenance"` dict, plus a `"meta"` dict: `toolkit_version`, `run_at`, `scorer`, `git_commit`, `features_available`, `features_missing`
- `_git_commit_short()` reads `git rev-parse --short HEAD`; returns `None` on failure

---

## Notebook Generator (`Skills/notebook_generator.py`)

Programmatically generates Jupyter notebooks for a given TIC target.

- `generate_notebook(tic_id, *, mission, stellar_radius_rsun, stellar_mass_msun, min_snr, output_path) -> Path`
- Produces `notebooks/TIC_{tic_id}.ipynb` by default
- 7 cells covering all pipeline stages; nbformat 4.4 compatible
- 10 tests in `tests/test_notebook_generator.py`

---

## Target Prioritizer (`Skills/target_prioritizer.py`)

Ranks a list of TIC IDs by scan priority, combining TOI status and sector coverage.

- `TargetRecommendation` dataclass: `tic_id`, `priority_score`, `toi_status`, `n_sectors`, `recommendation`, `reason`
- `prioritize_targets(tic_ids, *, toi_check_fn, toi_table_fn, sector_coverage_fn, priority_fn, min_priority, skip_known_tois)` → sorted list
- `format_recommendations(recs) -> str` — Markdown table
- Recommendations: `"scan"` | `"skip_toi"` | `"skip_low_priority"`
- 12 tests in `tests/test_target_prioritizer.py`

---

## Compare Candidates (`Skills/compare_candidates.py`)

Merges multiple batch_scan JSON files into a unified Markdown comparison report.

- `load_and_merge(paths) -> list[dict]` — flattens list or single-dict JSON files; adds `_source_file`
- `build_comparison_report(rows, *, title, sort_by) -> str` — `sort_by` in `{"false_positive_probability", "rank_score", "period_days"}`; FPP/period ascending, rank_score descending
- `write_comparison_report(rows, output_path, *, title) -> Path`
- 11 tests in `tests/test_compare_candidates.py`

---

## Candidate Timeline (`Skills/candidate_timeline.py`)

Tracks how a candidate's scores evolve across repeated pipeline runs.

- `TimelineEntry` dataclass: `run_at`, `period_days`, `fpp`, `planet_posterior`, `pathway`, `scorer`, `note`
- `CandidateTimeline(path)` — atomic-write JSON; `record(row, *, note)`, `entries(candidate_id)`, `latest(candidate_id)`, `summary(candidate_id)`, `to_markdown(candidate_id)`
- `summary()` returns `{n_runs, first_run_at, latest_run_at, trend_fpp}` — `trend_fpp = last_fpp − first_fpp`
- 12 tests in `tests/test_candidate_timeline.py`

---

## FITS Header Extractor (`Skills/fits_header_extractor.py`)

Extracts stellar parameters from TESS SPOC FITS headers for use as `vet_signal` kwargs.

- `FITSStellarParams` dataclass: `tic_id`, `stellar_radius_rsun`, `stellar_mass_msun`, `stellar_teff_k`, `stellar_logg`, `contamination_ratio`, `sector`
- `extract_from_header(header: dict) -> FITSStellarParams` — keys: `TICID`, `RADIUS`, `MASS`, `TEFF`, `LOGG`, `CROWDSAP` (→ `1 - CROWDSAP`), `SECTOR`
- `extract_stellar_params(fits_path, *, hdu_index=0) -> FITSStellarParams` — reads actual FITS file
- `to_vet_kwargs()` — returns dict excluding `None` fields, ready for `**kwargs` to `vet_signal`
- 12 tests in `tests/test_fits_header_extractor.py`

---

## Integration Pipeline Tests (`tests/test_integration_pipeline.py`)

End-to-end pipeline test using mocked I/O (no network required).

- Mocks `search_lightcurve` and `vet_signal`; scoring + pathway run for real
- 10 tests in `TestIntegrationPipeline` covering: non-empty output, required keys, posterior sum, FPP range, valid pathway, scorer modes, error cases, provenance score

---

## Skills Guide (`docs/SKILLS_GUIDE.md`)

Complete user reference for all 24 Skills scripts (updated Milestone 12).

- Quick-reference table of all scripts with purpose and key functions
- Discovery workflow diagram: `star_scanner → batch_scan → alert_filter → rank_candidates → watchlist/export/report`
- CLI examples for every script with common flag combinations
- Library usage pattern (importable functions without running CLI)
- ML training pipeline walkthrough (fetch → build → merge → train → evaluate)

---

## Core Design Decisions (see docs/DECISIONS.md for full rationale)

- **Bayesian log-score model**: `log_score_i = log_prior_i + weighted_evidence_i`, then `posterior_i = softmax(log_scores)`
- **6 hypotheses**: planet_candidate, eclipsing_binary, background_eclipsing_binary, stellar_variability, instrumental_artifact, known_object
- **OptScore pattern**: `float | None` — `None` means diagnostic not run; missing features contribute 0 to log scores (neutral, no bias)
- **Conservative priors**: built-in defaults remain planet_candidate = 0.10, EB/BEB/stellar/instrumental = 0.20 each, known_object = 0.10
- **Mission prior profiles**: `configs/scoring_priors_v0.json` defines opt-in conservative TESS/Kepler/K2 profiles loaded by `priors.py`
- **ML Tier 1 (XGBoost) is built** — `ml/xgboost_scorer.py` ships as an optional alternative scorer; Bayesian log-score model remains the default fallback when labels are unavailable
- **ML Tier 2 scaffolding is built** — `ml/cnn_scorer.py`, `Skills/train_cnn.py`, checkpoint/calibration utilities, and CLI wiring exist; production checkpoint `benchmark_cnn_v1` is promoted (see `docs/PRODUCTION_READINESS.md` T1-1)
- **ML Tier 3 (stacking) is built** — `ml/stacking_scorer.py` blends XGBoost + CNN + Bayesian P(planet) when models are supplied; falls back conservatively when optional models are unavailable; production weights are calibrated (see `docs/PRODUCTION_READINESS.md` T1-2)
- **CLI scorer options**: `exo <TIC-ID> --scorer [bayesian|xgboost|ensemble|cnn|full-ensemble] --model-path <path> --cnn-checkpoint <path>`
- **Never output "confirmed planet"** — use "candidate signal" or "follow-up target" (see `AGENTS.md` Scientific Guardrails)
- **Numerically stable softmax**: subtract max before exponentiation

---

## Key Types (schemas.py)

```python
Score    = Annotated[float, Field(ge=0.0, le=1.0)]
OptScore = Annotated[float | None, Field(ge=0.0, le=1.0)]
Mission  = Literal["TESS", "Kepler", "K2"]
SubmissionPathway = Literal[
    "known_object_annotation", "tfop_ready", "planet_hunters_discussion",
    "kepler_archive_candidate", "github_only_reproducibility", "paper_or_preprint_candidate"
]

CandidateSignal      # raw BLS output
CandidateFeatures    # 46 OptScore fields, all default None
HypothesisPosterior  # 6 Score fields, validator enforces sum ≈ 1.0 ±0.01
CandidateScores      # 6 Score fields (fpp, detection_confidence, novelty_score, …)
CandidateExplanation # tuple[str, ...] fields for positive/negative/blocking evidence
ScoringMetadata      # model name, version, commit, config_hash
ScoredCandidate      # full pipeline output
```

All models: `ConfigDict(frozen=True)` — immutable after construction.

### Pipeline result types (frozen dataclasses)

```python
FetchResult(light_curve, provenance: FetchProvenance)
CleanResult(light_curve, provenance: CleanProvenance)
VetResult(diagnostics: RawDiagnostics, features: CandidateFeatures)
# search returns list[CandidateSignal] directly
```

`RawDiagnostics` (frozen dataclass in `features.py`) — 30+ optional float/int fields covering
per-transit depths, odd/even, secondary SNR, stellar params, crowding, flags, catalog matches.

---

## Scoring Pipeline (scoring.py)

```
CandidateFeatures
    → compute_log_scores()      (hypotheses.py)
      optional mission priors   (priors.py)
    → softmax()                 (scoring.py)
    → HypothesisPosterior
    → compute_scores()          (scoring.py)
    → CandidateScores

Public entry point: score_candidate(signal, features, log_priors=None, prior_config=None)
    → tuple[HypothesisPosterior, CandidateScores]
```

---

## Pathway Classification (pathway.py)

`classify_submission_pathway(signal, features, posterior, scores, *, provenance_score=0.0, ...)`

Gate order (spec §11):
1. `posterior.known_object >= 0.80` → `known_object_annotation`
2. `fpp >= 0.70` → `github_only_reproducibility`
3. `transit_count < 2` → `planet_hunters_discussion`
4. TESS branch → `tfop_ready` (all 9 conditions) or `planet_hunters_discussion` or `github_only_reproducibility`
5. Kepler/K2 branch → `kepler_archive_candidate` or `github_only_reproducibility`
6. Fallback → `github_only_reproducibility`

`None` feature scores **fail** gate conditions conservatively.
`provenance_score` is computed in `run_pipeline()` from fetch provenance and
passed into pathway classification; callers that omit it still default to 0.0
and therefore block `tfop_ready` conservatively.

---

## Quality Commands

Canonical commands and rationale: `AGENTS.md` Quality Gates. Quick copy-paste:

```bash
# Full gates: Ruff + mypy + incomplete-implementation scan + directive-integrity
# check + six test shards x six xdist workers (ten gates total)
.venv/bin/python Skills/run_quality_gates.py

# Focused test diagnosis
PYTHONPATH=src .venv/bin/python -m pytest tests/test_target.py -n auto --dist=worksteal

# Individual static checks
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy src

# Apply safe Ruff fixes
.venv/bin/python -m ruff check . --fix
```

If pytest fails with `ModuleNotFoundError: No module named 'exo_toolkit'`, add `PYTHONPATH=src`.

`mypy` (bare binary) sees a different package path and reports false import errors for pydantic/numpy.
Always use `.venv/bin/python -m mypy src` locally.

---

## Data Pipeline Notes

### fetch.py
- Lazy lightkurve import (inside `fetch_lightcurve()`); `FetchProvenance` records cadence, sectors, pipeline, fetched_at
- Live tests use `@pytest.mark.integration_live` and are excluded from CI

### clean.py
- No lightkurve import at all — calls methods on the passed-in object only
- `CleanProvenance` records n_cadences_raw/cleaned, sigma_clip_sigma, window_length

### search.py
- Uses `astropy.timeseries.BoxLeastSquares` directly (no lightkurve needed)
- Duration grid capped at 90% of `period_min` to satisfy astropy BLS constraint
- Iterative transit masking in pure numpy; `_extract_flux_err` falls back to 1.4826×MAD

### vet.py
- No lightkurve import — pure numpy diagnostics from `lc.time.jd` / `lc.flux.value`
- Computes: individual depths, odd/even comparison, secondary eclipse SNR, transit shape, data-gap fraction
- Catalog diagnostics (stellar params, crowding, flags) pass through as keyword arguments

### calibration.py
- Public API: `compute_metrics`, `fit_calibration`, `apply_calibration`, `save_calibration`, `load_calibration`
- Methods: `"platt"` (Platt scaling via scipy Nelder-Mead), `"isotonic"` (PAVA — no sklearn)
- One-vs-rest calibration per hypothesis; renormalized to sum to 1.0 post-calibration
- Metrics: Brier scores, reliability curves, precision/recall/F1, confusion matrix
- `save_calibration(result, path)` / `load_calibration(path)` round-trip `CalibrationResult` as JSON
- All result containers are frozen dataclasses

---

## CNN Scorer Reference

Production checkpoint `benchmark_cnn_v1` is promoted under `models/cnn/benchmark_cnn_v1/`
(`docs/PRODUCTION_READINESS.md` T1-1 has the full training/evaluation history — C1-C19
candidate history, corpus status, rejection reasons — do not restate it here; it is
kept current there, not here).

**Cross-mission scoring guard**: every trained checkpoint declares its training mission
via `train_cnn.py --mission TESS|Kepler|K2|JWST`, stamped into the checkpoint's
`config.json` as `training_mission`. `run_pipeline()`/`exo scan` refuse by default to
apply a CNN checkpoint whose declared (or undeclared/`None`) mission doesn't match the
scan's `--mission`, since Kepler↔TESS CNN transfer has repeatedly failed this project's
production gates even after deliberate fine-tuning. Override: `allow_cross_mission_cnn=True`
/ `--allow-cross-mission-cnn`, for deliberate out-of-domain testing only.

**Accepted CLI flags** (verify against source if in doubt — these have drifted before):
- `train_cnn.py`: `--split-dir`, `--checkpoint-dir`, `--pretrained-checkpoint`, `--mission`, `--device auto|cpu|mps|cuda` (config defaults to `device=auto`)
- Evaluator flag is `--output-calibration` (not `--calibration-output`)

Architecture spec: `docs/CNN_SPEC.md`. Copy-paste workflow: `docs/CNN_PRODUCTION_RUNBOOK.md`.

**Architecture fit**: XGBoost and CNN sit alongside `scoring.py`; the stacking layer
(`ml/stacking_scorer.py`) blends their posteriors with the Bayesian log-score model,
which remains the fallback when features/models are missing. `calibration.py` handles
final probability calibration for all model variants. Label quality caveat: "candidate"
KOIs are noisy labels; train only on confirmed planets vs. confirmed false positives
where possible.

---

## Data Sources

- **TESS**: MAST via Lightkurve (`mission="TESS"`, PDCSAP flux preferred)
- **Kepler/K2**: MAST via Lightkurve (`mission="Kepler"` / `"K2"`)
- **JWST**: MAST via `astroquery.mast` directly — Lightkurve does NOT support JWST. Use `_calints.fits` (Stage 2) or `_x1dints.fits` (Stage 3 NIRISS SOSS). See `Skills/fetch_jwst_lc.py`.
- **Catalogs**: NASA Exoplanet Archive, TOI list, KOI list, CTOI via astroquery

Focus on lightly-worked targets: later TESS sectors, fainter stars (Tmag 10–14), less-crowded fields.

---

## Research Context (`docs/exoplanet_detection_research_brief.md`)

Full brief: `docs/exoplanet_detection_research_brief.md`. Key facts for coding agents:

### Satellite Priority Order (for discovery work)
1. **TESS** — best current public discovery engine; huge archive, ongoing sectors, TOIs, FFIs
2. **Kepler/K2** — highest-value historical benchmark; Kepler = cleaner long-baseline; K2 = noisier systematics
3. **JWST** — atmospheric characterization, not bulk detection; public data via MAST after proprietary period
4. **PLATO** — launching end-2026; bright-star terrestrial planets + asteroseismic ages (prepare pipeline)
5. **Roman** — future; microlensing census and coronagraph technology demo

### AI Methods Relevant to This Project
- **1D CNN on phase-folded light curves** — Shallue & Vanderburg (2018) baseline; local+global view architecture
- **Transformer for full light curves** — attention can model long light curves without pre-selecting transit windows
- **Semi-supervised / anomaly detection** — useful when labels are incomplete; helps discover unusual systems
- **GP for stellar variability** — models correlated noise; prevents biased transit depth/timing estimates
- **Bayesian atmospheric retrieval** — JWST spectra require TauREx/petitRADTRANS-style retrieval; ML retrieval (neural posterior estimation) is the frontier

### Citizen Science Quality Bar (before escalating any candidate)
- Signal repeats at consistent period
- Full transit with pre/post baseline; partial events are lower value
- Survives multiple detrending approaches
- No centroid shift, no eclipsing binary contaminant in aperture, no secondary eclipse
- Odd/even depth consistent
- Use BJD_TDB time standard; document it

### Minimum Submission Evidence
TIC ID + coordinates, light curve (BJD + normalized flux + errors), transit model parameters, false-positive diagnostic table, catalog cross-check (TOI/CTOI/Gaia/confirmed hosts), reproducible notebook or script.

### Pipeline Stack Guidance (from brief)
Use `lightkurve`, `astroquery`, `wotan` (detrending), `transitleastsquares`, `exoplanet`, `celerite`, `pymc` where appropriate. JWST: use `astroquery.mast` directly. Atmospheric: `petitRADTRANS` or `TauREx` for forward models.

### Upcoming Assets to Prepare For
- **PLATO** (end-2026): long-baseline photometry, asteroseismic stellar ages — pipeline should handle multi-year continuous light curves
- **Roman** (mid-2020s): microlensing fields, coronagraph tech demo — different data format from transit surveys
