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

Its release gate passed the unchanged 2,759 default tests plus Ruff/mypy as
8/8 supervised gates in 27.3 seconds under the canonical 6×6 topology.
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
- 14 tests in `tests/test_batch_scan.py`

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

- `priority_score(tmag, teff, n_sectors, contratio)` → float in [0, 1]
- `ScanLog(path)` — atomic-write JSON log; `record()`, `is_scanned()`, `scanned_ids()`, `summary()`
- `select_targets(n, tmag_range, exclude_tic_ids)` — TIC query, ranked, filtered
- `scan_star(tic_id, *, log, ...)` → dict with status/n_signals/best_fpp/best_pathway
- `run_background_scan(log_path, ...)` — iterates until Ctrl-C or max stars reached
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
CandidateFeatures    # 44 OptScore fields, all default None
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
# Full gates: Ruff + mypy + six test shards x six xdist workers
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
