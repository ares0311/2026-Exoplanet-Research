# PROJECT STATUS

## Status: Active Development
## Phase: Phase 4 individual anomalous-transit diagnostics
## Last Updated: 2026-07-16

---

## Current Snapshot

The repository contains a reproducible TESS/Kepler exoplanet candidate toolkit with:

- Core pipeline: Fetch -> Clean -> Search -> Vet -> Score -> Classify
- Bayesian log-score model over six hypotheses
- Optional XGBoost and stacking scorer modes (Tier 1 model trained: Kepler KOI AUC=0.992)
- SQLite-backed background automation with top-level logs
- 126 standalone `Skills/` Python files (`rg --files Skills -g '*.py'`)
- 149 top-level test files
- 32 package Python modules under `src/exo_toolkit/`
- JWST time-series ingestion wired into the CLI with `--mission JWST`
- Novel TESS target scanning that excludes TOI, CTOI, and confirmed-host catalogs

Local validation note: validated on Python 3.14.3 in `.venv` with `xgboost` dependency restored and macOS OpenMP runtime (`libomp`) installed. System Python is never used.

Phase 3 source gate: version 0.2.55 pins the 56,036,648-byte direct package and
model footprint for Chronos-Bolt tiny plus Astromer2 and adds a metadata-only,
fail-closed verifier. Its first merged run failed closed on followed Hugging
Face HEAD redirects without writing evidence or downloading payloads. Version
0.2.56 preserves the authoritative resolver headers and passes 2,734 tests plus
Ruff/mypy in 26.1 seconds. The merged full verifier then passed 7/7 operations
in 4.94 seconds: five
sources verified, 56,036,648 projected direct bytes, and zero payload bytes
downloaded. Artifact SHA-256 is `5610bbb8…3042`; Run Report commit `ae4e659`.
Source identity/footprint is complete, while dependencies, weights, inference,
training, stellar-variability labels, and injection comparison remain gated.

Version 0.2.57 adds the next bounded gate: one cached TESS product, two exact
ONNX revisions, isolated one-thread CPU inference, finite 256-dimensional
output checks, and structured timing/memory evidence. The dependencies are an
optional group and the model cache is ignored/ledgered. At that point merged
installation and smoke evidence were next; broad extraction and training remain
unauthorized.
The 0.2.57 full 6×6 gate passed 2,743 tests plus Ruff/mypy in 24.2 seconds with
the optional packages absent.

The first merged smoke failed closed before payload download because Xet tried
to write a log below sandbox-blocked `~/.cache/huggingface`; only 8 KB of
ignored metadata was created. Version 0.2.58 relocates Hub and Xet state into
the ignored model cache before import. The pinned optional group is installed,
`pip check` passes, and the 0.2.58 6×6 gate passed 2,743 tests plus Ruff/mypy in
26.2 seconds. The merged retry passed both exact models in 26.875 seconds with
finite `(1,1,1,256)` embeddings; max child RSS was 186,204,160 bytes and the
ignored cache totals 29,960,842 file bytes. Artifact SHA-256 is
`1cc59ab3…5de5d10`; Run Report commit `f8a7207`. Version 0.2.59 records the
runtime PASS. Variability-label and injection-recovery scientific gates remain
before broad extraction or training. The 0.2.59 6×6 evidence-release gate
passed 2,743 tests plus Ruff/mypy in 34.3 seconds.

Version 0.2.60 pins the publication-backed Drake et al. Catalina
stellar-variability source (47,055 rows, 17 classes, 1,166,660 compressed
bytes) and adds a five-operation, zero-full-payload verifier with eight offline
tests. Merged-code verification is next. The later 2,790-TIC metadata
crossmatch and embedding-aware injection comparison remain gated; independent
TIC work uses the single-parent six-shard/six-worker pattern. The version
0.2.60 release gate passed 2,751 default tests plus Ruff/mypy in 25.2 seconds.

The merged source verifier passed 5/5 operations in 3.334 seconds, validating
47,055 rows, all 17 class counts, schema, delivery metadata, and three sample
rows with zero full-catalog bytes. Artifact SHA-256 is `eb5d4bc6…39b9a`; Run
Report commit `b0003bb`. Version 0.2.61 records source identity complete. The
2,790-TIC crossmatch and embedding-aware injection comparison remain gated.
The 0.2.61 evidence-release gate passed 2,751 tests plus Ruff/mypy as 8/8
supervised gates in 40.2 seconds under the canonical 6×6 topology.

Version 0.2.62 adds the next bounded gate: a deterministic 216-TIC
TESS-Catalina crossmatch pilot using six shards × six exact-ID MAST batch
workers, one locked 1.17 MB source cache, precommitted positional/magnitude/
duplicate/blend safeguards, and training-disabled outputs. Full 2,790-TIC
execution awaits merged pilot evidence and global one-to-one reconciliation.
The 0.2.62 release gate passed 2,759 tests plus Ruff/mypy as 8/8 supervised
gates in 34.3 seconds under the canonical 6×6 topology.
Version 0.2.63 fixes the merged-main launch blocker by ignoring the shared
Catalina runtime cache. The supervisor stopped before issuing MAST requests,
so the authorized 216-TIC pilot remains pending a clean merged-main retry.
Version 0.2.64 fixes the second fail-closed integration defect from that run:
valid CDS rows without the optional class flag are 71 rather than 73 bytes.
The bounded parser now accepts 71-73 bytes, pads only omitted trailing fields,
and still rejects malformed shorter or longer rows before MAST access.
The 0.2.64 release gate passed 2,760 tests plus Ruff/mypy as 8/8 supervised
gates in 25.2 seconds under the canonical 6×6 topology; the real pinned
catalog parsed all 47,055 rows.
Version 0.2.65 closes the third fail-closed integration defect. MAST rejects
v1's `duplicate_i` field and accepts `duplicate_id`; immutable contract v1 is
retained for audit, v2 is active, and the request column list now comes from
the contract. The failed attempt wrote no pilot artifacts.
The v2 six-ID probe returned all rows in 1.5 seconds, and the 0.2.65 release
gate passed 2,761 tests plus Ruff/mypy as 8/8 gates in 26.2 seconds.
The merged pilot completed 216/216 queries and wrote every shard artifact.
Version 0.2.66 fixes only the final sandbox-lock handling so Run Report git
failure warns without converting successful data acquisition into exit 1.
The 0.2.66 release gate passed 2,762 tests plus Ruff/mypy as 8/8 supervised
gates in 34.3 seconds under the canonical 6×6 topology.
Version 0.2.67 closes the Catalina pilot as evidence-limited: 216/216 unique
TICs, 38 completed batches, 8.519s observed wall, and zero candidates within
the precommitted 3-arcsecond search radius. Global reconciliation passes with
zero accepted or duplicate sources; full execution/training stays blocked.
The 0.2.67 release gate passed 2,763 tests plus Ruff/mypy as 8/8 supervised
gates in 24.2 seconds under the canonical 6×6 topology.
Version 0.2.68 adds the replacement source preflight for ASAS-SN Catalog X.
Its immutable contract pins 378,861 VizieR rows, delivery/schema/distribution
metadata, exact TIC joins, and the fact that catalog classes are automated ML
outputs rather than ground truth. An exploratory zero-payload six-worker query
found 48 matches among the full 2,790-TIC inventory in 7.86 seconds. The
durable merged-main run uses six shards x six workers and must globally
reconcile all TICs before it can authorize follow-up benchmark design. It
cannot authorize training, extraction, promotion, or production scoring.
The 0.2.68 release gate passed 2,772 default tests plus Ruff/mypy as 8/8
supervised gates in 31.3 seconds under the canonical 6x6 topology.
The merged run passed all six shards and the global gate: 2,790 unique TICs,
48 unique ASAS-SN matches (44 known variables, four discoveries), minimum
probability 0.902, no duplicate TIC/source IDs, and no catalog payload bytes.
Observed wall time was 6.762 seconds. Version 0.2.69 commits the shard outputs,
summaries, aggregate, and integrity test. Follow-up embedding-aware benchmark
design is now the active scientific gate; training remains unauthorized.
The 0.2.69 evidence-release gate passed 2,773 default tests plus Ruff/mypy as
8/8 supervised gates in 32.3 seconds under the canonical 6x6 topology.
Version 0.2.70 freezes the next cache-only gate: all 48 matched TICs, four
bounded injection cells, 192 unique blind-BLS trials, and 384 paired rows from
the exact cached Chronos-Bolt tiny and Astromer2 models. The implementation is
allowlisted in the single-parent 6x6 launcher and persists no embeddings or
modified light curves. Its results are
descriptive only; training, broad extraction, promotion, and production scoring
remain unauthorized. See
`docs/REPRESENTATION_VARIABILITY_INJECTION_BENCHMARK.md`.
The 0.2.70 release gate passed 2,780 default tests plus Ruff/mypy as 8/8
supervised gates in 33.3 seconds under the canonical 6x6 topology.
Version 0.2.71 requires the benchmark's label rows to match all six paths and
hashes pinned by the ASAS-SN aggregate before any FITS/model work. The 6x6
release gate passed 2,781 tests plus Ruff/mypy in 30.1 seconds.
The merged run passed all six shards and global reconciliation: 48 TICs,
192 trials, 384 unique model rows, 96/96 higher-depth larger-shift comparisons
for each frozen model, 13/192 blind-BLS recoveries, and zero failures,
duplicates, downloads, or persisted embeddings. Version 0.2.72 commits the
evidence and integrity regression. The bounded gate is complete; training,
broad extraction, promotion, and production scoring remain unauthorized.
The 0.2.72 evidence-release gate passed 2,782 default tests plus Ruff/mypy as
8/8 supervised gates in 37.2 seconds under the canonical 6x6 topology.
Version 0.2.73 implements the next separately contracted grouped benchmark on
1,536 unique cache-local Kepler KICs. It compares frozen Chronos-Bolt tiny and
Astromer2 linear probes with the frozen calibrated CNN and a statistical
ephemeris baseline under train/validation/test separation. The exact selected
10.398 GB cache inventory is read-only, the run is allowlisted for the 6x6
supervisor, and aggregate reconciliation must remove every temporary embedding
array. Merged-main execution is next; all training and production authorization
flags remain false.
The 0.2.73 release gate passed 2,789 default tests plus Ruff/mypy as 8/8
supervised gates in 34.1 seconds under the canonical 6x6 topology.
The first merged v1 benchmark launch failed closed before processing because
Kepler files expose `SAP_QUALITY`, not v1's TESS-style `QUALITY`. No benchmark
data was downloaded or persisted. Version 0.2.74 preserves v1 and activates
immutable v2 with the corrected column and a hash-pinned exception for exactly
111 known truncated cache products. All other FITS errors remain fatal; the
merged v2 evidence run is pending.
The 0.2.74 release gate passed 2,790 default tests plus Ruff/mypy as 8/8
supervised gates in 36.3 seconds under the canonical 6x6 topology.
The first merged v2 run failed closed on its over-strict 95% 2,048-bin
occupancy rule before embeddings or durable output. Version 0.2.75 preserves
v2 and activates immutable v3 with neutral median physical-flux fill for empty
phase bins. A full cache-only 1,536-KIC preparation preflight passed in 158.4
seconds with exactly 111 pinned skips, zero writes/downloads, and finite smoke
embeddings from both external models. Merged v3 evidence remains pending.
The 0.2.75 release gate passed 2,791 default tests plus Ruff/mypy as 8/8
supervised gates in 36.3 seconds under the canonical 6x6 topology.
The merged v3 benchmark passed 1,536 unique KICs with 111 pinned skips, zero
failures/downloads/persisted embeddings, and one test opening. The frozen CNN
retained a wide lead: AUC/AP/top-100 0.923096/0.899184/91 versus Chronos-Bolt
tiny 0.722778/0.696344/71 and Astromer2 0.708984/0.659679/67. Version 0.2.76
records `no_external_added_value`; broad extraction/training remains blocked.
The 0.2.76 evidence-release gate passed 2,797 default tests plus Ruff/mypy as
8/8 supervised gates in 36.3 seconds under the canonical 6x6 topology.
Version 0.2.77 activates the previously unreachable per-transit duration-
consistency and TTV features in production vetting. Measurements use local
sidebands and noise-gated half-depth cadences, require two resolved events, and
otherwise remain unavailable. Flat noise is rejected and a 30-minute shifted
event is recovered in focused regression evidence.
The 0.2.77 release gate passed 2,801 default tests plus Ruff/mypy as 8/8
supervised gates in 35.3 seconds under the canonical 6x6 topology.
The 0.2.63 release gate passed the unchanged 2,759 tests plus Ruff/mypy as 8/8
supervised gates in 27.3 seconds under the canonical 6×6 topology.

---

## What Is Complete

| Area | Key Files | Status |
|------|-----------|--------|
| Scoring engine | `schemas.py`, `features.py`, `hypotheses.py`, `priors.py`, `scoring.py`, `pathway.py` | Complete |
| Data pipeline | `fetch.py`, `clean.py`, `search.py`, `vet.py`, `calibration.py` | Complete |
| Transit scan CLI | `cli.py` — `exo <TIC-ID>` with `--scorer`, `--model-path`, `--output`; JWST via `--mission JWST` | Complete |
| Background automation CLI | `cli.py` — `background-run-once`, summaries, integrity, validation | Complete |
| Background automation module | `background/` — config, fixtures, priority, runner, storage, reports | Complete |
| SQLite runtime state | `logs/background_search.sqlite3` schema v2 | Complete |
| Background config | `configs/background_search_v0.json` | Complete |
| Scoring prior config | `configs/scoring_priors_v0.json` — conservative default plus TESS/Kepler/K2 profiles | Complete |
| Scheduler docs | `docs/SCHEDULER.md` — cron, launchd, systemd | Complete |
| ML Tier 1 | `ml/xgboost_scorer.py` + `models/xgboost_koi.json` | Complete — trained on 7,586 Kepler KOIs (AUC=0.992) |
| ML Tier 3 | `ml/stacking_scorer.py` | Complete, includes optional 3-tier XGBoost/CNN/Bayesian blend |
| ML Tier 2 scaffolding | `ml/cnn_scorer.py`, `Skills/train_cnn.py`, CNN data utilities | Complete, no production checkpoint; active work follows `docs/exoplanet_exomoon_dataset_handoff.md` |
| Training/evaluation Skills | Kepler, TESS, combined training, CNN data assembly/validation/training support, XGBoost training, scorer evaluation | Complete |
| Discovery workflow Skills | star scanner, batch scan, alert filter, ranking, watchlist, exports, reports | Complete; run006/run008 evidence retained as historical provenance, not the active production blocker |
| Additional Skills | Analysis, vetting, observability, ML, physics, reporting, scheduling, and follow-up utilities retained after production-scope cleanup | Complete |
| Milestone 19a Skill | `multi_sector_phase_compare.py` — offline per-sector phase-fold comparison | Complete |
| Milestone 19b Skill | `candidate_dashboard_export.py` — static conservative candidate dashboard with optional plot artifacts | Complete |
| Milestone 19c Skill | `candidate_api.py` — local read-only candidate API plus optional background SQLite summaries | Complete |
| Milestone 19d Skill | `candidate_browser_ui.py` — interactive local candidate browser with optional plot previews | Complete |
| Milestone 30 Skills | 15 diagnostics + scheduling tools including `flux_anomaly_detector`, `candidate_confidence_tracker`, `uncertainty_propagator`, `multi_target_scheduler`, `candidate_archive`, and 10 more | Complete |
| Milestones 34-39 Skills | 90 additional ML evaluation, photometry quality, transit vetting, noise budget, orbit simulation, stellar physics, TTV, occurrence-rate, and planning utilities | Complete |
| CTOI source contract | `docs/CTOI_SOURCE_CONTRACT.md`, `Skills/fetch_exofop_ctoi.py`, `tests/fixtures/exofop_ctoi_sample.csv`, `tests/fixtures/exofop_ctoi_labels_sample.json` — opt-in fixture-backed community candidate labels | Complete, excluded from default training |
| Project MCP bootstrap | `.mcp.json`, `.codex/config.toml`, `Skills/mcp_bootstrap_server.py` — project-scoped file, git-read, and fixed validation MCP servers | Complete, offline by default |
| Live label-check audit | `Skills/count_tess_labels.py`, `Skills/tess_label_check_summary.py` — opt-in live ExoFOP gate check plus read-only SQLite log summary | Complete, live access requires intentional approval |
| Docs | `README.md`, `docs/`, `CLAUDE.md`, `AGENTS.md`, `CONTRIBUTING.md` | Active maintenance |

---

## Background Automation State

The implementation follows `docs/BACKGROUND_SEARCH_AUTOMATION_BLUEPRINT.md`:

- Single bounded command: `exo background-run-once`
- Top-level SQLite logs under `logs/`
- Top-level config under `configs/`
- Known TESS fixtures plus synthetic edge cases
- Composite target priority with exposed component factors and reason codes
- Exactly-one primary outcome target: reviewed or needs-follow-up
- Mandatory follow-up test records for escalated targets
- Conservative draft reports exported as Markdown and HTML
- Ranked top-three submission recommendations
- Explicit human-approval records; no external submission command is implemented
- Scheduler-friendly exit codes behind `--scheduler-exit-codes`
- Non-overlap locking with a brief wait

Runtime artifacts remain ignored by default:

```text
logs/background_search.sqlite3
reports/
checkpoints/
data/*.jsonl
data/*_splits/
models/cnn*/
```

Production-relevant ignored artifact state is committed in
`docs/LOCAL_ARTIFACT_LEDGER.md` and
`artifacts/manifests/local_artifacts.json` so GitHub-only agents can continue
without relying on chat context or local terminal output.

---

## Active Production Blocker

**T1-1 — production trained model checkpoint**

- The project reset on 2026-07-01 wholly adopts
  `docs/exoplanet_exomoon_dataset_handoff.md` as the active path to a trained
  model.
- Source-contract-first data/ML hardening is complete enough to produce a
  passing checkpoint: provider schemas were verified, source snapshots and
  manifests were committed, leakage-safe splits were enforced, storage remained
  bounded, and the master Kepler corpus trained successfully.
- `checkpoints/cnn_t1_1_kepler_master/best.pt` is promoted as
  `benchmark_cnn_v1` under `models/cnn/benchmark_cnn_v1/` after explicit human
  approval on 2026-07-09. It passed held-out gates with raw test AUC 0.9572,
  calibrated F1 0.8347, Brier 0.0580, ECE 0.0142, and temperature T=1.0.
- The active blocker is no longer data acquisition, another training loop, or
  checkpoint promotion. The next production gap is T1-2 stacking calibration
  after the promotion PR is merged and CI-clean.
- Do not repeat old C1-C19/C20-style retraining or Kepler batch processing
  unless a named promotion validation gate fails.
- Do not use synthetic examples as supervised training positives in this phase;
  synthetics remain CI/background fixtures only.
- Any local long-running command must be resumable, print progress/ETA, use
  top-level logs where applicable, and use the M4 Max/MPS path from
  `docs/SYSTEM_PROFILE.md` for training.

## Historical Discovery Evidence

**T1-0 — first real discovery scan evidence**

- The project mission has been realigned to discovering previously unknown transit candidates before doing more CNN work.
- Option A JWST integration is merged: A1/A2 via PR #133 and A3 CLI wiring via PR #141.
- Option B TESS novelty targeting is merged via PR #139: `star_scanner.py` excludes TOI, CTOI, and confirmed-host catalogs, and defaults to Tmag 12.0-14.5.
- Live scanner startup/target-selection hardening is merged via PR #143: ExoFOP SSL loading, Python 3.14 helper imports, bounded TIC target selection, and `no_data` classification are fixed; a one-target live smoke on `main` selected TIC 425884922 and completed with `1 no-data | 0 errors`.
- K2 overlap corpus collection is complete locally with 2,086 snippets; do not re-fetch it.
- A SPOC-only 200-target attempt completed on 2026-06-28 as `logs/discovery_run_001.json`, but it did not close T1-0: 198 targets had no SPOC long-cadence light curve, 2 hit transient remote disconnects, and there were 0 clear scans / 0 candidates.
- A QLP attempt started on 2026-06-28 as `logs/discovery_run_002_qlp.json`, but it did not close T1-0: it recorded 3 corrupt local Lightkurve cache errors, 0 clear scans, and 0 candidates. Root cause: interrupted prior QLP downloads left corrupt FITS files under `~/.lightkurve/cache/mastDownload/HLSP/...`, and the shared fetch path did not delete the named corrupt file and retry.
- A cache-repair QLP attempt started on 2026-06-28 as `logs/discovery_run_003_qlp_cache_repair.json`, but it did not close T1-0: it recorded 1 error, 0 clear scans, and 0 candidates before crashing with `ValueError: I/O operation on closed file`. Root cause: Lightkurve public download methods use `suppress_stdout`, which mutates process-global `sys.stdout`; that is unsafe under worker-thread downloads while the main thread prints progress.
- A stdout-safe QLP attempt completed on 2026-06-28 as `logs/discovery_run_004_qlp_stdout_safe.json`, but it did not close T1-0: it recorded 200 total entries, 0 candidates, 0 clear scans, 1 no-data row, and 199 errors. Root cause: the shared fetch path requested SPOC-style `pdcsap_flux`; valid QLP HLSP products provide `KSPSAP_FLUX`, `DET_FLUX`, `SYS_RM_FLUX`, or `SAP_FLUX`, not `PDCSAP_FLUX`.
- A flux-safe QLP attempt started as `logs/discovery_run_005_qlp_flux_safe.json`, but it did not close T1-0: the pasted console showed third-party MAST download chatter and warnings but no per-target scanner progress, and no durable log existed before the first completed target.
- PR #150 is merged on `main`: scanner logs are created immediately, active targets are checkpointed separately, per-target startup/progress is printed, and third-party MAST download banners are suppressed.
- Run006 completed locally on 2026-06-29 and produced durable scan evidence:
200 entries, 192 `candidate_found`, 6 `scanned_clear`, 1 `no_data`, 1 `error`,
and 0 active targets. The filtered output contains two rows: TIC 201252011
(period 227.39056281978395 d, FPP 0.1160636155807766) and TIC 257712351
(period 142.95415231096942 d, FPP 0.12672985673564718).
- Version 0.2.8 fixes two production-review blockers found during targeted
follow-up: Lightkurve stitch no longer normalizes QLP products before project
sigma-clipping, and `exo --output` now serializes computed vetting features for
`Skills/false_positive_vetter.py`.
- Version 0.2.9 adds candidate-review diagnostics: `exo --output` now
  serializes raw vetting diagnostics, fetch provenance, and missing-feature
  names, and `Skills/false_positive_vetter.py` explains why missing
  diagnostics are unavailable.
- Version 0.2.10 adds bounded retry/backoff for transient MAST/Lightkurve
  connection disconnects during candidate-review packet regeneration.
- Run008 targeted follow-up reproduced both filtered candidates under the fixed
path: `logs/discovery_run_008_targeted_qlp_stitch_safe.json` has 2
`candidate_found` entries and active `{}`. SHA-256:
`8626587c4fe59565132e078273763c7beac4a0a88597615f71e147a5134d1b0a`.
Filtered output SHA-256:
`574a4cf188faa9e273128496fcd23b27cb8369a3e9d2ad2c1b5bbaedd9effed4`.

This loop is now historical. It is useful scanner evidence, but later v0.2.10
candidate packet regeneration moved the two filtered candidates above the prior
FPP < 0.15 escalation threshold. Do not submit/contact externally from this
evidence without explicit human approval, and do not use it to block T1-1.

## Trained-Model History

**Production ML Tier 2 — checkpoint generalization**

- The label gate is open, but no CNN checkpoint has passed the production gate.
- The first seed-42 checkpoint completed training but was rejected on 2026-06-10.
- Held-out test AUC was 0.7404 and calibrated F1 was 0.6297, below the documented 0.85 and 0.80 targets.
- Validation-fitted Platt calibration worsened test Brier score and ECE, so no calibration or checkpoint artifact was promoted into `models/`.
- A 2026-06-10 audit found that every nominally usable snippet had `epoch_bjd=0.0`, so catalog transit events were not centered in phase.
- The old corpus, original seed-42 split, and temporary replacement split are retired.
- The local TESS v2 corpus is complete.
- The pre-fix local Kepler corpus was rejected on 2026-06-17: it had 7,454 rows, but 7,132 rows contained non-finite flux and only 322 finite examples survived builder filtering.
- The rebuilt local Kepler corpus was validated on 2026-06-17: 6,837 parseable finite snippets, zero duplicate resume keys, labels negative=4,280 and positive=2,557.
- `data/kepler_cnn_splits` passed validation on 2026-06-17 with train/val/test = 4,741 / 1,060 / 1,036.
- Tiny corrupt Kepler Lightkurve cache files were quarantined locally before training resumed.
- Kepler pretraining completed locally on 2026-06-18:
  `checkpoints/cnn_kepler_pretrain/best.pt`, SHA-256
  `c782d7af61171b3f58447f7a49343c86618c447292a71bd28d540807835787c7`;
  startup banner `device=mps`; best epoch 19, best validation loss 0.3905,
  best validation AUC 0.9186.
- `data/tess_cnn_splits` passed validation on 2026-06-18 with total examples
  2,110 and train/val/test = 1,477 / 318 / 315.
- Kepler->TESS fine-tuning completed locally on 2026-06-18 using
  `device=mps`; `checkpoints/cnn_tess_finetuned/best.pt` SHA-256
  `3fc115b3623b2485373aefef30a7aa901e1183cc77ef4b57ce6c1f2219f49214`;
  best epoch 22, best validation AUC 0.8408.
- Production evaluation rejected that fine-tuned checkpoint: raw test AUC
  0.8115, raw test F1 0.7523, calibrated test F1 0.7508, calibrated Brier
  0.1966, and calibrated ECE 0.1152. It must not be promoted into `models/`.
- Path A TESS expansion was approved on 2026-06-18 as the next T1-1 strategy
  and started with an ExoFOP TOI/CTOI inventory against
  `data/tess_snippets_v2.jsonl`.
- The first Path A inventory completed locally on 2026-06-18 and found only 56
  new labeled TIC IDs (16 positive, 40 negative). This is too small to justify
  a long MAST fetch or candidate-12 training as a production-closing attempt.
- The historical ExoMAST TESS TCE endpoint used by `Skills/tess_tce_fetcher.py`
  returned HTTP 404 on 2026-06-18. The helper now reports `Flag: UNAVAILABLE`
  instead of hiding the stale provider behind a generic invalid/empty result.
- Architecture details: `docs/CNN_SPEC.md`.
- Human local runbook: `docs/CNN_PRODUCTION_RUNBOOK.md`.
- Source access smoke test passed end-to-end on 2026-07-02 with TAP schemas/rows,
  ExoFOP CSV, and Lightkurve Kepler/TESS searches verified. The next blocker is
  leakage-safe manifest and cleanup-path verification before any bulk download.
- Storage/source snapshot planning passed on 2026-07-02: committed metadata
  records source row counts, sample MAST product metadata, and an under-cap
  92,093,823,360-byte combined Kepler-long-cadence plus TESS estimate.

---

## Next Actions

1. Promotion tooling compatibility is complete: `Skills/promote_cnn_checkpoint.py`
   verifies the current `method: temperature` calibration JSON produced by
   `Skills/evaluate_cnn_checkpoint.py`, preserves legacy Platt support, accepts
   explicit `--model-id`, and prints the required intentional `git add -f`
   checkpoint staging step.
2. The evidence package is on GitHub:
   `models/benchmark_cnn_v1/MODEL_CARD.md`,
   `models/benchmark_cnn_v1/REPRODUCIBILITY_MANIFEST.json`, and
   `data_selection/data_role_registry.yaml` record the selected artifact scope,
   exact SHA-256 hashes, data roles, limitations, and `git add -f` requirement.
3. Human promotion approval was granted on 2026-07-09 for checkpoint SHA
   `f29e6891c255289fa1e2eddad1fb6ca131c063cf11c24b8113e0e29d049441c5` as
   `benchmark_cnn_v1`.
4. Selected checkpoint/calibration/config/metrics/manifest artifacts are copied
   into `models/cnn/benchmark_cnn_v1/`, and `models/registry.json` registers
   `benchmark_cnn_v1`.
5. After this promotion PR merges and post-merge CI is clean, start T1-2
   stacking calibration on a held-out
   calibration set; do not tune full-ensemble weights before the CNN artifact
   exists.

Remote sync note: local `main` is synced with `origin/main` as of the latest
handoff.


## Latest Local Validation

Validated on 2026-07-13:

```bash
.venv/bin/python Skills/run_quality_gates.py
```

Result: Ruff and mypy passed; six disjoint pytest shards × six xdist workers
passed 2,726 default tests in 34.1 seconds total. Two `integration_live` tests
remain excluded by default.

---

## Key Design Decisions In Effect

- Bayesian log-score model is default; XGBoost and ensemble are opt-in.
- `OptScore = float | None`: missing diagnostics contribute neutrally to log scores.
- Missing diagnostics fail threshold gates conservatively.
- Conservative priors keep false positives prominent; mission-specific prior profiles are opt-in through `configs/scoring_priors_v0.json`.
- `provenance_score` is computed from cadence, sector count, and pipeline quality.
- `toi_checker.py` should be consulted before investing pipeline time on any new target.
- Default tests must mock external services; live tests require `integration_live`.
- **Never output "confirmed planet"** — always "candidate signal" or "follow-up target". This is a hard guardrail, not a preference.
- Background automation uses SQLite for durable state and deterministic fixtures by default.
- Background automation obeys the human-approval gate; no external submission without review.

---

## Project Mission (clarified 2026-06-22)

This project **identifies transit candidates for human review**. It does not confirm discoveries and makes no discovery claims.

The intended workflow is:

```
Pipeline output (candidates + FPP + pathway)
    → Human review process (separate, owner-defined)
        → Submission of vetted candidates only
```

The pipeline's role ends at outputting a ranked candidate list with supporting diagnostics. What happens next — reviewing phase-fold plots, checking against known catalogs, consulting experts, deciding what to submit — is entirely outside this codebase and is the human operator's responsibility.

This framing matters for how we evaluate data quality and model performance:
- **TESS SPOC photometry** is the same public data professional astronomers use for transit detection. It is appropriate for identifying transit-like signals.
- **It is not sufficient for confirmation.** Confirmation requires RV measurements, high-resolution imaging, and/or spectroscopy — none of which this pipeline provides or implies.
- The CNN model's role is to reduce false positives in the candidate list, not to confirm planets. An AUC of 0.85 means the model ranks 85% of real transits above 85% of false positives. It is a triage tool.
- All submission pathway labels (`tfop_ready`, `planet_hunters_discussion`, etc.) describe what kind of follow-up is appropriate, not what has been found.
