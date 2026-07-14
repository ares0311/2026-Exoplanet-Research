# ROADMAP

This roadmap is the executable, repo-specific projection of
`docs/astrometrics_coding_agents_master_guide.md`. The master guide controls
strategic phase ordering; this file records what this repository has actually
implemented and the remaining production work. A utility's existence does not
by itself satisfy a master-guide evidence requirement.

## Current Master-Guide Alignment — Production Priority Order

1. **Phase 1 — validated manifest contract (CONTRACT COMPLETE):** version 0.2.30 adds
   the shared schema, frozen Pydantic contract, fail-closed checksum/path
   validator, stable dataset IDs, and data-role-registry links for the committed
   Kepler/K2 production datasets. Require the same contract when the first TESS
   or JWST row-level production manifest is created; do not invent placeholder
   datasets merely to claim mission coverage.
2. **Phase 1 — reproducible candidate ledger (LIVE EVIDENCE + FIRST REVIEW COMPLETE):** versions 0.2.31, 0.2.34, 0.2.36, and 0.2.37
   add a strict, mission-neutral, append-only provenance contract and SQLite
   table carrying the master-guide fields (`source_dataset_id`, raw URI,
   preprocessing/generator versions and parameters, model versions/scores,
   injection context, review state, and regeneration command). Next, wire every
   production scan path to write this contract rather than relying on the
   legacy TESS-only convenience table. Version 0.2.34 corrects the contract
   for stitched light curves by preserving every exact archive product URI.
   Version 0.2.36 supplies the first real row-level live-search source:
   `tess_live_search_v1`, an immutable 18-target queue with 103 exact QLP
   product URIs and a checksum-validated dataset manifest. Version 0.2.37 wires
   the frozen queue to schema-v2 SQLite writes for candidates, scientifically
   useful null results, and preprocessing failures. Execution fails closed if
   the dataset checksum, target membership, product inventory, or exact fetched
   URI tuple differs; process shards receive collision-free log, database, and
   run-report paths. The three-shard live execution completed successfully on
   2026-07-11: 18/18 targets, 56 schema-v2 rows, three high-priority nulls, and
   zero failures. The committed run summary preserves local-ledger/log hashes
   and places three low-FPP signals on a conservative review queue. The first
   review is complete: two eclipse-scale signals are `likely_false_positive`;
   the remaining signal is `plausible_but_weak` because limb-darkening fails
   and centroid/odd-even evidence is missing. Version 0.2.39 adds the bounded,
   Run-Report-enabled TESS-SPOC target-pixel centroid diagnostic needed for
   that review. Its live run established that neither available TPF sector
   covers a predicted 97.16-day event; version 0.2.40 persists that no-coverage
   result rather than raising without evidence. Odd/even is likewise
   observationally underdetermined: only two independent QLP events exist and
   the diagnostic requires four. The requested follow-up is complete to the
   limit of available observations; do not add more ledger scaffolding or run
   another blind scan.
3. **Phase 1 — canonical regression evals (COMPLETE):** version 0.2.41 adds
   an offline sample-level suite with separate real-only and synthetic-inclusive
   frozen-evaluation roles. It preserves the accepted pi Mensae c and TOI-146.01
   controls, reruns deterministic deep/subthreshold injected-transit controls,
   and reports per-case deltas against the committed v1 baseline.
4. **Phase 2 — production sensitivity evidence (BOUNDED V1 + EXPANDED V2
   COMPLETE):** generic BLS transit
   injection-recovery tooling exists, and version 0.2.42 adds a bounded,
   cache-only real-background runner tied to the production pipeline,
   frozen-eval dataset ID, and `benchmark_cnn_v1` promotion package. A
   36-trial pre-merge smoke completed with zero failures in 7.1 seconds using
   six workers. The first merged run was failed closed because quarter-filtered
   inputs retained all-quarter provenance URIs; version 0.2.43 fixes and tests
   aligned path/curve filtering. The corrected merged-code run is now committed
   at `artifacts/manifests/production_sensitivity_v1.json`: 23/36 recoveries,
   zero failures, and curves by period, depth, duration, and real-background
   label. This closes the bounded short-period evidence deliverable, not general
   completeness. The v1 depth bins are non-monotonic under a 12-sample/bin,
   strongest-peak-only search and must not be interpreted as a survey
   completeness law.
   Version 0.2.46 commits the merged-code v2 artifact: 8/16 recovered in 7.98
   seconds with zero failures. Moderate TTV and both gap scenarios recovered;
   strong TTV, 90-day periodic, single-transit, and low-depth variability cases
   did not. These are measured boundary cases, not favorable assumptions.
5. **Phase 2 — calibrated candidate context (COMPLETE):** extend production
   outputs with calibration dataset IDs, score quantiles, threshold versions,
   and empirical false-discovery context now that canonical eval and sensitivity
   sets exist. Version 0.2.47 adds a validated reference contract, builder, CLI
   output wiring, and schema-v2 ledger preservation for the exact calibrated
   full-ensemble score. It deliberately emits null probability calibration and
   null decision threshold while reporting empirical K2 score rank and observed
   tail-negative fraction with numerator/denominator and domain limitations.
   Version 0.2.48 commits `models/candidate_context_v1.json`, built from merged
   code over all 588 reference rows with exact source hashes and a successful
   Run Report. No probability calibration or threshold is claimed.
6. **Phase 3 representation benchmark (BOUNDED PILOT COMPLETE; CNN GATE NOT
   MET):** version 0.2.49 adds a bounded masked-reconstruction Transformer
   pilot over the existing predefined KIC-grouped Kepler splits. Labels are
   hidden during training-only pretraining; the encoder is frozen before a
   linear probe; validation selects states; and the frozen test split opens
   once against a period/duration/flux-summary linear baseline and
   `benchmark_cnn_v1`. Version 0.2.50 commits the merged-code MPS result:
   embedding test AUC 0.832630/F1 0.635135/top-100 yield 72%; tabular test AUC
   0.823495/F1 0.274336/top-100 yield 6%; CNN AUC 0.957211/F1 0.834688. The
   compact pilot is rejected as a CNN replacement and must not be repeated
   unchanged. Phase 3 remains active for a materially different experiment
   backed by broad unlabeled Kepler/TESS data, stellar-variability labels,
   injection-recovery comparison, and external foundation-model baselines.
   Version 0.2.51 adds the cache-only source-inventory builder needed before any
   follow-up: the 41 GB Kepler cache contains only 11 KICs outside the labeled
   master corpus, while the 37 GB TESS cache contains 2,797 TICs outside local
   labeled corpora before live-role exclusion. No new archive download is
   authorized while the shared cache remains above the 80 GB caution threshold.
   Version 0.2.52 commits the merged-code result: 11,960 exact SPOC products
   across 2,790 eligible TICs and 84 sectors,
   totaling 29.79762048 GB already cached. The dataset is registered training-
   only. Next, benchmark streaming/bounded-sample preprocessing size and
   throughput before authorizing any derived arrays or new model training.
   Version 0.2.53 removes the six-terminal operational burden for future
   shard-capable acquisition: one fail-closed supervisor now launches the
   measured six shards × six workers, isolates logs, enforces storage/repo
   preflight, propagates shard failures, and serializes shard Run Report git
   operations. It does not authorize a new download or change the next Phase 3
   evidence gate. The same release replaces sequential local validation with a
   single-parent quality runner: six disjoint pytest file shards × six xdist
   workers execute beside Ruff and mypy, with per-gate logs and a combined
   failure status. Its optimized run passed 2,718 tests plus both static gates
   in 34.1s, about 58% faster than the earlier 81.57s xdist baseline.
   This is an operability improvement, not new scientific evidence.
   Version 0.2.54 adds the bounded preprocessing benchmark that the inventory
   gate requires: one parent supervises six Python shard subprocesses with six
   FITS workers each, opens a deterministic 36-product/36-TIC sample spanning
   sectors 1-98, filters `QUALITY == 0`, robustly normalizes and resamples to
   2,048 float32 bins in memory, then discards every derived array. It verifies
   the committed training-only inventory hash, paths, and file sizes, downloads
   nothing, and projects full-corpus time/size from measured results. Tooling
   and a 6/6 real-cache process smoke are complete. The merged-code 36-product
   run then passed 36/36 with zero failures/downloads/persisted arrays at 85.77
   products/s. It projects the full 11,960-product normalized-flux-only
   transform at 97.98 MB and 139.44 seconds, so future work should stream it
   rather than create a durable derived corpus. This closes the preprocessing
   measurement gate, not Phase 3: stellar-variability labels,
   injection-recovery comparison, and an external foundation-model baseline
   remain required before a materially broader representation experiment can
   support any production-model decision. The release gate passed 2,726 tests
   plus Ruff/mypy in 34.1 seconds.
   Version 0.2.55 freezes the external-baseline source contract before any
   dependency or weight download: Chronos-Bolt tiny and Astromer2 provide the
   bounded general time-series and astronomy-native controls at exact commits,
   hashes, and sizes. Three Python 3.14-compatible direct wheels plus both ONNX
   files total 56,036,648 bytes. The metadata-only verifier fails closed and
   downloads zero payload bytes. Next, run that verifier from merged `main` and
   commit its evidence artifact; only then may a separate bounded inference
   smoke consider optional dependency installation. Scientific Phase 3 still
   requires variability labels and injection-recovery comparison. The local
   release gate passed 2,733 tests plus Ruff/mypy in 31.1 seconds.
   The first merged verifier run failed closed because the default Python
   opener followed Hugging Face's 302 into Xet and lost the resolver's pinned
   commit/size/hash headers. Version 0.2.56 preserves the authoritative initial
   HEAD response, covers it with an offline regression test, and passes a live
   read-only header smoke. Its 6×6 gate passed 2,734 tests plus Ruff/mypy in
   26.1 seconds. The merged full verifier subsequently passed 7/7 operations in
   4.94 seconds, verifying all five pinned sources and 56,036,648 projected
   direct bytes with zero payload downloads (artifact SHA `5610bbb8…3042`, Run
   Report commit `ae4e659`). Source identity/footprint is complete. Next design
   a bounded inference smoke that measures ONNX memory/numerics before any
   optional dependency or weight installation; the scientific experiment also
   still requires variability labels and injection-recovery comparison.

## Milestone 1 — Scoring and Classification Engine ✓ COMPLETE

- [x] `schemas.py` — typed Pydantic data contracts
- [x] `features.py` — 35+ normalized feature extraction functions
- [x] `hypotheses.py` — Bayesian log-score models for 6 hypotheses
- [x] `priors.py` — versioned conservative default and mission-specific prior profiles
- [x] `scoring.py` — softmax posterior + configurable priors + FPP, detection confidence, novelty, habitability
- [x] `pathway.py` — submission pathway classifier (SCORING_MODEL.md §11)
- [x] CI via GitHub Actions (ruff → mypy → pytest)
- [x] `CLAUDE.md` — project context for AI coding agents

---

## Milestone 2 — Data Pipeline ✓ COMPLETE

- [x] `fetch.py` — query MAST via Lightkurve; return LightCurve + provenance metadata
- [x] `clean.py` — NaN removal, sigma-clip, normalization, detrending
- [x] `search.py` — BLS search → `CandidateSignal` list; iterative masking for multi-planet
- [x] `vet.py` — compute `RawDiagnostics` from light curve + signal; call `extract_features()`
- [x] `@pytest.mark.integration_live` tests against real MAST data

---

## Milestone 3 — End-to-End Validation ✓ COMPLETE

- [x] `notebooks/pipeline_demo.ipynb` — TOI-700 (TIC 150428135) full pipeline walkthrough
- [x] All 6 stages covered: Fetch → Clean → Search → Vet → Score → Classify
- [x] Human-readable candidate report rendered as Markdown in notebook
- [x] Figures: raw vs. cleaned flux, phase-folded transit, posterior bar chart, all-signals grid

---

## Milestone 4 — Calibration ✓ COMPLETE

- [x] `calibration.py` — reliability curves, Platt scaling (scipy), isotonic regression (PAVA)
- [x] One-vs-rest calibration per hypothesis; renormalized to sum to 1.0
- [x] Metrics: Brier scores, reliability curves, precision/recall/F1, confusion matrix
- [x] `Skills/train_xgboost.py` — includes post-training Platt calibration step

---

## Milestone 5 — Reporting ✓ COMPLETE

- [x] Rich-formatted candidate report via `exo <TIC-ID>` CLI
- [x] JSON output via `--output`
- [x] Scorer selection via `--scorer [bayesian|xgboost|ensemble|cnn|full-ensemble]`, `--model-path`, and `--cnn-checkpoint`

---

## Milestone 6 — Injection-Recovery Tooling ✓ COMPLETE

- [x] `Skills/injection_recovery.py` — inject synthetic box transits, recover via BLS
- [x] Measures recovery rate by radius, period, noise level
- [x] 25 tests in `tests/test_injection_recovery.py`
- [x] Produce and commit bounded production-pipeline recovery curves on real-background
  canonical cases, with manifest IDs and explicit linkage to the frozen
  `benchmark_cnn_v1` evidence package (`production_sensitivity_v1`)
- [x] Expand bounded sensitivity coverage to TTV, single-transit, data-gap,
   stellar-variability, multi-quarter, and longer-period cases before making a
  general completeness claim. Version 0.2.45 adds the explicit 16-trial v2
  scenario contract across two Q1-Q4 backgrounds, including event-overlap
  recovery semantics for single transits and cadence accounting for gaps. A
  cache-only merged-code run completed in 7.98 seconds with 8/16 recoveries and
  zero failures; durable evidence is `production_sensitivity_v2.json`. This
  checks scenario coverage, not general survey completeness.

---

## Milestone 7 — ML Ensemble Scorer ✓ COMPLETE

- [x] Tier 1 — XGBoost on tabular features (`ml/xgboost_scorer.py`, 45 tests)
- [x] Tier 3 — Stacking scorer blending XGBoost + CNN + Bayesian (`ml/stacking_scorer.py`, 22 tests)
- [x] Kepler training pipeline (`Skills/fetch_kepler_tce.py`, `build_training_data.py`, `train_xgboost.py`)
- [x] TESS training pipeline (`Skills/fetch_tess_toi.py`, `build_tess_training_data.py`)
- [x] Evaluation framework (`Skills/evaluate_scorer.py`, ROC-AUC, F1, reliability diagrams)
- [x] Combined training data (`Skills/build_combined_training_data.py`)
- [x] Offline CNN snippet split assembly (`Skills/build_cnn_training_data.py`, 13 tests)
- [x] Offline CNN split validation (`Skills/cnn_split_validator.py`, 15 tests)
- [x] Tier 2 scaffolding — CNN scorer wrapper, training loop, checkpoint/calibration helpers, phase-folded snippet wiring, and `cnn/full-ensemble` CLI modes
- [x] First Tier 2 candidate — trained and evaluated on the deterministic seed-42 split; rejected because held-out AUC was 0.7404 and calibration worsened Brier/ECE
- [x] T1-1 source-contract reset — verified public NASA/MAST sources, committed source snapshots, leakage-safe manifests, bounded raw-FITS cache policy, and storage estimates
- [x] Master Kepler corpus and checkpoint — trained `checkpoints/cnn_t1_1_kepler_master/best.pt` from the combined KOI+DR24 corpus; held-out gates passed with raw test AUC 0.9572, calibrated F1 0.8347, Brier 0.0580, ECE 0.0142, and T=1.0
- [x] Production Tier 2 promotion readiness — complete the new Astrometrics-policy evidence package before copying any checkpoint into `models/`
  - [x] Fix promotion tooling to accept the current temperature-scaling calibration JSON, not only legacy Platt fields, and print the required intentional checkpoint `git add -f`
  - [x] Add a model card for the master checkpoint
  - [x] Add a reproducibility manifest linking source snapshots, manifests, splits, config, calibration, metrics, SHA-256, runtime, and MPS/Python assumptions
  - [x] Add `data_selection/data_role_registry.yaml` for training, validation, calibration, and frozen-eval roles
  - [x] Mark the promoted architecture/data/preprocessing combination as the frozen `benchmark_cnn_v1` measuring stick
  - Preserve raw FITS as re-downloadable cache only; commit only selected production artifacts after explicit human approval
- [x] Human-approved CNN artifact promotion — approved on 2026-07-09; `benchmark_cnn_v1` is registered under `models/registry.json` and selected artifacts are promoted under `models/cnn/benchmark_cnn_v1/`
- [x] T1-2 stacking calibration — completed 2026-07-10 on 588 held-out K2 examples; calibrated weights (XGBoost=0.95/CNN=0.00/Bayesian=0.05) are wired into production

---

## Milestone 8 — Background Automation ✓ COMPLETE

- [x] `background/` module — SQLite-backed durable state (run ledger, reviewed/needs-follow-up logs, follow-up tests, reports, approvals)
- [x] `background/runner.py` — `background_run_once()` — one-shot scheduler-friendly invocation
- [x] `background/priority.py` — composite priority scoring (8 factors) with reason codes
- [x] `background/storage.py` — `BackgroundStore` with schema v2 tables for the run ledger, priority evaluations, outcomes, follow-up tests, reports, approvals, locks, and migrations
- [x] `background/reports.py` — draft Markdown/HTML reports; human-approval gate enforced
- [x] `background/fixtures.py` + `fixtures/known_tess_examples.json` — deterministic offline target pool
- [x] CLI subcommands: `exo background-run-once`, `run-summary`, `sqlite-integrity`, `target-priority-summary`, and 13 others
- [x] Scheduler docs (`docs/SCHEDULER.md`): cron, launchd, systemd timer examples
- [x] System profile (`docs/SYSTEM_PROFILE.md`): hardware sizing and batch-run defaults
- [x] `configs/background_search_v0.json` — versioned, fingerprinted configuration
- [x] 16 tests in `tests/test_background_automation.py`

---

## Milestone 8b — Star Scanner ✓ COMPLETE

- [x] `Skills/star_scanner.py` — `priority_score()`, `ScanLog` (JSON), `select_targets()` (TIC query), `scan_star()`, `run_background_scan()`
- [x] Priority scoring: Tmag (0.30), Teff/stellar type (0.25), sector coverage (0.25), contamination ratio (0.20)
- [x] TOI exclusion at startup; already-scanned exclusion via log; graceful Ctrl-C resume
- [x] 38 tests in `tests/test_star_scanner.py`

---

## Milestone 9a — Provenance Score ✓ COMPLETE

- [x] `compute_provenance_score(provenance: FetchProvenance) -> float` in `fetch.py`
- [x] Wired into `run_pipeline()` in `cli.py`; `provenance_score` included in JSON output rows
- [x] `tfop_ready` pathway now correctly enabled/blocked based on cadence, sector count, pipeline quality
- [x] 15 unit tests in `tests/test_fetch.py`; 4 flow tests in `tests/test_cli.py`
- [x] Documented in `docs/SCORING_MODEL.md §21`

---

## Milestone 9b — Candidate Ranking ✓ COMPLETE

- [x] `Skills/rank_candidates.py` — composite rank score weighting FPP, detection confidence, novelty, provenance, pathway
- [x] `load_candidates()`, `compute_rank_score()`, `rank_candidates()`, `print_rank_table()`
- [x] 12 tests in `tests/test_rank_candidates.py`

---

## Milestone 9c — Batch Scan ✓ COMPLETE

- [x] `Skills/batch_scan.py` — scan TIC ID lists from text/CSV; incremental JSON output with `--resume`
- [x] `read_tic_ids()`, `batch_scan()` with mock-injectable pipeline function
- [x] 14 tests in `tests/test_batch_scan.py`

---

## Milestone 9d — Sector Coverage ✓ COMPLETE

- [x] `Skills/sector_coverage.py` — query available TESS sectors per target without downloading data
- [x] `get_sector_coverage()`, `format_coverage_table()`; CLI with `--json` output
- [x] 10 tests in `tests/test_sector_coverage.py`

---

## Milestone 10a — Depth Scatter Chi-Square ✓ COMPLETE

- [x] `depth_scatter_chi2_score(depths, errors, chi2_threshold=3.0)` in `features.py`
- [x] New `depth_scatter_chi2_score: OptScore` field in `CandidateFeatures` (schemas.py)
- [x] Error-weighted reduced chi-square test complements existing robust-CV `depth_consistency_score`
- [x] Wired into `log_score_instrumental()` (+0.90 weight) and `log_score_planet()` (−0.60 weight)
- [x] 8 tests in `tests/test_features.py`; 5 tests in `tests/test_hypotheses.py`

---

## Milestone 10b — Phase-Fold Plots ✓ COMPLETE

- [x] `Skills/plot_lc.py` — `phase_fold()`, `plot_candidate()`, `plot_all()`
- [x] Generates PNG for each candidate row from `exo --output` JSON
- [x] No-op when matplotlib is absent; 11 tests in `tests/test_plot_lc.py`

---

## Milestone 10c — Watchlist + Summary Report ✓ COMPLETE

- [x] `Skills/watchlist.py` — atomic JSON watchlist; `add`, `remove`, `contains`, `list_ids`, `entries`, `clear`, `summary`; 13 tests
- [x] `Skills/summary_report.py` — `load_results`, `build_report`, `write_report`; partitions by status; candidates sorted by FPP; 14 tests

---

## Milestone 11a — Transit Timing Variation Score ✓ COMPLETE

- [x] `transit_timing_variation_score(midpoints, period_days, epoch_bjd, rms_threshold_minutes=10.0)` in `features.py`
- [x] New `transit_timing_variation_score: OptScore` field in `CandidateFeatures` (`schemas.py`)
- [x] O-C residuals in minutes; score = `clip(RMS_OC / threshold)` — saturates at threshold
- [x] Wired into `log_score_planet()` (−0.50 weight) and `log_score_instrumental()` (+0.60 weight)
- [x] 10 tests in `tests/test_features.py`; 3 tests in `tests/test_hypotheses.py`
- [x] Documented in `docs/SCORING_MODEL.md §22`

---

## Milestone 11b — TOI Checker + Export Candidates ✓ COMPLETE

- [x] `Skills/toi_checker.py` — `check_toi(tic_id)` queries ExoFOP TOI CSV; `format_toi_result()` one-liner status; handles column-name variations
- [x] `Skills/export_candidates.py` — `to_csv()`, `to_markdown_table()`, `to_summary_stats()`; 10-column export with display headers
- [x] 12 tests in `tests/test_toi_checker.py`; 13 tests in `tests/test_export_candidates.py`

---

## Milestone 11c — Alert Filter + Skills Guide ✓ COMPLETE

- [x] `Skills/alert_filter.py` — `filter_candidates()` AND-logic threshold filter (FPP, pathway, signals, rank, SNR); `apply_filters()` loads + writes JSON
- [x] `_fpp()` helper handles all dict shapes: `scores.false_positive_probability`, `best_fpp`, top-level
- [x] `docs/SKILLS_GUIDE.md` — workflow reference plus current inventory for 249 Skills; CLI examples; library usage pattern; ML pipeline walkthrough
- [x] 12 tests in `tests/test_alert_filter.py`

---

## Milestones 12-18 — Diagnostic And Operations Expansion ✓ COMPLETE

- [x] Milestone 12 diagnostic scores, CLI metadata, notebook generation, target prioritization, candidate comparison, timelines, and FITS helper utilities
- [x] Milestone 13 follow-up preparation, false-positive vetting, data quality, detrending comparison, recovery completeness, and HTML report utilities
- [x] Milestone 14 caching, period aliases, multi-planet checks, centroid analysis, catalog crossmatch, transit modeling, candidate database, follow-up scheduling, config management, alerts, and scorecards
- [x] Milestone 15 light-curve statistics, depth correction, nearby-star checks, binned exports, uncertainty, timing, candidate merging, multi-sector stacking, metadata, notes, TOI watching, contamination correction, benchmarking, and phase plots
- [x] Milestone 16 transit analysis and follow-up preparation utilities including radius, odd/even, secondary eclipse, momentum dump, duplicate TOI, activity, RV, impact parameter, observing request, ephemeris uncertainty, photometry combination, transit windows, labelled snippets, CNN augmentation, and report cards
- [x] Milestone 17 geometry, noise, period analysis, visibility, ground-truth matching, scatter metrics, centroid checks, and evidence aggregation utilities
- [x] Milestone 18 observability and analysis utilities including equilibrium temperature, TSM, airmass, moon separation, telescope time, false alarm probability, chi-square period checks, deduplication, run diffs, FITS exports, asymmetry scoring, trapezoid comparison, leaderboard, email formatting, and transmission-window prediction

---

## Future

- [x] Calibrate full-ensemble weights on a held-out set after CNN promotion — completed 2026-07-10 on 588 K2 examples; production weights are XGBoost=0.95, CNN=0.00, Bayesian=0.05.

---

## Milestone 19a — Multi-Sector Phase-Fold Comparison ✓ COMPLETE

- [x] `Skills/multi_sector_phase_compare.py` — offline per-sector phase-fold comparison for transit depth and phase centroid consistency
- [x] Flags insufficient coverage, weak/inverted signals, depth mismatches, and phase shifts conservatively
- [x] 12 tests in `tests/test_multi_sector_phase_compare.py`

---

## Milestone 19b — Static Candidate Dashboard Foundation ✓ COMPLETE

- [x] `docs/DASHBOARD_SPEC.md` — local-first dashboard data contract and guardrails
- [x] `Skills/candidate_dashboard_export.py` — static HTML dashboard from existing local candidate JSON rows
- [x] Preserves false-positive evidence, negative evidence, missing-score states, and blocking issues
- [x] Renders optional local phase-fold plot artifacts when supplied
- [x] 23 tests in `tests/test_candidate_dashboard_export.py`

---

## Milestone 19c — Local Read-Only Candidate API ✓ COMPLETE

- [x] `docs/API_SPEC.md` — local API contract, endpoints, guardrails, and non-goals
- [x] `Skills/candidate_api.py` — standard-library read-only HTTP API for local candidate JSON rows
- [x] Endpoints: `/health`, `/summary`, `/candidates`, `/candidates/<id>`, `/dashboard`, `/background/summary`, `/background/latest`
- [x] Static local review bundle endpoint: `/artifact.json`
- [x] Opt-in CORS headers for separate local frontends (`--cors-origin`)
- [x] Optional background SQLite summaries are read-only and do not create or mutate runtime databases
- [x] Candidate payloads carry optional phase-fold plot artifact paths
- [x] 33 tests in `tests/test_candidate_api.py`

---

## Milestone 19d — Interactive Local Candidate Browser ✓ COMPLETE

- [x] `Skills/candidate_browser_ui.py` — dependency-free browser UI for candidate review
- [x] Supports embedded-data mode for offline file viewing and API mode for `candidate_api.py`
- [x] Includes search, risk filtering, pathway filtering, summary metrics, detail panel, and optional plot previews
- [x] 20 tests in `tests/test_candidate_browser_ui.py`

---

## Milestone 20 — Slick Animated Command-Line UI

Unlocked by the formal production-ensemble acceptance PASS in version 0.2.33.
This milestone improves operator clarity without changing scientific scoring,
thresholds, or classifications.

- [ ] Add a polished terminal presentation for Fetch → Clean → Search → Vet → Score → Classify, with animated progress and useful elapsed-time/ETA status
- [ ] Preserve stable machine-readable JSON output, exit codes, redirected output, and non-TTY/CI behavior
- [ ] Provide an explicit `--no-animation`/reduced-motion path and graceful interruption/error rendering
- [ ] Add terminal-width, TTY/non-TTY, failure-path, and interruption regression tests
- [ ] Document the interactive and automation-safe CLI modes

---

## Decision Tree (current implementation)

```
known_object posterior ≥ 0.80      → known_object_annotation
FPP ≥ 0.70                         → github_only_reproducibility
transit_count < 2                  → planet_hunters_discussion

TESS:
  all 9 tfop conditions met        → tfop_ready
  detection_confidence ≥ 0.45      → planet_hunters_discussion
  otherwise                        → github_only_reproducibility

Kepler/K2:
  p_planet ≥ 0.65, novelty ≥ 0.70,
  FPP ≤ 0.35                       → kepler_archive_candidate
  otherwise                        → github_only_reproducibility
```

See `src/exo_toolkit/pathway.py` and `docs/SCORING_MODEL.md §11` for full threshold values.
