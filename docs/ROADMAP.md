# ROADMAP

This roadmap is the executable, repo-specific projection of
`docs/astrometrics_coding_agents_master_guide.md`. The master guide controls
strategic phase ordering; this file records what this repository has actually
implemented and the remaining production work. A utility's existence does not
by itself satisfy a master-guide evidence requirement.

## Current Master-Guide Alignment — Production Priority Order

1. **Phase 1 — validated manifest contract (IN PROGRESS):** version 0.2.30 adds
   the shared schema, frozen Pydantic contract, fail-closed checksum/path
   validator, stable dataset IDs, and data-role-registry links for the committed
   Kepler/K2 production datasets. Require the same contract when the first TESS
   or JWST row-level production manifest is created; do not invent placeholder
   datasets merely to claim mission coverage.
2. **Phase 1 — reproducible candidate ledger:** `Skills/candidate_database.py`
   provides a basic SQLite history, but it does not yet carry the master-guide
   provenance fields (`source_dataset_id`, raw URI, preprocessing/generator
   versions and parameters, model versions/scores, injection context, review
   state, and regeneration command) or receive every production candidate.
3. **Phase 1 — canonical regression evals:** no committed sample-level suite
   yet covers confirmed planets, known false positives, and injected controls
   with before/after pipeline comparisons.
4. **Phase 2 — production sensitivity evidence:** generic BLS transit
   injection-recovery tooling exists, but committed recovery curves and
   sensitivity context tied to the production pipeline and
   `benchmark_cnn_v1` promotion package do not. Tooling-complete is not
   evidence-complete.
5. **Phase 2 — calibrated candidate context:** extend production outputs with
   calibration dataset IDs, score quantiles, threshold versions, and empirical
   false-discovery context after the canonical eval and sensitivity sets exist.
6. **Phase 3 only after 1–5:** benchmark self-supervised light-curve embeddings
   against BLS/tabular/`benchmark_cnn_v1`; do not start frontier representation
   work while Phase 1–2 evidence gaps remain.

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
- [ ] Produce and commit production-pipeline recovery curves on real-background
  canonical cases, with manifest IDs and explicit linkage to the frozen
  `benchmark_cnn_v1` evidence package (master-guide Phase 2 requirement)

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

- [ ] Calibrate full-ensemble weights only after the CNN checkpoint is promoted and a held-out calibration set is available; do not tune stacking weights on training or frozen-eval data.

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
