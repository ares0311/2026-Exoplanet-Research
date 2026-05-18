# CLAUDE.md — Claude Code Project Context

This file is read automatically by Claude Code at session start.
It contains the facts a coding agent needs to work productively without re-reading every document.

---

## Standing Rules

- **Skills directory**: Any standalone `.py` utility script created to perform a task (data processing, report generation, injection-recovery, etc.) must be saved in `Skills/` at the project root. Create the directory if it does not exist. This allows scripts to be discovered and reused across sessions rather than recreated.

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
| `scoring.py` | **done** | `test_scoring.py` (45) — invariants + weight-sensitivity tests |
| `pathway.py` | **done** | `test_pathway.py` (60) — parametric + all-branch coverage |
| `fetch.py` | **done** | `test_fetch.py` (55, 2 live) |
| `clean.py` | **done** | `test_clean.py` (39) |
| `search.py` | **done** | `test_search.py` (43) |
| `vet.py` | **done** | `test_vet.py` (47) |
| `calibration.py` | **done** | `test_calibration.py` (70) — now includes `save_calibration`/`load_calibration` |
| `cli.py` | **done** | `test_cli.py` (50) — version flag, meta output, calibration integration |
| `ml/xgboost_scorer.py` | **done** | `test_xgboost_scorer.py` (45) |
| `ml/stacking_scorer.py` | **done** | `test_stacking_scorer.py` (22) |
| `background/` module | **done** | `test_background_automation.py` (16) |

**Total passing tests: ~2215 (+ 2 integration_live; 6 skipped without matplotlib)**
**Skills: `injection_recovery.py` (25), `fetch_kepler_tce.py`, `fetch_tess_toi.py` (11), `build_training_data.py` (34), `build_tess_training_data.py` (38), `build_combined_training_data.py` (13), `train_xgboost.py` (25), `evaluate_scorer.py` (14), `count_tess_labels.py`, `star_scanner.py` (38), `rank_candidates.py` (12), `batch_scan.py` (14), `sector_coverage.py` (10), `plot_lc.py` (11), `watchlist.py` (13), `summary_report.py` (14), `toi_checker.py` (12), `export_candidates.py` (13), `alert_filter.py` (12), `notebook_generator.py` (10), `target_prioritizer.py` (12), `compare_candidates.py` (11), `candidate_timeline.py` (12), `fits_header_extractor.py` (12), `ephemeris_predictor.py` (12), `stellar_params_fetcher.py` (12), `false_positive_vetter.py` (12), `sector_gap_finder.py` (13), `keplerian_fit.py` (11), `data_quality_checker.py` (12), `bulk_priority_update.py` (12), `multi_target_report.py` (13), `detrending_comparator.py` (12), `recovery_completeness_map.py` (12), `candidate_html_export.py` (13), `tess_year_planner.py` (11), `lightcurve_cache.py` (12), `period_alias_checker.py` (13), `multi_planet_checker.py` (11), `centroid_analyzer.py` (10), `catalog_crossmatch.py` (13), `transit_modeler.py` (12), `candidate_database.py` (12), `follow_up_scheduler.py` (13), `config_manager.py` (12), `signal_statistics.py` (11), `stellar_rotation.py` (12), `archive_lookup.py` (12), `vetting_scorecard.py` (15), `period_recovery_validator.py` (11), `alert_webhook.py` (13), `lc_statistics.py` (13), `transit_depth_corrector.py` (11), `nearby_star_checker.py` (12), `binned_lc_exporter.py` (11), `bootstrap_uncertainty.py` (12), `transit_timing_fitter.py` (12), `candidate_merger.py` (12), `multi_sector_stacker.py` (12), `target_metadata_fetcher.py` (13), `run_summary_exporter.py` (12), `candidate_notes.py` (13), `toi_watcher.py` (12), `flux_contamination_corrector.py` (11), `pipeline_benchmark.py` (12), `phase_plot_generator.py` (13), `planet_radius_estimator.py` (15), `odd_even_analyzer.py` (10), `secondary_eclipse_mapper.py` (11), `momentum_dump_flagger.py` (11), `duplicate_toi_detector.py` (13), `stellar_activity_filter.py` (13), `rv_semiamplitude_estimator.py` (13), `impact_parameter_refiner.py` (12), `obs_request_formatter.py` (12), `ephemeris_uncertainty_growth.py` (11), `multi_night_photometry_combiner.py` (11), `transit_window_extractor.py` (11), `labelled_lc_collector.py` (13), `cnn_feature_augmenter.py` (12), `candidate_report_card.py` (15), `limb_darkening_calculator.py` (18), `transit_duration_calculator.py` (13), `period_doubling_checker.py` (8), `stellar_density_calculator.py` (13), `eb_classifier.py` (13), `snr_estimator.py` (11), `phase_coverage_checker.py` (12), `photon_noise_estimator.py` (13), `harmonic_period_analyzer.py` (10), `tess_visibility_checker.py` (12), `ground_truth_matcher.py` (13), `transit_geometry_calculator.py` (13), `scatter_metric_calculator.py` (13), `pixel_level_centroid_checker.py` (11), `candidate_evidence_aggregator.py` (17), `equilibrium_temperature_calculator.py` (12), `tsm_calculator.py` (12), `airmass_calculator.py` (12), `moon_separation_checker.py` (11), `telescope_time_estimator.py` (12), `false_alarm_probability_estimator.py` (13), `chi_square_period_checker.py` (13), `candidate_deduplicator.py` (12), `pipeline_run_diff.py` (13), `fits_lightcurve_exporter.py` (15), `transit_asymmetry_scorer.py` (11), `trapezoid_box_comparator.py` (13), `leaderboard_generator.py` (14), `batch_email_formatter.py` (14), `transmission_window_predictor.py` (17)**

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
| `storage.py` | `BackgroundStore` — SQLite with 6 tables; schema v2 with v1→v2 migration |
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
- CLI: `python Skills/rank_candidates.py results/*.json --top 10 [--json]`
- 12 tests in `tests/test_rank_candidates.py`

---

## Batch Scan (`Skills/batch_scan.py`)

Scans a list of TIC IDs from a text or CSV file, writing incremental JSON results.

- `read_tic_ids(path)` — parse TIC IDs from plain text or CSV (skips comments, headers)
- `batch_scan(tic_ids, *, output_path, resume, run_pipeline_fn, ...)` — calls `run_pipeline` per target; writes after each result; `--resume` skips already-completed IDs
- Status per entry: `"candidate_found"` | `"scanned_clear"` | `"error"`
- CLI: `python Skills/batch_scan.py targets.txt --output results.json [--resume]`
- 14 tests in `tests/test_batch_scan.py`

---

## Sector Coverage (`Skills/sector_coverage.py`)

Queries MAST for which TESS sectors are available for a target without downloading data.

- `get_sector_coverage(target_id, *, pipeline, search_fn)` → `SectorCoverage`
- `format_coverage_table(coverages)` → plain-text table
- CLI: `python Skills/sector_coverage.py TIC 150428135 [--pipeline QLP] [--json]`
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
- CLI: `python Skills/watchlist.py add/remove/list/clear/summary`
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

- `exo --version` / `exo -V` — prints `exo-toolkit 0.1.0` (eager Typer callback)
- `__version__ = "0.1.0"` in `src/exo_toolkit/__init__.py` (importlib.metadata fallback)
- Each output row gains a `"meta"` dict: `toolkit_version`, `run_at`, `scorer`, `git_commit`, `features_available`
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
- **Conservative priors**: planet_candidate = 0.10, others = 0.20 each, known_object = 0.10
- **ML Tier 1 (XGBoost) is built** — `ml/xgboost_scorer.py` ships as an optional alternative scorer; Bayesian log-score model remains the default fallback when labels are unavailable
- **ML Tier 3 (stacking) is built** — `ml/stacking_scorer.py` blends XGBoost + Bayesian P(planet) with configurable weight; falls back to Bayesian-only when no model loaded
- **CLI scorer options**: `exo <TIC-ID> --scorer [bayesian|xgboost|ensemble] --model-path <path>`
- **Never output "confirmed planet"** — use "candidate signal" or "follow-up target"
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
CandidateFeatures    # 35 OptScore fields, all default None
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
    → softmax()                 (scoring.py)
    → HypothesisPosterior
    → compute_scores()          (scoring.py)
    → CandidateScores

Public entry point: score_candidate(signal, features, log_priors=None)
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
`provenance_score` defaults to 0.0 (blocks `tfop_ready` in v0 — not yet computed).

---

## Quality Commands

```bash
# Run tests (package not pip-installed — use PYTHONPATH)
PYTHONPATH=src python -m pytest

# Lint
ruff check .
ruff check . --fix

# Type-check (must use python -m mypy so stubs from site-packages are visible)
python -m mypy src

# All three together
ruff check . && python -m mypy src && PYTHONPATH=src python -m pytest
```

If pytest fails with `ModuleNotFoundError: No module named 'exo_toolkit'`, add `PYTHONPATH=src`.

`mypy` (bare binary) sees a different package path and reports false import errors for pydantic/numpy.
Always use `python -m mypy src` locally.

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

## What Is Not Yet Built

All pipeline modules are complete.

### Completed (2026-05-08)

**End-to-end example notebook** (`notebooks/pipeline_demo.ipynb`): ✅
- Target: TOI-700 (TIC 150428135) — M-dwarf with confirmed habitable-zone planet
- 25 cells covering all 6 pipeline stages with prose, code, and figures
- Figures: raw vs. cleaned flux, phase-folded transit, posterior bar chart, all-signals grid
- Human-readable candidate report rendered as Markdown inside the notebook

**Injection-recovery completeness mapping** (`Skills/injection_recovery.py`): ✅
- Injects synthetic box transits into real or simulated light curves
- Recovers via `search_lightcurve`; measures recovery rate vs. period and depth
- Usable as CLI script (`python Skills/injection_recovery.py`) or importable library
- 25 tests in `tests/test_injection_recovery.py`

**CLI entry point** (`src/exo_toolkit/cli.py`): ✅
- `exo <TIC-ID>` — runs full pipeline and prints Rich-formatted candidate report
- Options: `--mission`, `--min-snr`, `--max-peaks`, `--output` (JSON)
- Entry point registered in `pyproject.toml` as `exo = "exo_toolkit.cli:app"`
- 14 tests in `tests/test_cli.py`

**ML Ensemble Scorer — Tier 1: XGBoost** (`src/exo_toolkit/ml/xgboost_scorer.py`): ✅
- Binary XGBoost classifier (planet candidate vs false positive) on 35 OptScore fields
- Native XGBoost API (`xgb.DMatrix` + `xgb.train`) — no sklearn dependency
- `None` OptScores → `np.nan`; handled natively by XGBoost missing-value splitting
- Model serialised as paired metadata JSON + `.xgb.json` files
- 45 tests in `tests/test_xgboost_scorer.py`

**ML Ensemble Scorer — Tier 3: Stacking scorer** (`src/exo_toolkit/ml/stacking_scorer.py`): ✅
- Weighted average of XGBoost + Bayesian P(planet_candidate); weight configurable
- Falls back to Bayesian-only when no XGBoost model available
- `StackingScorer.from_model_path(path)` / `StackingScorer.bayesian_only()` factory methods
- 22 tests in `tests/test_stacking_scorer.py`

**Kepler training pipeline** (`Skills/`): ✅
- `fetch_kepler_tce.py` — downloads KOI cumulative table from NASA Exoplanet Archive
- `build_training_data.py` — maps 8 KOI columns → CandidateFeatures; 27 remain None
- `train_xgboost.py` — stratified k-fold CV, ROC-AUC/F1 metrics, saves final model
- 34 + 17 tests in `tests/test_build_training_data.py`, `tests/test_train_xgboost.py`

**TESS TOI training pipeline** (`Skills/`): ✅
- `fetch_tess_toi.py` — downloads TESS TOI table (CP/FP/EB) from ExoFOP-TESS
- `build_tess_training_data.py` — maps 5 TOI columns → CandidateFeatures; 30 remain None
- 38 tests in `tests/test_build_tess_training_data.py`

**Scorer evaluation** (`Skills/evaluate_scorer.py`): ✅
- Stratified k-fold cross-validation comparing Bayesian vs XGBoost ROC-AUC, F1, precision, recall
- Optional ROC curve PNG and reliability (calibration) diagram export (requires matplotlib)
- 14 tests in `tests/test_evaluate_scorer.py`

**CLI scorer options** (`src/exo_toolkit/cli.py`): ✅
- `--scorer [bayesian|xgboost|ensemble]`, `--model-path <path>`
- xgboost adds `xgb_planet_probability`; ensemble adds `ensemble_planet_probability`
- 20 tests in `tests/test_cli.py`

**ML Scoring Architecture docs** (`docs/ML_SCORING.md`): ✅
- Documents all scorer modes, training pipeline, column mappings, design decisions

**Platt calibration in training** (`Skills/train_xgboost.py`): ✅
- Collects out-of-fold predictions during k-fold CV
- Fits Platt scaling (A, B) via log-loss minimization (scipy Nelder-Mead)
- Saves `platt_calibration: {a, b}` to the model metadata JSON
- 25 tests in `tests/test_train_xgboost.py`

**Combined training dataset** (`Skills/build_combined_training_data.py`): ✅
- Merges Kepler KOI + TESS TOI training pickles
- Optional per-source cap with stratified subsampling
- 16 tests in `tests/test_build_combined_training_data.py`

**fetch_tess_toi offline tests** (`tests/test_fetch_tess_toi.py`): ✅
- 11 unit tests using monkeypatch to avoid live HTTP calls

**CNN Tier-2 gate tools**: ✅
- `Skills/count_tess_labels.py` — queries ExoFOP CP count, prints gate status
- `docs/CNN_SPEC.md` — full architecture spec (1D CNN, input format, training params)
- `docs/DATA_SOURCES.md` — MAST, ExoFOP, NExSci endpoints and caching guide
- `Skills/__init__.py` — makes Skills a proper Python package
- `docs/ROADMAP.md` + `docs/PROJECT_STATUS.md` — updated to current state

### Completed (2026-05-17) — Milestone 13

**12 new Skills + calibration integration + extended tests**: ✅

| Skill | Key Functions | Tests |
|---|---|---|
| `ephemeris_predictor.py` | `predict_transits`, `format_transit_table` | 12 |
| `stellar_params_fetcher.py` | `fetch_stellar_params`, `StellarParams.to_vet_kwargs` | 12 |
| `false_positive_vetter.py` | `vet_candidate`, `format_vetting_report` | 12 |
| `sector_gap_finder.py` | `find_sector_gaps`, `format_gap_report` | 12 |
| `keplerian_fit.py` | `fit_trapezoid`, `trapezoid_model` | 11 |
| `data_quality_checker.py` | `check_data_quality`, `format_quality_report` | 12 |
| `bulk_priority_update.py` | `update_priorities` (atomic write) | 12 |
| `multi_target_report.py` | `build_multi_target_report`, `write_multi_target_report` | 13 |
| `detrending_comparator.py` | `compare_detrending` (SG window selection by SNR) | 12 |
| `recovery_completeness_map.py` | `build_completeness_map`, `save_completeness_map`, `load_completeness_map` | 12 |
| `candidate_html_export.py` | `to_html_gallery`, `write_html_gallery` | 13 |
| `tess_year_planner.py` | `plan_sectors`, `format_sector_plan` | 11 |

**Calibration integration in `run_pipeline()`** (`src/exo_toolkit/cli.py`):
- New `calibration_path: Path | None = None` parameter
- When provided, applies `load_calibration()` + `apply_calibration()` from `calibration.py`
- Each output row gains `"calibrated_posterior"` dict (same 6-key structure as `"posterior"`)
- `calibration.py` gains `save_calibration(result, path)` and `load_calibration(path)` helpers

**Extended pathway tests** (`test_pathway.py`): +15 tests covering all 6 return values explicitly

### Completed (2026-05-17) — Milestone 14

**15 new Skills + 201 new tests**: ✅

| Skill | Key Functions | Tests |
|---|---|---|
| `lightcurve_cache.py` | `LightcurveCache.save/load/contains/clear`, `cache_key` | 12 |
| `period_alias_checker.py` | `check_period_alias`, `format_alias_result` | 13 |
| `multi_planet_checker.py` | `check_for_additional_planets`, `format_multi_planet_result` | 11 |
| `centroid_analyzer.py` | `analyze_centroid`, `format_centroid_result` | 10 |
| `catalog_crossmatch.py` | `crossmatch`, `format_crossmatch` | 13 |
| `transit_modeler.py` | `fit_transit_model`, `transit_model`, `format_model_result` | 12 |
| `candidate_database.py` | `CandidateDatabase.insert/latest/history/all_latest/export_csv` | 12 |
| `follow_up_scheduler.py` | `build_schedule`, `format_schedule` | 13 |
| `config_manager.py` | `load_config`, `validate_config`, `default_config` | 12 |
| `signal_statistics.py` | `compute_signal_stats`, `format_signal_stats` | 11 |
| `stellar_rotation.py` | `detect_rotation`, `format_rotation_result` | 12 |
| `archive_lookup.py` | `check_archive`, `format_archive_status` | 12 |
| `vetting_scorecard.py` | `build_scorecard`, `format_scorecard` | 15 |
| `period_recovery_validator.py` | `validate_period`, `format_validation_result` | 11 |
| `alert_webhook.py` | `build_alert_payload`, `send_alert`, `format_slack_payload` | 13 |

### Completed (2026-05-17) — Milestone 15

**15 new Skills + 177 new tests**: ✅

| Skill | Key Functions | Tests |
|---|---|---|
| `lc_statistics.py` | `compute_lc_stats`, `format_lc_stats` (CDPP, RMS, photon noise) | 13 |
| `transit_depth_corrector.py` | `correct_transit_depth`, `format_depth_correction` | 11 |
| `nearby_star_checker.py` | `check_nearby_stars`, `format_nearby_result` | 12 |
| `binned_lc_exporter.py` | `bin_lightcurve`, `export_binned_lc`, `load_binned_lc` | 11 |
| `bootstrap_uncertainty.py` | `bootstrap_uncertainty`, `format_bootstrap_result` | 12 |
| `transit_timing_fitter.py` | `fit_transit_times`, `format_timing_result` | 12 |
| `candidate_merger.py` | `merge_candidates`, `write_merged`, `format_merge_summary` | 12 |
| `multi_sector_stacker.py` | `stack_sectors`, `format_stack_summary` | 12 |
| `target_metadata_fetcher.py` | `fetch_target_metadata`, `format_target_metadata` | 13 |
| `run_summary_exporter.py` | `build_run_summary`, `write_run_summary`, `format_run_summary` | 12 |
| `candidate_notes.py` | `CandidateNotes.add/get/remove/search/summary` | 13 |
| `toi_watcher.py` | `watch_toi_list`, `format_watch_result` | 12 |
| `flux_contamination_corrector.py` | `correct_flux_contamination`, `format_contamination_result` | 11 |
| `pipeline_benchmark.py` | `benchmark_pipeline`, `format_benchmark_result` | 12 |
| `phase_plot_generator.py` | `generate_phase_plot`, `format_plot_result` | 13 |

### Completed (2026-05-18) — Milestone 16

**15 new Skills + 183 new tests**: ✅

| Skill | Key Functions | Tests |
|---|---|---|
| `planet_radius_estimator.py` | `estimate_planet_radius`, error propagation, `_classify` | 15 |
| `odd_even_analyzer.py` | `analyze_odd_even`, `_weighted_mean_err` | 10 |
| `secondary_eclipse_mapper.py` | `map_secondary_eclipse`, grid-search at phase 0.5 | 11 |
| `momentum_dump_flagger.py` | `flag_momentum_dumps`, periodic/explicit dump detection | 11 |
| `duplicate_toi_detector.py` | `detect_duplicate_toi`, period alias matching | 13 |
| `stellar_activity_filter.py` | `filter_stellar_activity`, `apply_activity_mask` | 13 |
| `rv_semiamplitude_estimator.py` | `estimate_rv_semiamplitude`, numerical error propagation | 13 |
| `impact_parameter_refiner.py` | `refine_impact_parameter`, Seager & MO (2003) geometry | 12 |
| `obs_request_formatter.py` | `build_obs_request`, RA/Dec sexagesimal, JSON payload | 12 |
| `ephemeris_uncertainty_growth.py` | `project_ephemeris_uncertainty`, σ_T(n) linear propagation | 11 |
| `multi_night_photometry_combiner.py` | `combine_photometry_nights`, per-night normalisation | 11 |
| `transit_window_extractor.py` | `extract_transit_windows`, per-transit + OOT arrays | 11 |
| `labelled_lc_collector.py` | `extract_snippet`, `build_dataset`, phase-fold+bin for CNN | 13 |
| `cnn_feature_augmenter.py` | `augment_snippet`, `augment_dataset`, noise/shift/scale/reverse | 12 |
| `candidate_report_card.py` | `build_report_card`, `save_report_card`, integrated vetting card | 15 |

### Completed (2026-05-18) — Milestone 17

**15 new Skills + 198 new tests**: ✅

| Skill | Key Functions | Tests |
|---|---|---|
| `limb_darkening_calculator.py` | `compute_limb_darkening`, bilinear Claret (2011) grid | 18 |
| `transit_duration_calculator.py` | `compute_transit_duration`, T14/T23/ingress-egress (Seager & MO) | 13 |
| `period_doubling_checker.py` | `check_period_doubling`, P/2 signal search | 8 |
| `stellar_density_calculator.py` | `compute_stellar_density`, photometric ρ★ from T14 | 13 |
| `eb_classifier.py` | `classify_eb`, heuristic EB probability from depth/odd-even/secondary | 13 |
| `snr_estimator.py` | `estimate_snr`, per-transit and combined SNR | 11 |
| `phase_coverage_checker.py` | `check_phase_coverage`, bin coverage fraction and gap phases | 12 |
| `photon_noise_estimator.py` | `estimate_photon_noise`, TESS photon/read/systematic noise model | 13 |
| `harmonic_period_analyzer.py` | `analyze_harmonics`, integer harmonic/sub-harmonic depth search | 10 |
| `tess_visibility_checker.py` | `check_tess_visibility`, ecliptic-coordinate sector model | 12 |
| `ground_truth_matcher.py` | `match_ground_truth`, period+epoch catalog matching | 13 |
| `transit_geometry_calculator.py` | `compute_transit_geometry`, Rp/R★, a/R★, inclination | 13 |
| `scatter_metric_calculator.py` | `compute_scatter_metrics`, RMS/MAD/CDPP/point-to-point | 13 |
| `pixel_level_centroid_checker.py` | `check_pixel_centroid`, in-transit vs OOT centroid shift | 11 |
| `candidate_evidence_aggregator.py` | `aggregate_evidence`, multi-diagnostic weighted evidence summary | 17 |

### Completed (2026-05-18) — Milestone 18

**15 new Skills + 206 new tests**: ✅

| Skill | Key Functions | Tests |
|---|---|---|
| `equilibrium_temperature_calculator.py` | `compute_equilibrium_temperature`, T_eq = Teff*(R/2a)^0.5*(1-AB)^0.25*f^0.25 | 12 |
| `tsm_calculator.py` | `compute_tsm`, Chen&Kipping M-R, Kempton+2018 TSM/ESM | 12 |
| `airmass_calculator.py` | `compute_airmass`, `compute_airmass_curve`, GMST-based LST | 12 |
| `moon_separation_checker.py` | `check_moon_separation`, Meeus low-precision lunar model | 11 |
| `telescope_time_estimator.py` | `estimate_telescope_time`, photon-noise SNR quadratic solver | 12 |
| `false_alarm_probability_estimator.py` | `estimate_fap`, Baluev (2008) analytic + empirical | 13 |
| `chi_square_period_checker.py` | `check_chi_square_period`, F-test via Lentz continued fraction | 13 |
| `candidate_deduplicator.py` | `deduplicate_candidates`, period+epoch+sky combined similarity | 12 |
| `pipeline_run_diff.py` | `diff_pipeline_runs`, ADDED/REMOVED/IMPROVED/DEGRADED/STABLE | 13 |
| `fits_lightcurve_exporter.py` | `export_lightcurve_to_fits`, injectable write_fn, BinTableHDU | 15 |
| `transit_asymmetry_scorer.py` | `score_transit_asymmetry`, ingress/egress weighted-mean comparison | 11 |
| `trapezoid_box_comparator.py` | `compare_trapezoid_box`, Δχ² + ΔBIC grid-search over ingress_frac | 13 |
| `leaderboard_generator.py` | `generate_leaderboard`, target/contributor modes, composite score | 14 |
| `batch_email_formatter.py` | `format_batch_email`, `format_single_candidate_email`, plain+HTML | 14 |
| `transmission_window_predictor.py` | `predict_transit_windows`, injectable airmass_fn + moon_fn | 17 |

### Next Step

**ML Ensemble Scorer — Tier 2: 1D CNN on phase-folded flux**
- Requires 5,000+ TESS labels before building
- Gate check: `python Skills/count_tess_labels.py`
- Architecture spec: `docs/CNN_SPEC.md`
- Training data infrastructure now available via `labelled_lc_collector.py` + `cnn_feature_augmenter.py`

### Future Enhancement: ML Ensemble Scorer (agreed 2026-05-01)

Replace or augment the Bayesian log-score model with a three-model ensemble once sufficient labeled data is available:

**Tier 1 — XGBoost on tabular features (build first)** ✅ DONE
**Tier 2 — 1D CNN on phase-folded flux (add after 5,000+ TESS labels)**
- Input: phase-folded, normalized flux array (treat as 1D image)
- Learns transit morphology directly; proven architecture (Shallue & Vanderburg 2018)
- Requires TESS-specific fine-tuning — Kepler-trained models are miscalibrated on TESS due to cadence, pixel scale, and systematics differences

**Tier 3 — Stacking meta-learner**
- Logistic regression (or simple average) over outputs of XGBoost + CNN + existing Bayesian log-score model
- Requires a separate held-out calibration set (~500+ examples)
- Final probabilities pass through existing `calibration.py` (Platt / isotonic)

**Architecture fit**
- XGBoost and CNN sit alongside `scoring.py`; stacking layer blends their posteriors
- Bayesian log-score model remains as fallback when features are missing (`None` scores)
- `calibration.py` handles final probability calibration for all model variants
- Label quality caveat: "candidate" KOIs are noisy labels; train only on confirmed planets vs. confirmed false positives where possible

---

## Guardrails (SCORING_MODEL.md §15)

- Never output "confirmed planet"
- Always expose false-positive evidence
- Suppress formal submission if key diagnostics are missing
- Store scoring model version with every candidate output
- Prefer conservative classifications

---

## Data Sources

- **TESS**: MAST via Lightkurve (`mission="TESS"`, PDCSAP flux preferred)
- **Kepler/K2**: MAST via Lightkurve (`mission="Kepler"` / `"K2"`)
- **Catalogs**: NASA Exoplanet Archive, TOI list, KOI list, CTOI via astroquery

Focus on lightly-worked targets: later TESS sectors, fainter stars (Tmag 10–14), less-crowded fields.
