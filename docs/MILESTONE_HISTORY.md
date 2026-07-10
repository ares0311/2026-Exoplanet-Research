# MILESTONE HISTORY

This file is a verbatim archive of the per-Skill Milestone changelog that
used to live inline in `CLAUDE.md`. It was moved here unedited (2026-07-10)
as part of a de-bloat pass on the project's directive files — every session
was re-reading ~400 lines of historical "Skill X added, N tests" tables that
have no bearing on current priorities. Nothing below was summarized or
deleted; it is relocated so `CLAUDE.md` stays focused on current architecture
and state.

For the current, authoritative "what shipped" summary at the right altitude
(one line per milestone), see `docs/PROJECT_STATUS.md`. For a workflow-oriented
(not exhaustive) Skills quick reference, see `docs/SKILLS_GUIDE.md`. For the
authoritative current Skill count and file list, run
`rg --files Skills -g '*.py' | sort`.

---

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
- Usable as CLI script (`.venv/bin/python Skills/injection_recovery.py`) or importable library
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
- Weighted average of XGBoost + CNN + Bayesian P(planet_candidate); weights configurable
- Falls back conservatively when optional XGBoost or CNN models are unavailable
- `StackingScorer.from_model_paths(...)` / `StackingScorer.bayesian_only()` factory methods
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
- `--scorer [bayesian|xgboost|ensemble|cnn|full-ensemble]`, `--model-path <path>`, `--cnn-checkpoint <path>`
- xgboost adds `xgb_planet_probability`; ensemble adds `ensemble_planet_probability`; CNN modes build phase-folded snippets when available and add experimental CNN/full-ensemble metadata when a checkpoint path is supplied
- 54 tests in `tests/test_cli.py`

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
- `Skills/count_tess_labels.py` — queries ExoFOP CP count, prints gate status, and writes `logs/tess_label_check.sqlite3`
- `Skills/tess_label_check_summary.py` — read-only local summary of the live-check SQLite audit log
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
- Each output row gains a `"calibrated_posterior"` dict (same 6-key structure as `"posterior"`)
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
| `stellar_rotation.py` | `detect_rotation`, `format_rotation_result` | 14 |
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
| `build_cnn_training_data.py` | `load_training_examples`, `write_training_splits`, offline train/val/test split assembly | 13 |
| `cnn_split_validator.py` | `validate_split_dir`, `validate_split_manifest`, offline split artifact validation | 15 |
| `candidate_dashboard_export.py` | `build_dashboard`, `write_dashboard`, conservative review dashboard | 15 |

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

### Completed (2026-05-19) — Milestone 19a

**1 new Skill + 12 new tests**: ✅

| Skill | Key Functions | Tests |
|---|---|---|
| `multi_sector_phase_compare.py` | `compare_sector_phase_folds`, `format_phase_comparison` | 12 |

### Completed (2026-05-19) — Milestone 19b

**1 new Skill + 23 tests**: ✅

| Skill | Key Functions | Tests |
|---|---|---|
| `candidate_dashboard_export.py` | `build_dashboard`, `write_dashboard`, `load_dashboard_rows`, optional phase-fold plot artifacts | 23 |

### Completed (2026-05-20) — Milestone 19c

**1 new Skill + 27 tests**: ✅

| Skill | Key Functions | Tests |
|---|---|---|
| `candidate_api.py` | `CandidateAPI`, `api_response`, `summary_payload`, `background_summary_payload`, `artifact_payload`, opt-in CORS headers | 33 |

### Completed (2026-05-20) — Milestone 19d

**1 new Skill + 20 tests**: ✅

| Skill | Key Functions | Tests |
|---|---|---|
| `candidate_browser_ui.py` | `build_browser_ui`, `write_browser_ui`, optional plot previews | 20 |

### Completed (2026-05-22) — Milestone 20

**15 new Skills + 201 new tests**: ✅

| Skill | Key Functions | Tests |
|---|---|---|
| `tess_sector_map.py` | `get_sector_map`, `format_sector_map` — ecliptic-coord sector model | 14 |
| `period_grid_search.py` | `search_period_grid`, `format_period_grid_result` — BLS power profile | 13 |
| `oot_rms_tracker.py` | `track_oot_rms`, `format_oot_rms_result` — per-sector OOT RMS + outlier flag | 13 |
| `phase_bin_snr.py` | `compute_phase_bin_snr`, `format_phase_bin_snr_result` — phased SNR profile | 12 |
| `centroid_offset_mapper.py` | `map_centroid_offsets`, `format_centroid_offset_result` — in-transit centroid shift | 13 |
| `tce_comparison_report.py` | `compare_tce`, `format_tce_comparison` — TCE table cross-match | 13 |
| `stellar_contamination_scorer.py` | `score_contamination`, `format_contamination_result` — composite aperture score | 15 |
| `transit_model_residual_tester.py` | `test_model_residuals`, `format_residual_test_result` — DW + runs + chi2 | 14 |
| `expected_depth_calculator.py` | `compute_expected_depth`, `format_expected_depth_result` — geometric + diluted depth | 15 |
| `snr_vs_period_plotter.py` | `compute_period_snr`, `format_period_snr_result` — SNR vs period grid | 11 |
| `multi_planet_period_checker.py` | `check_multi_planet_periods`, `format_multi_planet_check` — harmonic/alias detection | 13 |
| `sector_baseline_normalizer.py` | `normalize_sector_baselines`, `format_baseline_norm_result` — additive/mult norm | 13 |
| `false_positive_score_aggregator.py` | `aggregate_fp_scores`, `format_fp_aggregate_result` — weighted geom-mean FP prob | 15 |
| `candidate_csv_importer.py` | `import_candidates_csv`, `format_import_result` — CSV → ImportedCandidate | 13 |
| `noise_model_fitter.py` | `fit_noise_model`, `format_noise_model_result` — white + red noise (beta factor) | 13 |

### Completed (2026-05-22) — Milestone 21

**15 new Skills + 258 new tests**: ✅

| Skill | Key Functions | Tests |
|---|---|---|
| `polynomial_detrend.py` | `fit_polynomial_trend`, `apply_detrend` — piecewise polynomial detrending | 17 |
| `autocorrelation_period_finder.py` | `compute_acf`, `find_acf_period` — stellar rotation via ACF | 18 |
| `window_function_analyzer.py` | `compute_window_function`, `find_alias_periods` — spectral window / alias detection | 18 |
| `exclusion_zone_calculator.py` | `compute_exclusion_zone` — angular separation to exclude background source | 13 |
| `significance_threshold_calculator.py` | `compute_snr_threshold`, `compute_bls_threshold` — bootstrap significance thresholds | 19 |
| `candidate_similarity_scorer.py` | `score_similarity` — period/depth/duration similarity + duplicate/alias detection | 15 |
| `photometric_binary_checker.py` | `check_photometric_binary` — ellipsoidal variation at P/2 | 13 |
| `flux_ratio_calculator.py` | `compute_flux_ratios` — dilution factors from neighbour magnitudes | 16 |
| `period_refinement_calculator.py` | `refine_period_from_oc` — O-C grid search period refinement | 13 |
| `background_source_probability.py` | `estimate_bg_source_prob` — galactic source density bgEB prior | 15 |
| `observation_efficiency_calculator.py` | `compute_obs_efficiency` — phase coverage fraction from timestamps | 16 |
| `signal_comparison_reporter.py` | `compare_signals`, `format_signal_comparison` — side-by-side Markdown table | 17 |
| `tce_reliability_scorer.py` | `score_tce_reliability` — composite MES/n_transit/SES score | 16 |
| `spectral_type_classifier.py` | `classify_spectral_type` — OBAFGKM + luminosity class from Teff/logg | 19 |
| `barycentric_time_corrector.py` | `compute_barycentric_correction`, `apply_barycentric_correction` — BJD−JD Roemer delay | 16 |

### Completed (2026-05-23) — Milestone 22

**15 new Skills + 246 new tests**: ✅

| Skill | Key Functions | Tests |
|---|---|---|
| `transit_survey_planner.py` | `plan_transit_windows`, `format_survey_plan` — schedule upcoming transit windows | 13 |
| `period_commensurability_checker.py` | `check_commensurability`, `format_commensurability_result` — near-MMR resonant pair detection | 14 |
| `geometric_transit_probability.py` | `compute_transit_probability`, `format_transit_prob_result` — P_tr from Kepler 3rd law + ρ★ | 13 |
| `flux_periodogram.py` | `compute_dft_periodogram`, `find_periodogram_peaks`, `format_periodogram_result` — stdlib DFT | 16 |
| `kopparapu_hz_calculator.py` | `compute_hz_boundaries`, `classify_hz_position`, `format_hz_result` — Kopparapu (2013) HZ | 17 |
| `ttv_significance_tester.py` | `test_ttv_significance`, `format_ttv_test_result` — chi-square O-C significance test | 14 |
| `snr_sector_stacker.py` | `project_stacked_snr`, `format_stacked_snr_result` — √N SNR projection | 13 |
| `candidate_summary_card.py` | `build_summary_card`, `format_summary_card` — compact Markdown card from candidate dict | 17 |
| `multi_aperture_comparator.py` | `compare_apertures`, `format_aperture_compare_result` — depth/RMS discrepancy | 14 |
| `epoch_folding_optimizer.py` | `optimize_epoch`, `format_epoch_opt_result` — grid-search T0 minimising O-C RMS | 14 |
| `planet_occurrence_weight.py` | `compute_occurrence_weight`, `format_occurrence_weight_result` — w=1/(p_det×p_tr) | 15 |
| `data_gap_interpolator.py` | `characterize_gaps`, `fill_gaps_linear`, `format_gap_stats` — gap characterisation + fill | 16 |
| `sector_completion_tracker.py` | `SectorCompletionLog.mark_complete/is_complete/export_incomplete`, `format_completion_report` | 15 |
| `vetting_boolean_adapter.py` | `boolean_flags_to_entries`, `run_vetting_triage`, `format_triage_result` | 24 |
| `folded_transit_stack.py` | `stack_transit_windows`, `format_stack_result` — phase-align + stack for SNR | 16 |

### Completed (2026-05-23) — Milestone 23

**15 new Skills + 230 new tests**: ✅

| Skill | Key Functions | Tests |
|---|---|---|
| `votable_formatter.py` | `format_as_votable`, `format_votable_result` — VOTable 1.4 XML via stdlib ET | 16 |
| `astroimagej_region_writer.py` | `write_aij_region`, `format_aij_region_result` — AIJ `.apertures` file | 13 |
| `correlated_noise_estimator.py` | `estimate_correlated_noise`, `format_correlated_noise_result` — beta method red noise | 15 |
| `parameter_sweep_runner.py` | `run_parameter_sweep`, `format_sweep_result` — Cartesian grid sweep | 14 |
| `detection_efficiency_map.py` | `compute_detection_efficiency`, `format_efficiency_result` — 2-D period×depth grid | 13 |
| `false_negative_rate_estimator.py` | `estimate_false_negative_rate`, `format_fnr_result` — Type II error at threshold | 14 |
| `candidate_changelog_tracker.py` | `record_change`, `get_changelog`, `format_changelog_result` — field-level atomic JSON log | 13 |
| `disposition_recorder.py` | `record_disposition`, `get_disposition_history`, `format_disposition_result` — PC/FP/CP/EB/IS/UNK | 18 |
| `cadence_irregularity_scorer.py` | `score_cadence_irregularity`, `format_cadence_irregularity_result` — gap jitter scorer | 13 |
| `saturation_level_checker.py` | `check_saturation`, `format_saturation_result` — analytic TESS saturation from Tmag | 15 |
| `crowding_metric_calculator.py` | `compute_crowding_metric`, `format_crowding_result` — CROWDSAP-equivalent from catalog | 15 |
| `transit_ingress_timer.py` | `compute_ingress_duration`, `format_ingress_result` — T14/T23 Seager & M-O (2003) | 16 |
| `depth_period_correlation_scorer.py` | `score_depth_period_correlation`, `format_depth_period_result` — Pearson/Spearman/OLS | 15 |
| `multi_observatory_coordinator.py` | `coordinate_observations`, `format_coordination_result` — multi-site airmass+Moon | 15 |
| `fits_keyword_mapper.py` | `map_fits_keywords`, `format_keyword_map_result` — FITS header → canonical pipeline fields | 18 |

### Completed (2026-05-24) — Milestone 24

**15 new Skills + 211 new tests**: ✅

| Skill | Key Functions | Tests |
|---|---|---|
| `stellar_luminosity_calculator.py` | `compute_stellar_luminosity`, `format_luminosity_result` — L/L☉ from Stefan-Boltzmann | 14 |
| `contact_time_calculator.py` | `compute_contact_times`, `format_contact_times` — T1/T2/T3/T4 from T14/T23 | 14 |
| `target_coordinates_converter.py` | `convert_coordinates`, `format_coordinate_result` — RA/Dec → ecliptic + galactic (IAU 1958) | 14 |
| `stellar_surface_gravity_estimator.py` | `estimate_surface_gravity`, `format_surface_gravity_result` — log g + error propagation | 14 |
| `planet_mass_estimator.py` | `estimate_planet_mass`, `format_planet_mass_result` — Chen & Kipping (2017) M-R | 14 |
| `stellar_age_gyrochronology.py` | `estimate_stellar_age`, `format_gyro_result` — Barnes (2007) P_rot → age | 14 |
| `observation_window_merger.py` | `merge_windows`, `format_merged_windows` — merge overlapping/adjacent intervals | 14 |
| `rv_detectability_checker.py` | `check_rv_detectability`, `format_rv_detectability` — K amplitude + SNR decision | 14 |
| `phase_fold_quality_checker.py` | `check_phase_fold_quality`, `format_phase_fold_quality` — coverage/SNR/symmetry/A–D grade | 13 |
| `multi_band_depth_comparator.py` | `compare_multi_band_depths`, `format_multi_band_result` — chromaticity via inverse-variance mean | 14 |
| `aperture_optimization_scorer.py` | `score_apertures`, `format_aperture_result` — Gaussian PSF SNR-optimal aperture | 13 |
| `atmospheric_scale_height_calculator.py` | `compute_scale_height`, `format_scale_height_result` — H = kT/μg + transmission amplitude | 14 |
| `seasonal_visibility_planner.py` | `plan_seasonal_visibility`, `format_seasonal_visibility` — monthly ground-based observability | 14 |
| `rms_timescale_profiler.py` | `profile_rms_timescales`, `format_rms_timescale_result` — log-spaced RMS vs bin timescale | 13 |
| `candidate_submission_formatter.py` | `format_submission` — TFOP WG / Planet Hunters structured submission record | 15 |

### Completed (2026-05-24) — Milestone 25

**12 new Skills + 3 src modules updated + 191 new tests**: ✅

| Skill / Module | Key Functions | Tests |
|---|---|---|
| `fetch_exofop_ctoi.py` | `fetch_ctoi_table`, `ctoi_rows_to_label_rows`, `CtoisResult` — ExoFOP CTOI CSV download and opt-in label rows | 21 |
| `fetch_nea_koi_lc_index.py` | `fetch_koi_lc_index`, `KoiRecord` — NASA TAP ephemeris index | 13 |
| `multi_source_label_assembler.py` | `assemble_labels`, `LabelManifest`, `LabelRecord` — merge/dedup labels | 14 |
| `lc_snippet_batch_builder.py` | `build_snippet_batch`, checkpoint/resume batch extraction | 13 |
| `label_quality_controller.py` | `run_label_qc` — source agreement + ephemeris + confidence QC | 14 |
| `snippet_normalizer.py` | `normalize_snippet`, `normalize_batch` — Shallue & Vanderburg normalization | 13 |
| `training_data_monitor.py` | `monitor_training_data`, gate check (5000-label threshold) | 13 |
| `cnn_training_config.py` | `default_config`, `load_config`, `save_config`, `validate_config` | 18 |
| `train_cnn.py` | `train_cnn`, `CnnTrainingResult`, AUC via trapezoidal rule | 13 |
| `cnn_checkpoint_manager.py` | `list_checkpoints`, `select_best`, `prune_checkpoints` | 13 |
| `cnn_calibrator.py` | `fit_cnn_calibration`, `apply_cnn_calibration` — Platt scaling | 15 |
| `cnn_inference_batcher.py` | `run_cnn_inference`, injectable model_fn | 13 |
| `src/exo_toolkit/ml/cnn_scorer.py` | `CnnScorer.predict_proba/batch`, `from_checkpoint`, `unavailable` | 21 |
| `src/exo_toolkit/ml/stacking_scorer.py` | Updated: `from_model_paths`, 3-tier blend (XGB 0.35 + CNN 0.35 + Bayes 0.30) | 22 |
| `src/exo_toolkit/cli.py` | Updated: `--scorer cnn/full-ensemble`, `--cnn-checkpoint` flag | — |

### Completed (2026-05-25) — Milestone 26

**15 new Skills + 199 new tests**: ✅

| Skill | Key Functions | Tests |
|---|---|---|
| `snippet_quality_scorer.py` | `score_snippet_quality`, `score_snippet_batch` — CNN snippet coverage/depth_snr/noise composite | 13 |
| `ephemeris_drift_projector.py` | `project_ephemeris_drift`, `format_ephemeris_drift` — σ_T(n) uncertainty growth | 13 |
| `rv_phase_sampler.py` | `sample_rv_phases`, `format_rv_phases` — evenly spaced optimal RV phases | 13 |
| `planet_radius_gap_classifier.py` | `classify_radius_gap`, `format_radius_gap` — Fulton+2017 radius gap boundaries | 13 |
| `candidate_score_explainer.py` | `explain_candidate_score`, `format_score_explanation` — plain-English score breakdown | 13 |
| `transit_duration_anomaly_checker.py` | `check_duration_anomaly`, `format_duration_anomaly` — T14 vs Kepler 3rd law | 13 |
| `target_crowding_estimator.py` | `estimate_crowding`, `format_crowding` — flux_ratio + crowding_metric from neighbour mags | 13 |
| `json_to_csv_exporter.py` | `flatten_candidate`, `export_to_csv`, `format_export_result` — nested JSON → flat CSV | 13 |
| `toi_disposition_tracker.py` | `diff_toi_snapshots`, `format_toi_diff` — CSV snapshot diff (added/confirmed/FP/changed) | 13 |
| `multi_run_diff_reporter.py` | `diff_pipeline_runs`, `load_and_diff`, `format_run_diff` — pipeline JSON diff | 13 |
| `candidate_followup_prioritizer.py` | `prioritize_followup`, `format_followup_priorities` — composite priority scorer | 13 |
| `pipeline_dependency_checker.py` | `check_dependencies`, `format_dependency_check` — importlib feature matrix | 13 |
| `config_diff_tool.py` | `diff_configs`, `load_and_diff_configs`, `format_config_diff` — nested JSON config diff | 13 |
| `stellar_activity_index.py` | `compute_activity_index`, `format_activity_index` — RMS/MAD/outlier composite | 13 |
| `observation_log_parser.py` | `parse_obs_log`, `load_obs_log`, `format_obs_log` — CSV/TSV photometry log parser | 15 |


### Completed (2026-05-28) — Milestone 28

**15 Skills (Tier 2 data pipeline + CNN bridge tools; current test counts listed below)**: ✅

| Skill | Key Functions | Tests |
|---|---|---|
| `tess_tce_fetcher.py` | `fetch_tce_table`, `tce_to_label_rows`, `format_tce_summary` — SPOC TCE table from ExoMAST | 13 |
| `label_balance_analyzer.py` | `analyze_label_balance`, `format_balance_report` — class balance + weights | 13 |
| `snippet_deduplicator.py` | `deduplicate_snippets`, `apply_deduplication` — period-aware dedup | 13 |
| `validation_set_curator.py` | `curate_validation_set`, `format_curation_report` — leakage-free val split | 13 |
| `transfer_learning_config.py` | `TransferConfig`, `default_transfer_config`, `save/load`, `validate` — Kepler→TESS transfer | 12 |
| `cnn_prediction_uncertainty.py` | `estimate_uncertainty`, `batch_uncertainty` — MC dropout uncertainty | 13 |
| `training_data_stats_reporter.py` | `compute_training_stats`, `format_training_stats` — corpus statistics | 13 |
| `cnn_hyperparameter_config.py` | `HyperparamGrid`, `generate_candidates`, `save/load_grid` — arch search grid | 12 |
| `label_propagator.py` | `propagate_labels`, `format_propagation_report` — harmonic period propagation | 13 |
| `snippet_cache_manager.py` | `SnippetCacheManager.stats/contains/prune/export_manifest` — cache ops | 13 |
| `deployment_readiness_checker.py` | `check_deployment_readiness`, `format_readiness_report` — Tier 2 gate check | 14 |
| `cnn_threshold_optimizer.py` | `optimize_threshold`, `format_threshold_result` — F1/BA/Youden threshold sweep | 13 |
| `model_ensemble_evaluator.py` | `evaluate_ensemble`, `format_ensemble_eval` — AUC/PR/F1/Brier/ECE per tier | 13 |
| `training_resumption_manager.py` | `find_latest_checkpoint`, `plan_resumption` — resume from latest checkpoint | 13 |
| `tier2_progress_reporter.py` | `count_supervised_labels`, `build_tier2_status`, `status_to_dict`, `write_status_outputs`, `format_tier2_report` — unified Tier 2 progress dashboard | 19 |

### Completed (2026-05-25) — Milestone 27

**15 new Skills + 199 new tests**: ✅

| Skill | Key Functions | Tests |
|---|---|---|
| `cnn_model_config.py` | `CnnModelConfig`, `default_config`, `load_config`, `save_config` — 1D CNN architecture config | 13 |
| `label_coverage_reporter.py` | `report_label_coverage`, `format_coverage_report` — label counts by class/period/depth/source | 13 |
| `snippet_batch_progress.py` | `load_batch_progress`, `format_batch_progress` — checkpoint JSON progress tracker | 13 |
| `training_curve_logger.py` | `TrainingCurveLogger.log_epoch`, `load_curves`, `format_curves` — JSONL epoch log | 13 |
| `roc_auc_calculator.py` | `compute_roc_auc`, `format_roc_auc_result` — trapezoidal ROC-AUC + operating-point table | 13 |
| `pr_auc_calculator.py` | `compute_pr_auc`, `format_pr_auc_result` — precision-recall AUC + threshold sweep | 13 |
| `active_learning_scorer.py` | `score_active_learning`, `format_active_learning_result` — uncertainty sampling by \|score-0.5\| | 13 |
| `stratified_dataset_splitter.py` | `split_dataset`, `format_split_result` — stratified train/val/test split | 13 |
| `feature_importance_ranker.py` | `rank_feature_importance`, `format_importance_result` — permutation importance ranker | 13 |
| `model_performance_comparator.py` | `compare_model_performance`, `format_comparison_result` — AUC/F1/Brier side-by-side table | 13 |
| `model_registry.py` | `register`, `get_best`, `list_models`, `format_registry` — persistent JSON model registry | 13 |
| `prediction_batch_exporter.py` | `export_predictions`, `load_predictions`, `format_export_summary` — JSONL prediction export | 13 |
| `ensemble_weight_optimizer.py` | `optimize_weights`, `blend_scores`, `format_weight_result` — grid-search XGB/CNN/Bayes weights | 13 |
| `calibration_curve_reporter.py` | `compute_calibration_curve`, `format_calibration_curve` — reliability diagram data | 13 |
| `confusion_matrix_reporter.py` | `compute_confusion_matrix`, `format_confusion_matrix` — TP/FP/TN/FN + precision/recall/F1 | 13 |


### Completed (2026-05-29) — Milestone 29

**15 new Skills + 188 new tests (pipeline operations + session planning tools)**: ✅

| Skill | Key Functions | Tests |
|---|---|---|
| `pipeline_health_monitor.py` | `check_pipeline_health`, `format_health_report` — label/snippet/registry/calibration health dashboard | 15 |
| `candidate_significance_ranker.py` | `rank_by_significance`, `format_significance_table` — SNR+FPP+novelty composite significance rank | 13 |
| `data_freshness_checker.py` | `check_data_freshness`, `format_freshness_report` — artifact age vs configurable limits | 11 |
| `follow_up_checklist_generator.py` | `generate_checklist`, `format_checklist` — auto-generated prioritised observation checklist | 13 |
| `model_drift_detector.py` | `compute_baseline_stats`, `detect_drift`, `format_drift_report` — mean-shift + std-ratio drift detection | 12 |
| `candidate_cross_reference.py` | `cross_reference`, `format_cross_ref_result` — TIC+period catalog matching | 13 |
| `pipeline_throughput_tracker.py` | `ThroughputTracker.record/stats/clear`, `format_throughput_stats` — atomic JSON throughput log | 11 |
| `science_case_builder.py` | `build_science_case`, `format_science_case` — structured Markdown science case document | 14 |
| `lightcurve_segment_extractor.py` | `extract_transit_segments`, `format_segment_summary` — symmetric windows around transit mid-times | 12 |
| `multi_period_power_analyzer.py` | `analyze_multi_period_power`, `format_multi_period_result` — phase-fold SNR ranking over period grid | 12 |
| `target_selection_optimizer.py` | `optimize_target_selection`, `format_selection_result` — science/obs/stellar/pipeline composite scorer | 13 |
| `stellar_neighbor_vetter.py` | `vet_stellar_neighbors`, `format_neighbor_vetting` — aperture contamination from catalog neighbours | 13 |
| `period_alias_resolver.py` | `resolve_period_alias`, `format_alias_resolution` — harmonic/sub-harmonic alias detection | 12 |
| `candidate_prioritization_report.py` | `build_prioritization_report`, `write_prioritization_report` — full ranked Markdown planning report | 12 |
| `batch_result_archiver.py` | `archive_batch_results`, `format_archive_result` — dated archive dir + manifest for pipeline outputs | 12 |

### Completed (2026-05-29) — Milestone 30

**15 new Skills + 191 new tests**: ✅

| Skill | Key Functions | Tests |
|---|---|---|
| `signal_quality_grader.py` | `grade_signal_quality`, A–F grade from SNR/FPP/DC/novelty | 14 |
| `session_summary_generator.py` | `build_session_summary`, `format_session_summary` — session stats + next steps | 13 |
| `data_provenance_tracker.py` | `ProvenanceLog.record/get/history/summary`, MD5 checksums | 12 |
| `candidate_label_exporter.py` | `export_for_labeling`, `load_labeled`, suggested PC/FP labels | 12 |
| `pipeline_config_validator.py` | `validate_pipeline_config`, `load_and_validate` — required keys + ranges | 15 |
| `transit_baseline_comparator.py` | `compare_transit_baseline`, in-transit vs OOT depth ratio | 11 |
| `multi_mission_comparator.py` | `compare_multi_mission`, period/depth consistency across TESS/Kepler/K2 | 13 |
| `flux_anomaly_detector.py` | `detect_flux_anomalies`, OUTLIER/STEP/RAMP via median+MAD | 12 |
| `candidate_confidence_tracker.py` | `CandidateConfidenceTracker.record/trend/all_trends`, IMPROVING/DEGRADING | 13 |
| `observation_metadata_recorder.py` | `MetadataStore.record/get/list_by_tic/all_records` | 13 |
| `stellar_properties_reporter.py` | `build_stellar_report`, luminosity/HZ/spectral type from TIC params | 15 |
| `transit_ephemeris_updater.py` | `update_ephemeris`, linear O-C fit to refine period + epoch | 13 |
| `uncertainty_propagator.py` | `propagate_uncertainty`, finite-difference quadrature error propagation | 12 |
| `multi_target_scheduler.py` | `schedule_targets`, greedy priority-ordered nightly scheduler | 13 |
| `candidate_archive.py` | `CandidateArchive.insert/latest/history/search/export_csv` | 13 |
