# ROADMAP

## Milestone 1 — Scoring and Classification Engine ✓ COMPLETE

- [x] `schemas.py` — typed Pydantic data contracts
- [x] `features.py` — 35+ normalized feature extraction functions
- [x] `hypotheses.py` — Bayesian log-score models for 6 hypotheses
- [x] `scoring.py` — softmax posterior + FPP, detection confidence, novelty, habitability
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
- [x] Scorer selection via `--scorer [bayesian|xgboost|ensemble]` and `--model-path`

---

## Milestone 6 — Injection-Recovery ✓ COMPLETE

- [x] `Skills/injection_recovery.py` — inject synthetic box transits, recover via BLS
- [x] Measures recovery rate by radius, period, noise level
- [x] 25 tests in `tests/test_injection_recovery.py`

---

## Milestone 7 — ML Ensemble Scorer ✓ COMPLETE (Tier-2 CNN gated)

- [x] Tier 1 — XGBoost on tabular features (`ml/xgboost_scorer.py`, 45 tests)
- [x] Tier 3 — Stacking scorer blending XGBoost + Bayesian (`ml/stacking_scorer.py`, 22 tests)
- [x] Kepler training pipeline (`Skills/fetch_kepler_tce.py`, `build_training_data.py`, `train_xgboost.py`)
- [x] TESS training pipeline (`Skills/fetch_tess_toi.py`, `build_tess_training_data.py`)
- [x] Evaluation framework (`Skills/evaluate_scorer.py`, ROC-AUC, F1, reliability diagrams)
- [x] Combined training data (`Skills/build_combined_training_data.py`)
- [ ] Tier 2 — 1D CNN on phase-folded flux (**BLOCKED: requires 5,000+ TESS labels**)
  - Architecture spec: `docs/CNN_SPEC.md`
  - Gate check: `Skills/count_tess_labels.py`

---

## Milestone 8 — Background Automation ✓ COMPLETE

- [x] `background/` module — SQLite-backed durable state (run ledger, reviewed/needs-follow-up logs, follow-up tests, reports, approvals)
- [x] `background/runner.py` — `background_run_once()` — one-shot scheduler-friendly invocation
- [x] `background/priority.py` — composite priority scoring (8 factors) with reason codes
- [x] `background/storage.py` — `BackgroundStore` with 6 SQLite tables; schema v2 with migration
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

## Milestone 10 — Future

- [ ] Tier 2 CNN once TESS label threshold is met (5,000+ CP labels)
- [ ] Web API / dashboard for candidate browsing
- [ ] Per-star multi-sector phase-fold comparison across sectors

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
