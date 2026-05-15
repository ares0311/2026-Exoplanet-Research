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
| `features.py` | **done** | `test_features.py` (97) — includes `depth_scatter_chi2_score` |
| `hypotheses.py` | **done** | `test_hypotheses.py` (33) — `depth_scatter_chi2_score` wiring |
| `scoring.py` | **done** | `test_scoring.py` (25) |
| `pathway.py` | **done** | `test_pathway.py` (35) |
| `fetch.py` | **done** | `test_fetch.py` (55, 2 live) |
| `clean.py` | **done** | `test_clean.py` (39) |
| `search.py` | **done** | `test_search.py` (43) |
| `vet.py` | **done** | `test_vet.py` (47) |
| `calibration.py` | **done** | `test_calibration.py` (70) |
| `cli.py` | **done** | `test_cli.py` (24) |
| `ml/xgboost_scorer.py` | **done** | `test_xgboost_scorer.py` (45) |
| `ml/stacking_scorer.py` | **done** | `test_stacking_scorer.py` (22) |
| `background/` module | **done** | `test_background_automation.py` (16) |

**Total passing tests: 857 (+ 2 integration_live; 6 skipped without matplotlib)**
**Skills: `injection_recovery.py` (25), `fetch_kepler_tce.py`, `fetch_tess_toi.py` (11), `build_training_data.py` (34), `build_tess_training_data.py` (38), `build_combined_training_data.py` (13), `train_xgboost.py` (25), `evaluate_scorer.py` (14), `count_tess_labels.py`, `star_scanner.py` (38), `rank_candidates.py` (12), `batch_scan.py` (14), `sector_coverage.py` (10), `plot_lc.py` (11), `watchlist.py` (13), `summary_report.py` (14)**

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
- Public API: `compute_metrics`, `fit_calibration`, `apply_calibration`
- Methods: `"platt"` (Platt scaling via scipy Nelder-Mead), `"isotonic"` (PAVA — no sklearn)
- One-vs-rest calibration per hypothesis; renormalized to sum to 1.0 post-calibration
- Metrics: Brier scores, reliability curves, precision/recall/F1, confusion matrix
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

### Next Step

**ML Ensemble Scorer — Tier 2: 1D CNN on phase-folded flux**
- Requires 5,000+ TESS labels before building
- Gate check: `python Skills/count_tess_labels.py`
- Architecture spec: `docs/CNN_SPEC.md`

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
