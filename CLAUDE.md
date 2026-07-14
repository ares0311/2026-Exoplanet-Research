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

**Current test surface:** 138 top-level test files. Version 0.2.54 local 6×6 validation passed 2,726 default tests plus Ruff/mypy in 34.1s; 2 `integration_live` tests remain excluded from the default suite.
**Skills:** 118 standalone utility scripts live in `Skills/` (plus the package marker `Skills/__init__.py`). Use `rg --files Skills -g '*.py' | sort` for the authoritative current list, and see `docs/SKILLS_GUIDE.md` for workflow-oriented quick reference.

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
