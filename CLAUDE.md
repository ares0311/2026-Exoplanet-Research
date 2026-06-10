# CLAUDE.md — Claude Code Project Context

This file is read automatically by Claude Code at session start.
It contains the facts a coding agent needs to work productively without re-reading every document.

---

## PRIMARY DIRECTIVE — READ THIS BEFORE ANYTHING ELSE

**The only authorized work is work that advances this project to live production.**

Every session must begin by reading:
1. `AGENTS.md`
2. `docs/PRODUCTION_READINESS.md`

Before proposing or executing any task you must:
1. Name the highest-priority unresolved Tier 1 gap from `docs/PRODUCTION_READINESS.md`.
2. State explicitly how the proposed work closes or directly unblocks that gap.
3. If the proposed work does not close or unblock a named gap, **do not do it**.

### Prohibited work
- Adding Skills, modules, schemas, or scaffolding that do not directly close a named Tier 1 or Tier 2 gap.
- Repeating work already listed under "What Is Complete" in `docs/PRODUCTION_READINESS.md`.
- Writing "the next N utility scripts" when those scripts do not unblock a named gap.
- Treating "Apply All System Directives" as permission to add more code — it means read the gap list and work the highest-priority gap only.

### When the user says "Apply All System Directives"
1. Read `AGENTS.md` and `docs/PRODUCTION_READINESS.md`.
2. State the current Tier 1 and Tier 2 gaps in priority order.
3. For planning: propose tasks in priority order where every task closes or unblocks a named gap. Stop when gap-closing tasks run out — do not pad the list with non-gap work. Tasks may be agent-led (code) or human-led (data collection, API keys, expert review, network access) — both are valid plan items. Label each task clearly: [AGENT] or [HUMAN].
4. For each task, identify external dependencies (API keys, network access, GPU, human reviewer) and surface them as explicit questions before the DO phase.
5. Do not propose or execute work that does not close a named gap.

### Two-phase workflow: PLAN then DO

**PLAN phase** ("plan the next N tasks"):
- List all gap-closing tasks in priority order, labeled [AGENT] or [HUMAN].
- For every [HUMAN] task, provide exact step-by-step instructions so the human can act independently.
- Ask all questions about external dependencies upfront.
- Do not execute anything.

**Between PLAN and DO — resolve all [HUMAN] tasks first:**
- The human works through every [HUMAN] task using the instructions from the plan.
- All [HUMAN] blockers must be cleared before the DO phase begins.
- If a [HUMAN] task needs interactive help, work through it with the human until it is resolved.

**DO phase** ("DO the next N tasks"):
- By the time DO begins, all [HUMAN] blockers are already cleared.
- Execute only [AGENT] tasks.
- The DO phase should never contain a [HUMAN] blocker — if one appears, the PLAN phase was incomplete.

### When the highest-priority Tier 1 gap is blocked by an outside action
State the gap, name the blocker, and **immediately provide a complete step-by-step recipe** assuming the user has zero background knowledge of the specific task. Give exact commands to copy-paste, explain each in one plain-English sentence, and state exactly what output to paste back. Do not ask "do you want the commands?" — give them.

### Local–Remote Sync Policy — MANDATORY

The user's local Mac and GitHub `main` are the joint source of truth. Keep them in sync at all times.

**Agent rules (non-negotiable):**
1. Every code change must complete the full cycle: feature branch → commit → push → PR → CI green → merge to main → PR closed. Never leave a PR open at end of session.
2. Never tell the user to run a script that has not yet been merged to `main`.
3. Every recipe given to the user must begin with `git pull origin main`.
4. After every merge to `main`, remind the user: `git pull origin main`.

**Standard recipe header — prepend to EVERY user command:**
```bash
git pull origin main
```

**For long-running commands:**
```bash
git pull origin main
caffeinate -i python Skills/<script>.py [args]
```

### macOS Long-Running Process Policy — ALWAYS USE caffeinate
Any recipe for a Python command that runs longer than ~60 seconds **must** use `caffeinate -i`:
```bash
caffeinate -i python Skills/<script>.py [args]   # standard
caffeinate -dims python Skills/<script>.py [args] # lid-close safe
```
This applies to: light curve downloads, CNN training, batch scans, injection-recovery, and any repeated-network or long-compute script. **Never give a bare `python ...` recipe for these.**

### Python Environment Policy — NEVER TOUCH SYSTEM PYTHON
- Validated runtime: **Python 3.14.3** inside `.venv` — never use system Python
- All work happens inside the `.venv` virtual environment
- Never run `/Applications/Python*/Install\ Certificates.command`
- Never suggest `sudo pip install` or any path under `/Library/Frameworks/Python.framework/`
- `pip install` with `(.venv)` active is always venv-scoped and safe
- Fix SSL/package issues inside the venv only

---

## Standing Rules

- **Skills directory**: Any standalone `.py` utility script created to perform a task (data processing, report generation, injection-recovery, etc.) must be saved in `Skills/` at the project root. Create the directory if it does not exist. This allows scripts to be discovered and reused across sessions rather than recreated.

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

**Current test surface:** 432 top-level test files. Local validation on 2026-06-03 passed with 6385 default tests and 2 `integration_live` tests deselected.
**Skills:** 415 standalone utility scripts live in `Skills/` (plus the package marker `Skills/__init__.py`). See `docs/SKILLS_GUIDE.md` for workflow-oriented quick reference instead of relying on this file for exhaustive per-script counts.

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
- **Conservative priors**: built-in defaults remain planet_candidate = 0.10, EB/BEB/stellar/instrumental = 0.20 each, known_object = 0.10
- **Mission prior profiles**: `configs/scoring_priors_v0.json` defines opt-in conservative TESS/Kepler/K2 profiles loaded by `priors.py`
- **ML Tier 1 (XGBoost) is built** — `ml/xgboost_scorer.py` ships as an optional alternative scorer; Bayesian log-score model remains the default fallback when labels are unavailable
- **ML Tier 2 scaffolding is built** — `ml/cnn_scorer.py`, `Skills/train_cnn.py`, checkpoint/calibration utilities, and CLI wiring exist; production CNN use remains gated on 5,000+ labeled TESS light curves and calibration review
- **ML Tier 3 (stacking) is built** — `ml/stacking_scorer.py` blends XGBoost + CNN + Bayesian P(planet) when models are supplied; falls back conservatively when optional models are unavailable
- **CLI scorer options**: `exo <TIC-ID> --scorer [bayesian|xgboost|ensemble|cnn|full-ensemble] --model-path <path> --cnn-checkpoint <path>`
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

### Next Step

**Collect 5,000+ TESS labeled examples** — then run CNN training pipeline
- Gate check: `python Skills/count_tess_labels.py`
- Architecture spec: `docs/CNN_SPEC.md`
- Full pipeline: `fetch_ctoi_table` → `assemble_labels` → `run_label_qc` → `normalize_batch` → `monitor_training_data` → `train_cnn` → `cnn_calibrator` → `CnnScorer`

### ML Ensemble Scorer Status

The optional ML tiers now augment the Bayesian log-score model while preserving
Bayesian scoring as the default fallback:

**Tier 1 — XGBoost on tabular features (build first)** ✅ DONE
**Tier 2 — 1D CNN on phase-folded flux (scaffold built; production checkpoint after 5,000+ TESS labels)**
- Input: phase-folded, normalized flux array (treat as 1D image)
- Learns transit morphology directly; proven architecture (Shallue & Vanderburg 2018)
- Requires TESS-specific fine-tuning — Kepler-trained models are miscalibrated on TESS due to cadence, pixel scale, and systematics differences

**Tier 3 — Stacking meta-learner** ✅ DONE
- Simple weighted blend over outputs of XGBoost + CNN + existing Bayesian log-score model
- Production weights still require a separate held-out calibration set (~500+ examples)
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
