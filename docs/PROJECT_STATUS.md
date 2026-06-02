# PROJECT STATUS

## Status: Active Development
## Phase: Milestone 39 Complete — Physics, Vetting, Planning, and XGBoost Model Trained
## Last Updated: 2026-06-02

---

## Current Snapshot

The repository contains a reproducible TESS/Kepler exoplanet candidate toolkit with:

- Core pipeline: Fetch -> Clean -> Search -> Vet -> Score -> Classify
- Bayesian log-score model over six hypotheses
- Optional XGBoost and stacking scorer modes (Tier 1 model trained: Kepler KOI AUC=0.992)
- SQLite-backed background automation with top-level logs
- 413 standalone `Skills/` utility scripts
- 427 top-level test files, 6,367 default tests passing
- 27 package Python modules under `src/exo_toolkit/`

Local validation note: after restoring the declared `xgboost` dependency and installing the macOS OpenMP runtime (`libomp`), the default test suite passes locally on Python 3.13.12.

---

## What Is Complete

| Area | Key Files | Status |
|------|-----------|--------|
| Scoring engine | `schemas.py`, `features.py`, `hypotheses.py`, `priors.py`, `scoring.py`, `pathway.py` | Complete |
| Data pipeline | `fetch.py`, `clean.py`, `search.py`, `vet.py`, `calibration.py` | Complete |
| Transit scan CLI | `cli.py` — `exo <TIC-ID>` with `--scorer`, `--model-path`, `--output` | Complete |
| Background automation CLI | `cli.py` — `background-run-once`, summaries, integrity, validation | Complete |
| Background automation module | `background/` — config, fixtures, priority, runner, storage, reports | Complete |
| SQLite runtime state | `logs/background_search.sqlite3` schema v2 | Complete |
| Background config | `configs/background_search_v0.json` | Complete |
| Scoring prior config | `configs/scoring_priors_v0.json` — conservative default plus TESS/Kepler/K2 profiles | Complete |
| Scheduler docs | `docs/SCHEDULER.md` — cron, launchd, systemd | Complete |
| ML Tier 1 | `ml/xgboost_scorer.py` + `models/xgboost_koi.json` | Complete — trained on 7,586 Kepler KOIs (AUC=0.992) |
| ML Tier 3 | `ml/stacking_scorer.py` | Complete, includes optional 3-tier XGBoost/CNN/Bayesian blend |
| ML Tier 2 scaffolding | `ml/cnn_scorer.py`, `Skills/train_cnn.py`, CNN data utilities | Complete, production use gated on labeled TESS corpus |
| Training/evaluation Skills | Kepler, TESS, combined training, CNN data assembly/validation/training support, XGBoost training, scorer evaluation | Complete |
| Discovery workflow Skills | star scanner, batch scan, alert filter, ranking, watchlist, exports, reports | Complete |
| Milestones 13-39 Skills | 354 additional analysis, vetting, observability, ML, physics, reporting, scheduling, and follow-up utilities | Complete |
| Milestone 19a Skill | `multi_sector_phase_compare.py` — offline per-sector phase-fold comparison | Complete |
| Milestone 19b Skill | `candidate_dashboard_export.py` — static conservative candidate dashboard with optional plot artifacts | Complete |
| Milestone 19c Skill | `candidate_api.py` — local read-only candidate API plus optional background SQLite summaries | Complete |
| Milestone 19d Skill | `candidate_browser_ui.py` — interactive local candidate browser with optional plot previews | Complete |
| Milestone 30 Skills | 15 diagnostics + scheduling tools including `flux_anomaly_detector`, `candidate_confidence_tracker`, `uncertainty_propagator`, `multi_target_scheduler`, `candidate_archive`, and 10 more | Complete |
| Milestones 34-39 Skills | 90 additional ML evaluation, photometry quality, transit vetting, noise budget, orbit simulation, stellar physics, TTV, occurrence-rate, and planning utilities | Complete |
| CTOI source contract | `docs/CTOI_SOURCE_CONTRACT.md`, `Skills/fetch_exofop_ctoi.py`, `tests/fixtures/exofop_ctoi_sample.csv`, `tests/fixtures/exofop_ctoi_labels_sample.json` — opt-in fixture-backed community candidate labels | Complete, excluded from default training |
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
reports/background/*.md
reports/background/*.html
```

---

## Blocked

**Production ML Tier 2 — trained 1D CNN on phase-folded flux**

- Gate: 5,000+ labeled TESS light curves required before production training/use
- Offline readiness check: `python Skills/tier2_progress_reporter.py --labels data/exofop_ctoi_labels.json --output reports/tier2_status.md --json-output reports/tier2_status.json`
- Live gate check: `python Skills/count_tess_labels.py` when network access is intentionally approved
- Architecture spec: `docs/CNN_SPEC.md`
- Supporting implementation exists: `ml/cnn_scorer.py`, `Skills/train_cnn.py`, `labelled_lc_collector.py`, `cnn_feature_augmenter.py`, `build_cnn_training_data.py`, `cnn_split_validator.py`, and related label/QC/checkpoint/calibration utilities.

No active implementation blocker is known in the default local validation path.

---

## Next Actions

1. Run `python Skills/tier2_progress_reporter.py --labels data/exofop_ctoi_labels.json --output reports/tier2_status.md --json-output reports/tier2_status.json` to produce offline CNN readiness artifacts.
2. Run `python Skills/count_tess_labels.py` only when live ExoFOP access is intentionally approved.
3. Once the CNN gate opens, run the implemented CNN training/calibration pipeline per `docs/CNN_SPEC.md`.
4. Use `toi_checker.py` before investing pipeline time on new live targets.
5. Use `batch_scan.py` + `alert_filter.py` + `rank_candidates.py` + `watchlist.py` for systematic follow-up.
6. Use `multi_sector_phase_compare.py` to inspect sector-to-sector depth and phase consistency before advancing multi-sector follow-up targets.
7. Use `candidate_dashboard_export.py` to build static local review dashboards from existing candidate JSON outputs, including optional phase-fold plot artifact paths when available.
8. Use `candidate_api.py` to serve existing local candidate JSON, optional read-only background SQLite summaries, `/artifact.json` review bundles, and opt-in CORS for separate local frontends.
9. Use `candidate_browser_ui.py` for an interactive local browser UI with embedded-data/API modes and optional phase-fold plot previews.
10. Keep CTOI/community candidate ingestion opt-in and outside default training; the fixture-backed source contract now lives in `docs/CTOI_SOURCE_CONTRACT.md`.

Live-network note: the CNN gate check was not run during the latest local
maintenance pass because it queries ExoFOP and requires intentional live network
approval.

Remote sync note: local `main` is synced with `origin/main` as of the latest
handoff.


## Latest Local Validation

Validated on 2026-06-02:

```bash
.venv/bin/ruff check .
.venv/bin/python -m mypy src
.venv/bin/python -m pytest
```

Result: ruff passed, mypy passed, pytest passed with 6367 passed, 2 deselected, and no warnings.

---

## Key Design Decisions In Effect

- Bayesian log-score model is default; XGBoost and ensemble are opt-in.
- `OptScore = float | None`: missing diagnostics contribute neutrally to log scores.
- Missing diagnostics fail threshold gates conservatively.
- Conservative priors keep false positives prominent; mission-specific prior profiles are opt-in through `configs/scoring_priors_v0.json`.
- `provenance_score` is computed from cadence, sector count, and pipeline quality.
- `toi_checker.py` should be consulted before investing pipeline time on any new target.
- Default tests must mock external services; live tests require `integration_live`.
- Never output "confirmed planet"; use "candidate signal" or "follow-up target".
- Background automation uses SQLite for durable state and deterministic fixtures by default.
- Background automation obeys the human-approval gate; no external submission without review.
