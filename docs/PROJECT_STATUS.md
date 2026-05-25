# PROJECT STATUS

## Status: Active Development
## Phase: Milestone 27 Complete — CNN Training Infrastructure And Skills Expansion
## Last Updated: 2026-05-25

---

## Current Snapshot

The repository contains a reproducible TESS/Kepler exoplanet candidate toolkit with:

- Core pipeline: Fetch -> Clean -> Search -> Vet -> Score -> Classify
- Bayesian log-score model over six hypotheses
- Optional XGBoost and stacking scorer modes
- SQLite-backed background automation with top-level logs
- 249 standalone `Skills/` utility scripts
- 264 test files
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
| ML Tier 1 | `ml/xgboost_scorer.py` | Complete, requires `xgboost` dependency |
| ML Tier 3 | `ml/stacking_scorer.py` | Complete, includes optional 3-tier XGBoost/CNN/Bayesian blend |
| ML Tier 2 scaffolding | `ml/cnn_scorer.py`, `Skills/train_cnn.py`, CNN data utilities | Complete, production use gated on labeled TESS corpus |
| Training/evaluation Skills | Kepler, TESS, combined training, CNN data assembly/validation/training support, XGBoost training, scorer evaluation | Complete |
| Discovery workflow Skills | star scanner, batch scan, alert filter, ranking, watchlist, exports, reports | Complete |
| Milestones 13-18 Skills | 87 additional analysis, vetting, observability, reporting, and follow-up utilities | Complete |
| Milestone 19a Skill | `multi_sector_phase_compare.py` — offline per-sector phase-fold comparison | Complete |
| Milestone 19b Skill | `candidate_dashboard_export.py` — static conservative candidate dashboard with optional plot artifacts | Complete |
| Milestone 19c Skill | `candidate_api.py` — local read-only candidate API plus optional background SQLite summaries | Complete |
| Milestone 19d Skill | `candidate_browser_ui.py` — interactive local candidate browser with optional plot previews | Complete |
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
- Gate check: `python Skills/count_tess_labels.py`
- Architecture spec: `docs/CNN_SPEC.md`
- Supporting implementation exists: `ml/cnn_scorer.py`, `Skills/train_cnn.py`, `labelled_lc_collector.py`, `cnn_feature_augmenter.py`, `build_cnn_training_data.py`, `cnn_split_validator.py`, and related label/QC/checkpoint/calibration utilities.

No active implementation blocker is known in the default local validation path.

---

## Next Actions

1. Run `python Skills/count_tess_labels.py` periodically to monitor the CNN production-data gate.
2. Once the CNN gate opens, run the implemented CNN training/calibration pipeline per `docs/CNN_SPEC.md`.
3. Use `toi_checker.py` before investing pipeline time on new live targets.
4. Use `batch_scan.py` + `alert_filter.py` + `rank_candidates.py` + `watchlist.py` for systematic follow-up.
5. Use `multi_sector_phase_compare.py` to inspect sector-to-sector depth and phase consistency before advancing multi-sector follow-up targets.
6. Use `candidate_dashboard_export.py` to build static local review dashboards from existing candidate JSON outputs, including optional phase-fold plot artifact paths when available.
7. Use `candidate_api.py` to serve existing local candidate JSON and optional read-only background SQLite summaries for local browsing.
8. Use `candidate_browser_ui.py` for an interactive local browser UI with embedded-data/API modes and optional phase-fold plot previews.

Live-network note: the CNN gate check was not run during the latest local
maintenance pass because it queries ExoFOP and requires intentional live network
approval.

Remote sync note: local `main` contains commits that may need to be pushed to
`origin/main`. Pushing exports repository contents to GitHub and requires
explicit approval in restricted environments.

## Latest Local Validation

Validated on 2026-05-25:

```bash
.venv/bin/ruff check .
.venv/bin/python -m mypy src
.venv/bin/python -m pytest
```

Result: ruff passed, mypy passed, pytest passed with 4279 passed, 2 deselected, and 59 warnings.

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
