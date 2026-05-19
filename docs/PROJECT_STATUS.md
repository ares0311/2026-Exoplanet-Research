# PROJECT STATUS

## Status: Active Development
## Phase: Milestone 18 Complete — Background Automation And Skills Expansion
## Last Updated: 2026-05-19

---

## Current Snapshot

The repository contains a reproducible TESS/Kepler exoplanet candidate toolkit with:

- Core pipeline: Fetch -> Clean -> Search -> Vet -> Score -> Classify
- Bayesian log-score model over six hypotheses
- Optional XGBoost and stacking scorer modes
- SQLite-backed background automation with top-level logs
- 111 standalone `Skills/` utility scripts
- 124 test files
- 25 package Python modules under `src/exo_toolkit/`

Local validation note: after restoring the declared `xgboost` dependency and installing the macOS OpenMP runtime (`libomp`), the default test suite passes locally on Python 3.13.12.

---

## What Is Complete

| Area | Key Files | Status |
|------|-----------|--------|
| Scoring engine | `schemas.py`, `features.py`, `hypotheses.py`, `scoring.py`, `pathway.py` | Complete |
| Data pipeline | `fetch.py`, `clean.py`, `search.py`, `vet.py`, `calibration.py` | Complete |
| Transit scan CLI | `cli.py` — `exo <TIC-ID>` with `--scorer`, `--model-path`, `--output` | Complete |
| Background automation CLI | `cli.py` — `background-run-once`, summaries, integrity, validation | Complete |
| Background automation module | `background/` — config, fixtures, priority, runner, storage, reports | Complete |
| SQLite runtime state | `logs/background_search.sqlite3` schema v2 | Complete |
| Background config | `configs/background_search_v0.json` | Complete |
| Scheduler docs | `docs/SCHEDULER.md` — cron, launchd, systemd | Complete |
| ML Tier 1 | `ml/xgboost_scorer.py` | Complete, requires `xgboost` dependency |
| ML Tier 3 | `ml/stacking_scorer.py` | Complete |
| Training/evaluation Skills | Kepler, TESS, combined training, XGBoost training, scorer evaluation | Complete |
| Discovery workflow Skills | star scanner, batch scan, alert filter, ranking, watchlist, exports, reports | Complete |
| Milestones 13-18 Skills | 87 additional analysis, vetting, observability, reporting, and follow-up utilities | Complete |
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

**ML Tier 2 — 1D CNN on phase-folded flux**

- Gate: 5,000+ labeled TESS light curves required
- Gate check: `python Skills/count_tess_labels.py`
- Architecture spec: `docs/CNN_SPEC.md`
- Supporting data utilities exist: `labelled_lc_collector.py`, `cnn_feature_augmenter.py`

No active implementation blocker is known in the default local validation path.

---

## Next Actions

1. Run `python Skills/count_tess_labels.py` periodically to monitor the CNN gate.
2. Once the CNN gate opens, implement Tier 2 per `docs/CNN_SPEC.md`.
3. Use `toi_checker.py` before investing pipeline time on new live targets.
4. Use `batch_scan.py` + `alert_filter.py` + `rank_candidates.py` + `watchlist.py` for systematic follow-up.

Live-network note: the CNN gate check was not run during the latest local
maintenance pass because it queries ExoFOP and requires intentional live network
approval.

Remote sync note: local `main` contains commits that may need to be pushed to
`origin/main`. Pushing exports repository contents to GitHub and requires
explicit approval in restricted environments.

## Latest Local Validation

Validated on 2026-05-19:

```bash
.venv/bin/ruff check .
.venv/bin/python -m mypy src
.venv/bin/python -m pytest
```

Result: ruff passed, mypy passed, pytest passed with 2225 passed, 2 deselected, and 33 warnings.

---

## Key Design Decisions In Effect

- Bayesian log-score model is default; XGBoost and ensemble are opt-in.
- `OptScore = float | None`: missing diagnostics contribute neutrally to log scores.
- Missing diagnostics fail threshold gates conservatively.
- Conservative priors keep false positives prominent.
- `provenance_score` is computed from cadence, sector count, and pipeline quality.
- `toi_checker.py` should be consulted before investing pipeline time on any new target.
- Default tests must mock external services; live tests require `integration_live`.
- Never output "confirmed planet"; use "candidate signal" or "follow-up target".
- Background automation uses SQLite for durable state and deterministic fixtures by default.
- Background automation obeys the human-approval gate; no external submission without review.
