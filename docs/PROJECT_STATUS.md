# PROJECT STATUS

## Status: Active Development
## Phase: CNN Data Collection Complete — Training Pipeline Ready
## Last Updated: 2026-06-09

---

## Current Snapshot

The repository contains a reproducible TESS/Kepler exoplanet candidate toolkit with:

- Core pipeline: Fetch -> Clean -> Search -> Vet -> Score -> Classify
- Bayesian log-score model over six hypotheses
- Optional XGBoost and stacking scorer modes (Tier 1 model trained: Kepler KOI AUC=0.992)
- SQLite-backed background automation with top-level logs
- 415 standalone `Skills/` utility scripts
- 432 top-level test files, 6,385 default tests passing
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
| ML Tier 2 scaffolding | `ml/cnn_scorer.py`, `Skills/train_cnn.py`, CNN data utilities | Complete; label gate OPEN (2,668 labels); TESS light curve download complete (2,636 targets → `data/tess_snippets.jsonl`) |
| TESS light curve download | `Skills/download_tess_lightcurves.py` | Complete — 2,636 TESS TOI targets downloaded, `data/tess_snippets.jsonl` ready for CNN training |
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
| Project MCP bootstrap | `.mcp.json`, `.codex/config.toml`, `Skills/mcp_bootstrap_server.py` — project-scoped file, git-read, and fixed validation MCP servers | Complete, offline by default |
| Live label-check audit | `Skills/count_tess_labels.py`, `Skills/tess_label_check_summary.py` — opt-in live ExoFOP gate check plus read-only SQLite log summary | Complete, live access requires intentional approval |
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

## Active Blocker

**T1-1: Run CNN training pipeline** — data collection is complete, training has not yet run

- Label gate: OPEN — 2,668 labels (1,324 positive CP/KP + 1,344 negative FP/FA) as of 2026-06-06
- TESS download: COMPLETE — 2,636 TOI targets, `data/tess_snippets.jsonl`
- Code: all training/calibration code exists and is tested
- Blocker: training run requires PyTorch + CPU/GPU time; must be executed locally after `git pull origin main`
- Architecture spec: `docs/CNN_SPEC.md`

---

## Next Actions

**[HUMAN first]** Verify PyTorch is installed in the venv:
```bash
git pull origin main
python -c "import torch; print('PyTorch', torch.__version__)"
```
If missing: `pip install torch` with venv active.

**[AGENT then HUMAN]** Run CNN training pipeline in order:

1. `python Skills/build_cnn_training_data.py data/tess_snippets.jsonl --output-dir data/cnn_splits`
2. `python Skills/cnn_split_validator.py data/cnn_splits`
3. `caffeinate -i python Skills/train_cnn.py --splits-dir data/cnn_splits --output-dir models/cnn/`
4. `python Skills/cnn_calibrator.py --checkpoint models/cnn/best.pt --splits-dir data/cnn_splits --output models/cnn/calibration.json`
5. Test: `exo <known-TOI-TIC-ID> --scorer full-ensemble --cnn-checkpoint models/cnn/best.pt`

**[OPTIONAL CLEANUP]** One corrupt FITS cache entry may need manual clearing:
```bash
rm -rf "/Users/Rome/.lightkurve/cache/mastDownload/TESS/tess2021146024351-s0039-0000000220435095-0210-s"
caffeinate -i python Skills/download_tess_lightcurves.py --toi-csv data/tess_toi.csv --output data/tess_snippets.jsonl --resume
```

Remote sync note: local `main` is synced with `origin/main` as of 2026-06-09.


## Latest Local Validation

Validated on 2026-06-03:

```bash
.venv/bin/ruff check .
.venv/bin/python -m mypy src
.venv/bin/python -m pytest
```

Result: ruff passed, mypy passed, pytest passed with 6385 passed, 2 deselected, and no warnings.

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
