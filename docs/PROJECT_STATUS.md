# PROJECT STATUS

## Status: Active Development
## Phase: Milestone 39 Complete — Physics, Vetting, Planning, and XGBoost Model Trained
## Last Updated: 2026-06-18

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

Local validation note: validated on Python 3.14.3 in `.venv` with `xgboost` dependency restored and macOS OpenMP runtime (`libomp`) installed. System Python is never used.

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
reports/
checkpoints/
data/*.jsonl
data/*_splits/
models/cnn*/
```

Production-relevant ignored artifact state is committed in
`docs/LOCAL_ARTIFACT_LEDGER.md` and
`artifacts/manifests/local_artifacts.json` so GitHub-only agents can continue
without relying on chat context or local terminal output.

---

## Blocked

**Production ML Tier 2 — checkpoint generalization**

- The label gate is open, but no CNN checkpoint has passed the production gate.
- The first seed-42 checkpoint completed training but was rejected on 2026-06-10.
- Held-out test AUC was 0.7404 and calibrated F1 was 0.6297, below the documented 0.85 and 0.80 targets.
- Validation-fitted Platt calibration worsened test Brier score and ECE, so no calibration or checkpoint artifact was promoted into `models/`.
- A 2026-06-10 audit found that every nominally usable snippet had `epoch_bjd=0.0`, so catalog transit events were not centered in phase.
- The old corpus, original seed-42 split, and temporary replacement split are retired.
- The local TESS v2 corpus is complete.
- The pre-fix local Kepler corpus was rejected on 2026-06-17: it had 7,454 rows, but 7,132 rows contained non-finite flux and only 322 finite examples survived builder filtering.
- The rebuilt local Kepler corpus was validated on 2026-06-17: 6,837 parseable finite snippets, zero duplicate resume keys, labels negative=4,280 and positive=2,557.
- `data/kepler_cnn_splits` passed validation on 2026-06-17 with train/val/test = 4,741 / 1,060 / 1,036.
- Tiny corrupt Kepler Lightkurve cache files were quarantined locally before training resumed.
- Kepler pretraining completed locally on 2026-06-18:
  `checkpoints/cnn_kepler_pretrain/best.pt`, SHA-256
  `c782d7af61171b3f58447f7a49343c86618c447292a71bd28d540807835787c7`;
  startup banner `device=mps`; best epoch 19, best validation loss 0.3905,
  best validation AUC 0.9186.
- `data/tess_cnn_splits` passed validation on 2026-06-18 with total examples
  2,110 and train/val/test = 1,477 / 318 / 315.
- Architecture details: `docs/CNN_SPEC.md`.
- Human local runbook: `docs/CNN_PRODUCTION_RUNBOOK.md`.
- Next outside blocker: fine-tune from the MPS Kepler pretraining checkpoint
  on the validated TESS CNN splits.

---

## Next Actions

1. Update `docs/LOCAL_ARTIFACT_LEDGER.md` and `artifacts/manifests/local_artifacts.json` after each local artifact state change so GitHub records the current corpus/split/checkpoint status.
2. Run Step 5 TESS fine-tuning from
   `checkpoints/cnn_kepler_pretrain/best.pt`, then paste back the final
   training result plus SHA-256.
3. Run production gate evaluation. Promote nothing unless the evaluator reports
   `Flag: PASS`, raw test AUC is at least 0.85, calibrated test F1 is at least
   0.80, and calibrated Brier/ECE are no worse than raw.
4. If the gate passes, request explicit human approval to promote the checkpoint; the agent then updates `models/`, registry metadata, readiness docs, artifact ledger, and GitHub.
5. If the gate fails, document the rejection in `docs/PRODUCTION_READINESS.md` and the artifact ledger, then start the next T1-1 planning cycle from the observed failure mode.

Live-network note: the CNN gate check was not run during the latest local
maintenance pass because it queries ExoFOP and requires intentional live network
approval.

Remote sync note: local `main` is synced with `origin/main` as of the latest
handoff.


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
