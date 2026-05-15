# PROJECT STATUS

## Status: Active Development
## Phase: Phase 4 — Provenance Score, Batch Tools, Ranking Complete; CNN Tier-2 Gated on Data
## Last Updated: 2026-05-15

---

## What Is Complete

| Area | Key Files | Tests |
|------|-----------|-------|
| Scoring engine | `schemas`, `features`, `hypotheses`, `scoring`, `pathway` | 210 |
| Data pipeline | `fetch`, `clean`, `search`, `vet`, `calibration` | 236 |
| Provenance score | `fetch.py` — `compute_provenance_score()`; wired into `run_pipeline()` | 19 |
| CLI — transit scan | `cli.py` — `exo <TIC-ID>` with `--scorer`, `--model-path`, `--output` | 24 |
| CLI — background automation | `cli.py` — `exo background-run-once`, `run-summary`, `sqlite-integrity`, etc. | 16 |
| Background automation module | `background/` — SQLite store, runner, priority, reports, fixtures, followup | (above) |
| ML Tier 1 | `ml/xgboost_scorer.py` | 45 |
| ML Tier 3 | `ml/stacking_scorer.py` | 22 |
| Kepler training | `fetch_kepler_tce.py`, `build_training_data.py`, `train_xgboost.py` | 34+25 |
| TESS training | `fetch_tess_toi.py`, `build_tess_training_data.py` | 11+38 |
| Combined training | `build_combined_training_data.py` | 13 |
| Evaluation | `evaluate_scorer.py` — ROC-AUC, F1, reliability diagram | 14 |
| Injection-recovery | `injection_recovery.py` | 25 |
| Star scanner | `star_scanner.py` — TIC priority ranking, background scan loop, JSON log | 38 |
| Candidate ranking | `rank_candidates.py` — composite rank score, Rich table, `--top N` | 12 |
| Batch scan | `batch_scan.py` — text/CSV input, incremental JSON output, `--resume` | 14 |
| Sector coverage | `sector_coverage.py` — query TESS sector availability without download | 10 |
| CNN gate | `count_tess_labels.py` | — |
| Docs | `ML_SCORING.md`, `CNN_SPEC.md`, `DATA_SOURCES.md`, `DECISIONS.md`, `SCORING_MODEL.md §21` | — |
| Docs — automation | `BACKGROUND_SEARCH_AUTOMATION_BLUEPRINT.md`, `BACKGROUND_SEARCH_SQLITE_SCHEMA.md`, `SCHEDULER.md`, `SYSTEM_PROFILE.md` | — |
| README | 14-section rewrite with equations, MLA citations, submission guide, user guide | — |

**Total: 805 passing tests (+ 2 integration_live)**

---

## Blocked

**ML Tier 2 — 1D CNN on phase-folded flux**
- Gate: 5,000+ labeled TESS light curves required
- Current CP count: check with `python Skills/count_tess_labels.py`
- Architecture spec: `docs/CNN_SPEC.md`

---

## Next Actions

1. Run `python Skills/count_tess_labels.py` periodically to monitor CNN gate
2. Once gate opens: implement Tier 2 CNN per `docs/CNN_SPEC.md`
3. Compute `provenance_score` in `vet.py` to unlock `tfop_ready` pathway

---

## Key Design Decisions in Effect

- Bayesian log-score model is default; XGBoost/ensemble are opt-in via `--scorer`
- `OptScore = float | None` — missing diagnostics are neutral (contribute 0)
- Conservative priors: 10% planet_candidate, 20% each false-positive class
- `None` feature scores fail threshold gates conservatively (§15 Guardrails)
- `provenance_score` defaults to 0.0 (blocks `tfop_ready` until computed)
- Never output "confirmed planet" — use "candidate signal"
- Background automation uses SQLite for durable state; JSON fixtures for offline testing
- Background automation obeys human-approval gate — no external submission without review
