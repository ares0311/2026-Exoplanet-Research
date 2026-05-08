# PROJECT STATUS

## Status: Active Development
## Phase: Phase 2 — ML Ensemble Complete; CNN Tier-2 Gated on Data
## Last Updated: 2026-05-08

---

## What Is Complete

| Area | Key Files | Tests |
|------|-----------|-------|
| Scoring engine | `schemas`, `features`, `hypotheses`, `scoring`, `pathway` | 210 |
| Data pipeline | `fetch`, `clean`, `search`, `vet`, `calibration` | 236 |
| CLI | `cli.py` — `exo <TIC-ID>` with scorer options, batch, report-md | 20 |
| ML Tier 1 | `ml/xgboost_scorer.py` | 45 |
| ML Tier 3 | `ml/stacking_scorer.py` | 22 |
| Training pipelines | `Skills/` — Kepler KOI + TESS TOI + combined | 34+38+17 |
| Evaluation | `Skills/evaluate_scorer.py` — ROC-AUC, reliability diagram | 14 |
| Injection-recovery | `Skills/injection_recovery.py` | 25 |
| Docs | `ML_SCORING.md`, `CNN_SPEC.md`, `DATA_SOURCES.md`, `DECISIONS.md` | — |

**Total: 664 passing tests (+ 2 integration_live)**

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
