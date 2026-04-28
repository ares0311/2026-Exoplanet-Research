# PROJECT STATUS

## Status: Active Development
## Phase: Phase 0 — Scoring Engine Complete; Data Pipeline Next
## Last Updated: 2026-04-28

---

## Completed This Phase

The full scoring and classification engine is implemented, tested, and merged to the active PR.

| Module | Description | Tests |
|---|---|---|
| `schemas.py` | Pydantic v2 typed data contracts (frozen, validated) | 33 |
| `features.py` | 35+ feature extraction functions (`OptScore` pattern) | 89 |
| `hypotheses.py` | 6 Bayesian log-score models + `compute_log_scores()` | 28 |
| `scoring.py` | Softmax posterior + derived scores (FPP, detection confidence, novelty, etc.) | 25 |
| `pathway.py` | Submission pathway classifier (SCORING_MODEL.md §11) | 35 |

**Total: 210 tests — all passing, ruff-clean, mypy-clean**

CI is live via `.github/workflows/ci.yml` (ruff → mypy → pytest on every push).

---

## Current Focus

Build the data pipeline layer that feeds the scoring engine:

1. **`fetch.py`** — MAST/Lightkurve download + provenance
2. **`clean.py`** — detrending, sigma-clip, normalize
3. **`search.py`** — BLS search → `CandidateSignal` list
4. **`vet.py`** — compute `RawDiagnostics` → `CandidateFeatures` via `extract_features()`

---

## Next Milestone

**End-to-end detection of a known TESS planet**

Definition of Done:
- [ ] Fetch a real TESS light curve from MAST
- [ ] Clean and detrend the light curve
- [ ] Run BLS and recover a known transit signal
- [ ] Compute `CandidateFeatures` via `vet.py`
- [ ] Score with `score_candidate()` → posterior + derived scores
- [ ] Classify with `classify_submission_pathway()`
- [ ] Generate a human-readable candidate report
- [ ] Output matches expected pathway for a confirmed planet

---

## Key Design Decisions in Effect

- Bayesian log-score model (no ML classifiers yet)
- `OptScore = float | None` — missing diagnostics are neutral (contribute 0)
- Conservative priors: 10% planet_candidate, 20% each false-positive class
- `None` feature scores fail threshold gates conservatively (§15 Guardrails)
- `provenance_score` blocks `tfop_ready` in v0 (not yet computed)
- All scoring models store version hash for reproducibility

---

## Risks

- BLS detection on real TESS data depends on clean detrending (contamination, systematics)
- Feature extraction quality gates calibration — weights in hypotheses.py are v0 starting points
- Known-object matching requires catalog queries (Lightkurve / astroquery) — network-dependent
- `habitability_interest` v0 uses period as HZ proxy only; stellar parameters not yet in pipeline
