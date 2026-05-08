# ROADMAP

## Milestone 1 — Scoring and Classification Engine ✓ COMPLETE

- [x] `schemas.py` — typed Pydantic data contracts
- [x] `features.py` — 35+ normalized feature extraction functions
- [x] `hypotheses.py` — Bayesian log-score models for 6 hypotheses
- [x] `scoring.py` — softmax posterior + FPP, detection confidence, novelty, habitability
- [x] `pathway.py` — submission pathway classifier (SCORING_MODEL.md §11)
- [x] CI via GitHub Actions (ruff → mypy → pytest)
- [x] `CLAUDE.md` — project context for AI coding agents

---

## Milestone 2 — Data Pipeline ✓ COMPLETE

- [x] `fetch.py` — query MAST via Lightkurve; return LightCurve + provenance metadata
- [x] `clean.py` — NaN removal, sigma-clip, normalization, detrending
- [x] `search.py` — BLS search → `CandidateSignal` list; iterative masking for multi-planet
- [x] `vet.py` — compute `RawDiagnostics` from light curve + signal; call `extract_features()`
- [x] `@pytest.mark.integration_live` tests against real MAST data

---

## Milestone 3 — End-to-End Validation ✓ COMPLETE

- [x] `notebooks/pipeline_demo.ipynb` — TOI-700 (TIC 150428135) full pipeline walkthrough
- [x] All 6 stages covered: Fetch → Clean → Search → Vet → Score → Classify
- [x] Human-readable candidate report rendered as Markdown in notebook
- [x] Figures: raw vs. cleaned flux, phase-folded transit, posterior bar chart, all-signals grid

---

## Milestone 4 — Calibration ✓ COMPLETE

- [x] `calibration.py` — reliability curves, Platt scaling (scipy), isotonic regression (PAVA)
- [x] One-vs-rest calibration per hypothesis; renormalized to sum to 1.0
- [x] Metrics: Brier scores, reliability curves, precision/recall/F1, confusion matrix
- [x] `Skills/train_xgboost.py` — includes post-training Platt calibration step

---

## Milestone 5 — Reporting ✓ COMPLETE

- [x] Rich-formatted candidate report via `exo <TIC-ID>` CLI
- [x] JSON output via `--output`
- [x] Scorer selection via `--scorer [bayesian|xgboost|ensemble]` and `--model-path`

---

## Milestone 6 — Injection-Recovery ✓ COMPLETE

- [x] `Skills/injection_recovery.py` — inject synthetic box transits, recover via BLS
- [x] Measures recovery rate by radius, period, noise level
- [x] 25 tests in `tests/test_injection_recovery.py`

---

## Milestone 7 — ML Ensemble Scorer ✓ COMPLETE (Tier-2 CNN gated)

- [x] Tier 1 — XGBoost on tabular features (`ml/xgboost_scorer.py`, 45 tests)
- [x] Tier 3 — Stacking scorer blending XGBoost + Bayesian (`ml/stacking_scorer.py`, 22 tests)
- [x] Kepler training pipeline (`Skills/fetch_kepler_tce.py`, `build_training_data.py`, `train_xgboost.py`)
- [x] TESS training pipeline (`Skills/fetch_tess_toi.py`, `build_tess_training_data.py`)
- [x] Evaluation framework (`Skills/evaluate_scorer.py`, ROC-AUC, F1, reliability diagrams)
- [x] Combined training data (`Skills/build_combined_training_data.py`)
- [ ] Tier 2 — 1D CNN on phase-folded flux (**BLOCKED: requires 5,000+ TESS labels**)
  - Architecture spec: `docs/CNN_SPEC.md`
  - Gate check: `Skills/count_tess_labels.py`

---

## Milestone 8 — Future

- [ ] Provenance score computation (unlocks `tfop_ready` pathway)
- [ ] Tier 2 CNN once TESS label threshold is met
- [ ] Web API / batch processing service
- [ ] Per-star candidate ranking across all sectors

---

## Decision Tree (current implementation)

```
known_object posterior ≥ 0.80      → known_object_annotation
FPP ≥ 0.70                         → github_only_reproducibility
transit_count < 2                  → planet_hunters_discussion

TESS:
  all 9 tfop conditions met        → tfop_ready
  detection_confidence ≥ 0.45      → planet_hunters_discussion
  otherwise                        → github_only_reproducibility

Kepler/K2:
  p_planet ≥ 0.65, novelty ≥ 0.70,
  FPP ≤ 0.35                       → kepler_archive_candidate
  otherwise                        → github_only_reproducibility
```

See `src/exo_toolkit/pathway.py` and `docs/SCORING_MODEL.md §11` for full threshold values.
