# ROADMAP

## Milestone 1 — Scoring and Classification Engine ✓ COMPLETE

All modules implemented, tested (210 tests), ruff-clean, mypy-clean.

- [x] `schemas.py` — typed Pydantic data contracts
- [x] `features.py` — 35+ normalized feature extraction functions
- [x] `hypotheses.py` — Bayesian log-score models for 6 hypotheses
- [x] `scoring.py` — softmax posterior + FPP, detection confidence, novelty, habitability
- [x] `pathway.py` — submission pathway classifier (SCORING_MODEL.md §11)
- [x] CI via GitHub Actions (ruff → mypy → pytest)
- [x] `.gitignore` — proper Python template
- [x] `CLAUDE.md` — project context for AI coding agents

---

## Milestone 2 — Data Pipeline

Build the Fetch → Clean → Search → Vet chain that produces inputs for the scoring engine.

- [ ] `fetch.py` — query MAST via Lightkurve; return LightCurve + provenance metadata
- [ ] `clean.py` — NaN removal, sigma-clip, normalization, detrending (Lightkurve `flatten()`)
- [ ] `search.py` — BLS search → `CandidateSignal` list; iterative masking for multi-planet
- [ ] `vet.py` — compute `RawDiagnostics` from light curve + signal; call `extract_features()`
- [ ] Integration tests with mocked Lightkurve responses
- [ ] `@pytest.mark.integration_live` tests against real MAST data

---

## Milestone 3 — End-to-End Validation

Recover a known TESS or Kepler planet through the full pipeline.

- [ ] Select a confirmed planet as validation target (e.g. TOI-700d or TRAPPIST-1)
- [ ] Run full pipeline: Fetch → Clean → Search → Vet → Score → Classify
- [ ] Verify `pathway` output matches expected (e.g. `tfop_ready` or `kepler_archive_candidate`)
- [ ] Generate a human-readable candidate report (Markdown)

---

## Milestone 4 — Calibration

The v0 weights in `hypotheses.py` are starting points, not calibrated values.

- [ ] `calibration.py` — reliability curves, Platt scaling, isotonic regression
- [ ] Assemble validation set: confirmed planets, TOIs, KOIs, known false positives, known EBs
- [ ] Measure Brier score, precision-recall, ROC, confusion matrix by hypothesis type
- [ ] Calibrate by period/radius/SNR bins; update priors and weights with evidence

---

## Milestone 5 — Reporting

- [ ] Per-candidate Markdown/HTML report with explanation, plots, scoring breakdown
- [ ] Batch run support (list of targets → ranked candidate table)
- [ ] Export to standard formats (CSV, JSON, FITS header compatible)

---

## Milestone 6 — Injection-Recovery (Survey-Aware Reliability)

- [ ] Inject synthetic transit signals into real light curves
- [ ] Run full pipeline and measure recovery rate by radius, period, stellar type, noise
- [ ] Map completeness function; adjust detection confidence where recovery is poor

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
