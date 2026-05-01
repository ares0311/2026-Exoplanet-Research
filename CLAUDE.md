# CLAUDE.md — Claude Code Project Context

This file is read automatically by Claude Code at session start.
It contains the facts a coding agent needs to work productively without re-reading every document.

---

## Standing Rules

- **Skills directory**: Any standalone `.py` utility script created to perform a task (data processing, report generation, injection-recovery, etc.) must be saved in `Skills/` at the project root. Create the directory if it does not exist. This allows scripts to be discovered and reused across sessions rather than recreated.

---

## Project

**2026 Exoplanet Research**
Citizen-science toolkit for detecting and scoring exoplanet transit candidates from TESS and Kepler/K2 data.

Repository: `ares0311/2026-Exoplanet-Research`
Active branch: `main`
PR #1 merged 2026-04-28

---

## Architecture

```
Fetch → Clean → Search → Vet → Score → Classify
```

Python package: `src/exo_toolkit/`
Tests: `tests/`
Docs: `docs/`
CI: `.github/workflows/ci.yml`

### Module build order (each depends on prior)

| Module | Status | Tests |
|---|---|---|
| `schemas.py` | **done** | `test_schemas.py` (33) |
| `features.py` | **done** | `test_features.py` (89) |
| `hypotheses.py` | **done** | `test_hypotheses.py` (28) |
| `scoring.py` | **done** | `test_scoring.py` (25) |
| `pathway.py` | **done** | `test_pathway.py` (35) |
| `fetch.py` | **done** | `test_fetch.py` (40, 2 live) |
| `clean.py` | **done** | `test_clean.py` (39) |
| `search.py` | **done** | `test_search.py` (43) |
| `vet.py` | **done** | `test_vet.py` (47) |
| `calibration.py` | **done** | `test_calibration.py` (70) |

**Total passing tests: 406 (+ 2 integration_live)**

---

## Core Design Decisions (see docs/DECISIONS.md for full rationale)

- **Bayesian log-score model**: `log_score_i = log_prior_i + weighted_evidence_i`, then `posterior_i = softmax(log_scores)`
- **6 hypotheses**: planet_candidate, eclipsing_binary, background_eclipsing_binary, stellar_variability, instrumental_artifact, known_object
- **OptScore pattern**: `float | None` — `None` means diagnostic not run; missing features contribute 0 to log scores (neutral, no bias)
- **Conservative priors**: planet_candidate = 0.10, others = 0.20 each, known_object = 0.10
- **No ML classifiers** until labeled validation data and calibration infrastructure exist
- **Never output "confirmed planet"** — use "candidate signal" or "follow-up target"
- **Numerically stable softmax**: subtract max before exponentiation

---

## Key Types (schemas.py)

```python
Score    = Annotated[float, Field(ge=0.0, le=1.0)]
OptScore = Annotated[float | None, Field(ge=0.0, le=1.0)]
Mission  = Literal["TESS", "Kepler", "K2"]
SubmissionPathway = Literal[
    "known_object_annotation", "tfop_ready", "planet_hunters_discussion",
    "kepler_archive_candidate", "github_only_reproducibility", "paper_or_preprint_candidate"
]

CandidateSignal      # raw BLS output
CandidateFeatures    # 35 OptScore fields, all default None
HypothesisPosterior  # 6 Score fields, validator enforces sum ≈ 1.0 ±0.01
CandidateScores      # 6 Score fields (fpp, detection_confidence, novelty_score, …)
CandidateExplanation # tuple[str, ...] fields for positive/negative/blocking evidence
ScoringMetadata      # model name, version, commit, config_hash
ScoredCandidate      # full pipeline output
```

All models: `ConfigDict(frozen=True)` — immutable after construction.

### Pipeline result types (frozen dataclasses)

```python
FetchResult(light_curve, provenance: FetchProvenance)
CleanResult(light_curve, provenance: CleanProvenance)
VetResult(diagnostics: RawDiagnostics, features: CandidateFeatures)
# search returns list[CandidateSignal] directly
```

`RawDiagnostics` (frozen dataclass in `features.py`) — 30+ optional float/int fields covering
per-transit depths, odd/even, secondary SNR, stellar params, crowding, flags, catalog matches.

---

## Scoring Pipeline (scoring.py)

```
CandidateFeatures
    → compute_log_scores()      (hypotheses.py)
    → softmax()                 (scoring.py)
    → HypothesisPosterior
    → compute_scores()          (scoring.py)
    → CandidateScores

Public entry point: score_candidate(signal, features, log_priors=None)
    → tuple[HypothesisPosterior, CandidateScores]
```

---

## Pathway Classification (pathway.py)

`classify_submission_pathway(signal, features, posterior, scores, *, provenance_score=0.0, ...)`

Gate order (spec §11):
1. `posterior.known_object >= 0.80` → `known_object_annotation`
2. `fpp >= 0.70` → `github_only_reproducibility`
3. `transit_count < 2` → `planet_hunters_discussion`
4. TESS branch → `tfop_ready` (all 9 conditions) or `planet_hunters_discussion` or `github_only_reproducibility`
5. Kepler/K2 branch → `kepler_archive_candidate` or `github_only_reproducibility`
6. Fallback → `github_only_reproducibility`

`None` feature scores **fail** gate conditions conservatively.
`provenance_score` defaults to 0.0 (blocks `tfop_ready` in v0 — not yet computed).

---

## Quality Commands

```bash
# Run tests (package not pip-installed — use PYTHONPATH)
PYTHONPATH=src python -m pytest

# Lint
ruff check .
ruff check . --fix

# Type-check (must use python -m mypy so stubs from site-packages are visible)
python -m mypy src

# All three together
ruff check . && python -m mypy src && PYTHONPATH=src python -m pytest
```

If pytest fails with `ModuleNotFoundError: No module named 'exo_toolkit'`, add `PYTHONPATH=src`.

`mypy` (bare binary) sees a different package path and reports false import errors for pydantic/numpy.
Always use `python -m mypy src` locally.

---

## Data Pipeline Notes

### fetch.py
- Lazy lightkurve import (inside `fetch_lightcurve()`); `FetchProvenance` records cadence, sectors, pipeline, fetched_at
- Live tests use `@pytest.mark.integration_live` and are excluded from CI

### clean.py
- No lightkurve import at all — calls methods on the passed-in object only
- `CleanProvenance` records n_cadences_raw/cleaned, sigma_clip_sigma, window_length

### search.py
- Uses `astropy.timeseries.BoxLeastSquares` directly (no lightkurve needed)
- Duration grid capped at 90% of `period_min` to satisfy astropy BLS constraint
- Iterative transit masking in pure numpy; `_extract_flux_err` falls back to 1.4826×MAD

### vet.py
- No lightkurve import — pure numpy diagnostics from `lc.time.jd` / `lc.flux.value`
- Computes: individual depths, odd/even comparison, secondary eclipse SNR, transit shape, data-gap fraction
- Catalog diagnostics (stellar params, crowding, flags) pass through as keyword arguments

### calibration.py
- Public API: `compute_metrics`, `fit_calibration`, `apply_calibration`
- Methods: `"platt"` (Platt scaling via scipy Nelder-Mead), `"isotonic"` (PAVA — no sklearn)
- One-vs-rest calibration per hypothesis; renormalized to sum to 1.0 post-calibration
- Metrics: Brier scores, reliability curves, precision/recall/F1, confusion matrix
- All result containers are frozen dataclasses

## What Is Not Yet Built

All pipeline modules are complete.

### Next Step (agreed 2026-05-01)

**End-to-end example notebook** (`notebooks/pipeline_demo.ipynb`):
- Run a real TESS target through the full Fetch → Clean → Search → Vet → Score → Classify pipeline
- Produce a human-readable candidate report with posterior probabilities and explanation
- Stress-test module integration and surface any API rough edges
- Serves as the reference usage example for new contributors

After the notebook: injection-recovery testing, then a CLI entry point (`exo-scan <TIC-ID>`).

---

## Guardrails (SCORING_MODEL.md §15)

- Never output "confirmed planet"
- Always expose false-positive evidence
- Suppress formal submission if key diagnostics are missing
- Store scoring model version with every candidate output
- Prefer conservative classifications

---

## Data Sources

- **TESS**: MAST via Lightkurve (`mission="TESS"`, PDCSAP flux preferred)
- **Kepler/K2**: MAST via Lightkurve (`mission="Kepler"` / `"K2"`)
- **Catalogs**: NASA Exoplanet Archive, TOI list, KOI list, CTOI via astroquery

Focus on lightly-worked targets: later TESS sectors, fainter stars (Tmag 10–14), less-crowded fields.
