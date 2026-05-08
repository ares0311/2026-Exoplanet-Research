# ML Scoring Architecture

This document describes the machine-learning pipeline used to score exoplanet
transit candidates alongside the Bayesian log-score model.

---

## Overview

The toolkit supports three scorer modes, selectable via the CLI:

| Mode | Flag | Description |
|------|------|-------------|
| Bayesian (default) | `--scorer bayesian` | Log-score model over 6 hypotheses |
| XGBoost | `--scorer xgboost --model-path <path>` | Binary XGBoost on tabular features |
| Ensemble | `--scorer ensemble --model-path <path>` | Weighted average of XGBoost + Bayesian |

---

## Tier 1 — XGBoost on Tabular Features

**Module**: `src/exo_toolkit/ml/xgboost_scorer.py`  
**Tests**: `tests/test_xgboost_scorer.py` (45 tests)

### Input

The 35 `OptScore` fields from `CandidateFeatures`.  All fields are optional
(`float | None`); `None` is passed as `np.nan` and handled natively by
XGBoost's missing-value splitting — no imputation required.

### Model

- Binary classifier: planet candidate (1) vs. false positive (0)
- Native XGBoost API (`xgb.DMatrix` + `xgb.train`)
- Default params: `max_depth=4`, `eta=0.1`, `subsample=0.8`,
  `colsample_bytree=0.8`, `eval_metric=["logloss","auc"]`
- 100 boosting rounds by default

### Serialisation

Each saved model consists of two files:

```
model.json          # metadata: feature names, params, feature importance
model.xgb.json      # XGBoost booster weights (native format)
```

Load with `XGBoostScorer.load(path)`.

### CLI usage

```bash
exo TIC-150428135 --scorer xgboost --model-path data/kepler_model.json
```

Output row gains `"xgb_planet_probability"` key.

---

## Tier 3 — Stacking Scorer (Weighted Ensemble)

**Module**: `src/exo_toolkit/ml/stacking_scorer.py`  
**Tests**: `tests/test_stacking_scorer.py` (22 tests)

Blends XGBoost P(planet) with Bayesian P(planet_candidate):

```
P_blend = w * P_xgb + (1 - w) * P_bayesian
```

Default weight: `w = 0.5`.  Falls back to Bayesian-only when no XGBoost model
is loaded (`has_xgb == False`).

### Factory methods

```python
StackingScorer.from_model_path("model.json", xgb_weight=0.6)
StackingScorer.bayesian_only()   # no-op pass-through
```

### CLI usage

```bash
exo TIC-150428135 --scorer ensemble --model-path data/kepler_model.json
```

Output row gains `"ensemble_planet_probability"` key.

---

## Training Pipeline

### Data sources

| Source | Script | Description |
|--------|--------|-------------|
| Kepler KOI (DR25) | `Skills/fetch_kepler_tce.py` | ~8,000 confirmed + FP KOIs from NASA Exoplanet Archive |
| TESS TOI (ExoFOP) | `Skills/fetch_tess_toi.py` | CP + FP dispositions from ExoFOP-TESS |

### Feature extraction

| Script | Input | Output |
|--------|-------|--------|
| `Skills/build_training_data.py` | Kepler KOI CSV | `data/kepler_training.pkl` |
| `Skills/build_tess_training_data.py` | TESS TOI CSV | `data/tess_training.pkl` |

Both produce a pickle: `{"features_list": list[CandidateFeatures], "labels": list[int]}`.

#### Kepler column mapping

| KOI column | CandidateFeatures field |
|-----------|------------------------|
| `koi_model_snr` | `snr_score`, `log_snr_score` |
| `koi_count` | `transit_count_score` |
| `koi_duration` / `koi_period` | `duration_plausibility_score`, `duration_implausibility_score` |
| `koi_depth` | `large_depth_score` |
| `koi_prad` | `companion_radius_too_large_score` |
| `koi_dikco_msky` | `centroid_offset_score` |

27 of 35 `OptScore` fields remain `None` — handled natively by XGBoost.

#### TESS column mapping

| TOI column | CandidateFeatures field |
|-----------|------------------------|
| `snr` | `snr_score`, `log_snr_score` |
| `n_sectors` | `transit_count_score` (proxy) |
| `duration_hours` / `period_days` | `duration_plausibility_score`, `duration_implausibility_score` |
| `depth_mmag` × 1000 → ppm | `large_depth_score` |
| `planet_radius_earth` | `companion_radius_too_large_score` |

### Training

```bash
# Step 1 — fetch data
python Skills/fetch_kepler_tce.py --output data/kepler_koi.csv

# Step 2 — build features
python Skills/build_training_data.py --input data/kepler_koi.csv \
    --output data/kepler_training.pkl

# Step 3 — train & evaluate
python Skills/train_xgboost.py --data data/kepler_training.pkl \
    --output data/kepler_model.json --k-folds 5
```

`train_xgboost.py` prints stratified k-fold metrics (ROC-AUC, F1, precision,
recall) and saves the final model trained on the full dataset.

---

## Evaluation

```bash
python Skills/evaluate_scorer.py --data data/kepler_training.pkl \
    --model data/kepler_model.json --k-folds 5 --plot reports/roc.png
```

Prints a comparison table:

```
Scorer             auc          f1   precision      recall         acc
-----------------------------------------------------------------------
bayesian        0.7432      0.6841      0.7200      0.6511      0.7150
xgboost         0.8915      0.8203      0.8450      0.7971      0.8375
```

---

## Design Decisions

### Why XGBoost first?

- The 35 `OptScore` fields are already a clean tabular vector
- XGBoost handles `None` (→ `np.nan`) natively — no imputation logic
- Trains in seconds on ~8,000 Kepler examples
- Interpretable: feature importance via `booster.get_score(importance_type="gain")`

### Why Bayesian remains the default?

The Bayesian log-score model produces calibrated posteriors over all 6
hypotheses, not just planet vs. FP.  It works with zero training data and
produces human-readable evidence strings via `CandidateExplanation`.

XGBoost is an **optional augmentation** — it adds `xgb_planet_probability` to
the output row but does not replace the Bayesian posterior.

### Label quality

- "CANDIDATE" KOIs and "PC" TOIs are excluded — uncertain labels
- Train only on confirmed planets vs. confirmed false positives
- Kepler DR25 Robovetter labels (Thompson et al. 2018) are high-quality

### Tier 2 (CNN on phase-folded flux)

Not yet built.  Requires 5,000+ TESS labels.  See `docs/ROADMAP.md`.

---

## Guardrails

All ML scorer outputs are subject to the same guardrails as the Bayesian model
(see `docs/SCORING_MODEL.md §15`):

- Never output "confirmed planet"
- Always expose false-positive evidence
- Suppress formal submission when key diagnostics are missing
- Store scoring model version with every candidate output
