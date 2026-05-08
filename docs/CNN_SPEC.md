# ML Tier-2 — 1D CNN Architecture Specification

**Status: NOT YET BUILT — gated on 5,000+ labeled TESS light curves**

Check gate: `python Skills/count_tess_labels.py`

---

## Motivation

XGBoost (Tier 1) operates on 35 tabular `OptScore` features — statistical
summaries of the transit signal.  A 1D CNN operating directly on the
phase-folded flux array learns transit morphology without hand-crafted features,
capturing subtle shape signals that scalars cannot represent (limb-darkening
profile, ingress/egress asymmetry, secondary eclipse shape).

Reference architecture: Shallue & Vanderburg 2018 (AstroNet).

---

## Input Representation

```
phase-folded flux array — shape (L,) where L = 201 (configurable)
```

Construction:
1. Phase-fold the detrended light curve at the BLS period
2. Bin into L equal phase bins; use median per bin
3. Normalize: subtract median, divide by MAD × 1.4826
4. Pad or truncate to exactly L bins

This treats the folded transit as a 1D "image" of width L.

---

## Architecture

```
Input: (L, 1)
  Conv1D(16, kernel=5, padding=same) → ReLU → MaxPool(2)
  Conv1D(32, kernel=5, padding=same) → ReLU → MaxPool(2)
  Conv1D(64, kernel=3, padding=same) → ReLU → MaxPool(2)
  Flatten
  Dense(256) → ReLU → Dropout(0.5)
  Dense(64)  → ReLU → Dropout(0.3)
  Dense(1)   → Sigmoid
Output: P(planet_candidate) ∈ [0, 1]
```

This is a minimal adaptation of Shallue & Vanderburg 2018, reduced in depth
to fit the smaller TESS dataset.

---

## Training

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch (preferred) or pure NumPy for portability |
| Loss | Binary cross-entropy |
| Optimizer | Adam, lr=1e-3 |
| Batch size | 64 |
| Epochs | 50 (early stopping on val loss, patience=10) |
| Val split | 20% stratified hold-out |

---

## Data Requirements

- **Minimum**: 5,000 labeled TESS light curves (CP + FP)
- **Preferred**: 10,000+ with balanced classes
- Sources: ExoFOP CP/FP, Planet Hunters TESS, TFOP follow-up confirmed FPs
- Augmentation: time-shift, flux noise injection, phase jitter

---

## Integration

The CNN scorer fits alongside `scoring.py` and `ml/xgboost_scorer.py`:

```
CNN score → P(planet_candidate)
```

Stacking scorer (Tier 3, `ml/stacking_scorer.py`) blends:
```
P_blend = w_xgb * P_xgb + w_cnn * P_cnn + w_bayes * P_bayesian
```

Default weights (to be calibrated after training): `0.35 / 0.35 / 0.30`.

---

## Implementation Checklist

- [ ] `src/exo_toolkit/ml/cnn_scorer.py` — model definition, train, predict
- [ ] `Skills/build_cnn_training_data.py` — fetch + fold + normalize light curves
- [ ] `tests/test_cnn_scorer.py` — ≥ 30 tests
- [ ] Update `ml/stacking_scorer.py` — add CNN weight
- [ ] Update `docs/ML_SCORING.md` — document CNN mode
- [ ] Update CLI `--scorer` to accept `cnn` and `full-ensemble`
- [ ] `Skills/train_cnn.py` — training script with early stopping
