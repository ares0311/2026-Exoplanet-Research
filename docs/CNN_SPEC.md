# ML Tier-2 — 1D CNN Architecture Specification

**Status: IMPLEMENTATION SCAFFOLD BUILT — production training/use remains gated on 5,000+ labeled TESS light curves**

Track local Tier-2 readiness with the offline reporter:

```bash
python Skills/tier2_progress_reporter.py \
  --labels data/exofop_ctoi_labels.json \
  --output reports/tier2_status.md \
  --json-output reports/tier2_status.json
```

Check the live production-data gate with `python Skills/count_tess_labels.py`
only when live network access is intentionally approved. It queries ExoFOP and
must not run in default validation.

The repository now includes the CNN wrapper, training loop, checkpoint helpers,
calibration helpers, inference batcher, CLI flags, and 3-tier stacking support.
Those pieces are tested with offline fixtures and injectable models. They do
not imply that a production CNN checkpoint is available or that the CNN should
be used for scientific claims before the TESS label gate and calibration checks
are satisfied.

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
| Framework | PyTorch when installed; graceful no-op/unavailable paths when absent |
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

CLI integration is available through:

```bash
exo TIC-150428135 --scorer cnn --cnn-checkpoint path/to/best.pt
exo TIC-150428135 --scorer full-ensemble --model-path model.json --cnn-checkpoint path/to/best.pt
```

The CLI scorer path can load a CNN checkpoint and builds a normalized
phase-folded snippet from the cleaned light curve plus vetted
`CandidateSignal`. CNN probabilities remain experimental review metadata until
the production label gate opens and a calibrated checkpoint is registered; they
must not drive formal submission pathways.

---

## Implementation Checklist

- [x] `src/exo_toolkit/ml/cnn_scorer.py` — checkpoint wrapper and predict/batch interface
- [x] `Skills/labelled_lc_collector.py` — extract, phase-fold, bin, and store labelled snippets
- [x] `Skills/cnn_feature_augmenter.py` — noise, shift, scale, and reverse snippets for augmentation
- [x] `Skills/build_cnn_training_data.py` — assemble offline train/val/test splits from existing labelled snippet artifacts
- [x] `Skills/cnn_split_validator.py` — validate offline split manifests and train/val/test artifacts before training
- [x] `tests/test_cnn_scorer.py` — offline wrapper and stacking integration tests
- [x] Update `ml/stacking_scorer.py` — add CNN weight for 3-tier blends
- [x] Update `docs/ML_SCORING.md` — document CNN mode and production gate
- [x] Update CLI `--scorer` to accept `cnn` and `full-ensemble`
- [x] `Skills/train_cnn.py` — PyTorch training script with early stopping and graceful `NO_TORCH`
- [x] Wire vetted phase-folded snippets from the pipeline into CNN inference rows with neutral fallback when unavailable
- [ ] Train, calibrate, and register a production CNN checkpoint after the label gate opens
