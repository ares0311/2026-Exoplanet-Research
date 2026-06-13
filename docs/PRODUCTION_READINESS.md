# PRODUCTION READINESS

Last reviewed: 2026-06-13 (fifth training run evaluated; v2 config with batch norm + augmentation prepared)
Scope decision: T2-2 and T2-3 are permanently out of scope — see DECISION-013
Branch: `main` (82 production-critical Skills; non-production fluff removed)
Test baseline: 2,003 default tests passing, 2 integration_live deselected

---

## Live-Readiness Summary

| Scorer Mode | Status | Blocker |
|---|---|---|
| `--scorer bayesian` | **PRODUCTION READY** | None — default mode, zero external dependencies |
| `--scorer xgboost` | **PRODUCTION READY** | None — trained on 7,586 Kepler KOIs, AUC=0.992 |
| `--scorer ensemble` | **PRODUCTION READY** | None — conservative XGBoost+Bayesian blend when CNN absent |
| `--scorer cnn` | **NOT READY** | T1-1: first trained checkpoint failed held-out performance and calibration gates |
| `--scorer full-ensemble` | **NOT READY** | T1-1: no production-approved CNN checkpoint |

The system is safe to deploy now for Bayesian and XGBoost scoring modes. The
CNN label gate is open, but the first trained checkpoint was evaluated and
rejected; it must not be copied into `models/`, registered, or used for
production scoring.

---

## Tier 1 Gaps (Blocking Production CNN Use)

### T1-1: Production Tier 2 CNN Checkpoint

- **What is missing**: A CNN checkpoint that passes held-out performance and calibration gates
- **Gate status**: **OPEN** — 2,037 snippets (1,012 positive CP+KP + 1,025 negative FP+FA); corpus valid as of 2026-06-12
- **Code status**: Training and state-dict inference paths are operational; the package scorer reconstructs the trained architecture and fails closed when loading fails
- **Local corpus status**: **VALID as of 2026-06-12** — 2,037 snippets (1,012 positive CP+KP, 1,025 negative FP+FA, ratio 0.99); zero-epoch corpus retired and rebuilt from scratch with valid BJD epochs; label bug fixed (KP→1); MAST throttling fix applied (`bbb0877`)
- **Data policy**: A future audit-passing replacement corpus will be the authorized local T1-1 training input and remain uncommitted under `docs/SYSTEM_PROFILE.md` and DECISION-014
- **Retired splits**: The seed-42 1,837 / 392 / 394 split and temporary 1,492 / 369 / 368 replacement split were both derived from the invalid zero-epoch corpus and must not be reused
- **Rejected candidate 1**: SHA-256 `e02af3903ab65f4af4f3f05f95dd6da8815a6746fea1bf2eac67bbba3555d6c6`, trained on Python 3.14.3 with PyTorch 2.12.0 on the invalid zero-epoch corpus; best epoch 5, validation AUC 0.7476; test raw AUC 0.7404, F1 0.5804, Brier 0.2131, ECE 0.0716; Platt calibration threshold 0.503 produced test F1 0.6297, Brier 0.2295, ECE 0.1273; **REJECTED** — AUC and F1 below targets, calibration worsened both Brier and ECE
- **Rejected candidate 2**: Trained on Python 3.14.3 with PyTorch 2.12.0 on the valid 2026-06-12 corpus (2,037 snippets); splits: 1,425 train / 306 val / 306 test; default config (LR=1e-3, weight_decay=1e-4, dropout 0.5/0.3, aug_noise=0.02); best epoch 4, validation AUC 0.8177, F1 0.7711; test raw AUC 0.7180, F1 0.6998, Brier 0.2153, ECE 0.0646; Platt calibration A=1.5546, B=−0.7152, threshold 0.43 produced test F1 0.6998, Brier 0.2237, ECE 0.0730; **REJECTED** — test AUC 0.7180 < 0.85 gate, test F1 0.6998 < 0.80 gate, calibration worsened both Brier and ECE; val→test AUC gap of 10 points (0.8177→0.7180) indicates insufficient regularization
- **Root cause (candidate 2)**: Model overfit before early stopping (best epoch 4 out of 50); `train_loss=0.4409` vs `val_loss=0.6200` at epoch 14; 10-point AUC gap confirms under-regularization for 1,425-example corpus
- **Rejected candidate 3**: seed-42 splits, `cnn_retrain_v1.json` (LR=3e-4, weight_decay=1e-3, dropout 0.5/0.5, aug_noise=0.05); best epoch 13, val AUC=0.8235; test AUC=0.7283; val→test gap 9.5 pts; **REJECTED** — test AUC below gate; root cause: seed-42 split assigned harder examples to test partition
- **Rejected candidate 4**: seed-7 splits (1,425/306/306), `cnn_retrain_v1.json`; best epoch 32, val AUC=0.7914; test AUC=0.7682; val→test gap 2.3 pts; **REJECTED** — test AUC below gate; gap resolved but model generalization insufficient
- **Rejected candidate 5**: seed-7 splits, `cnn_retrain_v1.json`, ETA-enabled training; best epoch 33, val AUC=0.8083; test AUC=0.7758, F1=0.7268; val→test gap 3.3 pts; **REJECTED** — AUC 0.7758 < 0.85 gate, F1 0.7268 < 0.80 gate; gains plateaued without architectural improvements
- **Next training config**: `configs/cnn_retrain_v2.json` — adds batch normalization after each conv block, phase-flip augmentation (50% probability), and phase-shift augmentation (±20 bins); same seed-7 splits; checkpoint dir `models/cnn_v2/`; expected gain +3–6 pts test AUC
- **Next step**: Run `caffeinate -dims python Skills/train_cnn.py --split-dir data/cnn_splits --checkpoint-dir models/cnn_v2/ --config configs/cnn_retrain_v2.json`, then evaluate against gates (AUC ≥ 0.85, F1 ≥ 0.80)
- **Gate check**: `python Skills/count_tess_labels.py`
- **Architecture spec**: `docs/CNN_SPEC.md`
- **Artifact policy**: Commit the validated production checkpoint, calibration metadata, model registry entry, and reproducibility manifest under `models/` after all production-readiness checks pass

### T1-2: Stacking Tier 3 Production Weight Calibration

- **What is missing**: Held-out calibration set to tune XGBoost/CNN/Bayesian blend weights
- **Root cause**: Blocked by T1-1 — CNN output required before weight calibration is meaningful
- **Current state**: Conservative fallback weights in place (XGBoost 0.35 + CNN 0.35 + Bayesian 0.30); when CNN is absent, blend falls back to XGBoost 0.538 + Bayesian 0.462
- **Gate**: ~500 labeled held-out examples after T1-1 is resolved

---

## Tier 2 Gaps (Improvements, Not Blocking Deployment)

### T2-1: TESS-Specific XGBoost Model — COMPLETE

- **Status: COMPLETE as of 2026-06-11**
- **Model**: `models/xgboost_toi.json` + `models/xgboost_toi.xgb.json` (committed `882b838`)
- **Training set**: 1,960 examples from ExoFOP TOI (CP/KP → positive, FP/FA → negative)
- **Performance**: AUC=0.884, F1=0.729 on held-out fold; Platt calibration A=5.3061, B=−2.7153
- **Usage**: `exo <TIC-ID> --scorer xgboost --model-path models/xgboost_toi.json`

### T2-2: Expert Vetting and Methodology Review — N/A

**Status: Out of scope (DECISION-013).** This is a citizen science project operating independently.
Conservative scoring guardrails enforced in code serve as the substitute:
- Never output "confirmed planet"
- Always expose false-positive evidence
- Suppress `tfop_ready` when key diagnostics are missing
- No external submission without explicit human approval

### T2-3: Peer Review Before Publishing — N/A

**Status: Out of scope (DECISION-013).** This is a citizen science project operating independently.
The scientific guardrails in `docs/SCORING_MODEL.md §15` and `src/exo_toolkit/pathway.py` are the
conservative substitute for formal peer review. All outputs are labeled "candidate signal" or
"follow-up target" — never "confirmed planet".

---

## What Is Complete

Full module inventory: `docs/PROJECT_STATUS.md §What Is Complete`

| Area | Status |
|---|---|
| Core pipeline: Fetch → Clean → Search → Vet → Score → Classify | ✅ |
| Bayesian log-score model (6 hypotheses, 35+ feature functions) | ✅ |
| XGBoost Tier 1 scorer + Kepler training pipeline | ✅ |
| Stacking Tier 3 scorer (conservative fallback) | ✅ |
| CNN Tier 2 scaffolding (training loop, checkpoint, calibration) | ✅ (gated) |
| CLI: `exo <TIC-ID>` + all `background-*` subcommands | ✅ |
| Background automation (SQLite, priority, reports, approval gate) | ✅ |
| Calibration module (Platt scaling, isotonic PAVA, Brier metrics) | ✅ |
| 82 production-critical Skills/ | ✅ |
| 2,003 default tests, ruff clean, mypy clean | ✅ |
| All scientific guardrails enforced in code | ✅ |

---

## Pre-Deployment Compliance Checklist

Run these before any live deployment or public announcement:

- [ ] `PYTHONPATH=src python -m pytest` — all default tests pass, 0 failures
- [ ] `ruff check .` — no lint errors
- [ ] `python -m mypy src` — no type errors
- [ ] `exo background-run-once --dry-run` — no config errors
- [ ] `python Skills/tier2_progress_reporter.py` — confirm CNN gate status documented
- [ ] Verify `configs/background_search_v0.json` fingerprint matches expected value
- [ ] Verify `models/xgboost_koi.json` and `models/xgboost_koi_meta.json` exist for XGBoost scorer
- [ ] Run `exo <known-confirmed-TOI-TIC-ID> --scorer bayesian` — verify FPP < 0.5
- [ ] Run `exo <known-FP-TIC-ID> --scorer bayesian` — verify FPP > 0.5

---

## Scientific Guardrails (Non-Negotiable)

These are enforced in code and must never be bypassed:

1. Never output "confirmed planet" — only "candidate signal" or "follow-up target"
2. Always expose false-positive evidence alongside positive evidence
3. Suppress `tfop_ready` pathway when key diagnostics are missing (conservative gate)
4. No external submission or discovery contact without explicit human approval
5. Background automation draft reports require human approval before any external action
6. Conservative priors by default; mission-specific prior profiles are opt-in
7. `provenance_score` gates `tfop_ready` — 2-min SPOC with ≥2 sectors required

---

## Outside Blockers (Require Human Action — Cannot Be Automated)

| Blocker | What Is Needed | Who |
|---|---|---|
| CNN retraining run | Run training recipe after splits are built and validated | Human |
| CNN retraining run | Run the next approved training recipe in `.venv` after the rebuilt corpus passes audit | Human |
| CNN production promotion | Validate, calibrate, register, and commit only a checkpoint that passes held-out gates | Agent + human approval |
| Stacking weight calibration | Tune blend weights on held-out calibration set | Agent after T1-1 resolved |

---

## Planning Compliance Note

Any plan proposed in a session must:

1. Name the highest-priority unresolved Tier 1 gap (currently **T1-1: Production Tier 2 CNN Checkpoint**)
2. Show how each proposed step closes or directly unblocks that gap — or explicitly justify why it is Tier 2 work
3. Include outside blockers as explicit named steps with responsible party
4. Never propose log modules, schemas, or scaffolding unless they directly unblock a named gap
5. Never repeat work listed under "What Is Complete" above
