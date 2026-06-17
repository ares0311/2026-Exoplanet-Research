# PRODUCTION READINESS

Last reviewed: 2026-06-17 (pre-fix Kepler corpus rejected for non-finite flux; finite-flux rebuild required)
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
- **Gate status**: **OPEN** — awaiting finite-flux Kepler JSONL rebuild, Kepler pretraining, TESS fine-tuning, and held-out production evaluation from `docs/CNN_PRODUCTION_RUNBOOK.md`
- **Code status**: Training and state-dict inference paths are operational; the package scorer reconstructs the trained architecture and fails closed when loading fails
- **Prior local corpus status**: **VALID as of 2026-06-12** — 2,037 snippets (1,012 positive CP+KP, 1,025 negative FP+FA, ratio 0.99); zero-epoch corpus retired and rebuilt from scratch with valid BJD epochs; label bug fixed (KP→1); MAST throttling fix applied (`bbb0877`)
- **Local corpus status**: **KEPLER REBUILD REQUIRED** — TESS v2 complete at 2,619 snippets; pre-fix Kepler JSONL had 7,454 rows but 7,132 rows contained non-finite flux, leaving only 322 finite examples after filtering
- **Data policy**: Local CNN corpora, splits, checkpoints, and training logs remain uncommitted unless a future promotion decision explicitly commits a production checkpoint, calibration metadata, registry entry, and reproducibility manifest under `models/`; production-relevant local artifact state must remain visible in `docs/LOCAL_ARTIFACT_LEDGER.md` and `artifacts/manifests/local_artifacts.json`
- **Retired splits**: The seed-42 1,837 / 392 / 394 split and temporary 1,492 / 369 / 368 replacement split were both derived from the invalid zero-epoch corpus and must not be reused
- **Rejected candidate 1**: SHA-256 `e02af3903ab65f4af4f3f05f95dd6da8815a6746fea1bf2eac67bbba3555d6c6`, trained on Python 3.14.3 with PyTorch 2.12.0 on the invalid zero-epoch corpus; best epoch 5, validation AUC 0.7476; test raw AUC 0.7404, F1 0.5804, Brier 0.2131, ECE 0.0716; Platt calibration threshold 0.503 produced test F1 0.6297, Brier 0.2295, ECE 0.1273; **REJECTED** — AUC and F1 below targets, calibration worsened both Brier and ECE
- **Rejected candidate 2**: Trained on Python 3.14.3 with PyTorch 2.12.0 on the valid 2026-06-12 corpus (2,037 snippets); splits: 1,425 train / 306 val / 306 test; default config (LR=1e-3, weight_decay=1e-4, dropout 0.5/0.3, aug_noise=0.02); best epoch 4, validation AUC 0.8177, F1 0.7711; test raw AUC 0.7180, F1 0.6998, Brier 0.2153, ECE 0.0646; Platt calibration A=1.5546, B=−0.7152, threshold 0.43 produced test F1 0.6998, Brier 0.2237, ECE 0.0730; **REJECTED** — test AUC 0.7180 < 0.85 gate, test F1 0.6998 < 0.80 gate, calibration worsened both Brier and ECE; val→test AUC gap of 10 points (0.8177→0.7180) indicates insufficient regularization
- **Root cause (candidate 2)**: Model overfit before early stopping (best epoch 4 out of 50); `train_loss=0.4409` vs `val_loss=0.6200` at epoch 14; 10-point AUC gap confirms under-regularization for 1,425-example corpus
- **Rejected candidate 3**: seed-42 splits, `cnn_retrain_v1.json` (LR=3e-4, weight_decay=1e-3, dropout 0.5/0.5, aug_noise=0.05); best epoch 13, val AUC=0.8235; test AUC=0.7283; val→test gap 9.5 pts; **REJECTED** — test AUC below gate; root cause: seed-42 split assigned harder examples to test partition
- **Rejected candidate 4**: seed-7 splits (1,425/306/306), `cnn_retrain_v1.json`; best epoch 32, val AUC=0.7914; test AUC=0.7682; val→test gap 2.3 pts; **REJECTED** — test AUC below gate; gap resolved but model generalization insufficient
- **Rejected candidate 5**: seed-7 splits, `cnn_retrain_v1.json`, ETA-enabled training; best epoch 33, val AUC=0.8083; test AUC=0.7758, F1=0.7268; val→test gap 3.3 pts; **REJECTED** — AUC 0.7758 < 0.85 gate, F1 0.7268 < 0.80 gate; gains plateaued without architectural improvements
- **Rejected candidate 6**: `cnn_retrain_v2.json` (use_batch_norm=true, flip=true, shift=20); best epoch 1 (val AUC=0.7344); val loss exploded 0.68→1.07; val AUC below 0.5 by epoch 3; early stopping epoch 11; **REJECTED** — BatchNorm1d running stats do not converge in 22 mini-batches/epoch; stale stats misnormalize val set in eval mode; BN incompatible with this dataset size
- **Rejected candidate 7**: seed-7 splits, `cnn_retrain_v2b.json` (use_batch_norm=false, flip=true, shift=20); best epoch 23, val AUC=0.7887; test raw AUC=0.7527, F1=0.7214, Brier=0.2070, ECE=0.0990; Platt A=1.4732, B=−0.6896, threshold=0.50; calibrated test F1=0.7202, Brier=0.2200, ECE=0.1168; val→test gap 3.6 pts; **REJECTED** — test AUC 0.7527 < 0.85 gate, F1 0.7202 < 0.80 gate; root cause: model has ~435K parameters against 1,425 examples (massively overparameterized); augmentation reduced gap slightly but also depressed val AUC relative to candidate 5
- **Root cause (systematic, candidates 2–7)**: Dense(256) layer alone has 410K parameters; with 1,425 training examples the model cannot generalize past ~0.78 test AUC regardless of regularization; model capacity must be reduced to match the dataset size
- **Rejected candidate 8**: seed-7 splits, `cnn_retrain_v3.json` (Conv 8/16/32, Dense 64, dropout 0.3, flip+shift); best epoch 11, val AUC=0.7734; test raw AUC=0.7094, F1=0.6805, Brier=0.2238, ECE=0.0992; Platt A=1.3045, B=−0.6584; calibrated test F1=0.6792, Brier=0.2307, ECE=0.1282; val→test gap 6.4 pts; **REJECTED** — test AUC 0.7094 < 0.85 gate; root cause: halved conv channels reduced feature extraction capacity; flip+shift augmentation net harmful across all candidates where it was tested
- **Rejected candidate 9**: seed-7 splits, `cnn_retrain_v3b.json` (Conv 16/32/64, Dense 128, dropout 0.4, no flip/shift); best epoch 18, val AUC=0.7807; test raw AUC=0.7573, F1=0.7257, Brier=0.2051, ECE=0.0867; Platt A=1.5114, B=−0.7136; calibrated test F1=0.7207, Brier=0.2155, ECE=0.1230; val→test gap 2.3 pts; **REJECTED** — test AUC 0.7573 < 0.85 gate, F1 0.7207 < 0.80 gate; reducing Dense(256→128) slightly hurt val AUC without improving test AUC vs C5
- **Rejected candidate 10 (ensemble)**: 3-seed ensemble of `cnn_retrain_v1.json` (seeds 7, 13, 99); individual val AUCs: 0.7914, 0.7848, 0.8022; ensemble val AUC=0.8022; test raw AUC=0.7670, F1=0.7317, Brier=0.2057, ECE=0.1260; Platt A=1.5945, B=−0.7796; calibrated test F1=0.7317; **REJECTED** — test AUC 0.7670 < 0.85 gate, F1 0.7317 < 0.80 gate; ensemble is *worse* than best single model (C5: 0.7758); members too correlated on 1,425 examples to provide diversity gain
- **Systematic ceiling confirmed (candidates 2–10)**: All 10 candidates (single-model and 3-seed ensemble) produced test AUC 0.71–0.78; ceiling is a data-size constraint; 1,425 training examples cannot drive this architecture to 0.85 AUC regardless of tuning, regularization, augmentation, or ensembling strategy
- **Authorized path forward — BOTH Path A and Path B run in parallel**:
  - **Path A — More labeled TESS data**: Download additional phase-folded snippets from MAST for ExoFOP confirmed planets and confirmed false positives; target ≥ 5,000 training examples; retrain with expanded corpus
  - **Path B — Kepler→TESS transfer learning**: Pre-train CNN on Kepler TCE phase-folded light curves (tens of thousands of examples); fine-tune final dense layers on 1,425 TESS examples; most robust path to exceeding 0.85 test AUC on current TESS snippet count
- **Current authorized runbook**: `docs/CNN_PRODUCTION_RUNBOOK.md`
- **Current promotion gate**: raw held-out test AUC ≥ 0.85; calibrated held-out test F1 ≥ 0.80; Platt calibration must not worsen held-out test Brier score or ECE
- **Current data gate**: Kepler split validator must pass on a rebuilt finite-flux corpus before any Kepler pretraining
- **Gate check**: `.venv/bin/python Skills/evaluate_cnn_checkpoint.py --split-dir data/tess_cnn_splits --checkpoint checkpoints/cnn_tess_finetuned/best.pt --output-calibration checkpoints/cnn_tess_finetuned/calibration.json`
- **Architecture spec**: `docs/CNN_SPEC.md`
- **Artifact policy**: Keep `git add .` safe through `.gitignore`; commit local artifact status in the artifact ledger; commit the validated production checkpoint, calibration metadata, model registry entry, and reproducibility manifest under `models/` only after all production-readiness checks pass and the human approves promotion

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
| Kepler finite-flux rebuild | Run `docs/CNN_PRODUCTION_RUNBOOK.md` Steps 0-2 on the local Mac in `.venv` | Human |
| CNN split/training run | Run `docs/CNN_PRODUCTION_RUNBOOK.md` remaining training/evaluation steps after Kepler split validation passes | Human |
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
