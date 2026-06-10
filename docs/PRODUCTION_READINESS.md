# PRODUCTION READINESS

Last reviewed: 2026-06-10
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
- **Gate status**: **OPEN as of 2026-06-06** — 1,324 positive (CP+KP) + 1,344 negative (FP+FA) = 2,668 total
- **Code status**: Training and state-dict inference paths are operational; the package scorer reconstructs the trained architecture and fails closed when loading fails
- **Local corpus status**: **INVALID as of 2026-06-10** — all 2,623 nominally usable rows in `data/tess_snippets.jsonl` were phase-folded with `epoch_bjd=0.0` because the local `data/tess_toi.csv` predates epoch-column ingestion; the corpus and every split derived from it are retired
- **Data policy**: A future audit-passing replacement corpus will be the authorized local T1-1 training input and remain uncommitted under `docs/SYSTEM_PROFILE.md` and DECISION-014
- **Retired splits**: The seed-42 1,837 / 392 / 394 split and temporary 1,492 / 369 / 368 replacement split were both derived from the invalid zero-epoch corpus and must not be reused
- **Rejected candidate**: SHA-256 `e02af3903ab65f4af4f3f05f95dd6da8815a6746fea1bf2eac67bbba3555d6c6`, trained on Python 3.14.3 with PyTorch 2.12.0; best epoch 5, validation AUC 0.7476
- **Held-out test result**: raw AUC 0.7404, F1 0.5804, Brier 0.2131, ECE 0.0716; validation-fitted Platt calibration and threshold 0.503 produced test F1 0.6297, Brier 0.2295, and ECE 0.1273
- **Promotion decision**: **REJECTED** — AUC and F1 are below the documented 0.85 and 0.80 targets, and calibration worsened both Brier score and ECE
- **Root cause found**: Missing transit epochs left genuine transit events uncentered in phase; this is not an architecture-only generalization problem
- **Code remediation**: TOI fetch/download now fail closed on missing or non-BJD epochs, corpus audit reports invalid epochs, normalized padding is consistent at zero, and grouped development experiments cannot read the sealed promotion holdout
- **Next step**: Re-fetch the current ExoFOP TOI table, rebuild the local light-curve corpus from scratch with valid BJD epochs, audit it, and create a completely fresh grouped development/promotion split
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

### T2-1: TESS-Specific XGBoost Model

- **What is missing**: XGBoost model trained on TESS TOI labels (current model trained on Kepler KOIs)
- **Code status**: Complete — `Skills/build_tess_training_data.py`, `Skills/fetch_tess_toi.py`, `Skills/train_xgboost.py` all exist and are tested
- **Action**: Run training pipeline on TESS TOI CP/FP labels when labels are available
- **Outside blocker**: Sufficient TESS CP/FP label quality (partially available from ExoFOP now, but a larger confirmed set is better)

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
| CNN corpus rebuild | Re-fetch the epoch-bearing TOI table and rebuild all local phase-folded snippets from scratch | Human |
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
