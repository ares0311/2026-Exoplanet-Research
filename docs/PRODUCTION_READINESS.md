# PRODUCTION READINESS

Last reviewed: 2026-06-09
Scope decision: T2-2 and T2-3 are permanently out of scope — see DECISION-013
Branch: `claude/review-markdown-docs-SwVnR`
Test baseline: 1,939 default tests passing (core package), 2 integration_live deselected
Full suite: 6,385 default tests passing (2026-06-03)
TESS download: COMPLETE — 2,636 targets, `data/tess_snippets.jsonl`

---

## Live-Readiness Summary

| Scorer Mode | Status | Blocker |
|---|---|---|
| `--scorer bayesian` | **PRODUCTION READY** | None — default mode, zero external dependencies |
| `--scorer xgboost` | **PRODUCTION READY** | None — trained on 7,586 Kepler KOIs, AUC=0.992 |
| `--scorer ensemble` | **PRODUCTION READY** | None — conservative XGBoost+Bayesian blend when CNN absent |
| `--scorer cnn` | **NOT READY** | T1-1: CNN training pipeline must be run (gate is open, data ready) |
| `--scorer full-ensemble` | **NOT READY** | T1-1: CNN checkpoint required first |

The system is safe to deploy now for Bayesian and XGBoost scoring modes. The CNN label gate is open (2,668 quality labels as of 2026-06-06) and TESS download is complete — training pipeline can now be executed.

---

## Tier 1 Gaps (Blocking Production CNN Use)

### T1-1: Production Tier 2 CNN Checkpoint

- **What is missing**: A trained, calibrated 1D CNN checkpoint on TESS phase-folded flux
- **Gate status**: **OPEN as of 2026-06-06** — 1,324 positive (CP+KP) + 1,344 negative (FP+FA) = 2,668 total
- **Download status**: **COMPLETE** — 2,636 TESS TOI targets downloaded to `data/tess_snippets.jsonl`
- **What is missing**: A trained, calibrated 1D CNN checkpoint — the training pipeline must now be run
- **Code status**: Complete — `ml/cnn_scorer.py`, `Skills/train_cnn.py`, `labelled_lc_collector.py`, `snippet_normalizer.py`, `cnn_split_validator.py`, `cnn_calibrator.py`, `build_cnn_training_data.py` (now with JSONL support), and all supporting data pipeline utilities exist and are tested
- **Next step**: Run CNN training pipeline — see `docs/CNN_SPEC.md` for full recipe
- **Gate check**: `python Skills/count_tess_labels.py`
- **Architecture spec**: `docs/CNN_SPEC.md`
- **Outside blocker**: CNN training run (~hours) must run locally with PyTorch in venv

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
| 415+ Skills/ utility scripts | ✅ |
| 6,385+ default tests, ruff clean, mypy clean | ✅ |
| Milestones 1–47 | ✅ |
| TESS light curve download (2,636 targets → `data/tess_snippets.jsonl`) | ✅ |
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
| ~~TESS light curve download~~ | ~~DONE~~ — 2,636 targets in `data/tess_snippets.jsonl` | ✅ Complete |
| CNN training run | `caffeinate -i python Skills/train_cnn.py --splits-dir data/cnn_splits --output-dir models/cnn/` — requires PyTorch in venv | Agent + human (run locally) |
| CNN production training | Verify checkpoint quality, calibrate with `cnn_calibrator.py` | Agent after training complete |
| Stacking weight calibration | Tune blend weights on held-out calibration set | Agent after T1-1 resolved |

---

## Planning Compliance Note

Any plan proposed in a session must:

1. Name the highest-priority unresolved Tier 1 gap (currently **T1-1: Production Tier 2 CNN Checkpoint**)
2. Show how each proposed step closes or directly unblocks that gap — or explicitly justify why it is Tier 2 work
3. Include outside blockers as explicit named steps with responsible party
4. Never propose log modules, schemas, or scaffolding unless they directly unblock a named gap
5. Never repeat work listed under "What Is Complete" above
