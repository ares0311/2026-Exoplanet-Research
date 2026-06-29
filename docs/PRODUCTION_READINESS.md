# PRODUCTION READINESS

Last reviewed: 2026-06-29 (JWST A3 merged — PR #141; TESS novelty restructure B1-B4 merged — PR #139; live scanner startup/target-selection fix merged — PR #143; PR #145 worker/ETA fix merged; first SPOC-only B5 attempt, second QLP corrupt-cache attempt, third QLP stdout-race attempt, fourth QLP wrong-flux-column attempt, and fifth QLP no-progress/no-durable-log attempt did not close T1-0)
Scope decision: T2-2 and T2-3 are permanently out of scope — see DECISION-013
Branch: `main` (82 production-critical Skills; non-production fluff removed)
Test baseline: 2,328 default tests passing, 2 integration_live deselected

---

## Live-Readiness Summary

| Scorer Mode | Status | Blocker |
|---|---|---|
| `--scorer bayesian` | **PRODUCTION READY** | None — default mode, zero external dependencies |
| `--scorer xgboost` | **PRODUCTION READY** | None — trained on 7,586 Kepler KOIs, AUC=0.992 |
| `--scorer ensemble` | **PRODUCTION READY** | None — conservative XGBoost+Bayesian blend when CNN absent |
| `--scorer cnn` | **NOT READY** | T1-1: no trained checkpoint has passed held-out performance and calibration gates |
| `--scorer full-ensemble` | **NOT READY** | T1-1: no production-approved CNN checkpoint |

The system is safe to deploy now for Bayesian and XGBoost scoring modes. The
CNN label gate is open, but all evaluated CNN checkpoints remain rejected; they
must not be copied into `models/`, registered, or used for production scoring.

---

## Tier 1 Gaps (Blocking Live Discovery Production)

### T1-0: First Real Discovery Scan Evidence

- **What is missing**: A completed, reviewed discovery scan over novel TESS targets with actual downloadable light curves, excluding TOIs, CTOIs, and confirmed exoplanet hosts
- **Gate status**: **OPEN / HUMAN-RUN BLOCKED** — Option B1-B4 are merged; PR #143 fixed startup/target-selection failures; PR #145 added bounded workers and ETA output. `logs/discovery_run_001.json` completed locally on 2026-06-28 but does **not** close this gate because it used the old SPOC-only fetch path and produced 198 no-data rows plus 2 transient remote-disconnect errors, with 0 clear scans and 0 candidates. `logs/discovery_run_002_qlp.json` started the QLP path but also does **not** close this gate: it recorded 3 corrupt Lightkurve cache errors, 0 clear scans, and 0 candidates. `logs/discovery_run_003_qlp_cache_repair.json` also does **not** close this gate: it recorded 1 error, 0 clear scans, and 0 candidates before the process crashed on `ValueError: I/O operation on closed file`. `logs/discovery_run_004_qlp_stdout_safe.json` completed but also does **not** close this gate: it recorded 200 total entries, 0 candidates, 0 clear scans, 1 no-data row, and 199 errors caused by requesting the wrong QLP flux column. `logs/discovery_run_005_qlp_flux_safe.json` also does **not** close this gate: the pasted console showed third-party download chatter and warnings but no per-target progress, and no durable log existed before first target completion.
- **Why this is highest priority**: `docs/DISCOVERY_RUNBOOK.md` realigned production toward discovering previously unknown transit candidates. CNN work improves false-positive rejection, but it does not matter until the project has real candidates or a documented null discovery result.
- **Root cause of run 001**: `select_targets()` selected TIC catalog stars without requiring light-curve availability, while `run_pipeline()` could not override `fetch_lightcurve()` and therefore fetched only `author='SPOC', exptime='long'`.
- **Root cause of run 002**: interrupted prior QLP downloads left corrupt FITS files in the local Lightkurve MAST cache, and the shared fetch path surfaced Lightkurve's "This file may be corrupt due to an interrupted download" error as a terminal scan error instead of deleting the named cache file and retrying.
- **Root cause of run 003**: the shared fetch path still used Lightkurve public download methods. Lightkurve decorates `SearchResult.download()` and `download_all()` with `suppress_stdout`, which assigns `sys.stdout` process-wide. That is unsafe under `star_scanner.py` worker-thread downloads because the main thread prints progress while a worker can temporarily replace or close stdout.
- **Root cause of run 004**: the shared fetch path requested `pdcsap_flux`, which is a SPOC-style column. Valid QLP HLSP products do not include `PDCSAP_FLUX`; older sectors use `KSPSAP_FLUX`, newer sectors may use `DET_FLUX` or `SYS_RM_FLUX`, and `SAP_FLUX` remains a fallback. Lightkurve wrapped the missing-column `KeyError` as a misleading corrupt-download message.
- **Root cause of run 005**: `ScanLog` created no durable file until the first completed `record()`, `run_background_scan()` printed progress only after a future completed, and the Lightkurve per-product path still let Astroquery `Observations.download_products(verbose=True)` print MAST download banners from worker threads.
- **Required next batch**: After the scanner progress/quiet-download fix is merged, run `Skills/star_scanner.py --max-stars 200 --pipeline QLP --exptime long --workers 4 --request-delay 0.5 --log logs/discovery_run_006_qlp_progress_safe.json`, then rank and filter candidates with `Skills/rank_candidates.py` and `Skills/alert_filter.py`
- **Success evidence**: `logs/discovery_run_006_qlp_progress_safe.json` exists locally, includes clear scans or candidates instead of mostly no-data/cache-error/stdout-crash/wrong-flux-column/no-progress rows, top-ranked candidates or a meaningful null result are reviewed, and `docs/LOCAL_ARTIFACT_LEDGER.md` / `artifacts/manifests/local_artifacts.json` are updated with the batch result
- **Next escalation rule**: Do not resume CNN C20 training until at least the first real scan is complete and reviewed. If zero candidates emerge after at least 1,000 TIC targets, that null result becomes the next production-planning input.

### T1-1: Production Tier 2 CNN Checkpoint

- **What is missing**: A CNN checkpoint that passes held-out performance and calibration gates
- **Gate status**: **OPEN** — Kepler pretraining, TESS split generation, and two Kepler->TESS fine-tunes are locally complete; both fine-tuned checkpoints failed held-out production gates and must not be promoted
- **Current priority**: **PAUSED** until T1-0 is complete and reviewed
- **Code status**: Training and state-dict inference paths are operational; the package scorer reconstructs the trained architecture and fails closed when loading fails
- **Prior local corpus status**: **VALID as of 2026-06-12** — 2,037 snippets (1,012 positive CP+KP, 1,025 negative FP+FA, ratio 0.99); zero-epoch corpus retired and rebuilt from scratch with valid BJD epochs; label bug fixed (KP→1); MAST throttling fix applied (`bbb0877`)
- **Local corpus status**: **KEPLER LOCAL VALIDATED** — TESS v2 complete at 2,619 snippets; Kepler finite rebuild has 6,837 parseable snippets with zero non-finite flux rows, zero duplicate resume keys, labels negative=4,280 and positive=2,557; `data/kepler_cnn_splits` validator PASS with train/val/test = 4,741 / 1,060 / 1,036
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
- **Rejected candidate 11 (Kepler->TESS transfer)**: `checkpoints/cnn_tess_finetuned/best.pt`, SHA-256 `3fc115b3623b2485373aefef30a7aa901e1183cc77ef4b57ce6c1f2219f49214`; trained on Python 3.14.3 with PyTorch 2.12.0 using `device=mps`; initialized from Kepler pretrain SHA `c782d7af61171b3f58447f7a49343c86618c447292a71bd28d540807835787c7`; TESS splits train/val/test = 1,477 / 318 / 315; `configs/cnn_tess_finetune.json` with LR=1e-4, weight_decay=1e-3, batch=64, seed=7, frozen conv layers for 15 epochs; best epoch 22, validation loss 0.5255, validation AUC 0.8408; test raw AUC=0.8115, F1=0.7523, Brier=0.1818, ECE=0.0854; Platt A=1.80214901, B=-0.77900211, threshold=0.45; calibrated test F1=0.7508, Brier=0.1966, ECE=0.1152; **REJECTED** — test AUC 0.8115 < 0.85 gate, calibrated F1 0.7508 < 0.80 gate, and calibration worsened both Brier and ECE
- **Rejected candidate 12 (full-unfreeze Kepler->TESS transfer)**: `checkpoints/cnn_tess_c12/best.pt`, SHA-256 `cc8fbd2004e0fd41dc48bf7f48e3d6b552c75164c62556c3a016af3ca1642ff0`; trained on Python 3.14.3 with PyTorch 2.12.0 using `device=mps`; initialized from Kepler pretrain SHA `c782d7af61171b3f58447f7a49343c86618c447292a71bd28d540807835787c7`; TESS splits train/val/test = 1,477 / 318 / 315; `configs/cnn_tess_finetune_c12.json` with full unfreeze from epoch 1, LR=3e-5, weight_decay=1e-3, batch=32, seed=7; best epoch 20, validation loss 0.5302, validation AUC 0.8356; test raw AUC=0.8124, F1=0.7542, Brier=0.1780, ECE=0.0556; Platt A=1.73886384, B=-0.82141269, threshold=0.47; calibrated test F1=0.7516, Brier=0.1979, ECE=0.1283; **REJECTED** — validation AUC missed the 0.85 continuation threshold, test AUC and calibrated F1 remain below production gates, and calibration worsened both Brier and ECE
- **Transfer-learning result (C11/C12)**: Kepler pretraining lifted held-out test AUC above the prior TESS-only/ensemble range, but both C11/C12 transfer candidates plateaued near test AUC 0.812 and calibrated F1 0.75 on 1,477 training examples. Root cause was data-size ceiling.
- **Rejected candidate 13 (C13 — combined corpus, default LR)**: `checkpoints/cnn_tess_c13/best.pt`; trained on Python 3.14.3 with PyTorch 2.12.0 using `device=mps`; initialized from Kepler pretrain SHA `c782d7af61171b3f58447f7a49343c86618c447292a71bd28d540807835787c7`; combined splits train/val/test = 4,892 / 1,049 / 1,033 from `data/tess_combined_cnn_splits`; default config LR=1e-3 with half-LR schedule; best epoch 8, val_loss=0.8197, val_auc=0.8195; test raw AUC=0.8342, F1=0.7960, Brier=0.1664, ECE=0.0625; Platt A=1.78234347, B=-0.76273353, threshold=0.47; calibrated test F1=0.7960, Brier=0.1828, ECE=0.1334; **REJECTED** — test AUC 0.8342 < 0.85 gate, calibrated F1 0.7960 < 0.80 gate, and calibration worsened both Brier and ECE. Root cause: LR=1e-3 too high for pretrained init — val_loss spiked at epoch 5 (0.8421) then diverged to 1.3230 while train_loss fell to 0.2247; model overfit despite large dataset. Positive signs: test AUC improved +2.2 pts over C12; test raw F1 improved +4.2 pts; first candidate where test AUC exceeds val AUC (model generalizes when not derailed by LR).
- **C13 corpus expansion effect confirmed**: 4,892 vs 1,477 training examples drove AUC from 0.8124 to 0.8342 and F1 from 0.7542 to 0.7960. The data-size ceiling is broken; LR is now the limiting factor.
- **Path A inventory result**: Completed locally on 2026-06-18 against `data/tess_snippets_v2.jsonl`; ExoFOP TOI live counts CP=733, KP=591, FP=1,244, FA=100 (positive=1,324; negative=1,344; total=2,668); expansion inventory found only 56 new labeled TIC IDs (16 positive, 40 negative; 33 TOI, 23 CTOI). **Do not run the long MAST snippet fetch as a production-closing attempt**; even 100% fetch success would not materially move the CNN from 2,110 usable examples toward the ≥5,000 target.
- **TESS TCE source probe**: `Skills/tess_tce_fetcher.py` now fails closed with `Flag: UNAVAILABLE` for the stale historical ExoMAST TCE endpoint, which returned HTTP 404 on 2026-06-18. Do not treat that endpoint as the next large TESS-domain label source unless a current provider contract is found and documented.
- **Path B corpus result (2026-06-20)**: `Skills/fetch_tess_kepler_overlap_snippets.py` completed across multiple sessions (~11.8 h total, 4 workers, polite 0.25–1.0 s/worker request delay); `data/tess_kepler_overlap_snippets.jsonl` has **4,864 snippets**; `data/tess_kepler_overlap_snippets.jsonl.failures.jsonl` has ~2,716 terminal failures (NO_DATA/SHORT/NONFINITE/NO_LIGHTKURVE — correctly excluded). Combined corpus projection: TESS v2 (2,619) + overlap (4,864) = **~7,483 total snippets before dedup/filtering**, ~5× the 1,477 training examples that caused the systematic AUC ceiling.
- **Rejected candidate 14 (C14 — combined corpus, low LR)**: `checkpoints/cnn_tess_c14/best.pt`, SHA-256 `12fe6fe1004e1ea75b6fa5f244512cbe93e1b176bf3ec822ef5bd6df861d753d`; trained on Python 3.14.3 with PyTorch 2.12.0 using `device=mps`; combined splits train/val/test = 4,892 / 1,049 / 1,033; `configs/cnn_tess_finetune_c12.json` (LR=3e-5, batch=32, patience=20, full-unfreeze from epoch 1); best epoch 61, val_loss=0.7744, val_auc=0.8116; early stop epoch 81; test raw AUC=0.8319, F1=0.7859, Brier=0.1663, ECE=0.0273; Platt A=1.69593273, B=-0.73468346, threshold=0.47; calibrated test F1=0.7860, Brier=0.1932, ECE=0.1441; **REJECTED** — AUC 0.8319 < 0.85, cal F1 0.7860 < 0.80, Platt worsened Brier/ECE. Root cause: LR=3e-5 too conservative — LR scheduler decayed to 1.17e-7 by epoch 79, locking model in local optimum below C13 ceiling. Note: raw ECE=0.0273 is excellent (model well-calibrated without Platt); Platt is confirmed as systematic overcorrection (A≈1.7 sharpens already-calibrated probabilities) — 4 consecutive candidates affected.
- **Systematic calibration problem identified (C11–C14)**: Platt scaling has worsened Brier and ECE across all 4 Kepler→TESS transfer candidates. Raw ECE for C12=0.0556, C13=0.0625, C14=0.0273 — model probabilities are already well-calibrated. Platt A≈1.7–1.8 overcorrects by sharpening predictions that do not need sharpening. A future [AGENT] task should replace Platt with temperature scaling (single parameter) or skip calibration when raw ECE ≤ 0.05. This does not require a gate change, only a calibration method change in `evaluate_cnn_checkpoint.py`. Requires explicit human approval before modifying the production gate definition.
- **Rejected candidate 15 (C15 — combined corpus, intermediate LR)**: `checkpoints/cnn_tess_c15/best.pt`, SHA-256 `34f50183d19b73cdee48bbd1cc3a3680173c802faf5c9d4227369c75c772128c`; trained on Python 3.14.3 with PyTorch 2.12.0 using `device=mps`; combined splits train/val/test = 4,892 / 1,049 / 1,033; `configs/cnn_tess_c15.json` (LR=1e-4, min_lr=1e-6, lr_scheduler_patience=10, weight_decay=1e-3, use_batch_norm=false, augment=true, batch=32, patience=20, full-unfreeze from epoch 1); best epoch 16, val_auc=0.8162, val_loss=0.7663; early stop epoch 36; test raw AUC=0.8353, F1=0.7949, Brier=0.1642, ECE=0.0427; Platt A=1.73766998, B=-0.73987247, threshold=0.52; calibrated test F1=0.7938, Brier=0.1888, ECE=0.1389; **REJECTED** — test AUC 0.8353 < 0.85 gate, cal F1 0.7938 < 0.80 gate, Platt worsened Brier/ECE. Root cause: **LR tuning exhausted** — C13 (1e-3), C14 (3e-5), C15 (1e-4) all plateau at test AUC 0.83–0.84. Model peaked at epoch 16 (train_loss=0.46) then overfit hard (val_loss 0.77→1.03 while train_loss fell to 0.28 over 20 patience epochs). Primary bottleneck is insufficient regularization (weight_decay=1e-3, no batch norm), not LR. Platt A=1.74 is 5th consecutive calibration failure across C11–C15.
- **LR tuning trajectory**: C13 (LR=1e-3) → test AUC 0.8342; C14 (LR=3e-5) → test AUC 0.8319; C15 (LR=1e-4) → test AUC 0.8353. All three candidates converge at 0.83–0.84. Continued LR search is not productive.
- **Rejected candidate 16 (C16 — BN + strong L2)**: `checkpoints/cnn_tess_c16/best.pt`; trained on Python 3.14.3 with PyTorch 2.12.1 using `device=mps`; `configs/cnn_tess_c16.json` (LR=1e-4, weight_decay=1e-2, use_batch_norm=true, patience=25, augment=true); pretrain load: **8 tensors matched, 4 skipped** (shape mismatch) — BatchNorm layers shift Sequential indices so only `conv.0` and FC layers transferred; 2nd and 3rd conv layers trained from random init; best epoch 10, val_auc=0.6650, val_loss=1.0478; val_loss exploded 0.78→3.83 over 35 epochs while train_loss fell 1.77→0.33; early stop epoch 35; evaluator not run (val AUC far below 0.85 gate). **REJECTED** — val AUC 0.6650 < 0.85 gate; catastrophically worse than C13–C15. Root cause: BN index shift causes severely partial pretrain transfer (only first conv layer out of three); random-init 2nd+3rd conv layers combined with aggressive weight_decay=1e-2 caused immediate catastrophic overfitting.
- **Strategy exhaustion summary**: LR tuning (C13–C15, three orders of magnitude) → 0.83–0.84 ceiling. BN+WD regularization (C16) → catastrophic failure at 0.67. Both approaches exhausted. The 0.83–0.84 ceiling in C13–C15 is most likely a data ceiling, not a tuning problem.
- **C17 REJECTED (2026-06-21)** — joint Kepler+TESS fine-tuning:
  - **Result**: best val AUC 0.7859 (epoch 16), early stop epoch 46; val_loss 0.79→1.42 while train_loss 0.60→0.20
  - **Root cause**: domain mismatch. Kepler (30-min cadence) and TESS (2-min cadence) transit morphologies differ in noise profile, cadence aliasing, and phase-fold artifacts. Joint training caused the conv layers to drift toward mixed-domain representations that do not generalize to the TESS-only val set.
  - **Do not retry joint training**. Retain `data/joint_cnn_splits/` for reproducibility.
- **C18 REJECTED (2026-06-21)** — FC head warm-up with `freeze_conv_epochs=10`:
  - **Result**: SHA-256 `d33c15f45bd369d5eba4b87da3aa1908decc3baef5231dcff8544dd70987d496`; best epoch 22, val AUC 0.8262; early stop epoch 47. Test raw AUC=0.8439, F1=0.7979, Brier=0.1593, ECE=0.0301. Temperature T=1.61363521. Calibrated: threshold=0.46, Brier=0.1632, ECE=0.0667. **Flag: FAIL**.
  - **Best candidate of all 19**: test AUC improved from the 0.83–0.84 plateau (C13–C15) to 0.8439. `freeze_conv_epochs` confirmed as the right direction.
  - **Why it failed**: (1) raw AUC 0.8439 < 0.85 gate (short by 0.006). (2) T=1.61 fitted on overconfident val then applied to already-well-calibrated test (raw ECE=0.0301), worsening ECE to 0.0667.
  - **Do not rerun C18 unchanged.**
- **C19 REJECTED (2026-06-22)** — Extended FC head warm-up with `freeze_conv_epochs=20`:
  - **Result**: SHA-256 `65f3721fac577807f35e4edaeaa9cc0cd0f50959441344487f7c77f35a570436`; best epoch 29 (8 epochs after unfreeze at epoch 21); early stop epoch 54. Test raw AUC=0.8420, F1=0.7951, Brier=0.1606, ECE=0.0377. Temperature T=1.8785927. Calibrated: threshold=0.40, Brier=0.1658, ECE=0.0760. **Flag: FAIL**.
  - **Regressed from C18 in every metric**. Root cause: LR scheduler fires on val_auc plateaus; during 20 frozen epochs val_auc improved monotonically, so LR never decayed. Conv unfroze at epoch 21 with LR still at 1e-4 — same as C18 at epoch 11. Longer frozen phase over-adapted the FC head (T=1.88 vs T=1.61), producing worse calibration and lower test AUC.
  - **freeze_conv strategy exhausted.** Do not retry without a materially different corpus or training schedule.
- **Strategic decision (2026-06-22)**: Human chose Option C — more data. All training-side approaches exhausted on 4,892 examples. Next authorized corpus: K2 EPIC overlap (K2 KOI confirmed planets/FPs with TESS re-observations folded at K2 ephemerides). See runbook Step 7g.
- **ECE-skip gate fix (2026-06-22)**: `evaluate_cnn_checkpoint.py` now skips temperature scaling when raw test ECE < 0.05. Root cause of C11–C19 calibration doom loop confirmed: val is overconfident due to early-stopping selection bias; T > 1 applied to already-calibrated test structurally worsened ECE. With the fix, C20 gate is: raw AUC ≥ 0.85 AND raw F1 ≥ 0.80 (when ECE < 0.05, cal==raw). `Skills/fetch_tess_k2_overlap_snippets.py` and `configs/cnn_tess_c20.json` committed.
- **Current data gate**: TESS combined splits VALIDATED; Kepler splits VALIDATED; C13–C19 all rejected; no CNN checkpoint approved for promotion. K2 overlap corpus is complete at 2,086 snippets. C20 corpus assembly is blocked until T1-0 discovery-scan evidence exists.
- **Current authorized runbook**: `docs/CNN_PRODUCTION_RUNBOOK.md`
- **Current promotion gate**: raw held-out test AUC ≥ 0.85; calibrated held-out test F1 ≥ 0.80; temperature scaling calibration must not worsen held-out test Brier score or ECE
- **Calibration note**: Temperature scaling (T fitted via NLL on val split) replaced Platt scaling on 2026-06-21. Platt A≈1.7–1.8 consistently worsened calibration because raw predictions were already well-calibrated (ECE 0.02–0.06). Temperature scaling is the identity at T=1 and will not artificially sharpen probabilities.
- **Kepler pretraining gate**: **LOCAL PRETRAINED ON MPS** — `checkpoints/cnn_kepler_pretrain/best.pt`, SHA-256 `c782d7af61171b3f58447f7a49343c86618c447292a71bd28d540807835787c7`; Python 3.14.3 venv, PyTorch 2.12.1; startup banner `device=mps`; best epoch 19, best validation loss 0.3905, best validation AUC 0.9186; final epoch 34 val AUC 0.9123; retain as transfer-learning source
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
| 2,328 default tests, ruff clean, mypy clean | ✅ |
| All scientific guardrails enforced in code | ✅ |

---

## Pre-Deployment Compliance Checklist

Run these before any live deployment or public announcement:

- [ ] `PYTHONPATH=src .venv/bin/python -m pytest` — all default tests pass, 0 failures
- [ ] `.venv/bin/ruff check .` — no lint errors
- [ ] `.venv/bin/python -m mypy src` — no type errors
- [ ] `exo background-run-once --dry-run` — no config errors
- [ ] `.venv/bin/python Skills/tier2_progress_reporter.py` — confirm CNN gate status documented
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
| First real discovery scan | Run the 200-target `star_scanner.py` batch on the user's Mac and paste back the rank/filter summary | Human |
| Next CNN production plan | Wait for T1-0 scan evidence; only then decide whether C20 corpus assembly/training is still production-relevant | Agent + human approval |
| CNN production promotion | Validate, calibrate, register, and commit only a future checkpoint that passes held-out gates | Agent + human approval |
| Stacking weight calibration | Tune blend weights on held-out calibration set | Agent after T1-1 resolved |

---

## Planning Compliance Note

Any plan proposed in a session must:

1. Name the highest-priority unresolved Tier 1 gap (currently **T1-0: First Real Discovery Scan Evidence**)
2. Show how each proposed step closes or directly unblocks that gap — or explicitly justify why it is Tier 2 work
3. Include outside blockers as explicit named steps with responsible party
4. Never propose log modules, schemas, or scaffolding unless they directly unblock a named gap
5. Never repeat work listed under "What Is Complete" above
