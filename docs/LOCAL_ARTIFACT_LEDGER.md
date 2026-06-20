# LOCAL ARTIFACT LEDGER

## Purpose

This ledger makes local-only production artifacts visible to agents that can
only read GitHub. It is part of the project operating contract: bulky runtime
artifacts stay out of Git, but their status, expected paths, provenance, and
next actions must be committed here.

This directly supports T1-1, the Production Tier 2 CNN Checkpoint gap. The CNN
path depends on local corpora, splits, checkpoints, logs, and validation output
that are too large or too volatile to commit during development. Agents must be
able to reason about those artifacts from GitHub without relying on chat
context.

Machine-readable companion: `artifacts/manifests/local_artifacts.json`.

## Policy

- The standard operator cadence is `git add .`.
- `.gitignore` must make `git add .` safe for local data, splits, checkpoints,
  reports, SQLite logs, caches, and rejected experiments.
- If an ignored artifact affects production readiness, update this ledger in
  the same PR as any runbook, readiness, or artifact-policy change.
- Do not commit raw corpora, generated split files, intermediate checkpoints,
  SQLite runtime logs, generated reports, or rejected CNN experiments.
- Commit a CNN checkpoint only after it passes the production gate and the
  human explicitly approves promotion. At that point, also commit calibration
  metadata, registry entries, and a reproducibility manifest.
- Because CNN model experiment paths are ignored defensively, production
  promotion may require an explicit `git add -f` for the approved model
  artifacts. That exception must be documented in the promotion PR.

## Current Production-Relevant Local Artifacts

| Path | Status | Git policy | Last GitHub-visible state | Next action |
|---|---|---|---|---|
| `data/tess_snippets_v2.jsonl` | Source corpus present locally | Ignored | 2026-06-17 local inspection: 2,619 rows, 1,215 positive labels, 1,404 negative labels, 2,110 finite-flux rows, 509 rows rejected by finite-value filtering | Use only through validated splits; next T1-1 plan should expand usable TESS labels or improve label quality before another similar CNN promotion attempt |
| `data/kepler_snippets.jsonl` | Local validated source corpus | Ignored | 2026-06-17 local audit: 6,837 parseable rows; labels negative=4,280 and positive=2,557; zero JSON errors; zero non-finite flux rows; all flux vectors length 201; zero duplicate resume keys | Train from `data/kepler_cnn_splits`; retry missing rows only with explicit `--retry-failures` intent |
| `data/kepler_snippets.jsonl.failures.jsonl` | Failure sidecar expected for future fetches | Ignored | Current 6,837-row corpus predates the sidecar; 617 KOI signatures are absent/pending failure-sidecar review | Future fetch runs must write terminal failures here so ordinary resume does not reprocess them forever |
| `data/kepler_snippets_nan_corrupt_20260617.jsonl` | Rejected forensic artifact | Ignored | Preserved local copy of the pre-fix Kepler corpus when present | Do not train; keep local only unless explicitly requested for forensic review |
| `data/kepler_cnn_splits/` | Local validated split | Ignored | 2026-06-17 validator PASS; train/val/test = 4,741 / 1,060 / 1,036; train labels negative=2,984 positive=1,757; val negative=653 positive=407; test negative=643 positive=393 | Kepler pretraining complete; retain for reproducibility unless intentionally regenerating |
| `data/tess_cnn_splits/` | Local validated split | Ignored | 2026-06-18 validator PASS; total examples=2,110; train/val/test = 1,477 / 318 / 315; train labels negative=766 positive=711; val negative=166 positive=152; test negative=163 positive=152 | Retain for reproducibility; first Kepler->TESS fine-tune from these splits failed production gates |
| `data/new_tess_targets.txt` + `data/new_tess_targets.json` | Path A inventory complete / too small | Ignored | 2026-06-18 local inventory against `data/tess_snippets_v2.jsonl`: 56 new labeled TIC IDs; 16 positive, 40 negative; sources TOI=33, CTOI=23 | Do not run long MAST fetch as a production-closing attempt; start next T1-1 planning cycle |
| `data/tess_snippets_expansion_v3.jsonl` | Not recommended after inventory | Ignored | Not generated; Path A inventory found too few new labels to materially move the CNN training set toward the ≥5,000 target | Do not fetch unless a future planning cycle approves a forensic/non-production check |
| `data/tess_snippets_v3.jsonl` | Not generated | Ignored | Not generated because Path A inventory was too small | Do not build until a future materially larger TESS expansion is approved |
| `data/tess_cnn_splits_v3/` | Not generated | Ignored | Not generated because Path A inventory was too small | Do not train candidate 12 from v3 splits unless a future expansion is approved and validated |
| `checkpoints/cnn_kepler_pretrain/` | Local pretrained MPS checkpoint | Ignored | 2026-06-18 local training: startup banner `device=mps`; `best.pt` SHA-256 `c782d7af61171b3f58447f7a49343c86618c447292a71bd28d540807835787c7`; best epoch 19; best val loss 0.3905; best val AUC 0.9186; final epoch 34 val AUC 0.9123 | Retain as a local transfer-learning source; it does not satisfy production without a passing TESS fine-tuned checkpoint |
| `checkpoints/cnn_tess_finetuned/` | Rejected | Ignored | 2026-06-18 local MPS fine-tune from Kepler pretrain: `best.pt` SHA-256 `3fc115b3623b2485373aefef30a7aa901e1183cc77ef4b57ce6c1f2219f49214`; best epoch 22; val AUC 0.8408; test raw AUC 0.8115, F1 0.7523, Brier 0.1818, ECE 0.0854; Platt threshold 0.45 produced calibrated F1 0.7508, Brier 0.1966, ECE 0.1152; evaluator `Flag: FAIL` | Do not promote; next T1-1 planning cycle should add TESS signal or authorize a materially different CNN/transfer experiment |
| `checkpoints/cnn_tess_c12/` | Rejected | Ignored | 2026-06-19 local MPS full-unfreeze fine-tune from Kepler pretrain: `best.pt` SHA-256 `cc8fbd2004e0fd41dc48bf7f48e3d6b552c75164c62556c3a016af3ca1642ff0`; best epoch 20; val AUC 0.8356; test raw AUC 0.8124, F1 0.7542, Brier 0.1780, ECE 0.0556; Platt threshold 0.47 produced calibrated F1 0.7516, Brier 0.1979, ECE 0.1283; evaluator `Flag: FAIL` | Do not promote or rerun unchanged; proceed to `docs/CNN_PRODUCTION_RUNBOOK.md` Step 7c |
| `data/tess_kepler_overlap_snippets.jsonl` | **COMPLETE** — 4,864 snippets | Ignored | 2026-06-20: all 2,755 remaining KOIs processed in final session (wrote=1,512 skipped=1,243 terminal_failures_recorded=1,237 total_elapsed=15626s); combined with prior sessions = 4,864 snippets total; ~2,716 total terminal failures in sidecar; 3 fetch sessions spanning ~11.8 h total; 4 workers, polite 0.25–1.0 s/worker | Retain; combined corpus built; C13 trained and rejected; C14 is next |
| `data/tess_kepler_overlap_snippets.jsonl.failures.jsonl` | **COMPLETE** — ~2,716 terminal failures | Ignored | 2026-06-20: session 1 recorded 1,479 + session 3 recorded 1,237 = ~2,716 total; all are NO_DATA/SHORT/NONFINITE/NO_LIGHTKURVE (confirmed no TESS observations or data too sparse/noisy); ordinary reruns skip these by default | Do not retry without explicit `--retry-failures` intent; retain as durable resume sidecar |
| `data/tess_combined_snippets.jsonl` | **COMPLETE** — 7,483 rows | Ignored | 2026-06-20: built by concatenating TESS v2 (2,619) + overlap (4,864); row count verified; used as source for `data/tess_combined_cnn_splits/` | Retain; do not rebuild unless a new corpus expansion is authorized |
| `data/tess_combined_cnn_splits/` | **LOCAL VALIDATED** | Ignored | 2026-06-20 validator PASS; total examples=7,483 filtered to usable rows; train/val/test = 4,892 / 1,049 / 1,033; seed=7 | Retain; C13/C14/C15 all rejected; C16 uses same splits with batch normalization + stronger L2 regularization |
| `checkpoints/cnn_tess_c13/` | **Rejected** | Ignored | 2026-06-20 local MPS training from Kepler pretrain on combined splits; default config LR=1e-3; val_loss spiked epoch 5 (0.8421) while train_loss fell to 0.2247 (LR too high); best epoch 8 val_auc=0.8195; early stop epoch 18; test raw AUC=0.8342, F1=0.7960, Brier=0.1664, ECE=0.0625; Platt A=1.78234347 B=-0.76273353 threshold=0.47; calibrated F1=0.7960, Brier=0.1828, ECE=0.1334; evaluator `Flag: FAIL` | Do not promote; root cause: LR=1e-3 too high for pretrained init; C14 uses `configs/cnn_tess_finetune_c12.json` (LR=3e-5) on same splits |
| `checkpoints/cnn_tess_c14/` | **Rejected** | Ignored | 2026-06-20 local MPS training from Kepler pretrain on combined splits; `configs/cnn_tess_finetune_c12.json` (LR=3e-5, batch=32, patience=20, full unfreeze); SHA-256 `12fe6fe1004e1ea75b6fa5f244512cbe93e1b176bf3ec822ef5bd6df861d753d`; best epoch 61 val_auc=0.8116; early stop epoch 81 (LR decayed to 1.17e-7); test raw AUC=0.8319, F1=0.7859, Brier=0.1663, ECE=0.0273; Platt A=1.69593273 B=-0.73468346 threshold=0.47; calibrated F1=0.7860, Brier=0.1932, ECE=0.1441; evaluator `Flag: FAIL` | Do not promote; root cause: LR=3e-5 too conservative — scheduler decayed to near-zero, locking model below C13 ceiling; C15 uses LR=1e-4 + augmentation on same splits |
| `checkpoints/cnn_tess_c15/` | **Rejected** | Ignored | 2026-06-20 local MPS training from Kepler pretrain on combined splits; `configs/cnn_tess_c15.json` (LR=1e-4, min_lr=1e-6, lr_scheduler_patience=10, augment=true, batch=32, patience=20, full unfreeze); SHA-256 `34f50183d19b73cdee48bbd1cc3a3680173c802faf5c9d4227369c75c772128c`; best epoch 16 val_auc=0.8162; early stop epoch 36; test raw AUC=0.8353, F1=0.7949, Brier=0.1642, ECE=0.0427; Platt A=1.73766998 B=-0.73987247 threshold=0.52; calibrated F1=0.7938, Brier=0.1888, ECE=0.1389; evaluator `Flag: FAIL` | Do not promote; root cause: LR tuning exhausted — C13/C14/C15 all plateau at test AUC 0.83–0.84; primary bottleneck is overfitting (val_loss 0.77→1.03 while train_loss 0.46→0.28); C16 targets regularization with use_batch_norm=true + weight_decay=1e-2 |
| `checkpoints/cnn_tess_c16/` | **NOT TRAINED** — `configs/cnn_tess_c16.json` committed; ready to train | Ignored | Planned: train from `data/tess_combined_cnn_splits` with Kepler pretrain; `configs/cnn_tess_c16.json` (LR=1e-4, weight_decay=1e-2, use_batch_norm=true, min_lr=1e-6, lr_scheduler_patience=10, augment=true, batch=32, patience=25, full unfreeze from epoch 1); train command: `caffeinate -dims .venv/bin/python Skills/train_cnn.py --split-dir data/tess_combined_cnn_splits --checkpoint-dir checkpoints/cnn_tess_c16 --config configs/cnn_tess_c16.json --pretrained-checkpoint checkpoints/cnn_kepler_pretrain/best.pt --device auto` | [HUMAN] trains after `git pull origin main`; evaluate with `evaluate_cnn_checkpoint.py`; promote only after evaluator PASS plus explicit human approval |
| `models/cnn*/` | Ignored until promotion | Ignored by default | No CNN checkpoint is production-approved | Force-add only after evaluator PASS and explicit human approval |
| `logs/*.sqlite*` | Runtime state | Ignored | Background and live-check SQLite files are local runtime state | Summarize meaningful state in docs, not by committing SQLite DBs |
| `reports/` generated files | Runtime exports | Ignored | Generated reports are local artifacts unless promoted as documentation | Commit only curated docs, never unreviewed runtime exports |

## Required Update Points

Update this ledger whenever any of the following changes:

1. A local-only artifact becomes present, absent, rejected, validated, or
   promoted.
2. A corpus row count, label count, finite-filter count, split count, checkpoint
   SHA-256, or validation result changes.
3. A runbook command changes the expected path of a local artifact.
4. A production gate result accepts or rejects a checkpoint.
5. `.gitignore` changes any policy that controls production-relevant artifacts.

## T1-1 Handoff Rule

For T1-1, GitHub must always answer these questions without chat context:

1. Which local corpora are expected?
2. Which local corpora are valid, rejected, missing, or pending?
3. Which split directories are valid?
4. Which checkpoint, if any, is production-approved?
5. What command should the human run next?
6. What output should the human paste back?

If GitHub cannot answer those questions, the ledger is incomplete.
