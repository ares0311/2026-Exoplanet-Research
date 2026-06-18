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
| `checkpoints/cnn_kepler_pretrain/` | Local pretrained MPS checkpoint | Ignored | 2026-06-18 local training: startup banner `device=mps`; `best.pt` SHA-256 `c782d7af61171b3f58447f7a49343c86618c447292a71bd28d540807835787c7`; best epoch 19; best val loss 0.3905; best val AUC 0.9186; final epoch 34 val AUC 0.9123 | Retain as a local transfer-learning source; it does not satisfy production without a passing TESS fine-tuned checkpoint |
| `checkpoints/cnn_tess_finetuned/` | Rejected | Ignored | 2026-06-18 local MPS fine-tune from Kepler pretrain: `best.pt` SHA-256 `3fc115b3623b2485373aefef30a7aa901e1183cc77ef4b57ce6c1f2219f49214`; best epoch 22; val AUC 0.8408; test raw AUC 0.8115, F1 0.7523, Brier 0.1818, ECE 0.0854; Platt threshold 0.45 produced calibrated F1 0.7508, Brier 0.1966, ECE 0.1152; evaluator `Flag: FAIL` | Do not promote; next T1-1 planning cycle should add TESS signal or authorize a materially different CNN/transfer experiment |
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
