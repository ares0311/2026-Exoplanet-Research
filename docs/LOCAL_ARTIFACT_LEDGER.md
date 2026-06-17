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
| `data/tess_snippets_v2.jsonl` | Source corpus present locally | Ignored | 2026-06-17 local inspection: 2,619 rows, 1,215 positive labels, 1,404 negative labels, 2,110 finite-flux rows, 509 rows rejected by finite-value filtering | Build `data/tess_cnn_splits` only after Kepler pretraining is reviewed |
| `data/kepler_snippets.jsonl` | Rebuild required | Ignored | Current checkout snapshot: expected file absent; pre-fix 7,454-row file was rejected because 7,132 rows had non-finite flux | Rebuild with `docs/CNN_PRODUCTION_RUNBOOK.md` Step 1, then update this ledger with row counts and split validation result |
| `data/kepler_snippets_nan_corrupt_20260617.jsonl` | Rejected forensic artifact | Ignored | Preserved local copy of the pre-fix Kepler corpus when present | Do not train; keep local only unless explicitly requested for forensic review |
| `data/kepler_cnn_splits/` | Not valid yet | Ignored | Must be regenerated from rebuilt finite-flux Kepler JSONL | Build only after Kepler fetch reports `Flag: OK`; require `cnn_split_validator` PASS |
| `data/tess_cnn_splits/` | Pending | Ignored | Must be generated from `data/tess_snippets_v2.jsonl` for fine-tuning/evaluation | Build after Kepler pretraining review; require `cnn_split_validator` PASS |
| `checkpoints/cnn_kepler_pretrain/` | Pending | Ignored | No approved checkpoint | Train only from validated Kepler splits; paste final training result and SHA-256 into review |
| `checkpoints/cnn_tess_finetuned/` | Pending | Ignored | No approved checkpoint | Fine-tune from reviewed Kepler pretrain checkpoint; run production gate evaluator |
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
