# CNN PRODUCTION RUNBOOK

Purpose: close T1-1, the Production Tier 2 CNN Checkpoint gap.

This runbook is for the user's local Mac. The required training data files are
local-only artifacts and are not committed to the repository.
Their GitHub-visible status is tracked in `docs/LOCAL_ARTIFACT_LEDGER.md` and
`artifacts/manifests/local_artifacts.json`.

Production gates:

- Raw held-out test AUC must be at least `0.85`.
- Calibrated held-out test F1 must be at least `0.80`.
- Platt calibration must not worsen held-out test Brier score or ECE.
- A passing checkpoint still requires human approval before promotion.

## Current State

- `data/kepler_snippets.jsonl`: rejected on 2026-06-17 despite `7454` rows.
  The file contained non-finite flux values in `7132` rows; after finite-value
  filtering only `322` examples remained, which is not enough for transfer
  learning.
- Tiny corrupt Kepler Lightkurve cache files were moved to
  `$HOME/.lightkurve/cache/quarantine_corrupt_kepler_fits`.
- Fetch and split code now rejects non-finite flux, reconstructs the phase grid
  for Kepler snippets, and groups Kepler KOIs by `kepid` to avoid split leakage.
- Next human-at-Mac action: rebuild the Kepler JSONL with the fixed fetcher.
- After every local artifact state change, update the artifact ledger so agents
  that can only see GitHub know whether the corpus, splits, checkpoint, or
  promotion gate is missing, pending, valid, rejected, or approved.

## Step 0: Sync And Verify

```bash
git pull --ff-only origin main
.venv/bin/python -c "import sys, torch; assert sys.version_info[:3] == (3,14,3); assert sys.prefix != sys.base_prefix; print(sys.executable); print(torch.__version__)"
wc -l data/kepler_snippets.jsonl data/tess_snippets_v2.jsonl
```

Paste back the Python path, Torch version, and line counts if anything differs
from Python `3.14.3`, venv Python, Kepler `7454`, or TESS `2619`.

## Step 1: Rebuild Kepler JSONL

The previous `7454`-row file must not be used for training. Preserve it for
forensics, then rebuild from cached/downloaded Kepler light curves with finite
time/flux filtering enabled.

```bash
git pull --ff-only origin main
mv data/kepler_snippets.jsonl data/kepler_snippets_nan_corrupt_20260617.jsonl
caffeinate -dims .venv/bin/python Skills/fetch_kepler_lc_snippets.py --output data/kepler_snippets.jsonl --workers 3 --request-delay 0.5 --no-resume
wc -l data/kepler_snippets.jsonl
```

Stop and paste back the final fetch summary and line count. Do not build splits
unless the fetch reports `Flag: OK` and the line count is close to the expected
KOI table size. After agent review, update the artifact ledger with row counts,
finite-filter counts when available, and fetch status.

## Step 2: Build Kepler Splits

```bash
git pull --ff-only origin main
caffeinate -i .venv/bin/python Skills/build_cnn_training_data.py data/kepler_snippets.jsonl --output-dir data/kepler_cnn_splits
.venv/bin/python Skills/cnn_split_validator.py data/kepler_cnn_splits
```

Stop and paste back the split summary and validator result. Do not train if the
validator does not report `PASS`. After agent review, update the artifact
ledger with split counts and validator status.

## Step 3: Kepler Pretraining

```bash
git pull --ff-only origin main
caffeinate -dims .venv/bin/python Skills/train_cnn.py --split-dir data/kepler_cnn_splits --checkpoint-dir checkpoints/cnn_kepler_pretrain --config configs/cnn_kepler_pretrain.json
shasum -a 256 checkpoints/cnn_kepler_pretrain/best.pt
```

Paste back the final training result and SHA-256. The agent reviews this before
TESS fine-tuning. After agent review, update the artifact ledger with checkpoint
path, SHA-256, and pretraining status.

## Step 4: Build TESS Splits

```bash
git pull --ff-only origin main
caffeinate -i .venv/bin/python Skills/build_cnn_training_data.py data/tess_snippets_v2.jsonl --output-dir data/tess_cnn_splits
.venv/bin/python Skills/cnn_split_validator.py data/tess_cnn_splits
```

Stop and paste back the split summary and validator result. Do not fine-tune if
the validator does not report `PASS`. After agent review, update the artifact
ledger with split counts and validator status.

## Step 5: TESS Fine-Tuning

```bash
git pull --ff-only origin main
caffeinate -dims .venv/bin/python Skills/train_cnn.py --split-dir data/tess_cnn_splits --checkpoint-dir checkpoints/cnn_tess_finetuned --config configs/cnn_tess_finetune.json --pretrained-checkpoint checkpoints/cnn_kepler_pretrain/best.pt
shasum -a 256 checkpoints/cnn_tess_finetuned/best.pt
```

Paste back the final training result and SHA-256.
After agent review, update the artifact ledger with checkpoint path, SHA-256,
and fine-tuning status.

## Step 6: Production Gate Evaluation

```bash
git pull --ff-only origin main
.venv/bin/python Skills/evaluate_cnn_checkpoint.py --split-dir data/tess_cnn_splits --checkpoint checkpoints/cnn_tess_finetuned/best.pt --output-calibration checkpoints/cnn_tess_finetuned/calibration.json
```

Interpretation:

- Exit code `0` and `Flag: PASS`: paste the full evaluation output. The agent
  will prepare the promotion branch after explicit human approval and update
  the artifact ledger.
- Exit code `1` and `Flag: FAIL`: paste the full output. The checkpoint must be
  rejected and documented in readiness docs and the artifact ledger; do not
  promote it.
- Exit code `2`: paste the error. The run is blocked by environment, split, or
  checkpoint loading problems.

## Step 7: Promotion Only After Approval

Do not copy checkpoint artifacts into `models/` or update `models/registry.json`
until the evaluation passes all gates and the human explicitly approves
promotion. The agent performs the promotion branch, docs update, validation, and
GitHub merge after that approval. Because local CNN model paths are ignored to
make `git add .` safe, approved promotion may require an intentional
`git add -f` for the selected checkpoint artifacts; document that exception in
the promotion PR.
