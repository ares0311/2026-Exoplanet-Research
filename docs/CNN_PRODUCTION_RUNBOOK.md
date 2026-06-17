# CNN PRODUCTION RUNBOOK

Purpose: close T1-1, the Production Tier 2 CNN Checkpoint gap.

This runbook is for the user's local Mac. The required training data files are
local-only artifacts and are not committed to the repository.

Production gates:

- Raw held-out test AUC must be at least `0.85`.
- Calibrated held-out test F1 must be at least `0.80`.
- Platt calibration must not worsen held-out test Brier score or ECE.
- A passing checkpoint still requires human approval before promotion.

## Current State

- `data/kepler_snippets.jsonl`: complete locally at `7454` rows.
- Tiny corrupt Kepler Lightkurve cache files were moved to
  `$HOME/.lightkurve/cache/quarantine_corrupt_kepler_fits`.
- Next human-at-Mac action: build and validate Kepler CNN splits.

## Step 0: Sync And Verify

```bash
git pull --ff-only origin main
.venv/bin/python -c "import sys, torch; assert sys.version_info[:3] == (3,14,3); assert sys.prefix != sys.base_prefix; print(sys.executable); print(torch.__version__)"
wc -l data/kepler_snippets.jsonl data/tess_snippets_v2.jsonl
```

Paste back the Python path, Torch version, and line counts if anything differs
from Python `3.14.3`, venv Python, Kepler `7454`, or TESS `2619`.

## Step 1: Build Kepler Splits

```bash
git pull --ff-only origin main
caffeinate -i .venv/bin/python Skills/build_cnn_training_data.py data/kepler_snippets.jsonl --output-dir data/kepler_cnn_splits
.venv/bin/python Skills/cnn_split_validator.py data/kepler_cnn_splits
```

Stop and paste back the split summary and validator result. Do not train if the
validator does not report `PASS`.

## Step 2: Kepler Pretraining

```bash
git pull --ff-only origin main
caffeinate -dims .venv/bin/python Skills/train_cnn.py --split-dir data/kepler_cnn_splits --checkpoint-dir checkpoints/cnn_kepler_pretrain --config configs/cnn_kepler_pretrain.json
shasum -a 256 checkpoints/cnn_kepler_pretrain/best.pt
```

Paste back the final training result and SHA-256. The agent reviews this before
TESS fine-tuning.

## Step 3: Build TESS Splits

```bash
git pull --ff-only origin main
caffeinate -i .venv/bin/python Skills/build_cnn_training_data.py data/tess_snippets_v2.jsonl --output-dir data/tess_cnn_splits
.venv/bin/python Skills/cnn_split_validator.py data/tess_cnn_splits
```

Stop and paste back the split summary and validator result. Do not fine-tune if
the validator does not report `PASS`.

## Step 4: TESS Fine-Tuning

```bash
git pull --ff-only origin main
caffeinate -dims .venv/bin/python Skills/train_cnn.py --split-dir data/tess_cnn_splits --checkpoint-dir checkpoints/cnn_tess_finetuned --config configs/cnn_tess_finetune.json --pretrained-checkpoint checkpoints/cnn_kepler_pretrain/best.pt
shasum -a 256 checkpoints/cnn_tess_finetuned/best.pt
```

Paste back the final training result and SHA-256.

## Step 5: Production Gate Evaluation

```bash
git pull --ff-only origin main
.venv/bin/python Skills/evaluate_cnn_checkpoint.py --split-dir data/tess_cnn_splits --checkpoint checkpoints/cnn_tess_finetuned/best.pt --output-calibration checkpoints/cnn_tess_finetuned/calibration.json
```

Interpretation:

- Exit code `0` and `Flag: PASS`: paste the full evaluation output. The agent
  will prepare the promotion branch after explicit human approval.
- Exit code `1` and `Flag: FAIL`: paste the full output. The checkpoint must be
  rejected and documented; do not promote it.
- Exit code `2`: paste the error. The run is blocked by environment, split, or
  checkpoint loading problems.

## Step 6: Promotion Only After Approval

Do not copy checkpoint artifacts into `models/` or update `models/registry.json`
until the evaluation passes all gates and the human explicitly approves
promotion. The agent performs the promotion branch, docs update, validation, and
GitHub merge after that approval.
