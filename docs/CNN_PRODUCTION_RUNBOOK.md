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

- `data/kepler_snippets.jsonl`: locally validated on 2026-06-17 with `6837`
  parseable finite snippets, zero duplicate resume keys, labels negative=4,280
  and positive=2,557.
- `data/kepler_cnn_splits`: locally validated on 2026-06-17 with validator
  PASS; train/val/test = 4,741 / 1,060 / 1,036.
- Tiny corrupt Kepler Lightkurve cache files were moved to
  `$HOME/.lightkurve/cache/quarantine_corrupt_kepler_fits`.
- Fetch and split code rejects non-finite flux, reconstructs the phase grid
  for Kepler snippets, groups Kepler KOIs by `kepid` to avoid split leakage,
  and records terminal fetch failures in a JSONL sidecar so ordinary resume
  does not reprocess failed rows forever.
- Kepler pretraining completed locally on 2026-06-18 with the GPU-aware trainer
  and startup banner `device=mps`; checkpoint
  `checkpoints/cnn_kepler_pretrain/best.pt`, SHA-256
  `c782d7af61171b3f58447f7a49343c86618c447292a71bd28d540807835787c7`,
  best epoch 19, best validation loss 0.3905, and best validation AUC 0.9186.
- `data/tess_cnn_splits`: locally validated on 2026-06-18 with validator
  PASS; total examples = 2,110; train/val/test = 1,477 / 318 / 315.
- TESS fine-tuning from the MPS Kepler pretraining checkpoint completed locally
  on 2026-06-18 with startup banner `device=mps`; checkpoint
  `checkpoints/cnn_tess_finetuned/best.pt`, SHA-256
  `3fc115b3623b2485373aefef30a7aa901e1183cc77ef4b57ce6c1f2219f49214`,
  best epoch 22, best validation loss 0.5255, and best validation AUC 0.8408.
- Production gate evaluation rejected that checkpoint: test raw AUC 0.8115,
  raw F1 0.7523, calibrated F1 0.7508, calibrated Brier 0.1966, and calibrated
  ECE 0.1152. Do not promote this checkpoint.
- Path A inventory completed locally on 2026-06-18 and found only 56 new
  labeled TIC IDs absent from `data/tess_snippets_v2.jsonl` (16 positive, 40
  negative; 33 TOI, 23 CTOI). This is too small to justify a long MAST fetch
  as a production-closing candidate-12 run. Do not fetch v3 snippets for
  production from this inventory; start a new T1-1 planning cycle.
- Candidate 12 full-unfreeze fine-tuning completed locally on 2026-06-19 with
  startup banner `device=mps`; checkpoint `checkpoints/cnn_tess_c12/best.pt`,
  SHA-256 `cc8fbd2004e0fd41dc48bf7f48e3d6b552c75164c62556c3a016af3ca1642ff0`,
  best epoch 20, validation loss 0.5302, and validation AUC 0.8356.
  Production gate evaluation rejected that checkpoint: test raw AUC 0.8124,
  raw F1 0.7542, calibrated F1 0.7516, calibrated Brier 0.1979, and
  calibrated ECE 0.1283. Do not promote this checkpoint.
- After every local artifact state change, update the artifact ledger so agents
  that can only see GitHub know whether the corpus, splits, checkpoint, or
  promotion gate is missing, pending, valid, rejected, or approved.

## Step 0: Sync And Verify

```bash
git pull --ff-only origin main
.venv/bin/python -c "import sys, torch; assert sys.version_info[:3] == (3,14,3); assert sys.prefix != sys.base_prefix; print(sys.executable); print(torch.__version__)"
.venv/bin/python Skills/cnn_split_validator.py data/kepler_cnn_splits
wc -l data/kepler_snippets.jsonl data/tess_snippets_v2.jsonl
```

Paste back the Python path, Torch version, and line counts if anything differs
from Python `3.14.3`, venv Python, Kepler `6837`, TESS `2619`, or Kepler
split validator `PASS`.

## Step 1: Optional Kepler Missing-Row Retry

Skip this step when `data/kepler_snippets.jsonl` has `6837` rows and
`data/kepler_cnn_splits` validates. The current corpus is finite and usable for
Kepler pretraining.

If a human explicitly wants to retry missing rows, run one bounded fetch. Do
not use an infinite shell wrapper. Ordinary resume skips successful snippets
and terminal failures from `data/kepler_snippets.jsonl.failures.jsonl`; use
`--retry-failures` only for an intentional recheck.

```bash
git pull --ff-only origin main
caffeinate -dims .venv/bin/python Skills/fetch_kepler_lc_snippets.py --output data/kepler_snippets.jsonl --workers 3 --request-delay 0.5 --retry-failures
wc -l data/kepler_snippets.jsonl
```

Stop and paste back the final fetch summary and line count. After agent review,
rebuild and validate splits if the line count changed.

## Step 2: Build Kepler Splits

```bash
git pull --ff-only origin main
caffeinate -i .venv/bin/python Skills/build_cnn_training_data.py data/kepler_snippets.jsonl --output-dir data/kepler_cnn_splits
.venv/bin/python Skills/cnn_split_validator.py data/kepler_cnn_splits
```

This step already passed locally on 2026-06-17 for the 6,837-row Kepler JSONL.
Rerun it only if the Kepler JSONL changes. Do not train if the validator does
not report `PASS`. After agent review, update the artifact ledger with split
counts and validator status.

## Step 3: Kepler Pretraining

```bash
git pull --ff-only origin main
caffeinate -dims .venv/bin/python Skills/train_cnn.py --split-dir data/kepler_cnn_splits --checkpoint-dir checkpoints/cnn_kepler_pretrain --config configs/cnn_kepler_pretrain.json
shasum -a 256 checkpoints/cnn_kepler_pretrain/best.pt
```

This step completed locally on 2026-06-18 with startup banner `device=mps`,
best validation AUC `0.9186`, and SHA-256
`c782d7af61171b3f58447f7a49343c86618c447292a71bd28d540807835787c7`.
Rerun only if intentionally regenerating the checkpoint after a code or config
change. Future reruns should print `device=mps` on the recorded M4 Max when
Metal/MPS is available, unless the operator explicitly chooses CPU.

## Step 4: Build TESS Splits

```bash
git pull --ff-only origin main
caffeinate -i .venv/bin/python Skills/build_cnn_training_data.py data/tess_snippets_v2.jsonl --output-dir data/tess_cnn_splits
.venv/bin/python Skills/cnn_split_validator.py data/tess_cnn_splits
```

Stop and paste back the split summary and validator result. Do not fine-tune if
the validator does not report `PASS`. After review, update the artifact ledger
with split counts and validator status.

This step completed locally on 2026-06-18 with validator `PASS`; total
examples `2,110`; train/val/test `1,477 / 318 / 315`; train labels
negative=766 positive=711; val negative=166 positive=152; test negative=163
positive=152.

## Step 5: TESS Fine-Tuning

```bash
git pull --ff-only origin main
caffeinate -dims .venv/bin/python Skills/train_cnn.py --split-dir data/tess_cnn_splits --checkpoint-dir checkpoints/cnn_tess_finetuned --config configs/cnn_tess_finetune.json --pretrained-checkpoint checkpoints/cnn_kepler_pretrain/best.pt
shasum -a 256 checkpoints/cnn_tess_finetuned/best.pt
```

Paste back the final training result and SHA-256.
After agent review, update the artifact ledger with checkpoint path, SHA-256,
and fine-tuning status. On the recorded M4 Max, the training startup banner
should show `device=mps` when PyTorch Metal/MPS is available; paste back the
banner line too if it does not.

This step completed locally on 2026-06-18 with startup banner `device=mps`;
best epoch `22`; best validation AUC `0.8408`; and SHA-256
`3fc115b3623b2485373aefef30a7aa901e1183cc77ef4b57ce6c1f2219f49214`.

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

This step completed locally on 2026-06-18 with exit code `1` and `Flag: FAIL`.
The evaluated checkpoint is rejected: raw test AUC `0.8115`, raw test F1
`0.7523`, calibrated test F1 `0.7508`, calibrated Brier `0.1966`, and
calibrated ECE `0.1152`. The next runbook action is not to rerun this same
fine-tune; start a new T1-1 planning cycle around more usable TESS labels,
better label quality, or a materially different CNN/transfer strategy.

## Step 7: Candidate 12 Path A — Expand TESS Labels

This strategy was approved after candidate 11 failed. The goal was to increase
TESS-domain signal before training again. The first inventory found too few
new labels to justify a production-closing long fetch or training run.

First inventory newly labeled ExoFOP TOI/CTOI targets that are absent from the
current v2 corpus:

```bash
git pull origin main
.venv/bin/python Skills/count_tess_labels.py
.venv/bin/python Skills/fetch_additional_tess_labels.py --corpus data/tess_snippets_v2.jsonl --output data/new_tess_targets.txt
wc -l data/new_tess_targets.txt
```

Stop here and paste back the label-count output, expansion summary, and
`wc -l` result. The agent must review the count, positive/negative balance,
and source mix before any long MAST snippet fetch. If too few new usable
targets are available, start a new T1-1 planning cycle instead of training.

Result from 2026-06-18: **too few new targets**. ExoFOP TOI live counts were
CP=733, KP=591, FP=1,244, FA=100; expansion inventory found 56 new labeled TIC
IDs (16 positive, 40 negative). Do not continue to the fetch step below unless
a future planning cycle explicitly approves it for a non-production/forensic
check.

If the inventory justifies a long fetch, download snippets for the new target
JSON. This command is resumable and prints progress:

```bash
git pull origin main
caffeinate -dims .venv/bin/python Skills/fetch_tess_lc_snippets.py --rows data/new_tess_targets.json --output data/tess_snippets_expansion_v3.jsonl --max-errors 100
wc -l data/tess_snippets_expansion_v3.jsonl
```

Stop here and paste back the final fetch summary and line count. The agent must
audit the expansion JSONL before merge/split rebuild.

After agent approval, merge the expansion corpus and build validated v3 splits:

```bash
git pull origin main
cat data/tess_snippets_v2.jsonl data/tess_snippets_expansion_v3.jsonl > data/tess_snippets_v3.jsonl
caffeinate -i .venv/bin/python Skills/build_cnn_training_data.py data/tess_snippets_v3.jsonl --output-dir data/tess_cnn_splits_v3
.venv/bin/python Skills/cnn_split_validator.py data/tess_cnn_splits_v3
```

Stop here and paste back the split summary and validator output. Train a
candidate-12 checkpoint only if the expanded split passes validation and the
agent updates the artifact ledger with the new local artifact state.

## Step 7b: Candidate 12 — Full-Unfreeze Fine-Tune

**Status: REJECTED** — completed locally on 2026-06-19 and missed the gate.
Do not rerun this as a production-closing attempt without a new documented
hypothesis.

**Why this is different from candidate 11:** Candidate 11 (`configs/cnn_tess_finetune.json`)
froze the conv layers for 15 epochs, then fine-tuned at LR=1e-4. The Kepler conv
features never fully adapted to TESS cadence/noise.  This config (`configs/cnn_tess_finetune_c12.json`)
unfreezes all layers from epoch 1 with a much lower LR (3e-5), smaller batch (32), and
longer patience (20), giving the full network a gradual opportunity to adapt without
catastrophic forgetting of pretrained weights.

Result from 2026-06-19: best val AUC `0.8356`, test raw AUC `0.8124`,
calibrated F1 `0.7516`, calibrated Brier `0.1979`, and calibrated ECE
`0.1283`. This checkpoint is rejected. Proceed to Step 7c before any further
training.

## Step 7c: Kepler-TESS Overlap Corpus

**Status: SCRIPT READY** — `Skills/fetch_tess_kepler_overlap_snippets.py` written and tested.
This is Option B: downloads TESS light curves for ~6,500 Kepler KOI stars that MAST
has re-observed with TESS. Phase-folds using Kepler ephemerides (period + epoch
converted from BKJD to full BJD). Expected to yield 1,000–3,000 new labeled TESS
snippets with high-quality Kepler labels (confirmed planet vs confirmed FP).
The fetcher groups pending KOIs by KIC, fetches each KIC light curve once, folds
all KOIs for that KIC locally, and runs a bounded rolling thread pool. The main
thread writes successes and terminal failures as each KIC group completes, so an
interrupted run remains safely resumable.
It intentionally avoids Lightkurve `SearchResult.download_all()` because that
method mutates process-global `sys.stdout` and is unsafe with concurrent worker
threads.
Ordinary resume skips already written snippets and terminal failures recorded in
`data/tess_kepler_overlap_snippets.jsonl.failures.jsonl`; use
`--retry-failures` only for an intentional recheck of those terminal failures.

**Step 7b is rejected, so this is now the next authorized T1-1 runbook step.**
It takes ~12–24 hours.

```bash
git pull origin main
caffeinate -dims .venv/bin/python Skills/fetch_tess_kepler_overlap_snippets.py \
  --output data/tess_kepler_overlap_snippets.jsonl \
  --workers 4 \
  --request-delay 0.25
wc -l data/tess_kepler_overlap_snippets.jsonl
```

Stop and paste back the final fetch summary and line count. The agent must review
the count, terminal failure sidecar, and label balance before merge.

After agent approval, merge the overlap corpus with the v2 corpus and rebuild
splits:

```bash
git pull origin main
cat data/tess_snippets_v2.jsonl data/tess_kepler_overlap_snippets.jsonl > data/tess_combined_snippets.jsonl
caffeinate -i .venv/bin/python Skills/build_cnn_training_data.py data/tess_combined_snippets.jsonl --output-dir data/tess_combined_cnn_splits --seed 7
.venv/bin/python Skills/cnn_split_validator.py data/tess_combined_cnn_splits
```

Stop and paste back the split summary. Then train with the existing fine-tune config
and the Kepler pretrain checkpoint:

```bash
git pull origin main
caffeinate -dims .venv/bin/python Skills/train_cnn.py --split-dir data/tess_combined_cnn_splits --checkpoint-dir checkpoints/cnn_tess_c13 --config configs/cnn_tess_finetune_c12.json --pretrained-checkpoint checkpoints/cnn_kepler_pretrain/best.pt
```

Stop and paste back the training result. Then evaluate with Step 6 (substituting
`data/tess_combined_cnn_splits` and `checkpoints/cnn_tess_c13`).

## Step 8: Promotion Only After Approval

Do not copy checkpoint artifacts into `models/` or update `models/registry.json`
until the evaluation passes all gates and the human explicitly approves
promotion. The agent performs the promotion branch, docs update, validation, and
GitHub merge after that approval. Because local CNN model paths are ignored to
make `git add .` safe, approved promotion may require an intentional
`git add -f` for the selected checkpoint artifacts; document that exception in
the promotion PR.
