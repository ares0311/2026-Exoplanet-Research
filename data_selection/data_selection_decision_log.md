# Data Selection Decision Log

This log satisfies `docs/astrometrics_data_selection_policy.md` for production-relevant dataset choices. Keep entries concise, source-contract-based, and tied to committed manifests or run reports.

## 2026-07-08 Decision

Decision id: `t1-1-kepler-master-corpus-promotion-candidate`

Data role: `training`, `validation`, `calibration`, and `frozen_eval` via predefined train/val/test splits.

Data selected:
- Original KOI-based T1-1 Kepler snippets from `metadata/t1_1_kepler_training_manifest.jsonl`.
- DR24 TCE expansion snippets from `metadata/t1_1_kepler_dr24_expansion_manifest.jsonl`.
- Master combined corpus recorded locally as `data/processed/t1_1_kepler_master_combined.jsonl`.
- Validated split directory recorded locally as `data/t1_1_kepler_master_cnn_splits/`.

Why this data:
- Primary source tables were verified through NASA Exoplanet Archive TAP schema discovery before use.
- Labels are row-level Kepler KOI dispositions or DR24 TCE `av_training_set` labels with explicit label mappings.
- The split source is predefined by target/group, preserving leakage safety across examples from the same star.
- The master corpus materially improved the held-out CNN gate versus the prior passing Kepler checkpoint: raw test AUC 0.9572, calibrated F1 0.8347, Brier 0.0580, ECE 0.0142.

Rejected alternatives:
- Old C1-C19/C20-style TESS/combined retraining loops: rejected by measured gates and documented in `docs/PRODUCTION_READINESS.md`.
- `Q1_Q17_DR25_TCE` as a label expansion source: rejected because the relevant label columns were confirmed empty.
- Synthetic positives for this supervised training phase: rejected by `docs/exoplanet_exomoon_dataset_handoff.md`.
- Kaggle or mirror datasets: rejected while primary NASA/MAST sources are available.

Contamination controls:
- Do not use live-search candidate evidence as training data unless a future manifest demotes that data into a training role and removes it from blind-discovery claims.
- Do not mix exomoon residual/anomaly-ranking data into this supervised exoplanet classifier.
- Do not promote local checkpoint artifacts into `models/` without explicit human approval plus a promotion PR containing reproducibility metadata.

Storage and retention:
- Raw FITS products are treated as re-downloadable batch cache and are evicted after verified snippet extraction.
- Durable state is the committed manifests, run reports, local artifact ledger, source snapshots, and this decision log.
- The approved promotion package should include only selected production artifacts, not broad raw archives or scratch checkpoints.

Current status:
- `checkpoints/cnn_t1_1_kepler_master/best.pt` is the promotion candidate.
- Promotion is pending explicit human approval.

## 2026-07-10 Decision

Decision id: `t1-2-k2pandc-held-out-calibration-set`

Data role: `calibration` (T1-2 stacking Tier-3 weight calibration, not model training).

Data selected:
- 596 rows (356 CONFIRMED / 240 FALSE POSITIVE) from the NASA Exoplanet
  Archive K2 planets-and-candidates table (`k2pandc`), recorded in
  `metadata/t1_2_k2_calibration_manifest.jsonl`.
- Bayesian + XGBoost scores computed catalog-only in
  `metadata/t1_2_k2_calibration_partial_predictions.jsonl`.
- Native K2 light-curve snippets pending a bounded live fetch to
  `data/t1_2_k2_calibration_snippets.jsonl` (local, human-run).

Why this data:
- T1-2 requires a held-out set never used to train `xgboost_koi.json` or
  promote `benchmark_cnn_v1`. Both were trained/split entirely from Kepler
  prime-mission KIC targets, and `data_role_registry.yaml`'s existing
  training/validation/frozen_eval entries for that corpus explicitly forbid
  reuse for stacking calibration.
- K2's own planet catalog uses a disjoint EPIC target catalog observed
  during different campaigns/sky fields than the original Kepler mission,
  so `k2:epic:<id>` group keys can never collide with the `kepler:kic:<id>`
  namespace used by T1-1 — zero leakage risk by construction, not by
  post-hoc deduplication.
- `k2pandc` was already identified in the Label-Source Discovery Protocol
  (`CLAUDE.md`) as the one untried lever for Kepler-family data (native K2
  light curves, as opposed to the already-fetched TESS-re-observed subset).
  Using it for calibration rather than training expansion is recorded here
  so a future agent does not also try to consume it for T1-1 training.
- Live-verified via `tap_schema.columns` before use: 2,261 usable rows
  (CONFIRMED/FALSE POSITIVE with valid period+epoch) as of 2026-07-10,
  comfortably above the ~500-example gate noted across
  `docs/PRODUCTION_READINESS.md`/`CLAUDE.md`.

Rejected alternatives:
- Reusing the Kepler master corpus's own validation or frozen-eval splits:
  explicitly forbidden by `data_role_registry.yaml` (`forbidden_uses`
  already lists "stacking weight calibration" / "full-ensemble stacking
  calibration after CNN promotion" for those roles).
- TESS-domain calibration data: rejected because `benchmark_cnn_v1` is a
  Kepler-domain-only checkpoint; pairing it with a TESS calibration set
  would measure a domain mismatch, not real production performance.
- The already-fetched `data/tess_k2_overlap_snippets.jsonl` (K2 ephemerides,
  TESS light curves): rejected — that corpus is TESS-domain flux, not
  native K2 flux, and (see bug note below) was built with a corrupted
  epoch and needs re-fetching regardless.

Contamination controls:
- Do not reuse these 596 EPIC targets for any future T1-1-style Kepler/K2
  CNN training-corpus expansion — recorded as a `forbidden_uses` entry in
  `data_role_registry.yaml`.
- Do not report T1-2 calibration AUC/weights as a benchmark_cnn_v1
  performance claim; that claim belongs solely to the frozen-eval split.

Bugs found and fixed while building this dataset (see `AGENTS.md` /
`docs/PRODUCTION_READINESS.md` for full detail):
1. `pl_tranmid` in `k2pandc` is already full BJD_TDB, not BKJD. An existing
   script, `Skills/fetch_tess_k2_overlap_snippets.py`, assumed BKJD and
   added the 2454833-day offset a second time, corrupting the phase-fold
   epoch for every one of its already-fetched 2,086 TESS-domain K2-overlap
   snippets (`data/tess_k2_overlap_snippets.jsonl`, local-only). Fixed in
   both that script and the new T1-2 manifest builder. No promoted model
   was trained on the corrupted data (it predates any C20-style TESS
   combined-corpus assembly), but the local file must be re-fetched before
   any future use.
2. `pl_trandur` is empirically already in hours, not days as
   `tap_schema.columns` itself claims ("Transit Duration [day]", unit=
   "day") — cross-checked against literature-published transit durations
   for K2-18 b and K2-3 b. Fixed before any conversion was applied
   downstream.

Storage and retention:
- k2pandc catalog rows and derived manifests/predictions are small,
  committed metadata — no large raw archive to bound.
- Native K2 raw FITS products (once fetched) follow the same
  re-downloadable-batch-cache policy as other mission fetchers; only the
  phase-folded snippet JSONL is retained locally.

Current status:
- Manifest and catalog-only (Bayesian + XGBoost) scoring are complete and
  committed.
- `[HUMAN NEXT]`: run `Skills/fetch_t1_2_k2_calibration_snippets.py` live to
  fetch native K2 snippets, then fill in `cnn_prob` and run
  `Skills/calibrate_stacking_weights.py` to produce final T1-2 weights.
