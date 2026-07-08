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
