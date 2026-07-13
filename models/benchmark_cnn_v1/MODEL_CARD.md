# Benchmark CNN v1 Model Card

## Status

`benchmark_cnn_v1` is the frozen CNN benchmark for the T1-1 exoplanet
classifier path. The selected checkpoint passed the documented held-out gates,
received explicit human approval on 2026-07-09, and is promoted under
`models/cnn/benchmark_cnn_v1/`.

## Candidate Artifact

| Field | Value |
|---|---|
| Source checkpoint | `checkpoints/cnn_t1_1_kepler_master/best.pt` |
| Promoted checkpoint | `models/cnn/benchmark_cnn_v1/best.pt` |
| Checkpoint SHA-256 | `f29e6891c255289fa1e2eddad1fb6ca131c063cf11c24b8113e0e29d049441c5` |
| Local calibration | `checkpoints/cnn_t1_1_kepler_master/calibration.json` |
| Calibration SHA-256 | `357bab520edc183ea5917dee7561525a55c5f0f71ca399ad4acbf501ecc96c0d` |
| Calibration method | Temperature scaling |
| Temperature | `1.0` |
| Training strategy | Trained from scratch, not fine-tuned |
| Architecture config | `checkpoints/cnn_t1_1_kepler_master/config.json` |
| Config SHA-256 | `6a7593a7133eb6d02845d259ef6a5420249e8588644cbecd8bf901ba198abd2e` |
| Metrics file | `checkpoints/cnn_t1_1_kepler_master/metrics.json` |
| Metrics SHA-256 | `1056fad21a6155924e5b6613610862b7aeb484c317ef223d1343e634bf21908a` |

The promoted checkpoint remains under a normally ignored model path and must be
intentionally staged with `git add -f` in the promotion PR.

## Intended Use

- Rank and score Kepler-style phase-folded transit snippets as an internal CNN
  benchmark.
- Provide a frozen measuring stick for future exoplanet ML hardening work.
- Supply the CNN component needed before T1-2 stacking calibration can start.

## Not Intended For

- Confirming planets.
- Making external submissions without a complete candidate evidence packet and
  explicit human approval.
- Supervised exomoon classification.
- Live-search training or threshold tuning.
- Replacing false-positive diagnostics, catalog checks, centroid review, or
  follow-up photometry.

## Training Data

The model was trained on the local master Kepler corpus:

- Original KOI-based snippets: `data/processed/t1_1_kepler_snippets/combined.jsonl`
  with 7,442 rows.
- DR24 TCE expansion snippets:
  `data/processed/t1_1_kepler_dr24_expansion_snippets/combined.jsonl` with
  8,207 rows.
- Master combined corpus:
  `data/processed/t1_1_kepler_master_combined.jsonl` with 15,649 rows.
- Split manifest: `data/t1_1_kepler_master_cnn_splits/manifest.json`.

Split summary:

| Split | Examples | Negative | Positive |
|---|---:|---:|---:|
| Train | 10,800 | 8,442 | 2,358 |
| Validation | 2,393 | 1,875 | 518 |
| Test | 2,456 | 1,919 | 537 |

The split source is `predefined`, so examples inherit target/group-level split
assignments from committed manifests rather than being randomly re-shuffled.

## Evaluation

Production gate:

- Raw held-out test AUC >= 0.85.
- Calibrated held-out test F1 >= 0.80.
- Temperature calibration must not worsen held-out Brier score or ECE.

Observed held-out test metrics:

| Metric | Value |
|---|---:|
| Raw test AUC | 0.957211 |
| Raw test F1 | 0.834688 |
| Raw test Brier | 0.057970 |
| Raw test ECE | 0.014241 |
| Calibrated test F1 | 0.834688 |
| Calibrated test Brier | 0.057970 |
| Calibrated test ECE | 0.014241 |
| Evaluator flag | OK |

Temperature scaling selected `T=1.0`, so calibration was the identity and did
not degrade Brier score or ECE.

### Production-pipeline sensitivity evidence

Version 0.2.44 commits the bounded `production_sensitivity_v1` real-background
injection suite. It links `benchmark_cnn_v1`, `xgboost_koi`, and the calibrated
full ensemble to the `t1_1_kepler_master_frozen_eval` dataset role. On two
30-day Kepler Q1 backgrounds, the production pipeline recovered 23/36 injected
signals with zero execution failures. The durable sample-level artifact and
curves are `artifacts/manifests/production_sensitivity_v1.json`.

This is short-period context, not a new model-promotion gate or a survey
completeness claim. It covers 1, 3, and 10 days, two durations, three depths,
and only two backgrounds. The strongest-peak-only search produced non-monotonic
depth bins, so those 12-sample bins must not be extrapolated. TTV,
single-transit, gap, explicit variability, longer-period, and multi-quarter
coverage remain unmeasured.

Version 0.2.46 adds the expanded Q1-Q4 `production_sensitivity_v2` artifact.
Across 16 deterministic scenario trials the pipeline recovered 8 with zero
execution failures. Moderate TTV and both deterministic-gap scenarios were
recovered; strong TTV, 90-day periodic, single-transit, and 500-ppm transit over
5,000-ppm injected variability cases were missed on both backgrounds. These
misses are retained as model/pipeline boundary evidence and must not be
converted into favorable assumptions.

## Runtime And System Notes

- Expected local Python: 3.14.3 in `.venv`.
- Observed current local PyTorch while preparing this card: 2.12.1.
- Training config uses `device=auto`.
- `docs/SYSTEM_PROFILE.md` requires PyTorch training to resolve to MPS on the
  M4 Max when `torch.backends.mps.is_available()` is true and to print the
  resolved device in the startup banner.
- This card records the committed system expectation and the local artifact
  ledger evidence; it does not claim that every future runner has MPS.

## Limitations

- The current benchmark is Kepler-domain only. It should not be treated as a
  proven TESS production classifier.
- Positive labels are sparse relative to negatives in the master corpus
  (3,413 positive and 12,236 negative examples).
- The model consumes folded snippets, not full light curves, target pixel files,
  centroid diagnostics, or follow-up photometry.
- Metrics are held-out corpus metrics, not discovery-yield proof.
- Full-ensemble production uses the completed T1-2 stacking calibration, whose
  zero CNN blend weight means CNN probabilities remain review metadata rather
  than decision weight.

## Promotion Scope

The promotion PR copies and intentionally stages only the selected production
artifacts, not the whole checkpoint directory:

```bash
git add models/benchmark_cnn_v1/MODEL_CARD.md \
  models/benchmark_cnn_v1/REPRODUCIBILITY_MANIFEST.json \
  data_selection/data_role_registry.yaml \
  models/registry.json
git add -f models/cnn/benchmark_cnn_v1/best.pt \
  models/cnn/benchmark_cnn_v1/calibration.json \
  models/cnn/benchmark_cnn_v1/config.json \
  models/cnn/benchmark_cnn_v1/metrics.json \
  models/cnn/benchmark_cnn_v1/promotion_manifest.json
```

The exact destination may change in the promotion implementation, but any
ignored checkpoint artifact must be staged intentionally with `git add -f`.
