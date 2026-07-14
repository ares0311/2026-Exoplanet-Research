# Phase 3 Representation Benchmark

## Bounded pilot v1

`Skills/representation_pilot.py` tests whether a small masked-reconstruction
Transformer can learn useful representations from the existing 201-bin Kepler
phase-folded snippets. It is a go/no-go experiment for further data investment,
not a production model or a claim that the master-guide Phase 3 gate is closed.

The contract is leakage-safe:

1. Pretraining reads only `t1_1_kepler_master_train`; labels are never passed to
   the masked-reconstruction objective.
2. The encoder is frozen before fitting a linear classifier on train labels.
3. Validation selects the pretraining epoch and linear-probe state.
4. The predefined grouped test split is opened once for the versioned result.
5. The test AUC must strictly exceed `benchmark_cnn_v1` AUC 0.957211 to pass.

The comparison includes a small BLS-metadata/statistical linear baseline using
period, duration, and flux summaries. The result also records top-100 positive
yield. A passing pilot still cannot replace the promoted CNN without a separate
promotion package and human approval.

This design is grounded in primary work: [ASTROMER](https://arxiv.org/abs/2205.01677)
uses self-supervised Transformer representations for light curves;
[FALCO](https://arxiv.org/abs/2504.20290) pretrains a Transformer on unlabeled
Kepler light curves; and [Astroconformer](https://arxiv.org/abs/2309.16316)
tests attention-based long-range light-curve representations. The pilot is
intentionally smaller and does not claim architectural equivalence.

## Known missing Phase 3 evidence

- broad unlabeled Kepler and TESS pretraining data;
- stellar-variability labels;
- embedding-based injection-recovery comparison;
- measured inference and scientific comparison for the now-pinned external
  foundation-model baselines (Chronos-Bolt tiny and Astromer2);
- demonstrated improvement in grouped holdout, top-k review yield, or injection
recovery beyond the promoted benchmark.

Version 0.2.55 closes the source-identity and direct-footprint prerequisite for
the external arm. The immutable contract pins Python 3.14-compatible wheels and
exact ONNX commits/hashes for Chronos-Bolt tiny and Astromer2, totaling
56,036,648 direct bytes. The metadata-only verifier downloads no payloads and
must pass from merged code before dependencies or weights are introduced. This
is not inference evidence and does not close any scientific comparison item;
see `docs/REPRESENTATION_BASELINE_SOURCE_CONTRACT.md`.

The merged version 0.2.56 verification passed on 2026-07-14: 7/7 operations in
4.94 seconds, all five sources verified, 56,036,648 projected direct bytes, and
zero payload bytes downloaded. Evidence artifact SHA-256 is
`5610bbb859463e180bd9ee65ee7317518458560421487253d272b2d3b5753042`;
Run Report commit `ae4e659`. The source prerequisite is complete.

Version 0.2.57 adds the one-product/two-model inference smoke that addresses
the runtime-integration portion of the first remaining item. It pins optional
dependencies, exact weight revisions, a deterministic cached input, CPU thread
bounds, output shape/finiteness, and isolated memory/timing. Merged smoke
evidence will not itself satisfy grouped holdout, top-k, variability, or
injection-recovery comparison. The merged retry passed both exact models in
26.875 seconds with finite `(1,1,1,256)` embeddings and max child RSS
186,204,160 bytes. Artifact SHA-256 is `1cc59ab3…5de5d10`; Run Report commit
`f8a7207`. Version 0.2.59 records runtime integration complete while every
scientific comparison item above remains open.

Version 0.2.60 selects Drake et al. (2014), CDS/VizieR
`J/ApJS/213/9/table3`, for the missing variability-label source gate: 47,055
publication-backed rows, 17 classes, and a 1,166,660-byte compressed table.
The contract and zero-full-payload metadata verifier are documented in
`docs/STELLAR_VARIABILITY_LABEL_SOURCE_CONTRACT.md`. Merged verification is
next; a source PASS does not authorize the leakage-safe TIC crossmatch,
embedding extraction, or training.

The merged source verifier passed 5/5 operations in 3.334 seconds on
2026-07-14, verifying 47,055 rows, all 17 class counts, the required schema,
delivery metadata, and three labeled samples with zero full-catalog bytes.
Artifact SHA-256 is `eb5d4bc6…39b9a`; Run Report commit `b0003bb`. Version
0.2.61 closes source identity. The leakage-safe 2,790-TIC crossmatch and
embedding-aware injection comparison remain open.

The training loop is a single stateful optimization and is not shardable.
Batching uses the selected accelerator; `device=auto` prefers MPS, then CUDA,
then CPU. Every epoch prints loss, learning rate, patience state, and ETA. A
successful run writes a structured result plus a Run Report.

## Pilot v1 measured result

The merged-code run completed on 2026-07-12 in 33.5 seconds on Apple MPS. It
processed train/validation/test counts 10,800/2,393/2,456 and selected epoch 11
with validation masked MSE 125.091329.

| Model | Test AUC | F1 at 0.5 | Top-100 positive yield |
|---|---:|---:|---:|
| Frozen embedding + linear probe | 0.832630 | 0.635135 | 72% |
| Period/duration/flux-summary probe | 0.823495 | 0.274336 | 6% |
| `benchmark_cnn_v1` | 0.957211 | 0.834688 | not measured in this pilot |

Outcome: `does_not_beat_cnn`. The compact masked Transformer is rejected as a
CNN replacement and must not be tuned against or rerun on this frozen test
split. Its ranking improvement over the small tabular baseline supports a
materially different follow-up only after satisfying the missing Phase 3 data
contract. Durable evidence is
`artifacts/manifests/representation_pilot_v1.json`; the local checkpoint is
ignored and retained only for reproducibility.
