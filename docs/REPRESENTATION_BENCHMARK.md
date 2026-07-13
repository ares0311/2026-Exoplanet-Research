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
- a TimesFM-style external foundation-model baseline;
- demonstrated improvement in grouped holdout, top-k review yield, or injection
  recovery beyond the promoted benchmark.

The training loop is a single stateful optimization and is not shardable.
Batching uses the selected accelerator; `device=auto` prefers MPS, then CUDA,
then CPU. Every epoch prints loss, learning rate, patience state, and ETA. A
successful run writes a structured result plus a Run Report.
