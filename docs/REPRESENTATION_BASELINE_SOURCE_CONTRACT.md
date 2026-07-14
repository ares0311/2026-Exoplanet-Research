# Phase 3 External Embedding Source Contract

## Production outcome

This contract closes the reproducibility and storage-planning prerequisite for
the external foundation-model arm of the Phase 3 representation comparison. It
does not close the scientific benchmark and does not authorize training or a
production scorer change.

The immutable expected metadata is
`metadata/representation_baseline_source_contract_v1.json`. Run
`Skills/verify_representation_baseline_sources.py` to compare it with current
primary PyPI and Hugging Face metadata. The verifier reads metadata and pinned
file HEAD headers only; it installs nothing and downloads zero model bytes.

## Selected bounded baselines

| Role | Model | Frozen output | Context | Model bytes |
|---|---|---:|---:|---:|
| General time-series foundation model | Chronos-Bolt tiny | 256-d mean embedding | 2,048 observations | 13,901,747 |
| Astronomy-native comparator | Astromer2 | 256-d mean embedding | 200 observations | 15,989,097 |

The ONNX exports are supplied by the current `light-curve` embedding API.
Chronos-Bolt tiny represents the master guide's TimesFM-style external
time-series baseline without depending on a forecasting-only API. Astromer2
adds an astronomy-native control pretrained on 1.5 million MACHO light curves.

Full Chronos2 is deliberately excluded from the first comparison: its pinned
ONNX file is 463,834,607 bytes, versus 13,901,747 bytes for Chronos-Bolt tiny,
while both cover the same general time-series baseline role. It can be
reconsidered only if the bounded model exposes a measured capability gap.

## Runtime and storage boundary

The future optional runtime is pinned to:

- `light-curve==0.13.1`, whose macOS arm64 ABI3 wheel supports Python 3.10+;
- `onnxruntime==1.27.0`, with a native CPython 3.14 macOS arm64 wheel;
- `huggingface-hub==1.23.0`, supporting Python 3.10+.

The two model files plus the three direct wheels total exactly 56,036,648
bytes. Transitive package dependencies and cache overhead must be measured
before installation, but this bounded payload is far below the project's
100 GB ceiling. The dependencies remain absent from the default runtime until
merged source verification and a separate inference smoke pass justify adding
an optional dependency group.

## Frozen evaluation contract

1. Keep both external encoders frozen.
2. Apply an identical linear-probe protocol to each embedding source.
3. Group by target identity so no TIC or KIC crosses train, validation, or test.
4. Make every preprocessing and probe choice on train/validation only.
5. Open a new versioned test split once, after the comparison is frozen; do not
   reuse the consumed pilot-v1 test set for tuning.
6. Compare the promoted `benchmark_cnn_v1`, BLS/TLS/statistical features,
   Chronos-Bolt tiny, and Astromer2 on grouped holdout, top-k yield, and the
   versioned injection-recovery set.

Stellar-variability labels and the injection-recovery comparison remain open
prerequisites. A source verification PASS alone is not a scientific result.

## Parallelism decision

The verifier performs seven small metadata operations and is expected to
finish in seconds. Shard/process startup would dominate this one-off check, so
it is intentionally sequential. The later per-light-curve embedding extraction
is independently partitionable and must use the project's single-parent 6×6
pattern after inference throughput and memory are measured.

## Primary sources

- `light-curve` embedding API: <https://light-curve.snad.space/latest/embed/api/>
- `light-curve` package: <https://pypi.org/project/light-curve/0.13.1/>
- ONNX Runtime package: <https://pypi.org/project/onnxruntime/1.27.0/>
- Hugging Face Hub package: <https://pypi.org/project/huggingface-hub/1.23.0/>
- Chronos-Bolt upstream model: <https://huggingface.co/amazon/chronos-bolt-tiny>
- Astromer2 paper: <https://doi.org/10.1051/0004-6361/202554026>
