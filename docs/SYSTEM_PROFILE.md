# LOCAL SYSTEM PROFILE

## Purpose

This file records the local development machine profile so pipeline code, tests, and notebooks can be sized sensibly for this project.

This file is a committed source of truth for local optimization defaults. Do
not replace it with chat context, private notes, or untracked local state when
choosing worker counts, batch sizes, cache behavior, or long-running job
recipes.

Use this as an optimization guide and default-sizing directive, not as a
portability requirement. The codebase should still run on smaller systems
unless a task explicitly documents a higher local resource target.

Sensitive machine identifiers such as serial number, hardware UUID, provisioning UDID, and user account name are intentionally not recorded.

---

## Profile Snapshot

**Last verified:** 2026-05-01  
**Sources:** macOS About This Mac screenshot, `system_profiler SPHardwareDataType SPSoftwareDataType`, `system_profiler SPDisplaysDataType`

| Category | Local value |
|---|---|
| Machine | MacBook Pro, 16-inch, Nov 2024 |
| Model identifier | Mac16,5 |
| Chip | Apple M4 Max |
| CPU cores | 16 total: 12 performance cores, 4 efficiency cores |
| GPU cores | 40-core integrated Apple GPU |
| Memory | 64 GB unified memory |
| Metal | Supported |
| Startup disk | Phi |
| macOS | macOS 26.4.1 |
| Darwin kernel | Darwin 25.4.0 |

---

## Local Optimization Defaults

Prefer these defaults when running project code on this machine:

- Keep default CPU-bound worker counts below full saturation. Start with `12` workers for local batch jobs and increase only after measuring.
- Keep at least `2` CPU cores free during interactive work.
- For I/O-heavy work, external-service queries, or live catalog access, use lower concurrency first, usually `4` to `6` workers, because remote service limits and disk throughput can dominate.
- Target peak memory below `48 GB` for routine local runs, leaving about `16 GB` for macOS, browser windows, notebooks, and the editor.
- Chunk large target or sector sweeps by target, sector, or candidate batch rather than loading all mission data into memory at once.
- Prefer memory-mapped arrays, columnar files, or streaming reads for large intermediate products.
- Cache downloaded raw data and expensive intermediate products locally, but do not commit large mission data or generated cache directories.
- For AI/ML training, prefer the 40-core GPU through PyTorch Metal/MPS when
  available. Training code should expose `device=auto` and print the resolved
  device at startup; CPU should be an explicit override or fallback, not the
  silent default on this Mac.
- For non-training CPU-local batch work, prefer bounded multiprocessing or
  multithreading over strictly serial loops when it is scientifically safe and
  the workload is large enough to benefit. Start near the worker counts above,
  keep concurrency configurable, and measure before raising defaults.
- For live external-service workloads, keep concurrency lower and polite even
  on this machine. Avoid defaults that invite throttling or repeated failed
  network calls.

---

## Numerical Threading Guidance

Avoid accidental oversubscription when combining process-level parallelism with NumPy, SciPy, Astropy, Lightkurve, or other native numerical libraries.

For multi-process workloads, set native numerical libraries to one thread per process unless profiling shows a better setting:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
```

For a single large numerical job, allow native libraries to use more threads, commonly `8` to `12`, then benchmark before raising the limit.

---

## Project-Specific Guidance

### Fetch

- Treat live MAST, NASA Exoplanet Archive, ExoFOP, Gaia, and similar calls as rate-limited external services.
- Default tests should mock these services.
- Live integration runs should be marked and opt-in.

### Clean

- Process light curves sector-by-sector or target-by-target.
- Preserve transit-like signals before applying aggressive detrending.
- Write cleaned artifacts incrementally so long runs can resume.

### Search

- Box Least Squares sweeps can use local parallelism, but period-grid size should be explicit in configuration.
- For large sweeps, prefer bounded worker pools and progress checkpoints over one monolithic run.

### Vet

- Centroid, contamination, and systematics checks may require additional data products. Keep those downloads explicit and cacheable.

### Score and Classify

- Scoring and pathway classification should stay lightweight, deterministic, and pure where possible.
- These modules should run comfortably in unit tests without using the machine's full parallel capacity.

### Reports and Notebooks

- Notebooks may use this system's memory and CPU headroom for exploration, but production code should keep resource limits explicit.
- Reports should record enough provenance to reproduce results on this or another machine.

### Test Suite

- Run the full local gate through `.venv/bin/python
  Skills/run_quality_gates.py`. It partitions test modules across six pytest
  shards, runs six xdist workers per shard with work stealing, and supervises
  Ruff/mypy concurrently. It also caps native numeric backends at one inner
  thread per test worker to avoid nested oversubscription.
- Measured 2026-07-11 on this 16-CPU M4 Max: 2,630 tests took 226.45s serial,
  137.53s with 4 workers/load scheduling, 136.48s with 8 workers/load,
  138.88s with 16 workers/load, 81.54s with 8 workers/work stealing, and
  81.57s with 16 workers/work stealing. All runs passed identically.
- The 16-worker work-stealing default uses the maximum available parallelism
  with effectively identical best wall time; portable environments adapt via
  xdist's `auto` worker count.
- Measured 2026-07-13: the optimized six-shard × six-worker supervisor passed
  2,718 default tests plus Ruff/mypy in 34.1s, about 58% faster than the 81.57s
  single-universe baseline. Direct `-n auto --dist=worksteal` remains the
  focused-test and constrained-environment fallback.
- Version 0.2.54 validation on the same date passed 2,726 default tests plus
  Ruff/mypy in 34.1s with the same six-shard × six-worker configuration.
- Version 0.2.55 validation on 2026-07-14 passed 2,733 default tests plus
  Ruff/mypy in 31.1s with the same six-shard × six-worker configuration.
- Version 0.2.56 validation on 2026-07-14 passed 2,734 default tests plus
  Ruff/mypy in 26.1s with the same six-shard × six-worker configuration.
- Version 0.2.57 validation on 2026-07-14 passed 2,743 default tests plus
  Ruff/mypy in 24.2s with the same six-shard × six-worker configuration.
- Version 0.2.58 validation on 2026-07-14 passed 2,743 default tests plus
  Ruff/mypy in 26.2s with the same configuration and the optional
  representation dependencies installed.
- Version 0.2.59 evidence-release validation on 2026-07-14 passed 2,743 default
  tests plus Ruff/mypy in 34.3s with the same configuration.
- Version 0.2.60 validation on 2026-07-14 passed 2,751 default tests plus Ruff
  and mypy as 8/8 supervised gates in 25.2 seconds using six pytest shards and
  six xdist workers per shard. This remains the canonical full local release
  topology.
- Version 0.2.61 evidence validation passed the unchanged 2,751 tests plus Ruff
  and mypy as 8/8 gates in 40.2 seconds. All six test shards slowed broadly
  (25.1-40.1 seconds) without errors, timeouts, or a single-shard imbalance.
  Retain 6×6; this run does not justify increasing concurrency beyond 36 test
workers.

### TESS-Catalina metadata pilot

- Version 0.2.62 uses six modulo process shards with six I/O-bound exact-ID
  MAST batch workers each. The authorized 216-TIC pilot supplies about six
  six-ID batches per shard, exercising 36 concurrent requests once.
- The 1,166,660-byte Catalina gzip is a shared read-only input protected by a
  cross-process lock and atomic promotion. Do not multiply this download by
  shards or workers.
- Compare the pilot's aggregate TIC/s and service errors before authorizing a
  larger run. Back off on throttling/timeouts; do not scale beyond 6×6 from
  assumption.
- Version 0.2.62 validation passed 2,759 tests plus Ruff/mypy as 8/8 supervised
  gates in 34.3 seconds with the canonical six-shard/six-worker test topology.
- Version 0.2.63 ignores the shared `.cache/stellar_variability_labels/`
  runtime cache so the single-parent launcher sees a clean tree and its six
  child Run Reports can retain exact-file-only git ownership.
- Version 0.2.64 accepts the pinned Catalina payload's valid 71-byte unflagged
  records while retaining strict 71- to 73-byte bounds. The merged 6x6 pilot
  retry remains the next throughput and service-error measurement.
- Version 0.2.64 validation passed 2,760 tests plus Ruff/mypy as 8/8 supervised
  gates in 25.2 seconds with the canonical 6×6 test topology.
- Version 0.2.65 uses MAST's live-verified `duplicate_id` TIC column through
  contract v2; v1's rejected `duplicate_i` field is retained as audit evidence.
  Validate one six-ID request before the next 6×6 merged-main retry.
- The v2 six-ID probe returned all rows in 1.5 seconds. Version 0.2.65
  validation passed 2,761 tests plus Ruff/mypy as 8/8 gates in 26.2 seconds.
- Version 0.2.66 treats denied `.git/exo-run-report.lock` acquisition as a
  report-push failure: no unlocked git race, no data loss, warning plus exit 0.
- Version 0.2.66 validation passed 2,762 tests plus Ruff/mypy as 8/8 supervised
  gates in 34.3 seconds with the canonical 6×6 test topology.
- The merged Catalina pilot completed 216 TICs across 38 requests in 8.519s
  observed wall time (25.36 TIC/s). All six shards completed in 3.06-3.83s
  with no final service errors/timeouts; zero candidates were found.
- Version 0.2.67 validation passed 2,763 tests plus Ruff/mypy as 8/8 supervised
  gates in 24.2 seconds with the canonical 6×6 test topology.
- Version 0.2.68 applies the same single-parent 6x6 shape to the exact-TIC
  ASAS-SN metadata preflight. A preliminary six-worker measurement completed
  56 VizieR TAP requests for all 2,790 TICs in 7.86 seconds with zero failures,
  zero catalog payload bytes, and 48 matches. The merged reproduction uses six
  modulo shards with six workers each, 50 IDs per request, disjoint outputs,
  and global reconciliation. Do not increase beyond 36 request workers unless
  that run remains error-free and measured throughput justifies another step.
- Version 0.2.68 validation passed 2,772 default tests plus Ruff/mypy as 8/8
  supervised gates in 31.3 seconds under the canonical 6x6 test topology.
- The merged ASAS-SN preflight processed 2,790 TICs in 58 exact-ID batches plus
  six sequential source checks. First-shard-start to last-shard-completion wall
  time was 6.762 seconds (412.6 TIC/s); shard elapsed times were 1.665-5.876
  seconds. The one-second process staggering and shard-zero source checks
  explain the sub-linear improvement over the 7.86-second preliminary six-
  worker probe. All shards passed without final errors. Retain 6x6 and do not
  scale higher for this already-sub-10-second workload.
- Version 0.2.69 validation passed 2,773 default tests plus Ruff/mypy as 8/8
  supervised gates in 32.3 seconds under the canonical 6x6 test topology.
- Version 0.2.70's representation variability/injection harness retains the
  six-shard x six-worker shape for 48 independent cached TICs. FITS preparation
  and blind BLS own the six per-shard workers; frozen embedding inference is
  serialized through one Chronos-Bolt tiny and one Astromer2 session per shard
  so the process topology does not multiply into 72 ONNX sessions. Numeric and
  ONNX inner threads remain one. A one-TIC cache-only smoke read 12,508 clean
  cadences, completed all four injection/BLS trials, and returned finite
  256-element outputs from both models in under two seconds without writes.
- Version 0.2.70 validation passed 2,780 default tests plus Ruff/mypy as 8/8
  supervised gates in 33.3 seconds under the canonical 6x6 test topology.
- Version 0.2.63 validation passed the unchanged 2,759 tests plus Ruff/mypy as
  8/8 supervised gates in 27.3 seconds with the canonical 6×6 test topology.

### Cache-local representation preprocessing

- `Skills/benchmark_representation_preprocessing.py` uses six ordinary Python
  shard subprocesses with six FITS I/O/preprocessing threads per shard. This
  avoids Python process-pool semaphore requirements while preserving the 6×6
  single-parent model inside the approved sandbox.
- Numeric backend inner threads are capped at one in every shard. The bounded
  production sample is 36 products, one per TIC group, so every worker owns at
  most one product in the default run.
- A six-product real-cache process smoke passed 6/6 with no derived-array
  persistence on 2026-07-13. Use the merged 36-product result—not the smoke—to
  decide whether the measured throughput and memory justify further scaling.

### External representation baselines

- Version 0.2.55's source contract selects the 13.9 MB Chronos-Bolt tiny ONNX
  model and 16.0 MB Astromer2 ONNX model. With the three pinned direct wheels,
  the bounded payload is 56,036,648 bytes before transitive dependency/cache
  overhead.
- Source verification is seven small metadata operations and intentionally
  sequential; process startup would dominate. It downloads zero payload bytes.
- The merged version 0.2.56 run measured 4.94 seconds for all seven operations,
  confirming that sequential metadata verification is the correct shape.
- After a separate inference smoke establishes memory and throughput, extract
  embeddings across independent light curves with one parent, six shards, and
  six workers per shard. Keep ONNX Runtime intra/inter-op thread counts bounded
  so 36 workers do not create nested native-thread oversubscription.
- Version 0.2.57's one-product smoke runs each selected model in a separate CPU
  process with one ONNX intra-op and one inter-op thread. Sequential children
  make peak RSS attributable; this is a measurement exception, not the future
  extraction topology.
- Keep `HF_HOME` and `HF_XET_CACHE` under the ignored in-repo model cache
  before Hub import. The 0.2.57 smoke proved that Xet otherwise falls back to a
  sandbox-blocked home-cache log path; 0.2.58 makes containment explicit.
- The merged bounded retry completed in 26.875s. Chronos-Bolt tiny used
  126,058,496 bytes peak RSS and Astromer2 used 186,204,160 bytes; both emitted
  finite `(1,1,1,256)` embeddings with one ONNX intra/inter-op thread. Version
  0.2.59 records this runtime baseline for later 6×6 extraction sizing.

- The five-request variability-source verifier remains sequential because it
  is a sub-three-minute dependent metadata gate.

### AI/ML Training

- PyTorch training should use `device=auto` defaults that resolve to `mps` on
  this Apple Silicon Mac when `torch.backends.mps.is_available()` is true.
- Save checkpoints in a portable CPU state-dict format even when training uses
  MPS or another accelerator.
- Preserve deterministic seeds where supported, but record device and runtime
  details because accelerator kernels may differ numerically from CPU kernels.
- Keep batch size configurable. Increase batch size only after a measured run
  confirms that validation behavior and checkpoint quality remain acceptable.

---

## Portability Rule

Optimizing for this MacBook Pro means choosing good defaults for local development. It does not mean hardcoding Apple-specific assumptions into scientific logic.

When performance-sensitive code needs system-specific behavior, expose it through configuration or documented runtime defaults.
