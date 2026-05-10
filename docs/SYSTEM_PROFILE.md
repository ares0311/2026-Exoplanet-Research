# LOCAL SYSTEM PROFILE

## Purpose

This file records the local development machine profile so pipeline code, tests, and notebooks can be sized sensibly for this project.

Use this as an optimization guide, not as a portability requirement. The codebase should still run on smaller systems unless a task explicitly documents a higher local resource target.

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
- Use the 40-core GPU or Metal acceleration only as an optional optimization path. CPU implementations should remain the default baseline until GPU behavior is tested and reproducible.

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

---

## Portability Rule

Optimizing for this MacBook Pro means choosing good defaults for local development. It does not mean hardcoding Apple-specific assumptions into scientific logic.

When performance-sensitive code needs system-specific behavior, expose it through configuration or documented runtime defaults.
