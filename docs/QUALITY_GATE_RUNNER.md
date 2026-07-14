# Parallel Quality-Gate Runner

`Skills/run_quality_gates.py` replaces sequential local quality checks with one
supervisor. It assigns every `tests/test_*.py` file to exactly one of six test
shards, runs six pytest-xdist workers inside each shard (36 test workers total),
and runs Ruff and mypy concurrently with those shards.

The file-level partition prevents duplicated test execution. A greedy weight
based on each module's test definitions and size keeps shard loads reasonably
balanced without a preliminary collection run. Each test child also limits
OpenMP/BLAS-style numeric backends to one inner thread so 36 pytest workers do
not accidentally become hundreds of competing numeric threads.

Measured on the recorded M4 Max on 2026-07-13, the optimized 6×6 full run
passed 2,718 default tests, Ruff, and mypy in 34.1 seconds. The prior
single-pytest-universe xdist baseline was 81.57 seconds, so the supervisor
reduced measured wall time by about 58% while preserving the pass count.
Injection-recovery unit tests now stub the separately tested BLS search layer
when asserting only grid shape/count/bounds, eliminating repeated scientific
search computation that did not contribute to those unit assertions.

## Canonical full gate

```bash
git switch main
git pull --ff-only origin main
caffeinate -i .venv/bin/python Skills/run_quality_gates.py
```

The launcher prints a startup banner and heartbeat/ETA, writes one log per gate
under `logs/quality_gates/<timestamp>/`, writes a combined JSON summary, and
returns non-zero if Ruff, mypy, or any test shard fails. Ctrl-C or SIGTERM
terminates every active child.

Use `--tests-only` only after the static gates have already passed unchanged
code in the same work cycle:

```bash
git switch main
git pull --ff-only origin main
caffeinate -i .venv/bin/python Skills/run_quality_gates.py --tests-only
```

Use `--dry-run` to inspect all eight commands and their disjoint test-file
assignments without starting them. Direct pytest remains appropriate for a
focused test file, failure reproduction, or a documented constrained-machine
fallback; do not launch multiple unpartitioned full-suite pytest commands.
