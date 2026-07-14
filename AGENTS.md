# AGENTS.md — Instructions for AI Coding Agents

This file contains binding rules for AI coding agents working in this repository.
It is the **single canonical home for every operating rule and policy** in this
project — every MANDATORY policy below lives here once, not duplicated in
`CLAUDE.md` or elsewhere. `CLAUDE.md` covers architecture/module-map/current-state
and defers to this file for rules; read both, but if a rule ever appears to
conflict between the two, this file wins.

---

## PRIMARY DIRECTIVE — READ THIS BEFORE ANYTHING ELSE

**The only authorized work is work that advances this project to live production.**

Every session must begin by reading:
1. `AGENTS.md` (this file)
2. `docs/PRODUCTION_READINESS.md`
3. `docs/DISCOVERY_RUNBOOK.md`
4. `docs/exoplanet_exomoon_dataset_handoff.md`
5. `docs/exoplanet_detection_research_brief.md` (skim satellite table + AI methods)
6. `docs/astrometrics_coding_agents_master_guide.md`
7. `docs/astrometrics_data_selection_policy.md`
8. `docs/astrometrics_external_and_cloud_storage_policy.md`

Before proposing or executing any task you must:
1. Read the current state and roadmap in `docs/PRODUCTION_READINESS.md`,
   `docs/ROADMAP.md`, and the relevant runbook rather than relying on a stale
   numbered-gap summary.
2. Choose the highest-impact safe task that materially advances the project
   toward live production. Prefer, in order: an unresolved production blocker;
   a failing pre-deployment check or production defect; an unfinished roadmap
   item; then the highest-value validation, reliability, operability, or
   deployment-readiness improvement supported by current evidence.
3. State explicitly what production outcome the task closes, unblocks, or
   measurably improves. If it does none of those things, **do not do it**.

Tier 1 and Tier 2 gaps are priority signals, not an authorization whitelist.
When every named gap is closed but the project is not live in production, do
not stop: inspect readiness checks, roadmap state, real operator workflows,
open defects, and deployment evidence, then continue with the most impactful
production-advancing task. Add or update roadmap/readiness tracking when the
existing documents do not represent a newly discovered blocker or defect.

### Prohibited work

- Adding Skills, modules, schemas, or scaffolding that do not materially advance a concrete production outcome.
- Repeating work already listed under "What Is Complete" in `docs/PRODUCTION_READINESS.md`.
- Writing "the next N utility scripts" without a concrete production need.
- Treating "Apply All System Directives" as permission to add undirected code — it means assess current readiness and work the highest-impact production task.
- Running `exo background-run-once` expecting to discover new planets — background automation scans **7 static fixture targets** (3 known planets + 4 synthetics) and is a CI validation tool, not a discovery engine. See `docs/DISCOVERY_RUNBOOK.md §Background Automation`.
- Continuing the run006/run008 candidate-review loop as the primary production path. Those scans are historical evidence; the active production path is the dataset/model-training plan in `docs/exoplanet_exomoon_dataset_handoff.md`.
- Proposing ad hoc CNN retraining against the old rejected corpora without first satisfying the dataset/source-contract requirements in `docs/exoplanet_exomoon_dataset_handoff.md`.

### When the user says "Apply All System Directives"

1. Read `AGENTS.md` and `docs/PRODUCTION_READINESS.md`.
2. State the current production state, unresolved blockers/checks, and roadmap items in priority order.
3. For planning: propose tasks in impact order where **every task materially advances production**. Do not pad the list with speculative work. Tasks may be agent-led (code) or human-led (data collection, API keys, expert review, network access) — both are valid plan items. Label each task clearly: **[AGENT]** or **[HUMAN]**.
4. For each task, identify external dependencies (API keys, network access, GPU, human reviewer) and surface them as explicit questions before the DO phase.
5. Do not propose or execute work without a concrete production outcome.

### Two-phase workflow: PLAN then DO

**PLAN phase** ("plan the next N tasks"):
- List all production-advancing tasks in priority order, labeled **[AGENT]** or **[HUMAN]**.
- For every **[HUMAN]** task, provide exact step-by-step instructions so the human can act independently.
- Ask all questions about external dependencies upfront.
- Do not execute anything.

**Between PLAN and DO — resolve all [HUMAN] tasks first:**
- The human works through every **[HUMAN]** task using the instructions from the plan.
- All **[HUMAN]** blockers must be cleared before the DO phase begins.
- If a **[HUMAN]** task needs interactive help, work through it with the human until it is resolved.

**DO phase** ("DO the next N tasks"):
- By the time DO begins, all **[HUMAN]** blockers are already cleared.
- Execute only **[AGENT]** tasks.
- The DO phase should never contain a **[HUMAN]** blocker — if one appears, the PLAN phase was incomplete.

### Outside blockers are not code problems

If the highest-priority production task is blocked by a human action (data collection, network access, API key, expert review), state the outcome, name the blocker, and **immediately provide a complete step-by-step recipe** assuming the user has zero background knowledge of the specific task. Do not ask "do you want the commands?" — give them.

### Human-blocker recipe format

When the user must take an action to unblock a gap:
1. Give exact commands to copy-paste, in order, with no ambiguity
2. Explain what each command does in one plain-English sentence
3. State exactly what output to paste back so you can continue
4. Do not stop at "here's how to get started" — give the complete recipe through to the handoff point

---

## Branch Naming Rule — MANDATORY

**All feature branches MUST use the `claude/` prefix** (e.g., `claude/fix-ssl-cert`, `claude/add-training-config`).

**Why this is non-negotiable:** `.github/workflows/ci.yml` only triggers CI on push to `main` or `claude/**`:
```yaml
on:
  push:
    branches: [main, "claude/**"]
```
A bare branch name like `fix-something` will NEVER trigger CI on push. A bare-named branch is a broken branch — CI will not run, the full cycle cannot complete, and the change cannot be safely merged.

**Rules (non-negotiable):**
1. Before running `git checkout -b <name>`, verify the name starts with `claude/`.
2. If you discover a branch was created without the `claude/` prefix, the correct action is: **stop, document the root cause, create a correctly named replacement branch from the same diff, and close the incorrectly named branch**. Never create a workaround branch as a symptom treatment.
3. Never push to a bare-named branch for any purpose, including "empty commit" workarounds.

---

## Local–Remote Sync Policy — MANDATORY

The user's local Mac and GitHub `main` are the joint source of truth. The
agent's server environment is a temporary workspace only. Keep them in sync
at all times — never leave them diverged.

**Agent rules (non-negotiable):**
1. Every code change must complete the full cycle: feature branch → commit → push → PR → CI green → merge to main → PR closed. Never leave a PR open at end of session.
2. Never tell the user to run a script that has not yet been merged to `main`. If a script is still on a feature branch, merge it first.
3. Every recipe given to the user must begin by switching to `main` and fast-forwarding from `origin/main`, so their local is guaranteed current before any command executes.
4. PRs must be merged, not just approved — after CI passes, promote from draft, squash-merge, and confirm the PR is closed before the session ends.
5. After every merge, remind the user to run `git switch main` then `git pull --ff-only origin main` on their Mac if they have a terminal open.

**Standard recipe header — prepend to EVERY user command:**
```bash
git switch main
git pull --ff-only origin main
```

**For long-running commands:**
```bash
git switch main
git pull --ff-only origin main
caffeinate -i .venv/bin/python Skills/<script>.py [args]
```

**What Not To Do:**
- Do not tell the user to run `python Skills/foo.py` before `foo.py` is on `main`.
- Do not leave PRs in draft or open state at end of a session.
- Do not commit directly to `main` — always use a feature branch and PR.
- Do not assume the user's local is current — always prepend the branch-safe sync block.

---

## macOS Long-Running Process Policy — ALWAYS USE caffeinate

Any recipe for a Python command that runs longer than ~60 seconds **must** use `caffeinate -i`:
```bash
caffeinate -i .venv/bin/python Skills/<script>.py [args]   # standard
caffeinate -dims .venv/bin/python Skills/<script>.py [args] # lid-close safe
```
This applies to: light curve downloads, CNN training, batch scans, injection-recovery, and any repeated-network or long-compute script. **Never give a bare `python ...` recipe for these.**

---

## Console Output and ETA — MANDATORY

**Every script that iterates over N items or trains for N epochs must print real-time progress with ETA.**
A silent script is indistinguishable from a hung one.

**Item loop pattern** (download, batch scan, injection-recovery, etc.):
```python
import time
start = time.monotonic()
for i, item in enumerate(items, 1):
    # ... work ...
    elapsed = time.monotonic() - start
    rate = i / elapsed
    remaining = (n_total - i) / rate if rate > 0 else float("inf")
    eta = f"{remaining/60:.0f}m{remaining%60:.0f}s" if remaining > 90 else f"{remaining:.0f}s"
    print(f"  [{i}/{n_total}]  elapsed={elapsed:.0f}s  ETA={eta}", flush=True)
```

**Training loop pattern** — one line per epoch, minimum fields: epoch, train loss, val loss, primary metric, LR, best/patience:
```
Epoch  4/50  train=0.4123  val=0.5335  auc=0.8177  lr=3.00e-04  ← best
Epoch  5/50  train=0.3990  val=0.5610  auc=0.8041  lr=3.00e-04  (patience 1/10)
```

Print a startup banner before any loop, showing total size, batch size, max epochs, and patience.
Print a completion or early-stop line at the end.

**Rules:**
- Always `flush=True` — buffered output defeats the purpose.
- Print at every step, or every 10 items for fast loops.
- **Never commit a long-running script without console output.** If a script is silent, add progress prints before committing.
- When modifying any existing long-running script, check it meets this standard and fix if not.

---

## Run Report Policy — MANDATORY

**Every acquisition or processing script (sharded or single-threaded) must write a structured completion report and auto-commit + push it after each successful run.**
Console output is for watching a run live; it is not a record. The old
pattern — human pastes console output, agent manually transcribes it into
`AGENTS.md`/`docs/PRODUCTION_READINESS.md`/the local artifact ledger — is
lossy and does not scale to multiple concurrent console tabs. A script
reporting its own outcome (like a PR announcing its own merge) replaces that.

**Mechanism** (`Skills/run_report.py` — read it before adding a new
acquisition script or modifying an existing one):
```python
from run_report import RunReport, report_path_for, run_and_commit_report

report = RunReport(
    script="my_fetch_script", status="success",
    started_at=started_at, completed_at=completed_at,
    elapsed_seconds=elapsed, items_processed=n, items_written=w, items_failed=f,
    output_paths=(str(output_path),),
    shard_index=shard_index, shard_count=shard_count,  # None/1 if not sharded
)
path = report_path_for("my_fetch_script", shard_index=shard_index, shard_count=shard_count)
run_and_commit_report(report, path)  # appends JSON line, then commits+pushes ONLY that file
```

**Rules:**
- Call this at the end of every successful run (or, for sharded scripts, every shard's run) — not per-item.
- The report ledger lives at `artifacts/manifests/run_reports/<script>.jsonl` (or `.shardIofN.jsonl` when sharded, so concurrent shards never contend for one file) and is **committed to git, never gitignored**.
- `commit_and_push_report`/`run_and_commit_report` must stage **only** the exact report path (`git add -- <path>`, never `git add .`/`-A`) — it must never sweep up unrelated uncommitted work sitting in the operator's working tree.
- A report-push failure must never crash the script or discard the data it already fetched: print a warning and exit 0; the human can push manually later.
- This is a narrow, intentional exception to the branch/PR/CI cycle above: the report push goes directly to whatever branch is checked out (normally `main`). It must never be used to push code, data files, or training manifests — only the small structured JSON completion record.
- To check cumulative progress without asking for pasted console output, read the ledger: `.venv/bin/python Skills/run_report.py <script-name>`, or use a script's own `--status-only` flag where one exists.
- **Retrofit scope**: this applies to every existing acquisition/processing Skill, not just newly-sharded ones — `batch_scan.py`, `star_scanner.py`, `fetch_kepler_lc_snippets.py`, `fetch_tess_lc_snippets.py`, `fetch_tess_kepler_overlap_snippets.py`, `fetch_tess_k2_overlap_snippets.py`, `fetch_kepler_tce.py`, `fetch_tess_toi.py`, `fetch_exofop_ctoi.py`, `fetch_nea_koi_lc_index.py`, `fetch_additional_tess_labels.py`, `fetch_confirmed_hosts.py`, `fetch_jwst_targets.py`, `fetch_jwst_lc.py`, `tess_tce_fetcher.py`. See `docs/DISCOVERY_RUNBOOK.md` Rule 7 for full detail and retrofit tracking.
- Tests must never invoke the real git commit/push path — inject a fake runner (see `commit_and_push_report`'s `run_fn` parameter and `tests/test_run_report.py`).

---

## Parallelism-First Recipe Policy — MANDATORY

**Standing rule: use the maximum safe parallelism available.** Parallelize
independent tool calls, and use all safely usable workers/processes/shards for
independent workload units. Correctness, measured throughput, memory capacity,
and observed external-service throttling are the bounds; legacy conservative
worker counts are baselines to exceed when measurements remain clean, not caps.

For any embarrassingly-parallel-across-independent-units workload (targets,
files, folds, sectors, EPIC/TIC/KIC IDs, anything with a `--max-*`/`-n`/count-
style bound), process-level sharding (`--shard-index`/`--shard-count`, one
process per supervisor child) is the default shape, not merely something to weigh.
If a target acquisition/processing script doesn't yet support it, that is a
gap to close (mirroring the proven pattern below) before recommending a
live run — not a reason to fall back to a single-process recipe.

**Single-parent orchestration is the default.** When a reviewed workload can
use six independent shards with six workers each, run it through
`Skills/run_six_shards.py` (or a task-specific supervisor with the same safety
properties) from one terminal. Do not make the operator open six terminals for
work the agent can supervise itself. A workload may be added to that launcher's
allowlist only after its native shard flags, disjoint outputs, Run Reports,
storage behavior, and interruption handling are verified. Use separate tabs
only when cross-terminal coordination materially helps a large operation and
the operator approves that plan.

**Before giving the user any recipe expected to take longer than 3 minutes, always consider sharding, multiprocessing/multithreading, or other parallelism — do not default to a purely sequential recipe without first checking whether a faster shape exists.**
This applies to every recipe, not just Kepler/TESS batch processing: training runs, injection-recovery sweeps, catalog downloads, evaluation/validation passes, anything with a `--max-*`/`-n`/count-style bound.

**Before proposing a recipe:**
1. Estimate the wall-clock time. If it's over ~3 minutes, explicitly work through whether the task is:
   - **Embarrassingly parallel across independent units** (targets, files, folds, sectors) → sharding (`--shard-index`/`--shard-count`, one process per supervisor child) is the default, per the pattern in `Skills/process_t1_kepler_batch.py` (version 0.2.18). Add it to the target script if missing rather than settling for `--workers`-only.
   - **I/O-bound within one process** (network/catalog calls) → in-process worker concurrency (`--workers`, `ThreadPoolExecutor`) *within each shard*; sharding and per-shard workers compose and should scale upward until throughput stops improving or errors/throttling appear.
   - **CPU-bound** (local computation, no external service) → multiprocessing or vectorization using all available CPUs by default, then retain a lower measured count only when it is demonstrably faster or required by memory limits.
   - Genuinely sequential (state must build incrementally, e.g. O-C ephemeris refinement across epochs) → say so explicitly rather than silently defaulting to slow.
2. When a target script lacks sharding support, add it before recommending a live parallel run — mirroring the pattern already proven in `Skills/process_t1_kepler_batch.py` and `Skills/fetch_t1_2_k2_calibration_snippets.py` (sharding added 2026-07-10): partition by `id % shard_count`, auto-suffix each shard's output/failure files so concurrent processes never race on one file, wire `shard_index`/`shard_count` into the Run Report. The only exception is a genuinely trivial one-off task that will not clear the 3-minute bar even sequentially — say so explicitly rather than silently skipping sharding.
   - **Shard-capable scripts today**: `Skills/process_t1_kepler_batch.py` (`--shard-index`/`--shard-count`, version 0.2.18) and `Skills/fetch_t1_2_k2_calibration_snippets.py` (partitions by `epic_id % shard_count`, auto-suffixed per-shard output/failure files). Any other acquisition/processing script (see the Run Report Policy retrofit list above) still needs this pattern added the first time a live multi-target recipe for it would clear the 3-minute bar.
3. Every parallel/sharded recipe must use the Run Report Policy above so
progress is self-reporting across shards. For live services, start from the
highest previously clean measurement and scale further while throughput
improves; back off immediately on throttling, timeouts, or increased failures.

4. Full local quality-gate runs use
`Skills/run_quality_gates.py`: it partitions every `tests/test_*.py` module
exactly once across six pytest shards, gives each shard six pytest-xdist workers
with `--dist=worksteal` (36 test workers total), and concurrently supervises
Ruff and mypy. Scientific numeric backends are held to one inner thread per
pytest worker so BLAS/OpenMP cannot multiply the intended 36-way concurrency.
The six shard logs and combined summary remain under ignored
`logs/quality_gates/`. Do not start six unpartitioned full-suite pytest runs;
that would duplicate tests rather than shard them. Direct pytest remains valid
for a focused diagnosis, or when a constrained environment cannot safely run
the canonical 6×6 supervisor. The earlier single-universe xdist baseline on the
recorded 16-CPU Mac reduced 2,630 tests from 226.45s serial to 81.57s; compare
the 6×6 measurement against that baseline and investigate any regression.
The optimized 6×6 run on 2026-07-13 passed all 2,718 default tests plus
Ruff/mypy in 34.1s after limiting native numeric backends to one inner thread
and removing redundant real BLS searches from unit tests whose assertions only
exercise injection-grid orchestration. That is about 58% faster than the
81.57s single-universe baseline.
The latest version 0.2.55 run on 2026-07-14 passed all 2,733 default tests plus
Ruff/mypy in 31.1s with the same six-shard × six-worker configuration.
Version 0.2.56 passed 2,734 default tests plus Ruff/mypy in 26.1s with that
same configuration after adding the Hugging Face redirect regression test.
Version 0.2.57 passed 2,743 default tests plus Ruff/mypy in 24.2s with the same
configuration; optional inference packages remained absent from default tests.
Version 0.2.58 passed the same 2,743 tests plus Ruff/mypy in 26.2s after the
optional representation group was installed and the Hub/Xet cache-containment
regression was added.
Version 0.2.59 passed 2,743 tests plus Ruff/mypy in 34.3s while recording the
successful merged inference artifact and closing the bounded runtime gate.

**Phase 3 stellar-variability label-source gate:** version 0.2.60 pins the
publication-backed Drake et al. (2014) Catalina catalog in
`metadata/stellar_variability_label_source_contract_v1.json` and verifies it
with `Skills/verify_stellar_variability_label_source.py`. The source has 47,055
machine-readable rows, 17 published classes, inspection flags, and a 1,166,660
byte compressed table. The verifier performs five primary-source metadata/TAP
checks, reads only three sample rows, and downloads zero full-catalog payload
bytes. This bounded gate is intentionally sequential because it completes
below the three-minute threshold. Gaia DR3 automated class predictions are
not ground truth; the gated approximately 160 GB StarEmbed corpus is outside
the current auth/storage boundary. A PASS authorizes only source identity.
Crossmatch and training remain gated. The later independent-TIC metadata
resolution/crossmatch must use the single-parent six-shard/six-worker pattern
after a small live-service throughput measurement. See
`docs/STELLAR_VARIABILITY_LABEL_SOURCE_CONTRACT.md`.
The version 0.2.60 release gate passed 2,751 default tests plus Ruff/mypy as
8/8 supervised gates in 25.2 seconds using six pytest shards with six xdist
workers each.
The merged verifier then passed 5/5 primary-source operations on 2026-07-14 in
3.334 seconds: 47,055 rows and all 17 class counts matched, the required schema
and 1,166,660-byte delivery metadata were exact, three labeled rows validated,
and zero full-catalog bytes were downloaded. Durable artifact SHA-256 is
`eb5d4bc6ae02065752e515fff19ed9b012d163f1d82a2be958796a65ba339b9a`;
Run Report commit `b0003bb`. Version 0.2.61 records source identity complete.
This does not authorize crossmatch or training; the next gate is the bounded,
leakage-safe 2,790-TIC metadata/crossmatch design described above.
The version 0.2.61 evidence-release gate passed the unchanged 2,751 default
tests plus Ruff/mypy as 8/8 supervised gates in 40.2 seconds under the same
6×6 topology. All six test shards were slower than the immediately prior run
but remained balanced (25.1-40.1 seconds), with no errors or timeouts; this was
not a single-shard serialization regression and does not justify scaling above
36 test workers.

**Phase 3 TESS-Catalina crossmatch pilot:** version 0.2.62 adds
`metadata/tess_catalina_crossmatch_contract_v1.json` and
`Skills/crossmatch_tess_catalina_labels.py`. Only the deterministic 216-TIC
pilot is authorized. Use `Skills/run_six_shards.py` from clean merged `main`:
six modulo process shards, six exact-ID MAST batch workers per shard, and six
TICs per request. A locked atomic cache download shares the hash-pinned
1,166,660-byte Catalina gzip across every shard; never download one copy per
worker. Accept at most one source within 1 arcsecond, require the precommitted
V/TESS magnitude safeguard, reject TIC duplicates/non-stars and Catalina blend
flag `f`, preserve raw class codes, and keep `training_authorized=false`.
All six outputs require global one-to-one reconciliation before any label use.
Full 2,790-TIC execution, FITS reads, embedding extraction, and training remain
gated pending pilot throughput/error/overlap evidence. See
`docs/TESS_CATALINA_CROSSMATCH.md`.
The version 0.2.62 release gate passed 2,759 default tests plus Ruff/mypy as
8/8 supervised gates in 34.3 seconds under the canonical 6×6 topology.
Version 0.2.63 explicitly ignores the shared
`.cache/stellar_variability_labels/` runtime cache. This keeps merged `main`
clean so the six shard-local Run Reports can safely commit only their own
ledger files; do not remove that ignore rule or force-add the cached catalog.
Version 0.2.64 fixes the other fail-closed defect exposed by that first live
pilot: CDS legitimately omits the two trailing blank bytes on 44,538 rows when
the optional Catalina class flag is absent; 2,517 flagged rows retain byte 73.
The parser accepts only the documented 71- to 73-byte forms, pads omitted
optional bytes before fixed-width slicing, and
continues to reject any shorter or longer row before MAST queries begin.
The version 0.2.64 release gate passed 2,760 default tests plus Ruff/mypy as
8/8 supervised gates in 34.3 seconds under the canonical 6×6 topology.
The version 0.2.63 release gate passed the unchanged 2,759 default tests plus
Ruff/mypy as 8/8 supervised gates in 27.3 seconds under the canonical 6×6
topology.

This optimized single-parent shard/worker shape is the standing default
wherever work is safely partitionable: acquisition, processing, tests,
evaluation sweeps, and similar independent units. Equivalent new supervisors
must preserve disjoint ownership, bounded resources, progress/ETA, failure
propagation, interruption cleanup, and any applicable storage/Run Report
guards. Use sequential execution only for genuinely state-dependent work, a
focused diagnosis, a constrained environment, or a measured parallel
regression; record the reason rather than silently falling back.

`Skills/benchmark_representation_preprocessing.py` is the current proven
task-specific processing example: one parent supervises six ordinary Python
shard subprocesses, each with six bounded FITS I/O/preprocessing threads. It
validates the committed training-only inventory contract and cache containment,
downloads nothing, discards every derived array, writes one small aggregate
manifest, and emits one Run Report only after an accepted aggregate run. Use
this process-plus-worker pattern for similar cache-local bounded processing
when the generic acquisition launcher is not the right interface.

**Phase 3 external-baseline source gate:** use the immutable
`metadata/representation_baseline_source_contract_v1.json` and
`Skills/verify_representation_baseline_sources.py` before installing or loading
any external embedding model. The bounded first comparison is frozen to
Chronos-Bolt tiny (general time-series foundation baseline) and Astromer2
(astronomy-native comparator), at exact repository commits and hashes. The
verifier is metadata-only and intentionally sequential because its seven small
requests complete below the three-minute sharding threshold. A PASS verifies
source identity and the 56,036,648-byte direct payload only; it does not
authorize dependencies, weight downloads, model training, or promotion. Once
inference is separately proven, per-light-curve embedding extraction must use
the single-parent 6×6 pattern because those units are independently
partitionable. See `docs/REPRESENTATION_BASELINE_SOURCE_CONTRACT.md`.
The verifier must inspect the initial Hugging Face resolver response without
following its 302 redirect: commit, size, and content SHA live in
`x-repo-commit`, `x-linked-size`, and `x-linked-etag` on that authoritative
response, not on the final Xet object-store response.
The merged version 0.2.56 verifier passed on 2026-07-14: 7/7 metadata
operations in 4.94 seconds, five sources verified, 56,036,648 projected direct
bytes, and zero payload bytes downloaded. Durable evidence is
`artifacts/manifests/representation_baseline_source_verification_v1.json`
(SHA-256 `5610bbb859463e180bd9ee65ee7317518458560421487253d272b2d3b5753042`);
Run Report commit `ae4e659`. This closes source identity/footprint only. It
still does not authorize dependency installation, weights, inference, or
training.

**Phase 3 external-baseline inference gate:** after the source evidence above,
use `Skills/smoke_representation_baseline_inference.py` before any broader
embedding extraction. Version 0.2.57 pins its dependencies in the optional
`representation` group, downloads only the exact two ONNX revisions into
ignored `.cache/representation_models/`, opens one deterministic inventory-
owned TESS product, and runs each model in an isolated CPU child with ONNX
intra/inter-op threads set to one. Require finite `(1, 1, 1, 256)` outputs and
record peak RSS/timing. The two-model/one-product smoke is intentionally
sequential for attributable memory; a later inventory extraction must use the
single-parent 6×6 pattern. A smoke PASS still does not authorize training or a
production model change. See `docs/REPRESENTATION_INFERENCE_SMOKE.md`.
Version 0.2.58 also requires the smoke to set `HF_HOME` and `HF_XET_CACHE`
inside `.cache/representation_models/` before importing `huggingface_hub`.
This is a sandbox, secret-isolation, and `git add .` safety requirement: never
let the runtime fall back to `~/.cache/huggingface`.
The merged 0.2.58 retry passed both exact models on 2026-07-14 in 26.875
seconds: Chronos-Bolt tiny peak RSS 126,058,496 bytes and Astromer2 peak RSS
186,204,160 bytes, both finite `(1, 1, 1, 256)` embeddings. Exact weight
payload is 29,890,844 bytes; full ignored cache file content is 29,960,842
bytes. Durable artifact SHA-256 is `1cc59ab3…5de5d10`; Run Report commit
`f8a7207`. Runtime integration is complete. The next Phase 3 work is scientific
design/evidence for stellar-variability labels and injection recovery—not
unbounded extraction or training.

**Ask, don't assume, when:**
- The right shard/worker *count* (not whether to shard at all) depends on the operator's own tradeoffs — available machine capacity, concurrent work, and trust in the external service's rate limits.
- It's unclear whether the task is I/O-bound, CPU-bound, or a mix (measuring first, then asking whether to push further, beats guessing a big number).
- Two people, one Mac: if a task is already running, ask before starting more concurrent supervisors.

The 3-minute bar governs whether to shard at all for a given one-off task; it is not license to skip adding sharding to a script that will be reused for future multi-target batches just because today's invocation happens to be small.

**Measure-then-scale cadence — do not recommend more parallelism from assumption alone:**
- After any parallel/sharded run completes, compute its real per-item rate (from the run report or console output) and compare it against the last known baseline (sequential, prior worker count, or prior shard count) before proposing the next step up.
- A rate close to or better than the prior baseline, with no new errors/timeouts, is what justifies scaling further (more shards, more workers). A regressed rate, new `ERROR:`/timeout flags, or any sign of throttling is a stop signal — back off, don't push harder.
- Sub-linear scaling (e.g., 2 shards giving nowhere near ~2x throughput) is itself a bug worth investigating, not just a result to accept because nothing crashed. Check the code for an artificial bottleneck (a lock, a shared mutable global, a serialized third-party call) before concluding the external service itself is the ceiling — this project has twice found a real in-process serialization bug this way (`_DOWNLOAD_PRODUCTS_LOCK` in `exo_toolkit/fetch.py`, fixed in version 0.2.19) rather than the bottleneck being MAST/Astroquery itself.
- Never recommend "try more workers/shards" as a bare escalation without this comparison — ground the next recommended count in the last real measurement, not in optimism.

---

## Python Environment Policy — NEVER TOUCH SYSTEM PYTHON

- Validated runtime: **Python 3.14.3** inside `.venv` — never use system Python
- All work happens inside the `.venv` virtual environment
- Never run `/Applications/Python*/Install\ Certificates.command`
- Never suggest `sudo pip install` or any path under `/Library/Frameworks/Python.framework/`
- Fix SSL/package issues inside the venv only
- To verify the venv is active before suggesting any pip/python commands, check that the prompt starts with `(.venv)` — do not trust an ambiguous prompt

**Command rules — NON-NEGOTIABLE:**
- NEVER give a bare `python` command in any recipe. Always use `.venv/bin/python`.
- NEVER give a bare `pip` command in any recipe. Always use `.venv/bin/python -m pip`.
- NEVER give `pip install` even if the venv appears active — the prompt cannot be trusted.
- NEVER give `source .venv/bin/activate` as a precondition and then bare `python`/`pip` — the user may be in a different shell or the prompt may be misleading.
- Every recipe line that touches Python must use the explicit `.venv/bin/python` prefix, e.g.:
  ```bash
  .venv/bin/python -m pip install torch
  .venv/bin/python Skills/train_cnn.py ...
  .venv/bin/python -m pytest
  ```

**`.venv` management rules — NON-NEGOTIABLE:**
- NEVER delete or recreate `.venv` without explicit human approval.
- NEVER create a venv with `--prompt <name>` unless that name is exactly the directory (`.venv`). A mismatched prompt causes the shell to show the wrong environment name and can mislead both the user and the agent.
- If `.venv` appears broken (wrong Python version, missing pip, missing torch), diagnose first — state the exact observed vs. expected Python version — then ask the human for permission before proposing a rebuild.
- Confirmed `.venv` identity: `python3.14 -m venv .venv`; prompt line in `pyvenv.cfg` must read `prompt = .venv`.

---

## Local System Profile Optimization — MANDATORY

`docs/SYSTEM_PROFILE.md` is a committed production directive and the
authoritative local hardware profile for this project. It must remain in the
repository and must not be treated as a disposable local note.

Before performance-sensitive changes, long-running recipes, worker-count
defaults, batch-size defaults, cache layout changes, or AI/ML training work,
read it and optimize defaults for the recorded MacBook Pro M4 Max profile
while keeping the code portable and configurable.

AI/ML training must use accelerator-first defaults. PyTorch training should use
a configurable `device=auto` policy that resolves to Apple Metal/MPS on the
recorded M4 Max when available, then CUDA when available, and CPU only when no
accelerator is available or the operator explicitly requests CPU. Training
startup banners must print the resolved device.

Other performance-sensitive code should use bounded multiprocessing or
multithreading when it is safe and useful. Use `docs/SYSTEM_PROFILE.md` to
choose local worker defaults, keep at least two CPU cores free for interactive
work, and keep live external-service jobs polite and lower-concurrency. Never
hardcode Apple-only assumptions into scientific scoring, classification, or
pathway logic; expose system-specific behavior through config or CLI flags.

---

## Astrometrics Data And Storage Policy — MANDATORY

The three Astrometrics policy docs in `docs/` are production directives:

- `docs/astrometrics_coding_agents_master_guide.md`
- `docs/astrometrics_data_selection_policy.md`
- `docs/astrometrics_external_and_cloud_storage_policy.md`

Before changing datasets, training sets, live-search targets, model promotion
artifacts, storage layout, cloud/external-drive behavior, or large-download
recipes, apply these policies. Data roles must stay separated
(`training`, `validation`, `calibration`, `frozen_eval`, `live_search`,
`followup_live_search`); data-selection decisions must be recorded in
`data_selection/data_selection_decision_log.md`; public raw archives should be
bounded re-downloadable cache unless pinned by policy; manifests, ledgers,
configs, reports, model cards, calibration/eval artifacts, and candidate
evidence are the durable truth. The 4TB external SSD is the normal large local
workspace when available, while Dropbox-style sync must not be treated as the
authoritative dataset layer for raw archives, batch caches, model artifacts, or
candidate evidence ledgers.

---

## Git-Add-Safe Artifact Policy — MANDATORY

The standard operator cadence is `git add .`. The repository must make that
safe. If `git add .` would stage local corpora, generated splits, checkpoints,
runtime logs, generated reports, virtual environments, rejected experiments, or
cache files, fix `.gitignore` before continuing.

Other coding agents may only see GitHub. Local-only artifacts stay ignored, but
their production-relevant state must be committed in:

- `docs/LOCAL_ARTIFACT_LEDGER.md`
- `artifacts/manifests/local_artifacts.json`

Whenever an ignored artifact affects T1-1 or another production gate, update
the ledger in the same PR as the code, runbook, or readiness change. GitHub
must show expected paths, current status, counts/hashes/validation results,
approval state, and exact next commands. Do not leave artifact truth only in
chat context, terminal output, or local files.

A production-approved CNN checkpoint is the only CNN artifact class that may be
promoted from ignored local state into `models/`, and only after evaluator PASS,
a committed promotion readiness package, and explicit human approval. The
readiness package must include temperature-calibration-aware promotion tooling,
a model card, reproducibility manifest, data-role registry, storage/retention
ledger updates, exact selected artifact scope, and frozen benchmark designation.
Because CNN model paths are ignored defensively, promotion may require a
documented `git add -f`.

---

## Label-Source Discovery Protocol — MANDATORY

Before claiming any labeled data source (Kepler, K2, TESS, SETI/BL, or future
missions) is exhausted, follow the methodology in
`docs/seti_labeled_hit_data_research.md`'s "Comprehensive Protocol: Ensuring
All Labeled Kepler and TESS Datasets Have Been Found" section, generalized to
whatever mission/domain is in scope:

1. **Discover tables via schema introspection, never hardcoded names.** For
   NASA Exoplanet Archive TAP, query `TAP_SCHEMA.tables` with a broad
   `LIKE` sweep across every relevant mission keyword (e.g.
   `%kepler%`/`%koi%`/`%tce%`/`%k2%`/`%tess%`/`%toi%`), not just the term
   that matches your current best guess — a `%tce%`-only sweep in this
   project's own history missed the `_KOI` table family and `k2pandc`
   entirely until the broader sweep was run. A related, already-hit
   gotcha: `epic_id` appears in `tap_schema.columns` as an ADQL view alias
   for `k2pandc`, but the underlying Oracle column is `k2c_objid` —
   querying `epic_id` directly raises `ORA-00904: 'EPIC_ID': invalid
   identifier`. **Never use `epic_id` in a `k2pandc` query**; use
   `k2c_objid` (with `epic_candname`, parsed from `"EPIC 211311380.01"`, as
   fallback). `Skills/fetch_tess_k2_overlap_snippets.py` hit this along
   with two other TAP query bugs in sequence — a disposition value with a
   literal space (`'FALSE POSITIVE'`) not percent-decoded correctly inside
   a SQL `IN` clause (fixed by filtering disposition locally instead of in
   SQL, plus proper `urllib.parse.quote()` encoding), and the table-name
   case-sensitivity issue described below — all three fixed across PRs
   #131/#132/#134.
2. **Audit VizieR/CDS and the literature** for mission-specific
   publication-backed catalogs (e.g. Planet Hunters TESS/Kepler,
   robovetter releases) before concluding a mission's official archive
   tables are the only source.
3. **Accept a source only with positive evidence**: a real downloadable
   machine-readable table, row-level records, and a human/published-review
   verdict column that is not merely a model's own prediction. Log
   rejections with an explicit reason (`no_machine_readable_table_found`,
   `aggregate_counts_only`, `model_outputs_not_ground_truth`,
   `data_available_upon_request_only`, etc.) rather than silently dropping
   a checked lead.
4. **Do not repeat an already-logged investigation.** Current label-source
   completeness status as of 2026-07-05 (see `docs/PRODUCTION_READINESS.md`
   T1-1 for the full narrative):
   - **Kepler**: close to exhausted via NASA Exoplanet Archive TAP.
     `cumulative` (KOI) + `Q1_Q17_DR24_TCE` are both in use. `Q1_Q17_DR25_TCE`
     is confirmed live to have its label columns (`av_training_set`,
     `av_pred_class`) entirely empty; its ~6,382 kepids not in DR24 mostly
     have no usable label anywhere (only 535 have a KOI disposition, and
     those were already captured by the KOI-based manifest independent of
     any TCE table). `Q1_Q12_TCE`/`Q1_Q16_TCE` have the `av_training_set`
     column but are reasoned (not independently verified) to be redundant
     subsets of DR24's full-mission reprocessing. The one genuinely untried
     lever is fetching K2's own native light curves for every `k2pandc`
     entry (not just the TESS-re-observed subset already used).
   - **TESS**: three open, unresolved threads, do not re-search from
     scratch — pick these up directly. (a) The TEV TCE catalog
     (`tev.mit.edu`) has confirmed real disposition fields (EXOFOP/Group/
     Human-Triage) across SPOC+QLP+FAINT-search TCEs, but its data API has
     not been found (JS SPA; needs a real browser session with dev tools
     open, not a static fetch). (b) Planet Hunters TESS / NotPlaNET
     (`github.com/vtardugno/TESS-CNN`) has real human-vetted PC/EB/OTH
     labels, confirmed not publicly downloadable — the README states data
     is available only by emailing the authors. (c) The T16 Planet Hunt
     (arXiv:2604.18579, ~11,554 candidates from 83.7M light curves) is a
     very recent, potentially large lead whose data-availability statement
     could not be checked — both arXiv and IOPscience returned 403s to
     automated fetches (their own anti-scraper measures, not this
     project's network policy); needs a real browser check.

---

## Standing Rules

- **Skills directory**: Any standalone `.py` utility script created to perform a task (data processing, report generation, injection-recovery, etc.) must be saved in `Skills/` at the project root. Create the directory if it does not exist. This allows scripts to be discovered and reused across sessions rather than recreated.

---

## Quality Gates

Run the single-parent 6×6 gate supervisor before every commit when the local
environment supports it:

```bash
.venv/bin/python Skills/run_quality_gates.py
```

It runs these logically independent gates concurrently: `.venv/bin/python -m
ruff check .`, `.venv/bin/python -m mypy src`, and six disjoint pytest file
shards with six xdist workers each. Use `--tests-only` only when Ruff and mypy
have already passed unchanged code in the current work cycle. Use direct
`.venv/bin/python -m pytest ... -n auto --dist=worksteal` only for focused
diagnosis or a documented constrained-environment fallback.

If a gate cannot run because of a local environment issue, record the exact blocker in the handoff or commit message. Default tests must not require live external services.

If pytest fails with `ModuleNotFoundError: No module named 'exo_toolkit'`, add
`PYTHONPATH=src`. `mypy` (bare binary) sees a different package path and reports
false import errors for pydantic/numpy — always use `.venv/bin/python -m mypy
src`.

## Code Standards

- Python 3.14.3 (validated runtime; minimum acceptable is 3.11).
- Use `from __future__ import annotations` in Python modules.
- Prefer Pydantic v2 frozen models for structured data contracts.
- Public functions must be fully typed.
- `OptScore = float | None`: absent diagnostics contribute neutrally to log scores, while threshold gates treat missing participating diagnostics conservatively.
- Add comments for why, not for obvious what.
- Keep changes scoped to the task and existing architecture.

## Testing Standards

Every meaningful code change needs appropriate tests. Required coverage should scale with risk:

- Unit tests for public functions and numerical thresholds
- Fixture-based tests for complex interactions
- `None`-input tests for `OptScore` paths
- Conservation and sanity tests for posteriors and bounded scores
- Integration tests for pipeline behavior with mocked external services
- `@pytest.mark.integration_live` for tests requiring live network access

Do not include live service tests in the default suite.

---

## Scientific Guardrails

Follow `docs/SCORING_MODEL.md §15` (this list is the canonical restatement —
do not restate it a third time elsewhere; point here or at the spec instead):

- Never emit "confirmed planet" for internally detected signals.
- Use "candidate signal", "possible transit-like event", or "follow-up target".
- Always expose false-positive evidence alongside positive evidence.
- Preserve provenance for scores, thresholds, inputs, and generated reports.
- Suppress formal submission pathways (`tfop_ready`) if key diagnostics are missing.
- Prefer conservative classifications over optimistic ones.
- Conservative priors by default; mission-specific prior profiles are opt-in.
- `provenance_score` gates `tfop_ready` — 2-min SPOC with ≥2 sectors required.
- External submission, discovery claim, or contact requires explicit human approval — including background automation draft reports.

---

## Background Automation

Background search automation uses top-level configuration and top-level SQLite runtime logs:

```text
configs/background_search_v0.json
logs/background_search.sqlite3
```

Generated SQLite databases and background report exports are runtime artifacts. Do not commit them unless a future decision explicitly promotes a fixture artifact.

The authoritative one-shot command is:

```bash
exo background-run-once
```

It scans **7 static fixture targets only** (see Prohibited Work above) — it is a CI validation tool, not a discovery engine. Schedulers should call one bounded run at a time, capture stdout/stderr, and avoid overlaps. See `docs/BACKGROUND_SEARCH_AUTOMATION_BLUEPRINT.md`, `docs/BACKGROUND_SEARCH_SQLITE_SCHEMA.md`, and `docs/SCHEDULER.md`. Technical module map (submodules, CLI subcommands, exit codes): `CLAUDE.md`'s Background Automation Module section.

---

## MCP Server Usage

When configured and available in the agent's environment, prefer these MCP servers over guessing or ad hoc web search:

- **GitHub MCP** — issues, PRs, remote branches, repo metadata, commit/PR review, PR notes and links, branch health.
- **Context7 MCP** — current library/framework/API/CLI documentation (`resolve-library-id` then `query-docs`). Use instead of relying on training-data knowledge, since library versions and APIs change.
- **arXiv MCP** — preprint lookup, paper search, and research context for exoplanet/exomoon detection methods.
- **NASA ADS MCP** — astronomy/astrophysics literature search, bibcodes, citations, references, author metrics, and BibTeX export.

These are general-purpose research/collaboration tools, separate from this repo's project-scoped MCP servers (`exo_project_files`, `exo_git_read`, `exo_guard` — see `CLAUDE.md` "Project-Scoped MCP Servers"), which remain the sandboxed, offline-by-default servers for reading repo files, safe git inspection, and fixed validation commands. Availability of the general-purpose servers above is environment-dependent — check before assuming they're present, and fall back to `WebSearch`/`WebFetch` or manual `gh`/`git` commands if not.

## Multi-Agent Continuity

Multiple agents may work on this project across separate sessions, branches, and chat threads. Repository documentation is the continuity mechanism.

When durable instructions, architectural decisions, operating rules, or scientific assumptions are established, record them in the appropriate repository document instead of leaving them only in chat. If chat context conflicts with repository documentation, prefer repository documentation unless the user explicitly instructs otherwise in the current task.

Preserve enough rationale, provenance, and test evidence in commits, docs, and code comments for another agent to continue without needing the conversation that produced the change.

## Branch And Git Policy

Default development should happen on a non-`main` branch and be merged through review. Do not push directly to `main` unless the current user explicitly requests a direct `main` commit or push.

Before committing, check `git status --short --branch`. Do not overwrite or revert unrelated user changes.

---

## What Not To Do

- Do not add features, abstractions, or refactors beyond what the task requires.
- Do not skip validation silently.
- Do not claim a signal is a confirmed planet.
- Do not enable live network access in default tests.
- Do not hide durable rules in chat-only context.
- Do not touch system Python — all Python work happens inside the `.venv`.

---

## Read First

Before writing code, recover project context from committed files (in
addition to the PRIMARY DIRECTIVE reading list above). Read:

- `CLAUDE.md` — current codebase state, module map, type system, quality commands
- `docs/SCORING_MODEL.md` — mathematical specification for scoring and classification
- `docs/PIPELINE_SPEC.md` — end-to-end pipeline architecture
- `docs/PROJECT_STATUS.md` — current active state and next work
- `docs/DECISIONS.md` — durable architectural decisions
- `docs/LOCAL_ARTIFACT_LEDGER.md` — GitHub-visible state for ignored local artifacts
- `CONTRIBUTING.md` — setup, validation, and contribution policy

Do not rely on chat context, memory, or prior conversation history as the source of truth.

---

## Current Production Status

**No Tier 1 gaps are open as of 2026-07-10** (T1-0/T1-1/T1-2 all complete —
T1-1's `benchmark_cnn_v1` CNN checkpoint is promoted, T1-2's stacking
calibration is wired into production). Tier 2 is closed: T2-1 complete,
T2-2/T2-3 permanently out of scope (DECISION-013).

Closed numbered gaps do not mean the project is live or that work should stop.
Continue from the pre-deployment checklist, roadmap, open production defects,
and real workflow validation using the PRIMARY DIRECTIVE's impact ordering.

The full handoff narrative, version-by-version changelog, CNN candidate
history (C1–C19), and corpus/checkpoint status is **not repeated here** — it
is kept current in one place: `docs/PRODUCTION_READINESS.md` (re-read it
fresh every session; do not rely on a cached summary, including this one).
Local artifact/corpus/checkpoint file-by-file status:
`docs/LOCAL_ARTIFACT_LEDGER.md`. Per-Skill historical changelog (Milestones
12–30, archival only): `docs/MILESTONE_HISTORY.md`.

Before starting any new task, re-check `docs/PRODUCTION_READINESS.md`,
`docs/ROADMAP.md`, and relevant runbooks per the PRIMARY DIRECTIVE above — do
not assume the status noted here is still current by the time you read it.
