# AGENTS.md — Instructions for AI Coding Agents

This file contains binding rules for AI coding agents working in this repository.

---

## PRIMARY DIRECTIVE — READ THIS BEFORE ANYTHING ELSE

**The only authorized work is work that advances this project to live production.**

Every session must begin by reading:
1. `AGENTS.md` (this file)
2. `docs/PRODUCTION_READINESS.md`

Before proposing or executing any task you must:
1. Name the highest-priority unresolved Tier 1 gap from `docs/PRODUCTION_READINESS.md`.
2. State explicitly how the proposed work closes or directly unblocks that gap.
3. If the proposed work does not close or unblock a named gap in `docs/PRODUCTION_READINESS.md`, **do not do it**.

### Prohibited work

- Adding Skills, modules, schemas, or scaffolding that do not directly close a named Tier 1 or Tier 2 gap.
- Repeating work already listed under "What Is Complete" in `docs/PRODUCTION_READINESS.md`.
- Writing "the next N utility scripts" when those scripts do not unblock a named gap.
- Treating "Apply All System Directives" as permission to add more code — it means read the gap list and work the highest-priority gap only.

### When the user says "Apply All System Directives"

1. Read `AGENTS.md` and `docs/PRODUCTION_READINESS.md`.
2. State the current Tier 1 and Tier 2 gaps in priority order.
3. For planning: propose tasks in priority order where **every task closes or unblocks a named gap**. Stop when gap-closing tasks run out — do not pad the list with non-gap work. Tasks may be agent-led (code) or human-led (data collection, API keys, expert review, network access) — both are valid plan items. Label each task clearly: **[AGENT]** or **[HUMAN]**.
4. For each task, identify external dependencies (API keys, network access, GPU, human reviewer) and surface them as explicit questions before the DO phase.
5. Do not propose or execute work that does not close a named gap.

### Two-phase workflow: PLAN then DO

**PLAN phase** ("plan the next N tasks"):
- List all gap-closing tasks in priority order, labeled **[AGENT]** or **[HUMAN]**.
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

If the highest-priority Tier 1 gap is blocked by a human action (data collection, network access, API key, expert review), state the gap, name the blocker, and **immediately provide a complete step-by-step recipe** assuming the user has zero background knowledge of the specific task. Do not ask "do you want the commands?" — give them.

### Human-blocker recipe format

When the user must take an action to unblock a gap:
1. Give exact commands to copy-paste, in order, with no ambiguity
2. Explain what each command does in one plain-English sentence
3. State exactly what output to paste back so you can continue
4. Do not stop at "here's how to get started" — give the complete recipe through to the handoff point

---

## HANDOFF STATE — 2026-06-26 (READ THIS FIRST)

**The only active gap is T1-1: Production CNN Checkpoint (AUC ≥ 0.85, F1 ≥ 0.80).**

### What was done in the previous sessions (2026-06-21 – 2026-06-26)

- **Temperature scaling live** — PR #125 merged. Platt replaced by temperature scaling in evaluator.
- **C17 REJECTED** — joint Kepler+TESS training (9,633 examples) achieved only val AUC 0.7859, worse than C13–C15 (0.81–0.84). Root cause: domain mismatch. Do not retry joint training.
- **C18 REJECTED** — `freeze_conv_epochs=10` on TESS combined splits. Best candidate so far: test AUC 0.8439, test F1 0.7979 (raw). Failed gates: raw AUC 0.8439 < 0.85 (short by 0.006); T=1.61 worsened already-excellent test calibration (ECE 0.0301→0.0667). See runbook Step 7e.
- **C19 REJECTED** — `freeze_conv_epochs=20` (doubled from C18). SHA-256 `65f3721fac577807f35e4edaeaa9cc0cd0f50959441344487f7c77f35a570436`. Test AUC 0.8420 < C18's 0.8439 — regressed. T=1.88 worsened ECE further (0.0377→0.0760). Root cause: LR scheduler does not fire during the frozen phase (val_auc improving), so conv unfreezes at same LR=1e-4 as C18; longer frozen phase over-adapted the FC head. See runbook Step 7f.
- **freeze_conv strategy exhausted** — C18 (freeze 10) was better than C19 (freeze 20). No further freeze_conv variant is expected to break through the 0.8439 ceiling on current corpus.
- **Strategic decision (human)** — more data. The only unexploited TESS-domain labeled source is the K2 EPIC overlap corpus (K2 planets/FPs with TESS re-observations). See runbook Step 7g.
- **ECE-skip gate fix (2026-06-22)** — `evaluate_cnn_checkpoint.py` now skips temperature scaling when raw test ECE < 0.05. Root cause of C11–C19 calibration doom loop confirmed: val overconfident from early-stopping → T>1 fitted → applied to already-calibrated test → structurally worse ECE. With the fix, C20 gate is: raw AUC ≥ 0.85 AND raw F1 ≥ 0.80 (cal==raw when ECE < 0.05). Three new tests verify the skip path.
- **K2 fetcher written (2026-06-22)** — `Skills/fetch_tess_k2_overlap_snippets.py` committed.
- **K2 TAP schema discovery fix (2026-06-26, PR #131)** — first fetch attempt crashed with HTTP 400 because k2pandc uses `disposition`/`pl_orbper`/`pl_tranmid` not the guessed names. Added `_discover_k2_columns()` to query `tap_schema.columns` at startup.
- **K2 TAP query encoding fix (2026-06-26, PR #132)** — second HTTP 400: `'FALSE+POSITIVE'` in the SQL IN clause was not decoded as a space inside the SQL string literal by the TAP server. Fixed by: (1) dropping the disposition filter from SQL entirely — fetch all rows with valid period/epoch, filter by disposition locally; (2) using `urllib.parse.quote()` for proper percent-encoding of the query string.
- **C20 config committed (2026-06-22)** — `configs/cnn_tess_c20.json` (identical to C18, freeze_conv_epochs=10, checkpoint_dir=checkpoints/cnn_tess_c20).
- **Project version bumped to 0.2.0** — `pyproject.toml` updated; "citizen-science" keyword removed; status updated to "4 - Beta".

### Where things stand

| Item | State |
|---|---|
| TESS v2 snippets (`data/tess_snippets_v2.jsonl`) | **COMPLETE** — 2,619 snippets on user's Mac |
| Kepler snippets (`data/kepler_snippets.jsonl`) | **LOCAL VALIDATED** — 6,837 finite snippets on user's Mac |
| Kepler splits (`data/kepler_cnn_splits/`) | **LOCAL VALIDATED** — train 4,741 / val 1,060 / test 1,036 |
| Kepler pretraining checkpoint (`checkpoints/cnn_kepler_pretrain/best.pt`) | **LOCAL PRETRAINED** — SHA `c782d7af...`; best val AUC 0.9186 |
| Combined TESS splits (`data/tess_combined_cnn_splits/`) | **VALIDATED** — train 4,892 / val 1,049 / test 1,033 |
| Joint CNN splits (`data/joint_cnn_splits/`) | **LOCAL VALIDATED** — built 2026-06-21; retain for reproducibility; do not retrain from |
| C17 checkpoint (`checkpoints/cnn_tess_c17/`) | **REJECTED** — val AUC 0.7859; domain mismatch; do not retrain |
| C18 checkpoint (`checkpoints/cnn_tess_c18/`) | **REJECTED** — test AUC 0.8439, F1 0.7979 (raw); T=1.61 worsened ECE; SHA `d33c15f4...`; best candidate of 19 |
| C19 checkpoint (`checkpoints/cnn_tess_c19/`) | **REJECTED** — test AUC 0.8420, F1 0.7951 (raw); T=1.88 worsened ECE; SHA `65f3721f...`; regressed from C18 |
| ECE-skip gate fix | **LIVE** — `evaluate_cnn_checkpoint.py` updated 2026-06-22; 53/53 tests pass |
| `Skills/fetch_tess_k2_overlap_snippets.py` | **FIXED (2026-06-26)** — schema discovery + percent-encoded query; drop IN clause, filter locally |
| `configs/cnn_tess_c20.json` | **COMMITTED** — 2026-06-22; identical to C18, freeze_conv_epochs=10 |
| K2 overlap corpus (`data/tess_k2_overlap_snippets.jsonl`) | **NOT YET BUILT** — human must run fetch after pulling main; expected 500–1,500 new labeled TESS snippets |

### First action for the incoming agent

The K2 fetcher has been fixed twice (schema discovery in PR #131, query encoding in PR #132). Always pull main before running — an older version will hit HTTP 400. Give the human:

```bash
git pull origin main
caffeinate -dims .venv/bin/python Skills/fetch_tess_k2_overlap_snippets.py \
  --output data/tess_k2_overlap_snippets.jsonl \
  --workers 4 \
  --request-delay 0.25
wc -l data/tess_k2_overlap_snippets.jsonl
```

After the human pastes back the fetch summary and line count, audit the corpus (count, label balance, sidecar), then give the Step 7g-C merge+split+train+evaluate sequence.

### CNN production runbook

Use `docs/CNN_PRODUCTION_RUNBOOK.md` for the authoritative copy-paste workflow.
The correct CLI flags are `--split-dir`, `--checkpoint-dir`, and `--pretrained-checkpoint`.

Gate: raw held-out test AUC ≥ 0.85, raw held-out test F1 ≥ 0.80 (when raw ECE < 0.05,
temperature scaling is skipped and calibrated metrics equal raw metrics — see ECE-skip note above).

---

## Local-Only Data Artifacts — Read Before Any CNN Task

Large training data files are stored on the user's local Mac and are **never committed to the repository**. Before proposing or executing any CNN training task, ask the user to confirm current file state.

| File | Status | Description |
|---|---|---|
| `data/tess_snippets_v2.jsonl` | **COMPLETE** — 2,619 snippets | TESS phase-folded snippets; merged from two download runs; 56 targets had permanent MAST 404s |
| `data/kepler_snippets.jsonl` | **LOCAL VALIDATED** — 6,837 finite snippets as of 2026-06-17 | JSON parse PASS; zero non-finite flux rows; zero duplicate resume keys; split validator PASS |
| `data/kepler_cnn_splits/` | **LOCAL VALIDATED** — train 4,741 / val 1,060 / test 1,036 | Kepler split used for pretraining and as the Kepler component of `data/joint_cnn_splits/` |
| `data/tess_kepler_overlap_snippets.jsonl` | **COMPLETE** — 4,864 snippets as of 2026-06-20 | Kepler KOI stars folded at Kepler ephemerides; TESS-domain labels from KOI disposition; ~2,716 terminal failures in sidecar |
| `data/tess_combined_snippets.jsonl` | **BUILT** — 7,483 rows | Concatenation of TESS v2 + overlap; used for `data/tess_combined_cnn_splits/`; do not rebuild |
| `data/joint_cnn_splits/` | **LOCAL VALIDATED** — 9,633 train / 1,049 val / 1,033 test | C17 splits; retain for reproducibility; do NOT use as training input for future candidates — joint training caused domain mismatch |

The Kepler download uses `author="Kepler"` (prevents HLSP/IRIS cache corruption) and `socket.setdefaulttimeout(120)` (prevents WiFi-drop hangs). It resumes automatically from durable success keys plus the failure sidecar. The optimized path groups pending KOIs by `kepid`, fetches each KIC once, filters non-finite time/flux samples before phase binning, and supports polite bounded concurrency via `--workers 3 --request-delay 0.5`.

Any long-running local data pull must have durable resume state for both
successful outputs and terminal failures. Console progress is not a checkpoint.
If rerunning a downloader reprocesses completed or terminally failed work by
default, stop and fix resume state before asking the human to run it again.

**Do not assume these files are present on the agent's server.** They exist only
on the user's Mac. If the user is away from the Mac, agent-side work is limited
to runbook, validation, promotion-gate, and documentation hardening until the
human can run the local commands in `docs/CNN_PRODUCTION_RUNBOOK.md`.

## Git-Add-Safe Artifact Policy — Mandatory

The standard operator cadence is `git add .`. If that command would stage local
corpora, split outputs, checkpoints, runtime logs, generated reports, virtual
environments, rejected experiments, or cache files, the repository is wrong and
`.gitignore` must be fixed before continuing.

Other coding agents may only see GitHub. Therefore local-only artifacts must be
ignored, but their production-relevant state must be committed in
`docs/LOCAL_ARTIFACT_LEDGER.md` and
`artifacts/manifests/local_artifacts.json`.

When any ignored artifact affects T1-1 or another production gate, update the
ledger in the same PR as the code, runbook, or readiness change. The ledger must
answer, from GitHub alone:

1. Which local artifact paths are expected
2. Whether each artifact is missing, pending, valid, rejected, or promoted
3. The latest known counts, hashes, validation results, and approval state
4. The exact next command the human or agent should run

Do not rely on chat context, local terminal scrollback, or uncommitted files for
artifact state. A production-approved CNN checkpoint is the only CNN artifact
class that may move from ignored local state into `models/`, and only after the
evaluator passes and the human explicitly approves promotion. Because CNN model
paths are ignored defensively, approved promotion may require an intentional
`git add -f` that is documented in the promotion PR.

---

## Read First

Before writing code, recover project context from committed files. Read:

- `CLAUDE.md` — current codebase state, module map, type system, quality commands
- `docs/SCORING_MODEL.md` — mathematical specification for scoring and classification
- `docs/PIPELINE_SPEC.md` — end-to-end pipeline architecture
- `docs/PROJECT_STATUS.md` — current active state and next work
- `docs/DECISIONS.md` — durable architectural decisions
- `docs/LOCAL_ARTIFACT_LEDGER.md` — GitHub-visible state for ignored local artifacts
- `CONTRIBUTING.md` — setup, validation, and contribution policy

Do not rely on chat context, memory, or prior conversation history as the source of truth.

## Multi-Agent Continuity

Multiple agents may work on this project across separate sessions, branches, and chat threads. Repository documentation is the continuity mechanism.

When durable instructions, architectural decisions, operating rules, or scientific assumptions are established, record them in the appropriate repository document instead of leaving them only in chat. If chat context conflicts with repository documentation, prefer repository documentation unless the user explicitly instructs otherwise in the current task.

Preserve enough rationale, provenance, and test evidence in commits, docs, and code comments for another agent to continue without needing the conversation that produced the change.

## Branch And Git Policy

Default development should happen on a non-`main` branch and be merged through review. Do not push directly to `main` unless the current user explicitly requests a direct `main` commit or push.

Before committing, check `git status --short --branch`. Do not overwrite or revert unrelated user changes.

## Quality Gates

Run these before every commit when the local environment supports them:

```bash
ruff check .
python -m mypy src
PYTHONPATH=src python -m pytest
```

If a gate cannot run because of a local environment issue, record the exact blocker in the handoff or commit message. Default tests must not require live external services.

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

## Scientific Guardrails

Follow `docs/SCORING_MODEL.md` guardrails:

- Never emit "confirmed planet" for internally detected signals.
- Use "candidate signal", "possible transit-like event", or "follow-up target".
- Always expose false-positive evidence.
- Preserve provenance for scores, thresholds, inputs, and generated reports.
- Suppress formal submission pathways if key diagnostics are missing.
- Prefer conservative classifications over optimistic ones.
- External submission or contact requires explicit human approval.

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

Schedulers should call one bounded run at a time, capture stdout/stderr, and avoid overlaps. See `docs/BACKGROUND_SEARCH_AUTOMATION_BLUEPRINT.md`, `docs/BACKGROUND_SEARCH_SQLITE_SCHEMA.md`, and `docs/SCHEDULER.md`.

## Local System Profile

`docs/SYSTEM_PROFILE.md` is a committed production directive and the
authoritative local hardware profile for this project. It must remain in the
repository and must not be treated as a disposable local note or untracked
artifact.

Before performance-sensitive changes, large jobs, worker-count defaults,
batch-size defaults, cache layout changes, or long-running user recipes, read
`docs/SYSTEM_PROFILE.md` and optimize project defaults for the recorded MacBook
Pro M4 Max profile. Keep the scientific code portable and configurable: do not
hardcode local machine assumptions into candidate detection, scoring,
classification, or pathway logic. If code needs machine-specific behavior,
expose it through configuration, CLI flags, or documented runtime defaults.

AI/ML training code must prefer local acceleration by default. For PyTorch
training, use a configurable `device=auto` policy that selects Apple Metal/MPS
on the recorded M4 Max when available, then CUDA when available, and falls back
to CPU only when no accelerator is available or the operator explicitly selects
CPU. Startup banners for training runs must print the resolved device so the
operator can tell whether the GPU is actually in use.

Other performance-sensitive code should use bounded parallelism when it is
scientifically safe and operationally useful. Prefer multiprocessing or
multithreading over strictly serial loops for CPU-local batch work, starting
near the worker counts in `docs/SYSTEM_PROFILE.md`. Live external-service
workloads must remain polite and bounded; never use unbounded concurrency or a
worker default that risks throttling MAST, ExoFOP, NASA Exoplanet Archive, or
similar services. Every parallel default must remain configurable.

## macOS Long-Running Process Policy

Any Python command expected to run longer than ~60 seconds **must** be prefixed with `caffeinate -i` in recipes given to the user. This prevents macOS from sleeping and killing the process mid-run.

```bash
# Standard form for any long download or training run:
caffeinate -i python Skills/<script>.py [args]

# To keep running with lid closed, use -dims instead:
caffeinate -dims python Skills/<script>.py [args]
```

This applies to: light curve downloads, CNN training, batch scans, injection-recovery runs, and any other script that makes repeated network calls or runs for more than a minute. Never give a bare `python ...` recipe for these — always prepend `caffeinate -i`.

## Console Output and ETA — MANDATORY

**Every script that iterates over N items or trains for N epochs must print real-time progress.**
The operator cannot see internal state; silent scripts look identical to hung ones.

### Required pattern for item loops

```python
import time

start = time.monotonic()
for i, item in enumerate(items, 1):
    # ... do work ...
    elapsed = time.monotonic() - start
    rate = i / elapsed
    remaining = (n_total - i) / rate if rate > 0 else float("inf")
    eta = f"{remaining/60:.0f}m{remaining%60:.0f}s" if remaining > 90 else f"{remaining:.0f}s"
    print(f"  [{i}/{n_total}]  elapsed={elapsed:.0f}s  ETA={eta}", flush=True)
```

### Required pattern for training loops

Print one line per epoch with at minimum: epoch number, train loss, val loss, primary metric, learning rate, and whether this is a new best or how far patience has advanced:

```
Epoch  N/50  train=0.4123  val=0.5210  auc=0.8011  lr=3.00e-04  ← best
Epoch  N/50  train=0.3990  val=0.5350  auc=0.7944  lr=3.00e-04  (patience 1/10)
```

Print a startup banner before the loop showing total size, batch size, max epochs, and patience.
Print an explicit early-stopping or completion line at the end.

### Non-negotiable rules

- Always `flush=True` on every progress print — buffered output defeats the purpose.
- Print at every step, or at minimum every 10 items for very fast loops.
- **Never commit a long-running script that has no console output** — if a script is silent, add progress prints before committing.
- When reviewing or modifying any existing long-running script, verify it meets this standard and add output if missing.

## Local–Remote Sync Policy

The user's local Mac and GitHub `main` are the joint source of truth. The agent's server environment is a temporary workspace only. Keeping them in sync is a hard requirement — never leave them diverged.

### Rules for the agent

1. **All code changes must reach `main` before the user runs anything.** The full cycle is mandatory: feature branch → commit → push → PR → CI green → merge to main → PR closed. Never leave a PR open at the end of a session.
2. **Never tell the user to run a script that has not yet been merged to main.** If a script is still on a feature branch, merge it first.
3. **Every recipe given to the user must begin with `git pull origin main`** so their local is guaranteed current before any command executes.
4. **PRs must be merged, not just approved.** After CI passes, promote from draft, squash-merge, and confirm the PR is closed before the session ends.
5. **After every merge**, remind the user to run `git pull origin main` on their Mac if they have a terminal open.

### Standard recipe header (copy-paste this before every user command)

```bash
# Always sync first
git pull origin main
```

For long-running commands, the full header is:

```bash
git pull origin main
caffeinate -i python Skills/<script>.py [args]
```

### What Not To Do

- Do not tell the user to run `python Skills/foo.py` before `foo.py` is on `main`.
- Do not leave PRs in draft or open state at end of a session.
- Do not commit directly to `main` — always use a feature branch and PR.
- Do not assume the user's local is current — always prepend `git pull origin main`.

## Python Environment Policy

This project runs inside a `.venv` virtual environment. **Never touch or run system Python.**

- All `pip install` commands must be run with the venv active (`(.venv)` in the prompt)
- Never run `/Applications/Python*/Install\ Certificates.command` — this modifies system Python
- Never suggest `sudo pip install` or `pip install --system`
- Never reference `/Library/Frameworks/Python.framework/` paths — those are system Python
- If an SSL or package issue arises, fix it inside the venv: `pip install <package>` with venv active is always safe and venv-scoped
- To verify the venv is active before suggesting any pip/python commands, check that the prompt starts with `(.venv)`

## What Not To Do

- Do not add features, abstractions, or refactors beyond what the task requires.
- Do not skip validation silently.
- Do not claim a signal is a confirmed planet.
- Do not enable live network access in default tests.
- Do not hide durable rules in chat-only context.
- Do not touch system Python — all Python work happens inside the `.venv`.
