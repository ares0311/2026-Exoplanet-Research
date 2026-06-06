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

## Read First

Before writing code, recover project context from committed files. Read:

- `CLAUDE.md` — current codebase state, module map, type system, quality commands
- `docs/SCORING_MODEL.md` — mathematical specification for scoring and classification
- `docs/PIPELINE_SPEC.md` — end-to-end pipeline architecture
- `docs/PROJECT_STATUS.md` — current active state and next work
- `docs/DECISIONS.md` — durable architectural decisions
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

- Python 3.11+.
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

Before performance-sensitive changes or large jobs, read `docs/SYSTEM_PROFILE.md`.

Optimize local defaults for the recorded MacBook Pro M4 Max profile while keeping scientific code portable and configurable. Do not hardcode local machine assumptions into candidate detection, scoring, or pathway logic.

## macOS Long-Running Process Policy

Any Python command expected to run longer than ~60 seconds **must** be prefixed with `caffeinate -i` in recipes given to the user. This prevents macOS from sleeping and killing the process mid-run.

```bash
# Standard form for any long download or training run:
caffeinate -i python Skills/<script>.py [args]

# To keep running with lid closed, use -dims instead:
caffeinate -dims python Skills/<script>.py [args]
```

This applies to: light curve downloads, CNN training, batch scans, injection-recovery runs, and any other script that makes repeated network calls or runs for more than a minute. Never give a bare `python ...` recipe for these — always prepend `caffeinate -i`.

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
