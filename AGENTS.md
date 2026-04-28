# AGENTS.md — Instructions for AI Coding Agents

This file contains binding rules for AI coding agents (Claude Code, Codex, etc.) working in this repository.

---

## Read First

Before writing any code, read:
- `CLAUDE.md` — current codebase state, module map, type system, quality commands
- `docs/SCORING_MODEL.md` — mathematical specification for all scoring and classification logic
- `docs/PIPELINE_SPEC.md` — end-to-end pipeline architecture
- `docs/PROJECT_STATUS.md` — what is done and what is next
- `docs/DECISIONS.md` — durable architectural decisions; do not contradict these without adding a new DECISION-NNN entry

---

## Branch

Always develop on: `claude/review-markdown-docs-SwVnR`
Always push to: `claude/review-markdown-docs-SwVnR`
Never push to `main` directly.

---

## Quality Gates (mandatory before every commit)

```bash
ruff check .
mypy src
PYTHONPATH=src python -m pytest
```

All three must pass. Fix any errors before committing.

---

## Code Standards

- **Python 3.11+** — use `from __future__ import annotations`, `X | Y` unions, `match`/`case` where useful
- **Pydantic v2** — frozen models, `ConfigDict(frozen=True)`, `Annotated` types, `model_validator`
- **Type annotations** — all public functions must be fully annotated; run `mypy src` to verify
- **`OptScore = float | None`** — absent features contribute 0 to log scores (neutral)
- **No comments explaining WHAT** — good names do that; only add comments for WHY (hidden constraints, workarounds, invariants)
- **No ML classifiers** — until labeled validation data and calibration infrastructure exist (DECISION-002)
- **Never output "confirmed planet"** — use "candidate signal" or "follow-up target" (SCORING_MODEL.md §15)

---

## Testing Standards

Every module must have a corresponding `tests/test_<module>.py`.

Required coverage:
- Unit tests for every public function
- Fixture-based tests for complex interactions
- Boundary condition tests for numerical thresholds
- `None`-input tests for all `OptScore` parameters
- Conservation / sanity tests (e.g. posteriors sum to 1.0, scores in [0, 1])
- `@pytest.mark.integration_live` for any test requiring live network access

Do not use `@pytest.mark.integration_live` tests in the default suite.

---

## Scientific Guardrails (SCORING_MODEL.md §15)

- Never emit "confirmed planet" for internally detected signals
- Always expose false-positive evidence in explanations
- Suppress formal submission pathways if key diagnostics are missing
- Store the scoring model version + threshold hash with every scored candidate
- Conservative classifications preferred over optimistic ones
- `None` feature scores must fail threshold gates (not pass them silently)

---

## Commit Messages

Format:
```
<Short imperative summary (≤72 chars)>

<Body: what changed and why, not how>
<Reference to spec section if relevant>

https://claude.ai/code/session_014aY2VWw83fBCBSWqqTGxZW
```

---

## What Not To Do

- Do not add features, abstractions, or refactors beyond what the task requires
- Do not add error handling for scenarios that cannot happen in internal code
- Do not add comments that describe what code does (names do that)
- Do not skip ruff, mypy, or pytest before committing
- Do not push to `main`
- Do not claim a signal is a confirmed planet
