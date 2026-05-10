# DECISIONS

## Purpose

This file records durable architectural, scientific, and engineering decisions for the 2026 Exoplanet Research project.

This file should be append-only in spirit. If a decision changes, add a new decision entry that supersedes the earlier one rather than silently rewriting history.

---

## DECISION-001: Use a Bayesian Candidate-Scoring Framework

**Date:** 2026-04-25  
**Status:** Accepted

### Context

The project needs to evaluate transit-like signals while explicitly accounting for false positives. A simple score is insufficient because the same signal could plausibly be a planet candidate, eclipsing binary, background eclipsing binary, stellar variability, instrumental artifact, or known object.

### Options Considered

1. Simple heuristic score
2. Machine-learning classifier
3. Bayesian-style multi-hypothesis scoring model

### Decision

Use a Bayesian-style multi-hypothesis scoring framework.

### Rationale

- Produces interpretable probabilities.
- Makes uncertainty explicit.
- Supports scientific caution.
- Allows false-positive hypotheses to be represented directly.
- Can be implemented heuristically first and calibrated later.

---

## DECISION-002: Start with Interpretable Models Before Machine Learning

**Date:** 2026-04-25  
**Status:** Accepted

### Decision

Implement transparent log-score/Bayesian-style models first. Add machine learning later only after baseline scoring, validation, and calibration infrastructure exist.

### Rationale

- Easier to debug.
- Easier to test.
- Easier to explain in candidate reports.
- Reduces risk of false confidence.

---

## DECISION-003: Use Apache License 2.0 for Code

**Date:** 2026-04-25  
**Status:** Accepted

### Decision

Use Apache License 2.0 for code.

### Rationale

- Permissive.
- Research and industry friendly.
- Includes explicit patent protections.
- Compatible with broad adoption.

### Notes

Documentation and written reports may use CC-BY-4.0 where appropriate. Raw NASA, TESS, Kepler, MAST, or NASA Exoplanet Archive data is not relicensed by this repository.

---

## DECISION-004: Split Documentation by Function

**Date:** 2026-04-25  
**Status:** Accepted

### Decision

Use a structured documentation system:

- `README.md` — public project face
- `docs/PROJECT_STATUS.md` — current active state
- `docs/ROADMAP.md` — milestones and future work
- `docs/PIPELINE_SPEC.md` — system architecture
- `docs/SCORING_MODEL.md` — mathematical and scoring specification
- `docs/DECISIONS.md` — durable rationale and architectural decisions
- `AGENTS.md` — instructions for AI coding agents
- `CONTRIBUTING.md` — contribution and testing policy

### Rationale

- Prevents information decay.
- Reduces repeated design work.
- Improves agent continuity.
- Separates “what we decided” from “what we are doing.”

---

## DECISION-005: Treat Testing as a Non-Negotiable Development Requirement

**Date:** 2026-04-25  
**Status:** Accepted

### Context

The project involves scientific inference, numerical algorithms, and external astronomical data. Bugs could produce misleading candidate rankings or false-positive claims. Future coding agents must not treat tests as optional.

### Decision

Every meaningful code change must include appropriate testing.

Required testing layers:

- unit tests for functions, classes, and modules
- integration tests for pipeline interactions
- regression tests for bug fixes
- scientific sanity tests for astronomy-specific logic
- mocked tests for external services
- separately marked live integration tests when necessary

### Rationale

- Prevents silent scientific errors.
- Protects scoring and classification logic.
- Makes agent work auditable.
- Supports reproducibility.
- Reduces risk of false candidate claims.

---

## DECISION-006: Use `pytest`, `ruff`, and `mypy` as Core Quality Tools

**Date:** 2026-04-25  
**Status:** Accepted

### Decision

Use:

- `pytest` for tests
- `pytest-cov` for coverage
- `ruff` for linting
- `mypy` for static type checking
- optional `pre-commit` hooks later

### Standard Local Validation

```bash
pytest --cov=exo_toolkit --cov-report=term-missing
ruff check .
mypy src
```

---

## DECISION-007: Mock External Services in Default Tests

**Date:** 2026-04-25  
**Status:** Accepted

### Decision

Default tests must not require live external services.

Live external tests must be explicitly marked:

```python
@pytest.mark.integration_live
```

### Rationale

- Keeps test suite deterministic.
- Avoids network-dependent failures.
- Reduces CI fragility.
- Makes local development faster.

---

## DECISION-008: Use Config Files for Scientific Thresholds

**Date:** 2026-04-25  
**Status:** Proposed

### Proposed Decision

Store thresholds in versioned config files, such as:

- `configs/scoring_v0.yaml`
- `configs/bls_search_v0.yaml`
- `configs/pathway_v0.yaml`

### Rationale

- Improves reproducibility.
- Allows explicit model-version tracking.
- Makes calibration updates auditable.

---

## DECISION-009: Record Local System Profile for Runtime Sizing

**Date:** 2026-05-01
**Status:** Accepted

### Context

The project will run computationally meaningful light-curve cleaning, BLS searches, vetting, scoring, reporting, and eventually injection-recovery experiments. Local runtime defaults should reflect the development machine's capacity without making the scientific code depend on one workstation.

### Decision

Record the local development machine profile in `docs/SYSTEM_PROFILE.md` and use it to guide default worker counts, memory targets, threading limits, cache behavior, and benchmarking expectations.

### Rationale

- Makes local performance assumptions explicit.
- Reduces accidental CPU or memory oversubscription.
- Helps future agents choose sensible defaults for batch jobs.
- Keeps machine-specific optimization separate from scientific logic.
- Preserves portability by requiring system-specific behavior to remain configurable.

---

## DECISION-010: Implement Background Search Automation with Top-Level SQLite Logs

**Date:** 2026-05-09  
**Status:** Accepted

### Context

The project needs a conservative background search process that can run repeatedly, select promising targets, preserve provenance, record negative evidence, and stop before external submission unless a human explicitly approves. Multiple agents may work on this implementation, so the storage, fixture, and scheduler defaults must be documented in the repository rather than left in chat context.

### Decision

Implement the first background search automation against known TESS example fixtures.

Use a top-level `logs/` directory for runtime logs. Store the durable run ledger, reviewed outcomes, needs-follow-up outcomes, target priority evaluations, follow-up test records, and submission recommendations in SQLite, with an initial database path such as `logs/background_search.sqlite3`.

Expose one bounded command for a single run through the existing project CLI namespace, such as:

```bash
exo background-run-once
```

Scheduler documentation should remain broadly compatible across cron, systemd timers, launchd, and controlled workflow runners. Because the primary local development environment is macOS, include a concrete macOS `launchd` example.

Background automation configuration should live in a top-level `configs/` directory. Scheduled or manual runs should briefly wait for an active run lock before failing, rather than immediately failing on overlap. Report export should support both Markdown and HTML. Fixture coverage should include known TESS examples plus clearly labeled edge cases for weak signals, contamination, incomplete provenance, calibration uncertainty, reviewed outcomes, and guardrail behavior.

### Rationale

- SQLite gives durable, queryable, transactional logs without requiring a separate service.
- A top-level `logs/` directory makes automation state easy to find across agents and sessions.
- Known TESS examples allow deterministic, scientifically inspectable development before live data access.
- A single-run command keeps scheduled work auditable, restartable, and reproducible.
- Broad scheduler guidance avoids hardcoding the workflow to one operating system while still supporting macOS well.
- Top-level configs make scientific thresholds easy for multiple agents to inspect and version.
- A short lock wait is friendlier to schedulers while still preventing overlapping runs.
- Markdown and HTML exports support both review in Git and richer local inspection.
- Edge-case fixtures make conservative guardrails testable before live data is introduced.
