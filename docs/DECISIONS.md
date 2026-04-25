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
