# AGENTS.md

## Purpose

Instructions for AI coding agents working on the 2026 Exoplanet Research repository.

This is scientific software. Prioritize correctness, reproducibility, conservative scientific language, and test coverage over speed.

---

## Required Reading Order

Before making meaningful changes, read:

1. `README.md`
2. `docs/PROJECT_STATUS.md`
3. `docs/PIPELINE_SPEC.md`
4. `docs/SCORING_MODEL.md`
5. `docs/ROADMAP.md`
6. `docs/DECISIONS.md`
7. `CONTRIBUTING.md`

If files conflict, prioritize:

1. `docs/DECISIONS.md`
2. `docs/SCORING_MODEL.md`
3. `docs/PIPELINE_SPEC.md`
4. `docs/PROJECT_STATUS.md`

---

## Non-Negotiable Scientific Rules

- Never call an internally detected signal a confirmed exoplanet.
- Use conservative language: `candidate signal`, `possible transit-like event`, or `follow-up target`.
- Treat false positives as the default hypothesis.
- Preserve uncertainty in outputs and reports.
- Always expose negative evidence and blocking issues.
- Always preserve data provenance.
- Do not claim discovery without external validation.

---

## Non-Negotiable Engineering Rule

Do not mark a task complete if tests are failing or missing.

If implementation cannot be fully tested yet:

- add a pending test stub where appropriate
- document the blocker in `docs/PROJECT_STATUS.md`
- explain what evidence is required to complete testing

---

## Testing Requirements

All code changes must include appropriate tests.

### Unit Tests

Every new function, class, or module must include unit tests covering:

- expected behavior
- edge cases
- invalid inputs
- numerical stability where applicable
- deterministic output where applicable

Unit tests must be fast, isolated, and free of live network access.

### Integration Tests

Pipeline-facing code must include integration tests showing that components work together.

Examples:

- synthetic light curve → clean → search
- candidate metrics → scoring model → pathway classifier
- known-object match → `known_object_annotation`
- secondary eclipse signal → elevated eclipsing-binary probability

Use synthetic data or small fixtures.

### Regression Tests

Every bug fix must include a regression test that fails before the fix and passes after the fix.

### Scientific Sanity Tests

Astronomy-specific modules must include scientific sanity checks.

Examples:

- high-SNR clean transit increases planet-candidate probability
- secondary eclipse increases eclipsing-binary probability
- odd/even mismatch penalizes planet-candidate probability
- catalog match suppresses new-candidate routing
- contamination risk increases false-positive probability
- single-transit events do not route to formal submission by default

---

## Minimum Local Validation

Before reporting work as complete, run:

```bash
pytest
```

When available, also run:

```bash
pytest --cov=exo_toolkit --cov-report=term-missing
ruff check .
mypy src
```

If a command fails, fix the issue or document the failure and reason.

---

## Coverage Expectations

Target coverage:

- scoring and pathway logic: 90%+
- cleaning, search, and vetting modules: 80%+
- CLI and reporting: smoke tests required
- live external integrations: mocked by default

Coverage is not a substitute for meaningful tests.

---

## External Service Policy

Default tests must not require live access to:

- MAST
- NASA Exoplanet Archive
- ExoFOP
- Gaia services
- remote catalogs

Live-data tests must be explicitly marked:

```python
@pytest.mark.integration_live
```

Slow tests must be explicitly marked:

```python
@pytest.mark.slow
```

---

## Data Policy

Do not commit:

- large raw FITS files
- downloaded mission data
- private credentials
- API tokens
- generated cache directories
- large intermediate products

Small synthetic fixtures are allowed in `tests/fixtures/`.

---

## Documentation Update Rules

Update documentation when changing:

- pipeline architecture
- scoring logic
- thresholds
- submission pathway logic
- testing requirements
- data assumptions
- external dependencies

Use:

- `docs/DECISIONS.md` for architecture and rationale
- `docs/PROJECT_STATUS.md` for active state and blockers
- `docs/SCORING_MODEL.md` for scoring changes
- `docs/PIPELINE_SPEC.md` for pipeline changes
- `docs/ROADMAP.md` for milestone changes

---

## Definition of Done

A change is complete only when:

- code is implemented
- unit tests are added or updated
- integration tests are added or updated where relevant
- all standard tests pass
- public interfaces are typed
- scientific assumptions are documented
- provenance behavior is preserved
- relevant docs are updated
- no confirmation claims are made for candidate signals

---

## Preferred Development Style

- Keep functions small and testable.
- Prefer pure functions for scoring and decision logic.
- Separate data acquisition from analysis.
- Separate detection from inference.
- Separate scoring from pathway classification.
- Use typed data models for candidate signals and outputs.
- Use configuration files for thresholds rather than hardcoding.
- Make intermediate artifacts inspectable.
