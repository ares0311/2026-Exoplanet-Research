# CONTRIBUTING.md

## Purpose

This repository is scientific software for exoplanet candidate detection and evaluation. Contributions must prioritize correctness, reproducibility, testing, and conservative scientific interpretation.

---

## Development Principles

- Build small, testable components.
- Prefer explicit, interpretable logic over black-box behavior.
- Preserve provenance for every generated result.
- Avoid hardcoded scientific thresholds where configuration is appropriate.
- Use synthetic fixtures for tests whenever possible.
- Do not commit large raw mission data.
- Do not label internally detected signals as confirmed planets.

---

## Local Setup

Recommended Python version:

```bash
python --version
# Python 3.11+
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install editable package with development dependencies:

```bash
pip install -e ".[dev]"
```

---

## Quality and Testing Policy

This project follows a test-first or test-alongside development model.

Every meaningful code change must include tests. Untested code is incomplete.

### Required Test Types

| Change Type | Required Tests |
|---|---|
| Pure function | Unit tests |
| Scoring logic | Unit tests + scientific sanity tests |
| Pathway classifier | Unit tests + decision-tree tests |
| Pipeline stage | Unit tests + integration tests |
| Bug fix | Regression test |
| CLI command | Smoke test |
| External API logic | Mocked integration test |
| Report generation | Snapshot or smoke test |

---

## Standard Test Commands

Run default tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=exo_toolkit --cov-report=term-missing
```

Run linting:

```bash
ruff check .
```

Run type checks:

```bash
mypy src
```

Recommended full local check:

```bash
pytest --cov=exo_toolkit --cov-report=term-missing
ruff check .
mypy src
```

---

## Testing Principles

Tests should be:

- deterministic
- small and fast
- isolated from live network services
- based on synthetic examples or small fixtures
- specific enough to catch real bugs
- clear about scientific expectations

Any stochastic test must use a fixed random seed.

---

## Scientific Testing Examples

The test suite should verify that:

- clean repeated transit signals score better than noisy weak signals
- secondary eclipses increase eclipsing-binary likelihood
- odd/even mismatches reduce planet-candidate probability
- high contamination increases background-eclipsing-binary likelihood
- known-object matches route to `known_object_annotation`
- low-confidence signals route to `github_only_reproducibility`
- single-transit events do not route to formal submission by default
- physically implausible durations reduce planet-candidate probability

---

## External Service Tests

Tests must not require live network access by default.

Live integration tests should be marked:

```python
@pytest.mark.integration_live
```

Slow tests should be marked:

```python
@pytest.mark.slow
```

Standard test runs should exclude live external tests unless explicitly requested.

---

## Code Style

Use:

- `ruff` for linting
- `mypy` for type checking
- Python type hints for public interfaces
- clear docstrings for scientific assumptions
- small, composable functions

Avoid:

- hidden global state
- hardcoded thresholds without config support
- silent failure modes
- untracked randomness
- network calls inside unit tests

---

## Documentation Requirements

Update documentation when changing:

- scoring logic
- pathway classification
- pipeline architecture
- thresholds
- data assumptions
- testing policy
- dependencies

Relevant files:

- `docs/SCORING_MODEL.md`
- `docs/PIPELINE_SPEC.md`
- `docs/ROADMAP.md`
- `docs/PROJECT_STATUS.md`
- `docs/DECISIONS.md`
- `AGENTS.md`

---

## Commit Guidance

Use clear, focused commits.

Good examples:

```bash
git commit -m "Add scoring model tests"
git commit -m "Implement pathway classifier"
git commit -m "Document testing policy for agents"
```

Avoid vague commits:

```bash
git commit -m "updates"
git commit -m "stuff"
```

---

## Review Checklist

Before a change is ready:

- [ ] Code is typed where appropriate
- [ ] Unit tests are included
- [ ] Integration tests are included where relevant
- [ ] Regression tests are included for bug fixes
- [ ] `pytest` passes
- [ ] `ruff check .` passes
- [ ] `mypy src` passes or documented exceptions exist
- [ ] Documentation is updated
- [ ] Scientific claims remain conservative
- [ ] Provenance behavior is preserved

---

## Scientific Language Policy

Use:

- candidate signal
- possible transit-like event
- follow-up target
- planet-candidate hypothesis

Avoid unless sourced from authoritative external catalogs:

- confirmed planet
- discovered planet
- new exoplanet
