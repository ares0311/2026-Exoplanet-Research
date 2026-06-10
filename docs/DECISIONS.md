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

---

## DECISION-011: Keep Mission-Specific Priors Conservative And Opt-In

**Date:** 2026-05-21
**Status:** Accepted

### Context

The original Bayesian/log-score model used one conservative default prior
distribution for every mission. TESS, Kepler, and K2 have different observing
baselines, pixel scales, catalog maturity, and systematic-noise regimes, so the
scoring engine needs a durable path to mission-specific priors without making
default scoring less conservative or less reproducible.

### Decision

Add `configs/scoring_priors_v0.json` and `src/exo_toolkit/priors.py` as an
opt-in prior configuration layer. Default scoring still uses the built-in
priors unless a caller explicitly passes a validated `ScoringPriorConfig` into
`score_candidate(..., prior_config=config)`.

Configured profiles must provide all six hypothesis probabilities, sum to 1.0,
use positive values, keep the combined false-positive prior larger than the
planet-candidate prior, disable confirmation claims, and require human approval
for external submission.

### Rationale

- Preserves default behavior for existing users and tests.
- Allows TESS, Kepler, and K2 priors to be audited in versioned repository
  config instead of hidden in chat or ad hoc scripts.
- Keeps false-positive hypotheses prominent.
- Leaves room for future period-, radius-, stellar-type-, and completeness-
  dependent priors after calibration evidence exists.

---

## DECISION-012: Invoke Mypy Through Python Module Execution

**Date:** 2026-05-27
**Status:** Accepted

### Context

Local validation now runs inside the active Python environment, commonly through `.venv/bin/python`.
Bare `mypy` can resolve to a different executable or dependency context than the interpreter used for tests and project tooling.

### Decision

Use `python -m mypy src` for the standard type-check command, and `.venv/bin/python -m mypy src` when invoking the project virtual environment directly.
The older bare `mypy src` form is superseded for current local and CI validation guidance.

### Rationale

- Keeps type checking tied to the same interpreter and installed packages used by the rest of validation.
- Reduces cross-agent ambiguity when multiple shells, virtual environments, or package managers are present.
- Aligns contributor docs, CI, and handoff validation evidence.

---

## DECISION-013: Expert Vetting and Peer Review Are Out of Scope (Citizen Science)

**Date:** 2026-06-06
**Status:** Accepted

### Context

`docs/PRODUCTION_READINESS.md` originally listed T2-2 (Expert Vetting) and T2-3 (Peer Review Before
Publishing) as Tier 2 improvement gaps. Both required access to an independent exoplanet transit
astronomer or academic peer reviewer. The project owner confirmed that this is a citizen science
project operating independently — no expert reviewer is available.

### Decision

T2-2 and T2-3 are permanently out of scope. They are marked N/A in `docs/PRODUCTION_READINESS.md`
and will never appear in gap lists or task plans.

The code-enforced scientific guardrails already in place are the citizen-science substitute:

1. Never output "confirmed planet" — only "candidate signal" or "follow-up target"
2. Always expose false-positive evidence alongside positive evidence
3. Suppress `tfop_ready` pathway when key diagnostics are missing
4. No external submission or discovery contact without explicit human approval
5. Background automation draft reports require human approval before any external action
6. Conservative priors by default; mission-specific profiles are opt-in
7. `provenance_score` gates `tfop_ready` — 2-min SPOC with ≥2 sectors required

### Rationale

- The project is citizen science; external expert access is not feasible.
- Conservative guardrails enforced in code provide equivalent protection against false claims.
- Removing unfeasible gaps from the gap list keeps planning focused on actionable work.
- This decision must not be reversed by a future agent without explicit instruction from the user.

---

## DECISION-014: Keep Training Corpora Local and Version Production Model Artifacts

**Date:** 2026-06-09
**Status:** Accepted

### Context

T1-1 requires training a production Tier 2 CNN from the generated
`data/tess_snippets.jsonl` corpus. The repository already directs agents to
cache large mission data locally and avoid committing generated data or cache
directories. The completed local corpus contains all 2,636 eligible rows:
2,623 usable snippets and 13 recorded fetch or extraction errors.

Production scoring must also remain reproducible across machines and agents.
That requires the exact validated model and its calibration and provenance
metadata to be available from the repository rather than only from one local
workspace.

### Decision

1. Treat a generated TESS snippet corpus as authorized local training input for
   T1-1 only after ephemeris integrity, class-balance, and leakage checks pass.
2. Keep generated training corpora and downloaded mission data local and
   uncommitted. Ignore `data/*.jsonl` alongside the other generated `data/`
   formats.
3. After production-readiness validation, commit the selected production CNN
   checkpoint, calibration metadata, model registry entry, and reproducibility
   manifest under `models/`.
4. The reproducibility manifest must identify the source-catalog version or
   hash, corpus hash, split manifest hash, training configuration, code commit,
   metrics, and calibration artifact.
5. If a validated checkpoint is too large for ordinary Git hosting, use Git
   LFS or another explicitly approved versioned artifact store without
   committing the training corpus itself.

### Rationale

- Keeps large, regenerable mission-derived data out of Git.
- Allows T1-1 training to proceed from the completed local corpus.
- Makes the deployed scorer reproducible and available to every agent and
  production environment.
- Separates private or bulky training inputs from the small set of artifacts
  required to reproduce production inference.

---

## DECISION-015: Fail Closed on Missing Transit Epochs

**Date:** 2026-06-10
**Status:** Accepted

### Context

The first local TESS corpus was generated from a stale TOI CSV that lacked the
`epoch_bjd` column. The downloader silently substituted zero, causing all 2,623
nominally usable snippets to be phase-folded without centering the catalog
transit. The resulting checkpoint failed held-out discrimination and
calibration gates.

### Decision

1. TOI training inputs must contain finite `epoch_bjd` values on the BJD scale
   (`>= 2,000,000`) and positive finite periods.
2. Fetch and download stages fail closed when required ephemeris columns are
   absent and exclude rows with invalid ephemerides.
3. Every generated corpus must pass the offline corpus audit before splitting
   or training.
4. The zero-epoch corpus, all splits derived from it, and all resulting
   checkpoints are permanently retired.

### Rationale

- A phase-folded CNN cannot learn centered transit morphology from an
  arbitrary phase origin.
- Silent numeric defaults are unsafe for scientific ephemerides.
- Corpus validation must precede model tuning so held-out evidence remains
  meaningful.

---

## DECISION-016: Validated Runtime Is Python 3.14.3 in `.venv`

**Date:** 2026-06-10
**Status:** Accepted

### Context

The project was previously developed on Python 3.11 and later validated on
Python 3.13.12. The project owner and a coding agent jointly tested and
confirmed full compatibility with Python 3.14.3. The Python Environment Policy
(AGENTS.md) already prohibits touching system Python; this decision records
the validated runtime version and makes the venv-only constraint durable.

### Decision

1. The validated and required runtime is **Python 3.14.3** running inside the
   project `.venv` virtual environment.
2. No code, test, or agent workflow may use system Python for any project
   operation.
3. All `pip install` commands must run with `(.venv)` active in the prompt.
4. Recipes given to the user must assume `.venv` is active and must never
   reference `/Library/Frameworks/Python.framework/` or suggest
   `sudo pip install`.
5. If a future Python version is tested and validated by the project owner,
   record it here as a superseding decision entry — do not silently rewrite
   this entry.

### Rationale

- Eliminates ambiguity about which Python interpreter agents and contributors
  should use.
- System Python on macOS is managed by the OS and must not be modified.
- Recording the validated version allows future agents to diagnose
  compatibility regressions against a known-good baseline.
- Keeps scientific code, dependencies, and environment fully isolated in the
  project venv.
