# EXO-Hunter — 2026 Exoplanet Research

| Field | Value |
|---|---|
| Research domain | Transiting-exoplanet detection and candidate vetting (TESS, Kepler, K2) |
| Primary task | Given a request for `N` targets, select, freeze, execute, and persist the best available `N` New or Follow-up searches |
| Validated status | **Nonconforming** against contract `HUNTER-PROD-2026-07-30.3`; see §1.4 |
| Canonical CLI entry point | `EXO-Hunter` (`exo_toolkit.hunter_shell:exohunter_entry`) |
| Durable schema version | Hunter SQLite schema v6 (`HUNTER_SCHEMA_VERSION`, `src/exo_toolkit/search_lifecycle.py`) |
| Sibling repositories | `2026 Near Earth Objects` (NEOHunter), `2026 Technosignatures` (TechnoHunter) |
| Canonical documentation | `docs/HUNTER_PRODUCTION_WORKFLOW.md`, `docs/HUNTER_PROD_CONTRACT.md`, `docs/CLI_UX_SPEC.md` |
| Package | `exo-toolkit` 0.5.3, Python >= 3.11 |

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. CLI Tool Usage](#2-cli-tool-usage)
- [3. Analytics, Mathematics, and Theoretical Foundation](#3-analytics-mathematics-and-theoretical-foundation)
- [4. Sibling Repositories and Shared Data](#4-sibling-repositories-and-shared-data)

## 1. Executive Summary

### 1.1 Research Objective and Scientific Context

This repository detects and ranks transiting-exoplanet candidate signals in
public photometry from TESS, Kepler, and K2, and manages the durable lifecycle
of the searches that produce them.

The scientific question is a decision problem, not a classification problem
alone: given finite observing and compute budget, which targets should be
searched next, and what does the resulting evidence support? The pipeline
therefore pairs a transit search with an explicit six-hypothesis Bayesian model
that scores competing astrophysical and instrumental explanations for every
detected signal, instead of emitting an undifferentiated "planet score".

Target selection favors lightly-worked parts of the archive: later TESS
sectors, fainter stars (Tmag 10-14), and less-crowded fields.

### 1.2 Scope, Boundaries, and Exclusions

In scope: archival photometry acquisition, detrending, box-least-squares
transit search, false-positive vetting diagnostics, Bayesian hypothesis
scoring, submission-pathway classification, and the durable EXO-Hunter search
lifecycle.

Out of scope, and verified as such:

- Radial-velocity confirmation, astrometric confirmation, and dynamical mass
  determination — **Not applicable**.
- Atmospheric retrieval from JWST spectra — **Not applicable**. JWST light
  curves are acquired (`Skills/fetch_jwst_lc.py`) but no retrieval model ships.
- Near-Earth-object and technosignature analysis — **Not applicable**. These are
  the responsibilities of the two sibling repositories named in §4.1.
- Autonomous external submission or public discovery claims — **Not
  applicable** by policy. No command submits to an external authority; that
  boundary is human-gated.

This repository never emits the phrase "confirmed planet". Detected signals are
reported as candidate signals or follow-up targets.

Target selection is owned entirely by the canonical selector. `Create-New-Search`
does not accept operator-ranked candidate files: the retired `--candidate-file`
bypass is gone, so no externally-ranked list can substitute for catalog-wide
adaptive discovery, novelty exclusion, and deterministic ranking.

### 1.3 System and Workflow Overview

The scientific pipeline is a fixed sequence:

```
Fetch -> Clean -> Search -> Vet -> Score -> Classify
```

The Hunter lifecycle wraps that pipeline with durable state:

```
request -> adaptive discovery -> identity and history -> eligibility -> ranking
-> sufficiency and expansion -> exact frozen manifest -> durable search creation
-> acquisition -> scoring -> durable results -> history and follow-up update
```

Creating a search freezes the exact selected targets. Execution runs those exact
targets and never silently regenerates, substitutes, or reorders them.

### 1.4 Verified Capability Status

Status vocabulary is restricted to Implemented, Experimental, Deprecated,
Nonconforming, and Not applicable.

| Capability | Status | Evidence |
|---|---|---|
| Six-hypothesis Bayesian scoring | Implemented | `src/exo_toolkit/hypotheses.py`, `scoring.py`; `tests/test_hypotheses.py`, `tests/test_scoring.py` |
| BLS transit search and vetting diagnostics | Implemented | `src/exo_toolkit/search.py`, `vet.py`; `tests/test_search.py`, `tests/test_vet.py` |
| Submission-pathway classification | Implemented | `src/exo_toolkit/pathway.py`; `tests/test_pathway.py` |
| Durable search lifecycle (schema v6) | Implemented | `src/exo_toolkit/search_lifecycle.py`; `tests/test_search_lifecycle.py` |
| Exact-target freezing and resume | Implemented | `HunterStore.create_search`, `execute_search`; `tests/test_search_lifecycle.py` |
| Persistent slash terminal | Implemented | `src/exo_toolkit/hunter_shell.py`; `tests/test_hunter_shell.py` |
| Slash-command palette with described parameters | Implemented | `src/exo_toolkit/hunter_ux.py`; `tests/test_hunter_ux.py::TestFilterCommands` |
| Live validity sentinels and shared validators | Implemented | `hunter_ux.validate_target_count`; `tests/test_hunter_shell.py::TestSharedValidatorParity` |
| `/Inspect-Target` detail view | Implemented | `hunter_cli.inspect_target`; `tests/test_hunter_shell.py::TestInspectTargetCommand` |
| Width-aware results table | Implemented | `hunter_ux.render_results_table`; `tests/test_hunter_ux.py::TestTruncateAndColumns` |
| Follow-up ranking formula integrity | Implemented | `tests/test_search_lifecycle.py::TestRegistryExpectedInformationGainMatchesContract` |
| Repository-native `prod-check` gate | Implemented | `src/exo_toolkit/prod_check.py`; `tests/test_prod_check.py` |
| Golden UX assertions | Implemented | `tests/golden/`; `tests/test_hunter_ux.py::TestGoldenUx` |
| XGBoost, CNN, and stacking scorers | Experimental | `src/exo_toolkit/ml/`; outside the Bayesian default path |
| Representation-model baselines (Chronos-Bolt, Astromer2) | Deprecated | `docs/GROUPED_EXTERNAL_REPRESENTATION_BENCHMARK.md` records `no_external_added_value`; retained for reproducibility |
| Installation and launch across all seven execution surfaces | Nonconforming | Contract LAUNCH-02 requires a built wheel, a fresh install, an upgrade-in-place, and execution from an unrelated directory; not verified |
| Live-data New and Follow-up acceptance evidence | Nonconforming | Contract E2E-01, E2E-02, and E2E-04 require a retained live-MAST bundle; not verified |
| Adaptive-discovery sufficiency evidence | Nonconforming | Contract DISC-01, DISC-02, and DISC-03 require expansion-round and churn evidence; not verified |
| Cross-project identity and history completeness | Nonconforming | Contract IDENT-01 through IDENT-04; not independently verified |

### 1.5 Evidence and Reproducibility

The durable acceptance ledger is `configs/HUNTER_PROD_STATE.json`. It records,
per requirement, the exact command, environment, observable assertion, raw
evidence path, tested commit, and timestamp.

The machine-enforced gate is `prod-check` (§2.6). It exits nonzero while any
mandatory requirement is unsatisfied, and reports any check it could not run as
`NOT EXECUTED` with a reason. A `NOT EXECUTED` stage is never counted toward a
pass total.

Current gate result on this tree: **13 of 19 checks passed, 0 failed, 6 NOT
EXECUTED**, therefore `PROD gate: BLOCKED`. The six unexecuted checks are the
live-environment and live-data requirements listed as Nonconforming in §1.4;
they are reported honestly rather than assumed, and are excluded from the
passed total.

## 2. CLI Tool Usage

### 2.1 Prerequisites

- Python >= 3.11. The pinned environment uses 3.14.
- `uv` for environment synchronization.
- Network access to MAST and the NASA Exoplanet Archive for acquisition
  commands. Analysis of already-cached data needs no network.

### 2.2 Installation

```bash
uv sync --all-extras --all-groups
```

This installs `exo-toolkit` and every console script listed in §2.6.

### 2.3 Environment Setup

The durable database defaults to `data/hunter_searches.sqlite3`. Override it per
command with `--db`.

Lightkurve writes downloaded products to its own cache. To keep that cache
inside the repository, pre-create `.cache/lightkurve/` and export
`XDG_CACHE_HOME` to the repository-local cache root before running acquisition
commands. Lightkurve honors that variable when the directory already exists.

Accessibility and automation are controlled by `NO_COLOR`, `--no-color`,
`--no-animation`, and the `CI`, `REDUCE_MOTION`, and `EXOHUNTER_REDUCE_MOTION`
environment variables. Animation disables automatically on a non-TTY stream.

### 2.4 Command Structure

`EXO-Hunter` is a persistent terminal. It stays active until `/Exit`.

```
EXO-Hunter [--db PATH] [--command CMD] [--script FILE]
           [--history-file PATH] [--no-color] [--no-animation] [--version]
```

Typing `/` opens a searchable command palette listing every command with its
description and its required and optional parameters. Typing `/` followed by
text filters that palette. Discovery never requires `/Help`.

Interactive and scriptable operation share one validation layer, so a value
rejected interactively is rejected identically by the one-shot commands.

### 2.5 End-to-End Workflow

Invoke the terminal through the synchronized environment. Use
`.venv/bin/EXO-Hunter` when the environment is not active, so the command
resolves to this repository's interpreter rather than whatever `EXO-Hunter`
happens to be first on `PATH`:

```bash
.venv/bin/EXO-Hunter --version
```

```bash
# 1. Open the terminal
EXO-Hunter

# 2. Freeze the best available 5 never-before-searched targets
/New-Search 5

# 3. Review the frozen manifest, then inspect any row in full
/Inspect-Target 1

# 4. Execute the exact frozen manifest
/Run-Search

# 5. Review the follow-up recommendations the run registered
/Show-Follow-Ups

# 6. Freeze and run a follow-up search
/Follow-Up-Search 5
/Run-Search
```

Non-interactive equivalent:

```bash
EXO-Hunter --no-animation --script commands.txt
```

### 2.6 Command Reference

Interactive slash commands:

| Command | Action | Required | Optional |
|---|---|---|---|
| `/New-Search <N>` | Select and freeze the best available never-before-searched targets | targets | max-download-gb |
| `/Follow-Up-Search <N>` | Select and freeze the highest-value previously-searched targets | targets | max-download-gb |
| `/Run-Search` | Execute or resume the exact frozen manifest | none | none |
| `/Show-Follow-Ups` | Show follow-up evidence, priority, and next action | none | status |
| `/Inspect-Target <rank-or-id>` | Full identity, score components, provenance, prior-search evidence | rank-or-id | none |
| `/Create-New-Search` | Lower-level creation taking explicit `--targets` and `--mode` | none | none |
| `/Import-Follow-Up` | Import one checksum-verified reviewed prior result | evidence-file | none |
| `/Recheck-Follow-Ups` | Recheck deferred follow-ups for new MAST sectors | none | none |
| `/Help` | Full command surface with parameter shapes | none | none |
| `/Exit` | Close the terminal | none | none |

`/Run-New-Search` is a compatibility alias for `/Run-Search`.

The lower-level creation form takes mode explicitly, which is useful in scripted
operation where the mode is a parameter rather than a chosen command:

```
/Create-New-Search --targets 100 --mode new
/Run-New-Search
```

For `N > 100` the run writes a timestamped complete export, prints a concise
summary and the output path, and preserves the durable non-CSV system of record.

Console scripts registered by `pyproject.toml`:

| Script | Target |
|---|---|
| `EXO-Hunter`, `ExoHunter` | `exo_toolkit.hunter_shell:exohunter_entry` |
| `exo` | `exo_toolkit.cli:cli_entry` |
| `Create-New-Search` | `exo_toolkit.hunter_cli:create_new_search_entry` |
| `Run-New-Search` | `exo_toolkit.hunter_cli:run_new_search_entry` |
| `Show-Follow-Ups` | `exo_toolkit.hunter_cli:show_follow_ups_entry` |
| `Import-Follow-Up` | `exo_toolkit.hunter_cli:import_follow_up_entry` |
| `Inspect-Target` | `exo_toolkit.hunter_cli:inspect_target_entry` |
| `Recheck-Follow-Ups` | `exo_toolkit.hunter_cli:recheck_follow_ups_entry` |
| `prod-check` | `exo_toolkit.prod_check:main_entry` |

Run the production gate:

```bash
prod-check                                    # human-readable
prod-check --json                             # machine-readable report
prod-check --output artifacts/prod_check.json
```

### 2.7 Outputs and Artifacts

| Artifact | Location | Role |
|---|---|---|
| Durable search database | `data/hunter_searches.sqlite3` | System of record (schema v6) |
| Review manifest CSV | `artifacts/manifests/` | Operator review export, not the system of record |
| Acceptance manifests | `artifacts/manifests/hunter_live_acceptance_v*.json` | Retained acceptance evidence |
| Acceptance ledger | `configs/HUNTER_PROD_STATE.json` | Requirement status and evidence |
| Gate report | `prod-check --output <path>` | Versioned machine-readable gate result |
| Golden UX baselines | `tests/golden/` | Semantic interaction assertions |

The five durable record types are the candidate catalog, review manifest, search
run, target search history, and follow-up registry. CSV is an export only.

### 2.8 Exit Codes and Failure Behavior

Hunter commands and the shell:

| Code | Meaning |
|---|---|
| 0 | Success |
| 2 | Invalid input, unknown command, or command failure |
| 130 | Interrupted during a scan |

The background automation module (`exo background-run-once` and related
subcommands) uses a distinct set: `0` success, `20` needs follow-up, `30`
blocked, `40` configuration error, `50` internal error.

`prod-check` exits `0` only when every mandatory requirement passes, and `1`
otherwise.

Failure behavior: incomplete work is never presented as complete. A failed
target records a typed error message and a non-zero exit; the search remains
resumable, and re-running retries only the unfinished portion. Detailed
tracebacks belong in logs rather than in the interactive response.

### 2.9 Troubleshooting

| Symptom | Cause and resolution |
|---|---|
| `ModuleNotFoundError: exo_toolkit` under pytest | Prefix the command with `PYTHONPATH=src` |
| `mypy` reports missing pydantic or numpy imports | Use `.venv/bin/python -m mypy src`; the bare binary resolves a different package path |
| Unknown command in the shell | Enter `/` to open the palette. A token matching no command is a hard error, not a silent success |
| `Invalid - enter a positive whole number.` | The target count must be a positive integer. The value never reaches the canonical layer |
| Lightkurve cache permission error | Set `XDG_CACHE_HOME` to a writable repository-local cache root (§2.3) |

## 3. Analytics, Mathematics, and Theoretical Foundation

### 3.1 Problem Formulation

Two distinct problems are solved.

**Signal interpretation.** Given a detected periodic dip, infer a posterior over
six competing hypotheses: `planet_candidate`, `eclipsing_binary`,
`background_eclipsing_binary`, `stellar_variability`, `instrumental_artifact`,
and `known_object`.

**Target selection.** Given a request for `N` targets and a mode, return the
best available `N` by a deterministic ranking, reporting absolute quality
separately from relative rank. Fewer than `N` are returned only after
demonstrating that fewer valid candidates exist.

### 3.2 Inputs, Outputs, Labels, Units, and Provenance

| Quantity | Symbol | Unit | Source |
|---|---|---|---|
| Time | `t` | days (BJD_TDB) | MAST light curve |
| Normalized flux | `f` | dimensionless | PDCSAP preferred |
| Orbital period | `P` | days | BLS search |
| Transit duration | `T` | hours | BLS and individual-event measurement |
| Transit depth | `d` | ppm | Per-event measurement |
| Stellar radius | `R_star` | solar radii | TIC catalog |
| Stellar mass | `M_star` | solar masses | TIC catalog |
| Effective temperature | `T_eff` | K | TIC catalog |
| False-positive probability | `fpp` | dimensionless in [0,1] | `scoring.compute_scores` |
| Detection confidence | `confidence` | dimensionless in [0,1] | `scoring.compute_scores` |

Every scored candidate carries a `meta` block recording toolkit version, run
timestamp, scorer, git commit, and which features were available or absent.

### 3.3 Mathematical Notation

`log_score_i` is the unnormalized log score of hypothesis `i`; `w_ij` is the
weight of feature `j` under hypothesis `i`; `x_j` in `[0,1]` is a feature score,
or absent; `prior_i` is the prior probability of hypothesis `i`.

### 3.4 Models, Algorithms, and Scores

**Hypothesis scoring.** For each hypothesis,

```
log_score_i = log(prior_i) + sum_j (w_ij * x_j)
posterior   = softmax(log_score)
```

The softmax subtracts the maximum before exponentiating, for numerical
stability. Implementation: `src/exo_toolkit/hypotheses.py` and `scoring.py`.
Validation: `tests/test_hypotheses.py`, `tests/test_scoring.py`.

**Absent features.** A feature whose diagnostic did not run is `None` and
contributes exactly zero to every log score, so absence is neutral rather than
evidence. Diagnostics requiring stellar parameters return `None` when those
parameters are absent, rather than substituting solar values.

**Provenance score.** Data quality in `[0,1]`:

```
provenance = 0.40*cadence_sub + 0.35*sector_sub + 0.25*pipeline_sub
```

`cadence_sub` ramps linearly from 1.0 at 2-minute cadence to 0.0 at 30-minute;
`sector_sub = min(n_sectors/3, 1)`; `pipeline_sub` is 1.00 for SPOC, Kepler, and
K2, 0.85 for QLP, 0.75 for TGLC, and 0.60 for unknown. The `tfop_ready` pathway
requires `provenance >= 0.80`. Implementation:
`fetch.compute_provenance_score`.

**New-target ranking.**

```
score = 70*tic_priority + 20*availability + 10*EIG - storage_cost_penalty
EIG   = tic_priority * availability
```

**Follow-up ranking.**

```
priority = 100*(1 - fpp) + 10*confidence
EIG      = (1 - fpp) * confidence
```

`EIG` is expected information gain, dimensionless in `[0,1]`. Both the
history-derived and registry-derived persistence paths publish this identical
equation. Where the underlying evidence is absent, the metric is `null` rather
than a substituted proxy. Implementation: `src/exo_toolkit/hunter_ranking.py`
and `search_lifecycle.py`. Validation:
`tests/test_search_lifecycle.py::TestRegistryExpectedInformationGainMatchesContract`.

### 3.5 Assumptions, Objectives, and Statistical Methods

- Transit signals are approximately periodic and box-like. The BLS duration grid
  is capped at 90% of the minimum trial period.
- Stellar density consistency uses the central-transit approximation
  `a/R_star = P / (pi*T)` with impact parameter zero, which biases toward larger
  inferred densities for grazing geometries.
- Priors are deliberately conservative: `planet_candidate = 0.10`, each of the
  four astrophysical and instrumental alternatives `0.20`, and
  `known_object = 0.10`. Mission-specific profiles are opt-in through
  `configs/scoring_priors_v0.json`.
- Robust statistics are preferred throughout: median absolute deviation for
  noise, and inverse-variance weighting for depth comparison.

### 3.6 Thresholds, Calibration, and Uncertainty

| Threshold | Value | Role |
|---|---|---|
| `tfop_ready` provenance | >= 0.80 | Pathway gate |
| `known_object_annotation` | posterior >= 0.80 | Pathway gate |
| `github_only_reproducibility` | fpp >= 0.70 | Pathway gate |
| Strict follow-up bar | fpp < 0.15 and confidence > 0.40 | Reported, not a selection gate |
| Depth-scatter saturation | reduced chi-square = 3.0 | Feature saturation |
| Extra-event significance | 3 sigma below out-of-transit median | Event flagging |

Calibration supports Platt scaling and isotonic regression
(`src/exo_toolkit/calibration.py`), applied one-vs-rest per hypothesis and
renormalized to sum to 1. A non-converged or failed Platt fit raises rather than
silently returning the identity transform.

The strict follow-up bar is reported per candidate as
`meets_strict_follow_up_bar`. It does not remove candidates from the selectable
pool, because relative rank and absolute quality are reported separately.

### 3.7 Evaluation and Validation

Metrics implemented in `calibration.py`: Brier score, reliability curves,
precision, recall, F1, and confusion matrix.

Injection-recovery evidence is recorded in
`docs/REPRESENTATION_VARIABILITY_INJECTION_BENCHMARK.md`: 48 targets, 192
trials, with blind BLS recovering 13 of 192.

The strongest evaluated classifier remains the frozen calibrated CNN
(`benchmark_cnn_v1`): test AUC 0.923096, average precision 0.899184, and
top-100 yield 91, against Chronos-Bolt tiny at 0.722778 and Astromer2 at
0.708984 on the same grouped split. Details:
`docs/GROUPED_EXTERNAL_REPRESENTATION_BENCHMARK.md`.

### 3.8 Limitations and Failure Modes

- The sky sweep is a documented sample, not an exhaustive survey: 126 base tiles
  plus a 180-tile expansion ring.
- TIC catalog completeness is poorer for the faint stars this project targets,
  so stellar-parameter-dependent diagnostics are frequently absent.
- `stellar_density_consistency_score` assumes a central transit and is biased
  for grazing geometries.
- Automated variability classes (ASAS-SN `Class`) are machine output, not human
  ground truth, and do not authorize training.
- Metrics are frozen at selection time and are not recomputed on inspection.

### 3.9 Implementation and Test Traceability

| Component | Implementation | Validation |
|---|---|---|
| Schemas | `src/exo_toolkit/schemas.py` | `tests/test_schemas.py` |
| Features | `src/exo_toolkit/features.py` | `tests/test_features.py` |
| Hypotheses | `src/exo_toolkit/hypotheses.py` | `tests/test_hypotheses.py` |
| Scoring | `src/exo_toolkit/scoring.py` | `tests/test_scoring.py` |
| Pathway | `src/exo_toolkit/pathway.py` | `tests/test_pathway.py` |
| Search lifecycle | `src/exo_toolkit/search_lifecycle.py` | `tests/test_search_lifecycle.py` |
| Hunter CLI | `src/exo_toolkit/hunter_cli.py` | `tests/test_hunter_cli.py` |
| Shell | `src/exo_toolkit/hunter_shell.py` | `tests/test_hunter_shell.py` |
| CLI interaction contract | `src/exo_toolkit/hunter_ux.py` | `tests/test_hunter_ux.py` |
| Production gate | `src/exo_toolkit/prod_check.py` | `tests/test_prod_check.py` |

Quality gates:

```bash
.venv/bin/python Skills/run_quality_gates.py     # ten supervised gates, 6x6 topology
PYTHONPATH=src .venv/bin/pytest                  # test suite
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy src
prod-check
```

Recorded result on this tree: **3,297 tests passed and 4 failed**. All four
failures are environmental rather than defects in shipped code. Three
`tests/test_run_quality_gates.py::TestGitState` cases fail because `git init` is
denied inside the sandbox, returning exit status 128, and
`test_outside_repo_cross_project_history_is_rejected_before_db_creation` fails
because pytest's `tmp_path` resolves inside the repository tree, defeating that
test's own out-of-repository premise. Ruff and mypy both pass across 43 source
files.

Coverage claims in this repository must name their denominator. The configured
coverage source is the `exo_toolkit` package only; it does not measure
`Skills/`.

## 4. Sibling Repositories and Shared Data

### 4.1 Research Program and Repository Responsibilities

| Repository | Hunter | Scientific responsibility |
|---|---|---|
| `2026 Exoplanet Research` | EXO-Hunter | Transiting-exoplanet detection, vetting, and candidate ranking |
| `2026 Near Earth Objects` | NEOHunter | Near-Earth-object discovery, orbit resolution, and close-approach ranking |
| `2026 Technosignatures` | TechnoHunter | Technosignature search, RFI rejection, and signal scoring |

The three share one logical interaction architecture and one logical identity
and history contract. They do not share commands, algorithms, datasets, or
maturity labels.

### 4.2 Local Discovery and Configuration

Sibling locations are supplied explicitly per invocation. This repository
publishes no absolute personal path and performs no filesystem search for its
siblings.

Cross-project history is consumed from a verified repository-local copy whose
path is given by `--cross-project-history-path`. `--history-source-root`
overrides the repository-root resolution heuristic when a manifest is read from
outside this checkout.

### 4.3 Shared Artifacts, Ownership, and Access

| Artifact | Owner | Readers | Access |
|---|---|---|---|
| Target search history | Producing Hunter | All three, read-only | Verified repository-local copy |
| Follow-up registry | EXO-Hunter | EXO-Hunter | `data/hunter_searches.sqlite3` |
| Candidate catalog | EXO-Hunter | EXO-Hunter | `data/hunter_searches.sqlite3` |

Each Hunter publishes only the validated records it owns, and consumes sibling
records read-only.

### 4.4 Schemas, Provenance, Versioning, and Compatibility

The durable schema is Hunter SQLite schema v6. Imported history manifests are
checksum-verified before use; a hash mismatch fails closed and no records are
imported.

Decision validity is one of `valid`, `stale-but-usable`, `refresh-required`,
`invalid`, or `unknown`. A target cannot be treated as New when required history
is incomplete, malformed, incompatible, refresh-required, or known to omit newer
records. `stale-but-usable` cannot justify a known-incomplete novelty decision.

The interface contract is documented in
`docs/HUNTER_CROSS_PROJECT_INTERFACE.md`.

### 4.5 Availability, Failure Behavior, and Regeneration

If a sibling history copy is absent, malformed, or fails checksum verification,
the affected command fails closed before creating or mutating any durable
record, and reports the exact defect. It does not proceed with reduced history,
because that would let an already-searched target reappear as New.

Regeneration: re-export the history manifest from the owning repository and
re-import it with `--cross-project-history-path`. Import is idempotent and
append-only; existing records are never rewritten.

### 4.6 Cross-Repository Safety Boundaries

Exactly one repository is writable per session. This repository never modifies,
formats, migrates, commits, branches, pushes, or opens a pull request in a
sibling repository.

Prohibited, and verified absent from `src/` runtime code by the `prod-check`
`sibling_write_isolation` check: runtime imports from sibling repositories,
cross-repository symlinks, hard-coded personal paths, and undocumented
filesystem dependencies.

## License

Apache-2.0. See `LICENSE`.
