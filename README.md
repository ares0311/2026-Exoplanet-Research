# EXO-Hunter — 2026 Exoplanet Research

[![CI](https://github.com/ares0311/2026-Exoplanet-Research/actions/workflows/ci.yml/badge.svg)](https://github.com/ares0311/2026-Exoplanet-Research/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-0.3.9-blue.svg)](pyproject.toml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

EXO-Hunter is a reproducible exoplanet transit-search system for TESS,
Kepler, K2, and JWST photometry. It acquires archive data, cleans light curves,
finds periodic signals, computes interpretable diagnostics, scores competing
astrophysical hypotheses, preserves complete search provenance, and recommends
evidence-based follow-up without making confirmation or discovery claims.

The core application does not require AI. Its default Bayesian path and the
entire Hunter search lifecycle are deterministic, explainable, and testable.
XGBoost, CNN, and ensemble scorers are optional.

## Current production state

Version 0.3.11 is the current release state represented by this repository.

| Area | Status | Current evidence |
|---|---|---|
| EXO-Hunter lifecycle | **PROD accepted** | Best-available-N selection (new and follow-up), adaptive discovery expansion, and live create-through-execution evidence in [`hunter_live_acceptance_v9.json`](artifacts/manifests/hunter_live_acceptance_v9.json) (extends v8/v7; see `docs/HUNTER_PRODUCTION_WORKFLOW.md`) |
| Bayesian scorer | **Production ready** | Default non-ML scorer; no model artifact required |
| XGBoost scorer | **Production ready** | Trained on 7,586 Kepler KOIs; held-out AUC 0.992 |
| XGBoost + Bayesian ensemble | **Production ready** | Conservative fallback when no CNN is supplied |
| CNN scorer | **Production benchmark promoted** | `benchmark_cnn_v1`; validated for its Kepler domain, not unrestricted cross-mission use |
| Full ensemble | **Production ready** | Calibrated weights: XGBoost 0.95, CNN 0.00, Bayesian 0.05 |
| Dataset and model research | **Active, gated** | Manifested, leakage-safe experiments continue; failed model strategies are retained as evidence and not silently reused |
| External submission | **Not implemented or authorized** | The system produces recommendations and evidence packages only |

“PROD accepted” refers to the application and its defined operating contract.
It does not mean that every search must find a candidate, that a transit signal
is a confirmed planet, or that the software may submit an alert automatically.
Scientific null results and ineligible follow-up universes are valid outcomes
when every candidate was evaluated and the reasons are preserved.

The authoritative readiness narrative is
[`docs/PRODUCTION_READINESS.md`](docs/PRODUCTION_READINESS.md). The active
research sequence and completed bounded gates are recorded in
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## Production workflow

```text
candidate universe
→ identity and prior-history resolution
→ eligibility
→ deterministic ranking and selection
→ immutable manifest
→ durable pending search
→ data acquisition and preprocessing
→ signal search, vetting, and scoring
→ composite interpretation
→ append-only results and provenance
→ follow-up registration
→ recommended next action
```

Each stage has an explicit input and output contract. A later command executes
the exact pending manifest; it never regenerates or silently substitutes the
target list. Partial work is reported as partial, failed work remains visible,
and a restart resumes only unfinished or failed targets from that manifest.

## Install

Requirements:

- Python 3.11 or newer
- Git
- [`uv`](https://docs.astral.sh/uv/) for the supported development setup
- Internet access for live NASA/MAST catalog and light-curve retrieval

For a new checkout:

```bash
git clone https://github.com/ares0311/2026-Exoplanet-Research.git
cd 2026-Exoplanet-Research
git switch main
git pull --ff-only origin main
uv sync --all-extras --all-groups
```

For an existing checkout:

```bash
git switch main
git pull --ff-only origin main
uv sync --all-extras --all-groups
```

Verify the installed entry points:

```bash
git switch main
git pull --ff-only origin main
.venv/bin/exo --version
.venv/bin/Create-New-Search --help
.venv/bin/Run-New-Search --help
.venv/bin/Show-Follow-Ups --help
```

## Scan one target

The direct CLI runs fetch → clean → search → vet → score → classify for one
target. It supports TESS, Kepler, K2, and JWST.

```bash
git switch main
git pull --ff-only origin main
caffeinate -i .venv/bin/exo "TIC 150428135" \
  --mission TESS \
  --scorer bayesian \
  --output reports/tic_150428135.json
```

Useful options include:

- `--pipeline SPOC|QLP|TGLC` to constrain the TESS archive product author;
- `--exptime long|short|fast` to constrain cadence;
- `--max-peaks` and `--max-period-grid-points` to bound the BLS search;
- `--no-animation` for logs, CI, and redirected output;
- `--scorer bayesian|xgboost|ensemble|cnn|full-ensemble` to select a scorer.

Model-backed modes require the corresponding `--model-path` and/or
`--cnn-checkpoint`. Cross-mission CNN use fails closed unless explicitly
enabled because prior transfer experiments did not establish general validity.

## Run EXO-Hunter

The Hunter commands use `data/hunter_searches.sqlite3` by default. This ignored
runtime database is the local durable system of record; CSV files are review
exports only.

### 1. Create a new-target search

```bash
git switch main
git pull --ff-only origin main
caffeinate -i .venv/bin/Create-New-Search --targets 100 --mode new
```

The live selector requests at least 10,000 TIC candidates when data permits,
then applies deterministic metadata ranking, prior-search exclusions,
ASAS-SN variable screening, QLP product availability, and estimated data cost.
It freezes eligible and ineligible candidate snapshots with selection reasons.

Searches of at most 100 targets render a terminal table. Larger searches write
a timestamped review CSV under `reports/search_manifests/`; SQLite remains
authoritative in both cases.

### 2. Create a follow-up search

```bash
git switch main
git pull --ff-only origin main
.venv/bin/Create-New-Search --targets 10 --mode follow-up
```

The default command automatically and idempotently imports
[`hunter_prior_search_history_v1.json`](data_selection/hunter_prior_search_history_v1.json)
before ranking. That versioned contract normalizes seven preserved sources,
608 search events, and 200 unique TIC targets without rewriting the source
logs. Follow-up eligibility considers the latest result, prior work, evidence
quality, current registry disposition, revisit policy, and data availability.

If fewer targets qualify than requested, the command fails loudly and creates
no search. It does not replace a scientifically ineligible target with another
target or manufacture a result by rerunning frozen work.

### 3. Execute or resume the exact pending search

```bash
git switch main
git pull --ff-only origin main
caffeinate -i .venv/bin/Run-New-Search --workers 6 --scorer bayesian
```

The parent process owns all SQLite writes while workers independently acquire
and process targets. A partial or failed run returns nonzero. Repeating the
command resumes the same manifest, records interrupted attempts, skips only
terminal target outcomes, and retries unfinished or failed targets. A completed
manifest cannot be executed again.

### 4. Inspect follow-ups

```bash
git switch main
git pull --ff-only origin main
.venv/bin/Show-Follow-Ups --status all
```

Each row includes its target, evidence, reason, priority, prior-search
provenance, current disposition, revisit condition, and recommended action.
Lifecycle states are `open`, `scheduled`, `completed`, and `deferred`, with
append-only transition events and links to the consuming search.

### 5. Import a reviewed prior result

```bash
git switch main
git pull --ff-only origin main
.venv/bin/Import-Follow-Up \
  --evidence-file artifacts/manifests/hunter_reviewed_followup_import_v1.json
```

Import validates source hashes and writes a completed historical manifest,
run, history event, and stable follow-up row in one durable contract. Repeating
the same import is idempotent. Source drift fails before database mutation.

All four Hunter commands support `--json` and `--no-color` for automation.
`Create-New-Search` can also consume a provenance-complete JSON or CSV candidate
file. See the full operator contract in
[`docs/HUNTER_PRODUCTION_WORKFLOW.md`](docs/HUNTER_PRODUCTION_WORKFLOW.md).

## Durable state and provenance

The Hunter database separates concepts that must not be collapsed:

| Durable concept | Storage | Invariant |
|---|---|---|
| Candidate catalog | `candidate_catalog` | Immutable snapshot per candidate and search creation |
| Search manifest | `search_manifests`, `search_manifest_targets` | Exact ordered membership, configuration, selector version, and SHA-256 |
| Search run | `search_runs` | One row per attempt; interruption and partial completion remain explicit |
| Target search history | `target_search_history` | Append-only results, failures, method, source, timestamps, and provenance |
| Search lifecycle | `search_state_events` | Append-only pending/running/interrupted/partial/failed/completed transitions |
| Follow-up registry | `follow_up_registry` | Stable recommendation, disposition, revisit gate, and parent relationship |
| Follow-up lifecycle | `follow_up_events` | Append-only transitions linked to consuming searches |

Completed attempts also append small Git-visible Run Reports under
`artifacts/manifests/run_reports/`. Runtime databases, caches, downloaded
products, reports, and committed evidence have deliberately different
retention policies.

Every completed search preserves, where applicable:

- exact target order and candidate snapshots;
- configuration, selector, pipeline, code, scorer, and model versions;
- archive product URIs and acquisition metadata;
- preprocessing context and cadence counts;
- every signal's diagnostics and scores;
- composite result and conservative interpretation;
- failures and execution state;
- prior-search and follow-up relationships.

## Scientific pipeline

The package implementation lives in `src/exo_toolkit/`:

| Stage | Primary module | Responsibility |
|---|---|---|
| Fetch | `fetch.py` | Mission-aware archive retrieval and raw-product provenance |
| Clean | `clean.py` | finite-value filtering, clipping, normalization, and detrending |
| Search | `search.py` | bounded Box Least Squares search and iterative signal masking |
| Vet | `vet.py` | odd/even, secondary-eclipse, shape, contamination, event timing, duration, missing-event, asymmetry, and extra-event diagnostics |
| Features | `features.py` | normalized, typed scientific features with explicit missingness |
| Score | `scoring.py`, `hypotheses.py`, `ml/` | Bayesian and optional model-backed competing-hypothesis scores |
| Classify | `pathway.py` | ordered conservative recommendation gates |
| Orchestrate | `search_lifecycle.py`, `hunter_cli.py` | deterministic selection, durable execution, recovery, history, and follow-ups |

The scoring output is a candidate assessment, not a validation. The system
exposes positive evidence, false-positive evidence, missing diagnostics, FPP,
detection confidence, and the recommended review pathway. The unchanged
follow-up gate requires FPP below 0.15, detection confidence above 0.40, and an
eligible pathway.

Phase 4's bounded individual-event core now measures event midpoints and
durations plus `missing_transit_fraction`, `transit_asymmetry`, and
`extra_event_count`. These diagnostics help distinguish coherent planetary
events from instrumental, stellar, or ephemeris-mismatch behavior. The exact
scoring definitions and versioned weights are documented in
[`docs/SCORING_MODEL.md`](docs/SCORING_MODEL.md).

## Verified production evidence

The v0.3.8 baseline and current v0.3.9 acceptance artifacts record:

- a live 10,000-candidate new-target universe;
- a completed real follow-up search for TIC 237884073;
- nine exact QLP product URIs and 63,205 processed cadences;
- zero failures in that follow-up execution;
- schema-v5 SQLite integrity, zero foreign-key violations, and 12 append-only
  storage triggers;
- 608 normalized historical events plus existing live events, preserved as
  611 append-only history rows;
- a 202-target combined follow-up universe with an idempotent repeat import;
- a real reviewed recommendation for TIC 355651994 with preserved
  `open → deferred` lifecycle history;
- recomputed hashes for all 10 manifests and 10,610 candidate snapshots;
- byte-level SHA-256 verification of all seven committed history sources;
- all defined launch, selection, execution, provenance, follow-up, recovery,
  no-AI, no-manual-bridge, and no-substitution requirements represented by
  structured evidence rather than self-attested pass strings.

At that acceptance snapshot, zero of the 202 follow-up targets was eligible:
190 were incomplete or below the evidence gate, six had no signal, three had
already been followed up, one was deferred, one had failed, and one had no
data. That is not “there are no scientifically interesting targets in the
sky.” It means none of the currently imported, previously searched targets
meets every rule for scheduling an exact follow-up today. The software can
still create new-target searches, ingest additional reliable history sources,
and schedule a future follow-up when new evidence satisfies its revisit gate.

TIC 355651994 is deferred specifically because currently available MAST
products do not cover a predicted event and the independent-event requirement
is not met. The row remains visible with its evidence and revisit condition;
it is not erased, mislabeled as actionable, or replaced.

## Data sources and research scope

Supported mission paths:

- **TESS:** SPOC, QLP, and other explicitly selected MAST light-curve products;
- **Kepler and K2:** archive light curves, catalog labels, frozen calibration,
  sensitivity, and benchmark roles;
- **JWST:** time-series products converted to white-light curves before transit
  search; spectral series are not passed directly to BLS.

The project uses NASA Exoplanet Archive, MAST, ExoFOP-TESS, and verified public
catalogs. Dataset roles are explicit: training, validation, calibration,
frozen evaluation, live search, or follow-up live search. A live-search dataset
cannot silently become training data, and unlabeled examples are not treated as
negatives.

The exoplanet classifier/ranker is the production path. Exomoon work remains a
separate residual/anomaly-ranking research track because no large confirmed
real-positive exomoon label set exists. The project does not claim a supervised
exomoon detector.

See [`docs/exoplanet_exomoon_dataset_handoff.md`](docs/exoplanet_exomoon_dataset_handoff.md)
for the dataset contract and
[`docs/exoplanet_detection_research_brief.md`](docs/exoplanet_detection_research_brief.md)
for mission, method, literature, and responsible follow-up context.

## Known limits

- The live new-target selector's 126 cone-search tiles cover about 99 square
  degrees, not the whole sky. Achieved coverage is recorded in each selector log.
- Day-one Hunter execution uses archive-extracted light curves. Raw TESS FFI
  photometric extraction is a separate gated phase.
- Public catalog and MAST metadata can change. Reproducibility applies to the
  frozen snapshot and exact provenance, not to future archive responses.
- The committed follow-up history is complete for its seven known source logs,
  not for every search ever performed by every external project.
- `benchmark_cnn_v1` is a Kepler-domain benchmark. Cross-mission transfer is
  not silently assumed.
- The historical `star_scanner.py` run006/run008 scans are preserved evidence,
  not the primary production workflow. `exo background-run-once` exercises
  seven static fixtures and is a CI/automation check, not a discovery engine.
- The system does not confirm planets, publish discoveries, contact authorities,
  or submit candidates without independent human review and authorization.

## Quality and verification

The canonical local gate partitions all default test modules across six pytest
shards with six xdist workers each while Ruff and strict mypy run concurrently.
It also runs the repository's reliability controls and incomplete-implementation
checks.

```bash
git switch main
git pull --ff-only origin main
UV_CACHE_DIR=.uv-cache caffeinate -i \
  .venv/bin/python Skills/run_quality_gates.py
```

Focused tests remain appropriate while diagnosing a change, but a production
claim requires the full current-tree gate. Live-service tests are marked
`integration_live` and excluded from the default suite; committed acceptance
artifacts provide the audited live evidence.

The test philosophy is fail-loud and behavior-first:

- required dependency, schema, input, and archive failures return nonzero;
- partial success never appears complete;
- expected safe fallbacks are explicit and directly tested;
- manifests, histories, and source files are checksum-verified;
- selection is deterministic and exact manifests are immutable;
- restart, interruption, idempotency, and stale-candidate races are tested.

See [`docs/RELIABILITY_CONTROLS.md`](docs/RELIABILITY_CONTROLS.md) for the
agent/repository controls and [`AGENTS.md`](AGENTS.md) for binding contribution
rules.

## Repository map

```text
src/exo_toolkit/       installable scientific pipeline and Hunter lifecycle
tests/                 offline unit, integration, contract, and recovery tests
Skills/                bounded acquisition, processing, evaluation, and QA tools
docs/                  readiness, runbooks, methods, policies, and research plans
data_selection/        versioned candidate/history/dataset contracts and decisions
metadata/              frozen source, benchmark, and schema contracts
models/                promoted model and calibration artifacts
artifacts/manifests/   committed acceptance evidence and Run Reports
reports/               generated local review output and manifest CSV exports
data/                  ignored runtime databases and working data
```

Start with these documents:

1. [`docs/PRODUCTION_READINESS.md`](docs/PRODUCTION_READINESS.md) — current
   acceptance, scorer status, and evidence.
2. [`docs/HUNTER_PRODUCTION_WORKFLOW.md`](docs/HUNTER_PRODUCTION_WORKFLOW.md) —
   operator contract and durable lifecycle semantics.
3. [`docs/DISCOVERY_RUNBOOK.md`](docs/DISCOVERY_RUNBOOK.md) — live-search and
   responsible review procedure.
4. [`docs/ROADMAP.md`](docs/ROADMAP.md) — completed and next research gates.
5. [`docs/astrometrics_data_selection_policy.md`](docs/astrometrics_data_selection_policy.md)
   and [`docs/astrometrics_external_and_cloud_storage_policy.md`](docs/astrometrics_external_and_cloud_storage_policy.md)
   — data roles, download limits, cache, external-drive, and retention rules.

## Scientific and operational guardrails

- Never call a transit-like signal a confirmed planet without authoritative
  external confirmation.
- Never silently replace missing data, targets, model outputs, or failed work.
- Preserve every search event, failure, source, timestamp, and follow-up link.
- Keep frozen evaluation and live-search data isolated from model training.
- Verify public resource schemas before use; do not guess renamed fields or
  substitute mirrors.
- Estimate storage before acquisition and preserve enough provenance to
  reproduce or re-download exact products.
- Treat external submission, alerts, and authority-facing communication as
  separate human-approved actions.

## License

Licensed under the [Apache License 2.0](LICENSE).
